"""Optuna optimization using combined out-of-fold XGBoost predictions."""

from __future__ import annotations

import logging
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AutoMLConfig
from .data import PreparedData
from .exceptions import TrainingError, TrialTrainingError
from .metrics import MetricSpec, is_better, score_predictions, worst_score
from .modeling import (
    build_search_state,
    create_model,
    predict_for_score,
    refine_search_state,
    suggest_from_state,
    suggest_parameters,
)
from .recording import TrialRecorder


def _actual_target(data: PreparedData, config: AutoMLConfig) -> np.ndarray:
    """Return the target array CV scores are computed against.

    For regression the original target is used (predictions are inverted to the
    original scale by ``cross_validated_predictions``); for classification the
    0/1 encoded target is used.
    """
    if config.task == "regression":
        return data.target.to_numpy(dtype=float)
    return data.encoded_target.to_numpy(dtype=float)


# Number of consecutive failed feature-removal attempts tolerated before the
# importance-based backward elimination stops. A failure does not abort the
# search; the next least-important feature is tried instead.
ELIMINATION_FAILURE_PATIENCE = 3


@dataclass(slots=True)
class OptimizationOutcome:
    study: Any
    best_parameters: dict[str, Any]
    best_score: float
    best_trial_number: int
    selected_features: list[str] | None = None
    recorder_rows: pd.DataFrame | None = None
    studies_history: list[tuple[str, int, Any]] | None = None


def _format_params(parameters: dict[str, Any]) -> str:
    return " ".join(
        f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in sorted(parameters.items())
    )


def _format_search_bounds(search_state: dict[str, dict[str, Any]]) -> str:
    return " ".join(
        f"{key}=[{spec['low']:.6g},{spec['high']:.6g}]"
        for key, spec in sorted(search_state.items())
    )


def create_study(config: AutoMLConfig, metric: MetricSpec, run_dir: Path, run_id: str) -> tuple[Any, Path | None]:
    import optuna

    database: Path | None = None
    storage = config.optuna.storage
    if storage is None:
        database = run_dir / "optuna.db"
        storage = f"sqlite:///{database.resolve()}"
    study = optuna.create_study(
        study_name=config.optuna.study_name or f"automl_{run_id}",
        storage=storage,
        load_if_exists=config.optuna.load_if_exists,
        direction=metric.direction,
        sampler=optuna.samplers.TPESampler(seed=config.seed),
    )
    return study, database


def cross_validated_predictions(
    data: PreparedData,
    config: AutoMLConfig,
    parameters: dict[str, Any],
    selected_features: list[str] | None = None,
) -> np.ndarray:
    """Fit one model per shared fold and return predictions in original row order.

    Predictions are returned in the original target scale: when a regression
    target transformation chain is configured, the model is fit on the transformed
    target but the holdout predictions are inverted back to the original scale
    so every CV score is computed against the original actuals.
    """
    from .transforms import invert_transform_chain

    features = data.features if selected_features is None else data.features[selected_features]
    predictions = np.full(len(features), np.nan, dtype=float)
    for fold in sorted(data.fold_codes.unique()):
        train_mask = data.fold_codes.to_numpy() != fold
        holdout_mask = ~train_mask
        model = create_model(config, parameters, selected_features)
        model.fit(
            features.loc[train_mask],
            data.encoded_target.loc[train_mask],
            verbose=False,
        )
        predictions[holdout_mask] = predict_for_score(
            model,
            features.loc[holdout_mask],
            config.task,
        )
    if not np.isfinite(predictions).all():
        raise TrainingError("cross-validation did not produce one finite prediction per row")
    if config.task == "regression" and data.target_transforms:
        predictions = invert_transform_chain(predictions, data.target_transforms)
    return predictions


def optimize(
    study: Any,
    data: PreparedData,
    config: AutoMLConfig,
    metric: MetricSpec,
    run_id: str,
    logger: logging.Logger,
    selected_features: list[str] | None = None,
    seed_params: dict[str, Any] | None = None,
    search_state: dict[str, dict[str, Any]] | None = None,
    recorder: TrialRecorder | None = None,
    round_name: str = "round1",
    iteration: int = 1,
    n_trials: int | None = None,
) -> OptimizationOutcome:
    import optuna

    if seed_params is not None:
        study.enqueue_trial(dict(seed_params))
        logger.info("round1 seeded with incumbent params=%s", _format_params(seed_params))

    n_features = len(selected_features) if selected_features is not None else len(config.predictors)
    max_selected = config.max_selected_features

    def _apply_feature_penalty(score: float) -> float:
        if max_selected is None or n_features <= max_selected:
            return score
        excess = n_features - max_selected
        multiplier = excess + 1
        if metric.direction == "minimize":
            return score * multiplier
        return score / multiplier

    def objective(trial: Any) -> float:
        parameters = suggest_from_state(trial, search_state) if search_state is not None else suggest_parameters(trial)
        trial.set_user_attr("algorithm", "xgboost")
        trial.set_user_attr("normalized_parameters", parameters)
        started = time.perf_counter()
        logger.info(
            "trial_started params=%s",
            _format_params(parameters),
            extra={"run_id": run_id, "trial_number": trial.number},
        )
        try:
            predictions = cross_validated_predictions(data, config, parameters, selected_features)
            score = score_predictions(
                metric,
                _actual_target(data, config),
                predictions,
            )
            trial.set_user_attr("cv_score", score)
            penalized = _apply_feature_penalty(score)
            trial.set_user_attr("penalized_score", penalized)
            if recorder is not None:
                recorder.add(
                    trial,
                    round_name=round_name,
                    iteration=iteration,
                    raw_score=score,
                    penalized_score=penalized,
                    selected_features=selected_features,
                )
            logger.info(
                "trial_completed raw_score=%.6f penalized_score=%.6f duration=%.2fs",
                score,
                penalized,
                time.perf_counter() - started,
                extra={"run_id": run_id, "trial_number": trial.number},
            )
            return penalized
        except Exception as exc:
            trial.set_user_attr("failure_reason", str(exc)[:2000])
            if recorder is not None:
                recorder.add(
                    trial,
                    round_name=round_name,
                    iteration=iteration,
                    raw_score=None,
                    penalized_score=None,
                    selected_features=selected_features,
                )
            logger.exception("trial_failed", extra={"run_id": run_id, "trial_number": trial.number})
            raise TrialTrainingError(str(exc)) from exc

    trial_count = n_trials if n_trials is not None else config.n_trials
    logger.info(
        "study started: n_trials=%d timeout=%s n_features=%d",
        trial_count, config.timeout_seconds or "none", n_features,
    )
    study.optimize(
        objective,
        n_trials=trial_count,
        timeout=config.timeout_seconds,
        n_jobs=1,
        catch=(TrialTrainingError,),
    )
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        logger.info("study finished: 0/%d trials completed", len(study.trials))
        raise TrainingError("no Optuna trial completed successfully")
    best = study.best_trial
    parameters = dict(best.user_attrs.get("normalized_parameters") or {})
    if best.value is None or not parameters:
        raise TrainingError("best trial is missing its score or normalized parameters")
    raw_score = best.user_attrs.get("cv_score", best.value)
    logger.info(
        "study finished: %d/%d trials completed, best=%d raw=%.6f penalized=%.6f params=%s",
        len(complete),
        len(study.trials),
        best.number,
        float(raw_score),
        float(best.value),
        _format_params(parameters),
    )
    outcome = OptimizationOutcome(study, parameters, float(raw_score), best.number)
    if recorder is not None:
        outcome.recorder_rows = recorder.dataframe()
    return outcome


def refit_best(
    outcome: OptimizationOutcome,
    data: PreparedData,
    config: AutoMLConfig,
    metric: MetricSpec,
) -> tuple[Any, np.ndarray, float]:
    """Recompute winning OOF predictions and train one model on every row."""

    selected = outcome.selected_features
    oof = cross_validated_predictions(data, config, outcome.best_parameters, selected)
    score = score_predictions(metric, _actual_target(data, config), oof)
    if not math.isclose(
        score,
        outcome.best_score,
        rel_tol=config.refit_score_relative_tolerance,
        abs_tol=config.refit_score_absolute_tolerance,
    ):
        raise TrainingError(
            f"recomputed OOF score differs from best trial: {score} vs {outcome.best_score}"
        )
    features = data.features if selected is None else data.features[selected]
    model = create_model(config, outcome.best_parameters, selected)
    model.fit(features, data.encoded_target, verbose=False)
    return model, oof, score


def run_feature_selection(
    data: PreparedData,
    config: AutoMLConfig,
    metric: MetricSpec,
    best_parameters: dict[str, Any],
    base_score: float,
    all_features: list[str],
    active_features: list[str],
    iteration: int,
    logger: logging.Logger,
    shared_study: Any | None = None,
    recorder: TrialRecorder | None = None,
) -> tuple[list[str], float]:
    import optuna

    n_trials = config.feature_selection_trials or config.n_trials
    max_selected = config.max_selected_features
    feature_flag_keys = {
        f"feat__{idx:03d}": name for idx, name in enumerate(all_features)
    }

    if shared_study is not None:
        study = shared_study
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*experimental.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*experimental.*", module="optuna")
            sampler = optuna.samplers.TPESampler(
                seed=config.seed + 1000 * iteration,
                multivariate=True,
            )
        study = optuna.create_study(direction=metric.direction, sampler=sampler)

    trials_before = len(study.trials)

    active_set = set(active_features)
    anchor_on = {
        key: 1 if feat_name in active_set else 0
        for key, feat_name in feature_flag_keys.items()
    }
    study.enqueue_trial(anchor_on)
    logger.info(
        "round2 anchored on %d/%d active features: %s",
        len(active_set), len(all_features), ",".join(active_features),
    )

    anchor_trial_number = trials_before

    def objective(trial: optuna.trial.Trial) -> float:
        started = time.perf_counter()
        is_anchor = trial.number == anchor_trial_number
        selected = []
        for flag_key, feat_name in feature_flag_keys.items():
            if trial.suggest_int(flag_key, 0, 1) == 1:
                selected.append(feat_name)

        cap_violation = 0.0
        if max_selected is not None:
            cap_violation = max(0.0, float(len(selected) - max_selected))
        trial.set_user_attr("cap_violation", cap_violation)
        trial.set_user_attr("selected_features", selected)

        if not selected:
            if recorder is not None:
                recorder.add(
                    trial,
                    round_name="round2",
                    iteration=iteration,
                    raw_score=None,
                    penalized_score=worst_score(metric.direction),
                    selected_features=selected,
                )
            logger.info(
                "feature selection trial %d: empty subset, skipped (worst score)",
                trial.number,
            )
            return worst_score(metric.direction)

        if is_anchor:
            raw_score = base_score
        else:
            try:
                predictions = cross_validated_predictions(data, config, best_parameters, selected)
                raw_score = score_predictions(
                    metric,
                    _actual_target(data, config),
                    predictions,
                )
            except Exception as exc:
                if recorder is not None:
                    recorder.add(
                        trial,
                        round_name="round2",
                        iteration=iteration,
                        raw_score=None,
                        penalized_score=worst_score(metric.direction),
                        selected_features=selected,
                    )
                logger.warning(
                    "feature selection trial %d failed (%d features): %s",
                    trial.number, len(selected), exc,
                )
                return worst_score(metric.direction)

        score = raw_score
        if cap_violation > 0.0:
            penalty_multiplier = int(cap_violation) + 1
            if metric.direction == "minimize":
                score = raw_score * penalty_multiplier
            else:
                score = raw_score / penalty_multiplier
        if recorder is not None:
            recorder.add(
                trial,
                round_name="round2",
                iteration=iteration,
                raw_score=raw_score,
                penalized_score=score,
                selected_features=selected,
            )
        logger.info(
            "feature selection trial %d: %d/%d features raw=%.6f scored=%.6f "
            "excess=%.0f duration=%.2fs selected=%s",
            trial.number,
            len(selected),
            len(all_features),
            raw_score,
            score,
            cap_violation,
            time.perf_counter() - started,
            ",".join(selected),
        )
        return score

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    if shared_study is not None:
        logger.info(
            "feature selection iteration %d: %d new trials, %d total in shared study",
            iteration, len(study.trials) - trials_before, len(study.trials),
        )

    complete = [
        t for t in study.trials[trials_before:]
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and math.isfinite(t.value)
    ]
    logger.info(
        "feature selection round2 iteration %d finished: %d/%d new scored trials",
        iteration,
        len(complete),
        len(study.trials) - trials_before,
    )

    if not complete:
        logger.info(
            "feature-selection iteration %d: no scored trials, keeping %d active features",
            iteration, len(active_features),
        )
        return list(active_features), base_score

    best_trial = min(complete, key=lambda t: t.value) if metric.direction == "minimize" else max(complete, key=lambda t: t.value)
    best_selected = best_trial.user_attrs.get("selected_features", active_features)
    if not best_selected:
        logger.info(
            "feature selection iteration %d: winning subset empty, keeping %d active features",
            iteration, len(active_features),
        )
        return list(active_features), base_score

    logger.info(
        "feature selection iteration %d winner: trial=%d score=%.6f features=%s",
        iteration, best_trial.number, float(best_trial.value), ",".join(best_selected),
    )
    return list(best_selected), float(best_trial.value)


def run_round1b_elimination(
    data: PreparedData,
    config: AutoMLConfig,
    metric: MetricSpec,
    best_parameters: dict[str, Any],
    base_score: float,
    active_features: list[str],
    iteration: int,
    logger: logging.Logger,
    recorder: TrialRecorder | None = None,
) -> tuple[list[str], float]:
    """Backward feature elimination driven by XGBoost variable importance.

    Runs as "round1b" for the first ``config.round1b_iterations`` iterations,
    right after Round 1 and before the Optuna-based Round 2. Fits a model with
    the winning parameters on all active predictors, ranks them by
    fitted-model importance (most -> least), and tries dropping them in order
    of increasing importance (least important first). Each candidate subset is
    scored via the shared cross-validation folds. A drop is accepted when the
    score is the same (within tolerance) or better than the current score.
    A drop that degrades the score does NOT stop the search: that feature is
    kept and the next least-important feature is tried instead. The search
    stops only after ``ELIMINATION_FAILURE_PATIENCE`` consecutive failures, so
    a later (more important) feature can still be dropped if doing so improves
    the score. Every feature dropped here is permanently removed from the
    candidate pool for all subsequent iterations.
    """
    import types

    ranking_model = create_model(config, best_parameters, active_features)
    ranking_model.fit(
        data.features[active_features], data.encoded_target, verbose=False
    )
    importances = np.asarray(
        getattr(ranking_model, "feature_importances_", []), dtype=float
    )
    if importances.shape != (len(active_features),) or not np.isfinite(importances).all():
        raise TrainingError("XGBoost returned invalid feature importances for selection")

    # Rank most -> least important; tie-break by name (matches variable_importance.csv).
    ranked = [
        name for _, name in sorted(
            zip(importances, active_features),
            key=lambda pair: (-float(pair[0]), pair[1]),
        )
    ]
    # Removal order: least important first (the "last variable of the variable
    # importance" at each step). The importance ranking is computed once from the
    # Round 1 model and reused for every elimination step.
    removal_order = list(reversed(ranked))
    logger.info(
        "round2 importance ranking (iteration %d, %d features): %s",
        iteration, len(ranked), ",".join(ranked),
    )

    # Keep the surviving features in their original active_features order so that
    # the OOF score recomputed in refit_best matches the Round 1 score (XGBoost
    # column sampling by index is order-sensitive).
    current_features = list(active_features)
    current_score = float(base_score)
    rel_tol = config.refit_score_relative_tolerance
    abs_tol = config.refit_score_absolute_tolerance

    def _same_or_better(new: float, reference: float) -> bool:
        return is_better(new, reference, metric.direction) or math.isclose(
            new, reference, rel_tol=rel_tol, abs_tol=abs_tol
        )

    step = 0
    consecutive_failures = 0
    while (
        len(current_features) > 1
        and step < len(removal_order)
        and consecutive_failures < ELIMINATION_FAILURE_PATIENCE
    ):
        removed = removal_order[step]
        candidate = [feature for feature in current_features if feature != removed]
        started = time.perf_counter()
        try:
            predictions = cross_validated_predictions(
                data, config, best_parameters, candidate
            )
            score = score_predictions(
                metric, _actual_target(data, config), predictions
            )
        except Exception as exc:
            logger.warning(
                "round1b elimination iteration %d step %d failed removing %s: %s",
                iteration, step, removed, exc,
            )
            if recorder is not None:
                recorder.add(
                    trial=types.SimpleNamespace(number=step),
                    round_name="round1b",
                    iteration=iteration,
                    raw_score=None,
                    penalized_score=worst_score(metric.direction),
                    selected_features=candidate,
                )
            # Treat a training failure like a degradation: keep the feature,
            # count it toward the patience budget, and try the next candidate.
            consecutive_failures += 1
            step += 1
            continue

        accepted = _same_or_better(score, current_score)
        logger.info(
            "round1b elimination iteration %d step %d: removed=%s candidate=%d features "
            "score=%.6f previous=%.6f accepted=%s consecutive_failures=%d duration=%.2fs",
            iteration, step, removed, len(candidate), score, current_score, accepted,
            consecutive_failures + (0 if accepted else 1),
            time.perf_counter() - started,
        )
        if recorder is not None:
            recorder.add(
                trial=types.SimpleNamespace(number=step),
                round_name="round1b",
                iteration=iteration,
                raw_score=score,
                penalized_score=score,
                selected_features=candidate,
            )
        if accepted:
            current_features = candidate
            current_score = score
            consecutive_failures = 0
        else:
            # Degrade: keep `removed`, do not stop -- try the next least-important
            # feature. Stop only after ELIMINATION_FAILURE_PATIENCE consecutive
            # failures.
            consecutive_failures += 1
        step += 1

    logger.info(
        "round1b elimination iteration %d finished: %d features score=%.6f "
        "steps_tried=%d consecutive_failures=%d selected=%s",
        iteration, len(current_features), current_score, step, consecutive_failures,
        ",".join(current_features),
    )
    return current_features, current_score


def run_bottom_up_reduction(
    data: PreparedData,
    config: AutoMLConfig,
    metric: MetricSpec,
    best_parameters: dict[str, Any],
    active_features: list[str],
    base_score: float,
    logger: logging.Logger,
) -> tuple[list[str], float]:
    """Bottom-up variable reduction via an XGBoost importance threshold.

    Fits a model with the winning parameters on the active predictors, reads
    ``feature_importances_`` (normalized to sum to 1), drops every predictor
    whose importance is below the effective threshold, then rebuilds via the
    shared cross-validation folds on the survivors and returns the survivors
    together with their CV score. The effective threshold is
    ``config.bottom_up_importance_threshold`` when an explicit value is set;
    otherwise it is computed dynamically from the number of active predictors
    as ``(1 / n_predictors) / 2`` (half the average importance). If every
    predictor falls below the threshold the reduction is a no-op (all features
    kept).
    """
    ranking_model = create_model(config, best_parameters, active_features)
    ranking_model.fit(
        data.features[active_features], data.encoded_target, verbose=False
    )
    importances = np.asarray(
        getattr(ranking_model, "feature_importances_", []), dtype=float
    )
    if importances.shape != (len(active_features),) or not np.isfinite(importances).all():
        raise TrainingError("XGBoost returned invalid feature importances for bottom-up reduction")

    importance_by_feature = dict(zip(active_features, importances))
    # Dynamic default: half the average importance when no explicit threshold set.
    if config.bottom_up_importance_threshold is not None:
        threshold = float(config.bottom_up_importance_threshold)
    else:
        threshold = (1.0 / len(active_features)) / 2.0 if active_features else 0.0
    survivors = [feature for feature in active_features if importance_by_feature[feature] >= threshold]
    dropped = [feature for feature in active_features if importance_by_feature[feature] < threshold]
    if not survivors:
        logger.warning(
            "bottom-up reduction: all %d features below threshold %.6f; keeping all features",
            len(active_features), threshold,
        )
        return list(active_features), float(base_score)
    if not dropped:
        logger.info(
            "bottom-up reduction: no feature below threshold %.6f; keeping all %d features",
            threshold, len(active_features),
        )
        return list(active_features), float(base_score)

    predictions = cross_validated_predictions(data, config, best_parameters, survivors)
    score = score_predictions(metric, _actual_target(data, config), predictions)
    logger.info(
        "bottom-up reduction: dropped %d/%d features below threshold %.6f: %s",
        len(dropped), len(active_features), threshold, ",".join(dropped),
    )
    logger.info(
        "bottom-up reduction: %d survivors score=%.6f selected=%s",
        len(survivors), score, ",".join(survivors),
    )
    return survivors, score


# Modest default hyper-parameters for the multimodel reduction models.
# These models exist only to produce SHAP-based variable importance, not to be
# the final predictor, so a shallow, fast configuration is used.
_MULTIMODEL_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
}


def _build_variable_groups(
    predictors: list[str],
    n_models: int,
    appearances: int,
    seed: int,
) -> list[list[str]]:
    """Split predictors into ``n_models`` random, low-overlap groups.

    Each predictor appears in approximately ``appearances`` groups (within
    ±1) and every group has size ``len(predictors) * appearances // n_models``
    (±1 when the totals do not divide evenly). Predictors are assigned one
    at a time (in random order) to the `appearances` groups with the most
    remaining capacity that do not already contain the predictor; this balances
    group filling (no group fills too early and starves late predictors),
    guarantees no group contains a duplicate predictor, and keeps the
    per-model group size balanced (within ±1) and the per-predictor
    appearance count approximately `appearances`. Because the demand
    (n_predictors * appearances) equals the supply (sum of group
    capacities), a feasible assignment always exists; the greedy is retried
    with fresh shuffles (seeded deterministically) until it finds one, so the
    result is deterministic. Pairwise overlap between groups is random in
    expectation, which is the "as stratified as possible" given the counts.
    """
    if n_models > len(predictors) * appearances:
        raise ValueError(
            f"multimodel_n_models({n_models}) cannot exceed the number of "
            f"predictor appearances ({len(predictors) * appearances})"
        )
    if appearances > n_models:
        raise ValueError(
            f"multimodel_variable_appearances({appearances}) cannot exceed "
            f"multimodel_n_models({n_models})"
        )

    total_slots = len(predictors) * appearances
    base_size = total_slots // n_models
    remainder = total_slots % n_models
    # Distribute the remainder so group sizes vary by at most 1.
    capacities = [
        base_size + (1 if index < remainder else 0) for index in range(n_models)
    ]

    def _attempt(attempt_seed: int) -> list[list[str]] | None:
        rng = np.random.default_rng(attempt_seed)
        # Process predictors in random order; shuffle so no predictor
        # is systematically favored for the early (still-empty) groups.
        order = list(predictors)
        rng.shuffle(order)
        groups: list[list[str]] = [[] for _ in range(n_models)]
        remaining_capacity = list(capacities)
        for predictor in order:
            need = appearances
            # Pick `need` groups with the most remaining capacity that do
            # not already contain this predictor. Ties in capacity are
            # broken randomly so the search is diversified across attempts.
            chosen: list[int] = []
            for _ in range(need):
                eligible = [
                    index for index in range(n_models)
                    if remaining_capacity[index] > 0
                    and index not in set(chosen)
                    and predictor not in set(groups[index])
                ]
                if not eligible:
                    return None
                # Prefer the groups with the most remaining capacity.
                max_cap = max(remaining_capacity[index] for index in eligible)
                top = [index for index in eligible if remaining_capacity[index] == max_cap]
                pick = top[int(rng.integers(len(top)))]
                chosen.append(pick)
            for index in chosen:
                groups[index].append(predictor)
                remaining_capacity[index] -= 1
        if sum(remaining_capacity) == 0:
            return groups
        return None

    for offset in range(200):
        groups = _attempt(seed + offset * 7919)
        if groups is not None:
            return groups
    raise ValueError(
        "multimodel grouping failed to find a feasible assignment after 200 "
        "attempts; use a different combination of models/appearances/"
        "predictors"
    )


def _compute_shap_importance(
    model: Any,
    data: PreparedData,
    group: list[str],
    config: AutoMLConfig,
) -> dict[str, float]:
    """Mean absolute SHAP value per predictor for one fitted model.

    SHAP is computed on a sample of the training rows (controlled by
    ``explain_sample_size``) to keep the multimodel stage affordable.
    Returns a mapping ``{predictor: mean_abs_shap}`` for the predictors
    in ``group`` (predictors not in the group are absent).
    """
    import shap

    sample_size = config.explain_sample_size or len(data.features)
    if len(data.features) > sample_size:
        rng = np.random.default_rng(config.seed)
        indices = np.sort(rng.choice(len(data.features), size=sample_size, replace=False))
    else:
        indices = np.arange(len(data.features))
    background = data.features.iloc[indices][group]
    try:
        explainer = shap.TreeExplainer(model, model_output="raw")
        explanation = explainer(background)
    except Exception as exc:
        raise TrainingError(f"SHAP TreeExplainer failed in multimodel stage: {exc}") from exc
    values = np.asarray(explanation.values)
    if values.ndim == 3 and values.shape[-1] == 2:
        values = values[:, :, 1]
    if values.shape != (len(background), len(group)):
        raise TrainingError(
            f"unexpected SHAP shape {values.shape} in multimodel stage; "
            f"expected {(len(background), len(group))}"
        )
    if not np.isfinite(values).all():
        raise TrainingError("SHAP returned non-finite values in multimodel stage")
    return {
        feature: float(np.mean(np.abs(values[:, column])))
        for column, feature in enumerate(group)
    }


def run_multimodel_variable_reduction(
    data: PreparedData,
    config: AutoMLConfig,
    metric: MetricSpec,
    run_dir: Path,
    run_id: str,
    logger: logging.Logger,
) -> tuple[list[str], list[str], dict[str, float]]:
    """Multimodel variable reduction run at the start of the pipeline.

    Splits the predictors into ``multimodel_n_models`` random, low-overlap
    groups (each predictor appears in
    ``multimodel_variable_appearances`` groups), fits one XGBoost
    model per group, computes SHAP-based variable importance for each model
    and saves it, averages importance across the models, excludes the
    bottom ``multimodel_exclude_bottom`` predictors (an int count or a
    float bottom fraction), and splits the survivors into the top
    ``multimodel_top_k`` (available from iteration 1) and the rest
    (held out of iteration 1 and introduced in the Round 2 candidate pool
    of iteration 2). Returns ``(top_k_features, rest_features, importance)``
    where ``importance`` maps every predictor to its average importance.
    The excluded bottom predictors are permanently removed from the
    pipeline (the caller adds them to ``eliminated_by_round1b``).
    """
    predictors = list(config.predictors)
    n_models = config.multimodel_n_models
    appearances = config.multimodel_variable_appearances
    groups = _build_variable_groups(predictors, n_models, appearances, config.seed)
    logger.info(
        "multimodel variable reduction: %d models, each predictor in %d/%d groups "
        "(group size = %d)",
        n_models, appearances, n_models, len(groups[0]) if groups else 0,
    )

    importance_rows: list[dict[str, Any]] = []
    per_model_importance: list[dict[str, float]] = []
    for index, group in enumerate(groups):
        started = time.perf_counter()
        model = create_model(config, dict(_MULTIMODEL_DEFAULT_PARAMS), group)
        model.fit(data.features[group], data.encoded_target, verbose=False)
        importance = _compute_shap_importance(model, data, group, config)
        per_model_importance.append(importance)
        importance_rows.append({
            "model": index,
            "group_size": len(group),
            "group": ",".join(group),
            **{feature: importance.get(feature, 0.0) for feature in predictors},
        })
        logger.info(
            "multimodel model %d/%d fitted on %d features in %.1fs",
            index + 1, n_models, len(group), time.perf_counter() - started,
        )

    # Average importance across the n_models. Predictors that never appeared
    # in a group contribute 0 for the models they were absent from, so
    # dividing by n_models is the straightforward average across models.
    avg_importance: dict[str, float] = {
        feature: sum(model_importance.get(feature, 0.0) for model_importance in per_model_importance) / n_models
        for feature in predictors
    }

    # Persist per-model and average importance.
    importance_frame = pd.DataFrame(importance_rows)
    importance_path = run_dir / "multimodel_importance.csv"
    temporary = importance_path.with_name(f".{importance_path.name}.tmp")
    try:
        importance_frame.to_csv(temporary, index=False)
        temporary.replace(importance_path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise TrainingError(f"could not write {importance_path.name}: {exc}") from exc
    logger.info(
        "multimodel importance written to %s (%d rows)",
        importance_path, len(importance_frame),
    )

    # Exclude the bottom N predictors (count or fraction).
    exclude_count = config._multimodel_exclude_count(len(predictors))
    ranked = sorted(predictors, key=lambda feature: (-avg_importance[feature], feature))
    bottom_n = ranked[-exclude_count:] if exclude_count else []
    survivors = [feature for feature in ranked if feature not in set(bottom_n)]
    top_k = survivors[: config.multimodel_top_k]
    rest = [feature for feature in survivors if feature not in set(top_k)]
    logger.info(
        "multimodel variable reduction finished: excluded %d bottom predictors, "
        "survivors=%d (top_k=%d for iteration 1, rest=%d for iteration 2 round 2)",
        len(bottom_n), len(survivors), len(top_k), len(rest),
    )
    if bottom_n:
        logger.info(
            "multimodel permanently excluded bottom %d predictors: %s",
            len(bottom_n), ",".join(sorted(bottom_n)),
        )
    logger.info(
        "multimodel top %d for iteration 1: %s",
        len(top_k), ",".join(top_k),
    )
    if rest:
        logger.info(
            "multimodel rest %d (introduced in iteration 2 round 2): %s",
            len(rest), ",".join(rest),
        )
    return top_k, rest, avg_importance


def run_iterative_optimization(
    data: PreparedData,
    config: AutoMLConfig,
    metric: MetricSpec,
    run_dir: Path,
    run_id: str,
    logger: logging.Logger,
) -> OptimizationOutcome:
    import optuna

    active_features: list[str] = list(config.predictors)
    best_params: dict[str, Any] | None = None
    best_score: float | None = None
    best_trial_number: int = 0
    best_features: list[str] = list(config.predictors)
    last_study: Any = None
    prev_iter_best_params: dict[str, Any] | None = None
    max_selected = config.max_selected_features
    recorder = TrialRecorder(config)
    trials_history: list[tuple[str, int, Any]] = []

    # Features permanently removed by the round1b elimination (first iteration
    # only). They are excluded from the Round 2 candidate pool in every
    # subsequent iteration so they can never be re-selected. Bottom-up
    # dropped features are NOT added here (they stay selectable).
    eliminated_by_round1b: set[str] = set()

    # Multimodel-reduction "rest" predictors: held out of iteration 1
    # and released (introduced into the Round2 candidate pool) at the
    # end of iteration 1, i.e. they become selectable in iteration 2.
    held_from_iteration1: set[str] = set()

    # Wall-clock budget for the whole optimization. Checked after each
    # iteration completes; remaining iterations are skipped when exceeded.
    run_started_perf = time.perf_counter()

    # Multimodel variable reduction runs once at the very start of the
    # pipeline (before iteration 1). It permanently excludes the bottom-N
    # predictors and splits the survivors into top_k (active from iteration 1)
    # and rest (introduced in iteration 2 round 2).
    if config.multimodel_variable_reduction:
        logger.info("multimodel variable reduction stage started")
        top_k, rest, importance = run_multimodel_variable_reduction(
            data, config, metric, run_dir, f"{run_id}_multimodel", logger,
        )
        eliminated_by_round1b |= set(config.predictors) - set(top_k) - set(rest)
        active_features = list(top_k)
        held_from_iteration1 = set(rest)
        logger.info(
            "multimodel stage set iteration 1 active features=%d (top_k), "
            "rest=%d held for iteration 2 round 2, permanently excluded=%d",
            len(active_features), len(rest), len(eliminated_by_round1b),
        )

    r2_shared_study: Any | None = None
    # Round 1 study carried across iterations. When the previous
    # iteration's Round 2 did NOT improve the score, the next iteration
    # continues this study (preserving TPE sampler state and trial
    # history) instead of starting a fresh one. When Round 2 improved
    # (or on iteration 1), a new Round 1 study is created.
    round1_study: Any | None = None
    prev_round2_won = False
    search_state: dict[str, dict[str, Any]] | None = None
    if config.iterations > 1:
        search_state = build_search_state()
        logger.info("built search state for round1 refinement (refine_factor=%.2f)", config.refine_factor)
    if config.feature_selection and config.iterations > 1:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*experimental.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*experimental.*", module="optuna")
            r2_sampler = optuna.samplers.TPESampler(
                seed=config.seed + 9999,
                multivariate=True,
            )
        r2_shared_study = optuna.create_study(direction=metric.direction, sampler=r2_sampler)
        logger.info("created shared round2 study for feature selection across iterations")

    def _penalize(score: float, n_features: int) -> float:
        if max_selected is None or n_features <= max_selected:
            return score
        excess = n_features - max_selected
        multiplier = excess + 1
        if metric.direction == "minimize":
            return score * multiplier
        return score / multiplier

    for iteration in range(1, config.iterations + 1):
        iteration_started = time.perf_counter()
        logger.info(
            "iteration %d/%d: round1 (hyper-parameters) with %d features: %s",
            iteration, config.iterations, len(active_features), ",".join(active_features),
        )

        # Round 1 study: continue the previous iteration's study when
        # Round 2 did not improve the score last iteration; start a
        # fresh study when Round 2 improved (or on iteration 1).
        if round1_study is not None and not prev_round2_won:
            study = round1_study
            logger.info(
                "iteration %d/%d: round1 continues previous iteration's optuna study",
                iteration, config.iterations,
            )
            new_round1_study = False
        else:
            study, _ = create_study(config, metric, run_dir, f"{run_id}_iter{iteration}")
            round1_study = study
            logger.info(
                "iteration %d/%d: round1 started a new optuna study",
                iteration, config.iterations,
            )
            new_round1_study = True
        # Bottom-up variable reduction uses a larger Round 1 trial budget in
        # iteration 1 so a stronger importance ranking feeds the reduction.
        round1_n_trials = None
        if (
            config.bottom_up_variable_reduction
            and iteration == 1
            and config.bottom_up_n_trials is not None
        ):
            round1_n_trials = config.bottom_up_n_trials
        round1_outcome = optimize(
            study, data, config, metric, run_id, logger,
            selected_features=active_features,
            seed_params=prev_iter_best_params,
            search_state=search_state,
            recorder=recorder,
            round_name="round1",
            iteration=iteration,
            n_trials=round1_n_trials,
        )
        # Only record a study in the fallback history once (when it is
        # first created); a continued study is already in the history and
        # its trials accumulate in the same object.
        if new_round1_study:
            trials_history.append(("round1", iteration, study))
        last_study = study
        round1_params = round1_outcome.best_parameters
        round1_raw_score = round1_outcome.best_score
        round1_penalized = _penalize(round1_raw_score, len(active_features))
        logger.info(
            "iteration %d round1 best raw_score: %.6f penalized: %.6f (%d features)",
            iteration, round1_raw_score, round1_penalized, len(active_features),
        )

        iter_params = round1_params
        iter_score = round1_penalized
        iter_features = list(active_features)
        iter_trial = round1_outcome.best_trial_number

        # Round 1b: importance-based backward elimination. Runs in the first
        # ``config.round1b_iterations`` iterations (0 disables it). Features
        # dropped here are permanently removed from all future iterations.
        if config.feature_selection and iteration <= config.round1b_iterations:
            logger.info(
                "iteration %d/%d: round1b (importance-based feature elimination) from %d "
                "currently active features",
                iteration, config.iterations, len(active_features),
            )
            r1b_features, r1b_score = run_round1b_elimination(
                data, config, metric, round1_params, round1_raw_score,
                active_features=active_features,
                iteration=iteration,
                logger=logger,
                recorder=recorder,
            )
            newly_eliminated = set(active_features) - set(r1b_features)
            if newly_eliminated:
                eliminated_by_round1b |= newly_eliminated
                logger.info(
                    "iteration %d round1b permanently eliminated %d features: %s",
                    iteration, len(newly_eliminated), ",".join(sorted(newly_eliminated)),
                )
            # The survivors become the active set for round2 and all future rounds.
            active_features = list(r1b_features)
            iter_features = list(r1b_features)
            iter_score = _penalize(r1b_score, len(r1b_features))
            # Round 2 anchors on the post-elimination baseline.
            round1_raw_score = r1b_score
            logger.info(
                "iteration %d round1b finished: %d features score=%.6f selected=%s",
                iteration, len(r1b_features), r1b_score, ",".join(r1b_features),
            )

        # Bottom-up variable reduction (iteration 1 only). After Round 1 (or
        # Round 1b if it ran), fit the best model, drop predictors whose
        # importance is below ``bottom_up_importance_threshold``, rebuild via CV
        # on the survivors, and use that survivor model as the starting point
        # (anchor) for Round 2. The dropped predictors are NOT permanently
        # removed: they remain available in Round 2's candidate pool, but the
        # Round 2 anchor trial starts with only the survivors selected.
        if config.bottom_up_variable_reduction and iteration == 1:
            # Match the dynamic default used inside run_bottom_up_reduction so
            # the log line always has a real number to format (config value may
            # be None, which would crash the %.6f format specifier).
            if config.bottom_up_importance_threshold is not None:
                effective_threshold = float(config.bottom_up_importance_threshold)
            else:
                effective_threshold = (
                    (1.0 / len(active_features)) / 2.0 if active_features else 0.0
                )
            logger.info(
                "iteration %d/%d: bottom-up variable reduction from %d active features "
                "(importance threshold=%.6f)",
                iteration, config.iterations, len(active_features),
                effective_threshold,
            )
            bu_features, bu_score = run_bottom_up_reduction(
                data, config, metric, round1_params,
                active_features=active_features,
                base_score=round1_raw_score,
                logger=logger,
            )
            dropped = set(active_features) - set(bu_features)
            if dropped:
                logger.info(
                    "iteration %d bottom-up dropped %d features (kept available for "
                    "round2): %s",
                    iteration, len(dropped), ",".join(sorted(dropped)),
                )
            # The survivors become the active set / Round 2 anchor. Round 2's
            # candidate pool still includes the dropped predictors, so they can be
            # re-selected if doing so improves the score.
            active_features = list(bu_features)
            iter_features = list(bu_features)
            iter_score = _penalize(bu_score, len(bu_features))
            iter_trial = round1_outcome.best_trial_number
            round1_raw_score = bu_score
            logger.info(
                "iteration %d bottom-up reduction finished: %d features score=%.6f "
                "selected=%s",
                iteration, len(bu_features), bu_score, ",".join(bu_features),
            )
            # The bottom-up model becomes the global best / Round 2 starting point.
            best_params = round1_params
            best_score = iter_score
            best_trial_number = iter_trial
            best_features = list(bu_features)
            logger.info(
                "iteration %d: bottom-up model adopted as global best score=%.6f",
                iteration, best_score,
            )

        # When feature selection is disabled, Round 2 never runs, so it can
        # never improve the score: carry the Round 1 study forward.
        if not config.feature_selection:
            prev_round2_won = False

        if config.feature_selection:
            # Round 2 candidate pool: all original predictors except those
            # permanently eliminated by round 1b (bottom-up dropped features
            # are NOT in this set, so they remain selectable in Round 2).
            round2_candidates = [
                feature for feature in config.predictors
                if feature not in eliminated_by_round1b
                and (iteration > 1 or feature not in held_from_iteration1)
            ]
            logger.info(
                "iteration %d/%d: round2 (feature selection) from %d candidates, "
                "%d currently active",
                iteration, config.iterations, len(round2_candidates),
                len(active_features),
            )
            r2_features, r2_score = run_feature_selection(
                data, config, metric, round1_params, round1_raw_score,
                all_features=round2_candidates,
                active_features=active_features,
                iteration=iteration,
                logger=logger,
                shared_study=r2_shared_study,
                recorder=recorder,
            )
            logger.info(
                "iteration %d round2 best score: %.6f with %d features: %s",
                iteration, r2_score, len(r2_features), ",".join(r2_features),
            )
            if is_better(r2_score, iter_score, metric.direction):
                iter_features = r2_features
                iter_score = r2_score
                logger.info("iteration %d: round2 wins", iteration)
                prev_round2_won = True
            else:
                logger.info("iteration %d: round1/round1b wins", iteration)
                prev_round2_won = False

        active_features = list(iter_features)
        prev_iter_best_params = dict(iter_params)

        # Release the multimodel "rest" predictors after iteration 1 so
        # they re-enter the Round 2 candidate pool in iteration 2.
        if iteration == 1 and held_from_iteration1:
            logger.info(
                "iteration 1 complete: releasing %d held 'rest' predictors for "
                "iteration 2 round 2",
                len(held_from_iteration1),
            )
            held_from_iteration1 = set()

        if search_state is not None:
            refine_search_state(search_state, iter_params, config.refine_factor)
            logger.info(
                "refined round1 search bounds around iteration %d best params: %s",
                iteration, _format_search_bounds(search_state),
            )

        if is_better(iter_score, best_score, metric.direction):
            best_params = iter_params
            best_score = iter_score
            best_trial_number = iter_trial
            best_features = list(iter_features)
            logger.info("iteration %d: new global best score %.6f", iteration, best_score)

        logger.info(
            "iteration %d complete in %.1fs: score=%.6f global_best=%.6f",
            iteration, time.perf_counter() - iteration_started, iter_score, best_score,
        )

        # Stop the remaining iterations when the wall-clock run budget is exceeded.
        if config.max_run_seconds is not None:
            elapsed = time.perf_counter() - run_started_perf
            if elapsed > config.max_run_seconds:
                logger.info(
                    "max_run_seconds=%.1f exceeded after iteration %d (elapsed=%.1fs); "
                    "stopping iterations, keeping current best",
                    config.max_run_seconds, iteration, elapsed,
                )
                break

    if best_params is None or best_score is None:
        raise TrainingError("no iteration completed successfully")

    logger.info(
        "iterative optimization finished: best_score=%.6f best_trial=%d features=%s params=%s",
        best_score, best_trial_number, ",".join(best_features), _format_params(best_params),
    )

    if config.feature_selection and r2_shared_study is not None:
        trials_history.append(("round2", 0, r2_shared_study))

    return OptimizationOutcome(
        study=last_study,
        best_parameters=best_params,
        best_score=best_score,
        best_trial_number=best_trial_number,
        selected_features=best_features if config.feature_selection else None,
        recorder_rows=recorder.dataframe(),
        studies_history=trials_history,
    )

