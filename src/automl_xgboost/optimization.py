"""Optuna optimization using combined out-of-fold XGBoost predictions."""

from __future__ import annotations

import logging
import math
import time
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
    """Fit one model per shared fold and return predictions in original row order."""

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
                data.encoded_target.to_numpy(dtype=float),
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

    logger.info(
        "study started: n_trials=%d timeout=%s n_features=%d",
        config.n_trials, config.timeout_seconds or "none", n_features,
    )
    study.optimize(
        objective,
        n_trials=config.n_trials,
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
    score = score_predictions(metric, data.encoded_target.to_numpy(dtype=float), oof)
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


def run_importance_based_feature_selection(
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

    After Round 1, fit a model with the winning parameters on all active
    predictors, rank them by fitted-model importance (most -> least), and
    repeatedly drop the least-important one. Each candidate subset is scored
    via the shared cross-validation folds. A drop is accepted when the score
    is the same (within tolerance) or better than the current score; the first
    drop that degrades the score stops the search, and the kept feature set is
    the one that still includes that feature.
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
    while len(current_features) > 1 and step < len(removal_order):
        removed = removal_order[step]
        candidate = [feature for feature in current_features if feature != removed]
        started = time.perf_counter()
        try:
            predictions = cross_validated_predictions(
                data, config, best_parameters, candidate
            )
            score = score_predictions(
                metric, data.encoded_target.to_numpy(dtype=float), predictions
            )
        except Exception as exc:
            logger.warning(
                "round2 elimination iteration %d step %d failed removing %s: %s",
                iteration, step, removed, exc,
            )
            if recorder is not None:
                recorder.add(
                    trial=types.SimpleNamespace(number=step),
                    round_name="round2",
                    iteration=iteration,
                    raw_score=None,
                    penalized_score=worst_score(metric.direction),
                    selected_features=candidate,
                )
            break

        accepted = _same_or_better(score, current_score)
        logger.info(
            "round2 elimination iteration %d step %d: removed=%s candidate=%d features "
            "score=%.6f previous=%.6f accepted=%s duration=%.2fs",
            iteration, step, removed, len(candidate), score, current_score, accepted,
            time.perf_counter() - started,
        )
        if recorder is not None:
            recorder.add(
                trial=types.SimpleNamespace(number=step),
                round_name="round2",
                iteration=iteration,
                raw_score=score,
                penalized_score=score,
                selected_features=candidate,
            )
        if accepted:
            current_features = candidate
            current_score = score
            step += 1
        else:
            # Degrade: stop and keep the set that still includes `removed`.
            break

    logger.info(
        "round2 elimination iteration %d finished: %d features score=%.6f selected=%s",
        iteration, len(current_features), current_score, ",".join(current_features),
    )
    return current_features, current_score


def run_iterative_optimization(
    data: PreparedData,
    config: AutoMLConfig,
    metric: MetricSpec,
    run_dir: Path,
    run_id: str,
    logger: logging.Logger,
) -> OptimizationOutcome:
    active_features: list[str] = list(config.predictors)
    best_params: dict[str, Any] | None = None
    best_score: float | None = None
    best_trial_number: int = 0
    best_features: list[str] = list(config.predictors)
    last_study: Any = None
    prev_iter_best_params: dict[str, Any] | None = None
    recorder = TrialRecorder(config)
    trials_history: list[tuple[str, int, Any]] = []

    search_state: dict[str, dict[str, Any]] | None = None
    if config.iterations > 1:
        search_state = build_search_state()
        logger.info("built search state for round1 refinement (refine_factor=%.2f)", config.refine_factor)

    for iteration in range(1, config.iterations + 1):
        iteration_started = time.perf_counter()
        logger.info(
            "iteration %d/%d: round1 (hyper-parameters) with %d features: %s",
            iteration, config.iterations, len(active_features), ",".join(active_features),
        )

        study, _ = create_study(config, metric, run_dir, f"{run_id}_iter{iteration}")
        round1_outcome = optimize(
            study, data, config, metric, run_id, logger,
            selected_features=active_features,
            seed_params=prev_iter_best_params,
            search_state=search_state,
            recorder=recorder,
            round_name="round1",
            iteration=iteration,
        )
        trials_history.append(("round1", iteration, study))
        last_study = study
        round1_params = round1_outcome.best_parameters
        round1_raw_score = round1_outcome.best_score
        logger.info(
            "iteration %d round1 best raw_score: %.6f (%d features)",
            iteration, round1_raw_score, len(active_features),
        )

        iter_params = round1_params
        iter_score = round1_raw_score
        iter_features = list(active_features)
        iter_trial = round1_outcome.best_trial_number

        if config.feature_selection:
            logger.info(
                "iteration %d/%d: round2 (importance-based feature elimination) from %d "
                "currently active features",
                iteration, config.iterations, len(active_features),
            )
            r2_features, r2_score = run_importance_based_feature_selection(
                data, config, metric, round1_params, round1_raw_score,
                active_features=active_features,
                iteration=iteration,
                logger=logger,
                recorder=recorder,
            )
            logger.info(
                "iteration %d round2 best score: %.6f with %d features: %s",
                iteration, r2_score, len(r2_features), ",".join(r2_features),
            )
            # Elimination guarantees r2_score is same-or-better vs round1_raw_score,
            # so the eliminated subset becomes the iteration result.
            iter_features = r2_features
            iter_score = r2_score
            logger.info("iteration %d: round2 elimination result adopted", iteration)

        active_features = list(iter_features)
        prev_iter_best_params = dict(iter_params)

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

    if best_params is None or best_score is None:
        raise TrainingError("no iteration completed successfully")

    logger.info(
        "iterative optimization finished: best_score=%.6f best_trial=%d features=%s params=%s",
        best_score, best_trial_number, ",".join(best_features), _format_params(best_params),
    )

    return OptimizationOutcome(
        study=last_study,
        best_parameters=best_params,
        best_score=best_score,
        best_trial_number=best_trial_number,
        selected_features=best_features if config.feature_selection else None,
        recorder_rows=recorder.dataframe(),
        studies_history=trials_history,
    )

