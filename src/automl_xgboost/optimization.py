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
                    data.encoded_target.to_numpy(dtype=float),
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

    round2_best_value: float | None = None
    round2_best_trial: int | None = None
    optuna_logger = optuna.logging.get_logger("automl_xgboost.round2")

    def log_round2_best(_study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal round2_best_value, round2_best_trial, round2_best_features
        if (
            trial.state != optuna.trial.TrialState.COMPLETE
            or trial.value is None
            or not math.isfinite(trial.value)
        ):
            return
        if not is_better(float(trial.value), round2_best_value, metric.direction):
            return
        selected = list(trial.user_attrs.get("selected_features") or [])
        round2_best_value = float(trial.value)
        round2_best_trial = trial.number
        message = (
            "round2 iteration %d best is trial %d with value %.6f (%d features)"
        )
        logger.info(message, iteration, round2_best_trial, round2_best_value, len(selected))
        optuna_logger.info(message, iteration, round2_best_trial, round2_best_value, len(selected))

    study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=[log_round2_best])

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

    r2_shared_study: Any | None = None
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
        round1_penalized = _penalize(round1_raw_score, len(active_features))
        logger.info(
            "iteration %d round1 best raw_score: %.6f penalized: %.6f (%d features)",
            iteration, round1_raw_score, round1_penalized, len(active_features),
        )

        iter_params = round1_params
        iter_score = round1_penalized
        iter_features = list(active_features)
        iter_trial = round1_outcome.best_trial_number

        if config.feature_selection:
            logger.info(
                "iteration %d/%d: round2 (feature selection) from %d candidates, "
                "%d currently active",
                iteration, config.iterations, len(config.predictors),
                len(active_features),
            )
            r2_features, r2_score = run_feature_selection(
                data, config, metric, round1_params, round1_raw_score,
                all_features=list(config.predictors),
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
            if is_better(r2_score, round1_penalized, metric.direction):
                iter_features = r2_features
                iter_score = r2_score
                logger.info("iteration %d: round2 wins", iteration)
            else:
                logger.info("iteration %d: round1 wins", iteration)

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

