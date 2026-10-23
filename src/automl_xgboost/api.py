"""Public end-to-end pipeline API."""

from __future__ import annotations

import importlib.metadata
import platform
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import (
    atomic_json, atomic_yaml, close_logger, configure_logging, prepare_run_directory,
    utc_now, write_dataframe, write_manifest,
)
from .config import AutoMLConfig
from .data import PreparedData, load_dataset, prepare_data
from .diagnostics import write_partial_dependence_plot, write_performance_plot
from .exceptions import AutoMLError, TrainingError
from .metrics import get_metric
from .modeling import predict_for_score
from .optimization import create_study, optimize, refit_best, run_iterative_optimization
from .recording import merge_trials_dataframe, write_trials_artifacts
from .reporting import compute_shap_values, write_shap_report
from .result import AutoMLArtifacts, AutoMLResult


def run_automl(config: AutoMLConfig | dict[str, Any]) -> AutoMLResult:
    if not isinstance(config, AutoMLConfig):
        config = AutoMLConfig.model_validate(config)
    metric = get_metric(config.task, config.scorer)
    data = prepare_data(load_dataset(config.dataset), config)
    run_id = _run_id()
    run_dir = prepare_run_directory(config, run_id)
    logger, log_path = configure_logging(run_dir, run_id)
    logger_closed = False
    started_at = utc_now()
    run_started_perf = time.perf_counter()
    try:
        logger.info("run_started", extra={"run_id": run_id})
        logger.info(
            "config: task=%s scorer=%s direction=%s n_trials=%d n_folds=%d iterations=%d "
            "feature_selection=%s max_selected_features=%s seed=%d n_jobs=%d",
            config.task, config.scorer, metric.direction, config.n_trials, config.n_folds,
            config.iterations, config.feature_selection, config.max_selected_features,
            config.seed, config.n_jobs,
        )
        logger.info(
            "data: rows=%d predictors=%d (%s) monotonicity=%s",
            len(data.source), len(config.predictors), ",".join(config.predictors),
            config.monotonicity_constraints or "none",
        )
        logger.info(
            "data: missing_counts=%s fold_mapping=%s class_levels=%s",
            {k: v for k, v in data.missing_counts.items() if v} or "none",
            data.fold_mapping or "none",
            data.class_levels or "n/a",
        )
        config_path = atomic_yaml(run_dir / "resolved_config.yaml", config.redacted_dict())
        folds_frame = pd.DataFrame({"row_id": data.row_ids, "fold": data.fold_values})
        folds_path = write_dataframe(folds_frame, run_dir / "folds", config.output_format)

        study, optuna_database = create_study(config, metric, run_dir, run_id)
        use_iterative = config.iterations > 1 or config.feature_selection
        optimization_started = time.perf_counter()
        logger.info("optimization started: iterative=%s", use_iterative)
        if use_iterative:
            outcome = run_iterative_optimization(data, config, metric, run_dir, run_id, logger)
            optuna_database = None
        else:
            outcome = optimize(study, data, config, metric, run_id, logger)
        logger.info(
            "optimization completed in %.1fs: best_score=%.6f best_trial=%d selected_features=%s",
            time.perf_counter() - optimization_started, outcome.best_score,
            outcome.best_trial_number,
            ",".join(outcome.selected_features) if outcome.selected_features is not None else "all",
        )
        selected_features = outcome.selected_features
        active_predictors = selected_features if selected_features is not None else config.predictors
        if outcome.recorder_rows is not None:
            trials_frame = outcome.recorder_rows
        else:
            studies = outcome.studies_history or [("single", 1, outcome.study)]
            trials_frame = merge_trials_dataframe(
                studies, iterative=config.iterations > 1 or config.feature_selection
            )
        trials_path, history_path, progress_path = write_trials_artifacts(
            trials_frame, config.scorer, metric.direction, run_dir
        )
        refit_started = time.perf_counter()
        model, oof, score = refit_best(outcome, data, config, metric)
        logger.info(
            "refit verified and full-data model fitted in %.1fs: recomputed_oof_score=%.6f",
            time.perf_counter() - refit_started, score,
        )
        features_for_predict = data.features if selected_features is None else data.features[selected_features]
        full_prediction = predict_for_score(model, features_for_predict, config.task)

        cv_frame = _prediction_frame(data, oof, config.task, include_fold=True)
        full_frame = _prediction_frame(data, full_prediction, config.task, include_fold=False)
        cv_path = write_dataframe(cv_frame, run_dir / "cv_predictions", config.output_format)
        full_path = write_dataframe(full_frame, run_dir / "full_data_predictions", config.output_format)

        performance_path = write_performance_plot(data, oof, config, run_dir)
        if selected_features is not None:
            sliced_data = _slice_data(data, selected_features)
            sliced_config = config.model_copy(update={"predictors": list(selected_features)})
        else:
            sliced_data = data
            sliced_config = config
        partial_dependence_path, partial_dependence_features = write_partial_dependence_plot(
            model, sliced_data, sliced_config, run_dir
        )

        shap_started = time.perf_counter()
        explanation, shap_frame = compute_shap_values(model, sliced_data)
        shap_path = write_dataframe(shap_frame, run_dir / "shap_values", config.output_format)

        model_path = run_dir / "model.json"
        temporary_model = run_dir / ".model.tmp.json"
        model.save_model(temporary_model)
        temporary_model.replace(model_path)
        model_metadata = {
            "model_format": "XGBoost JSON",
            "task": config.task,
            "target": config.target,
            "predictors": list(active_predictors),
            "all_predictors": config.predictors,
            "selected_features": list(selected_features) if selected_features is not None else None,
            "category_levels": {k: v for k, v in data.category_levels.items() if k in set(active_predictors)},
            "class_levels": data.class_levels,
            "positive_class": data.class_levels[1] if data.class_levels else None,
            "missing_value": "NaN passed natively to XGBoost",
            "parameters": outcome.best_parameters,
        }
        model_metadata_path = atomic_json(run_dir / "model_metadata.json", model_metadata)

        report_metadata = {
            "run_id": run_id,
            "algorithm": "xgboost",
            "best_parameters": outcome.best_parameters,
            "best_combined_oof_score": score,
            "scorer": config.scorer,
            "direction": metric.direction,
            "rows": len(data.source),
            "predictors": len(active_predictors),
            "selected_features": list(selected_features) if selected_features is not None else None,
            "rows_with_any_missing_predictor": int(data.features[list(active_predictors)].isna().any(axis=1).sum()),
            "missing_counts": {k: v for k, v in data.missing_counts.items() if k in set(active_predictors)},
            "partial_dependence_features": partial_dependence_features,
            "partial_dependence_background_rows": min(len(data.features), 1_000),
            "shap_scale": "raw XGBoost margin; SHAP values plus BiasTerm equal raw prediction",
        }
        report_path, report_assets = write_shap_report(
            explanation, shap_frame, sliced_data, sliced_config, run_dir, report_metadata
        )
        logger.info(
            "shap computed for %d rows and report written in %.1fs",
            len(shap_frame), time.perf_counter() - shap_started,
        )
        metadata = {
            **report_metadata,
            "status": "success",
            "started_at": started_at,
            "completed_at": utc_now(),
            "best_trial_number": outcome.best_trial_number,
            "fold_mapping": data.fold_mapping,
            "class_levels": data.class_levels,
            "category_levels": data.category_levels,
            "seed": config.seed,
            "python_version": platform.python_version(),
            "package_versions": _versions(),
            "mean_residual_deviance_definition": (
                "mean squared residual deviance for the reg:squarederror objective"
            ),
        }
        metadata_path = atomic_json(run_dir / "run_metadata.json", metadata)
        logger.info("artifacts_validated", extra={"run_id": run_id})
        for handler in logger.handlers:
            handler.flush()

        table_media = "application/vnd.apache.parquet" if config.output_format == "parquet" else "text/csv"
        inventory: list[tuple[Path, str, int | None]] = [
            (config_path, "application/yaml", None),
            (metadata_path, "application/json", None),
            (model_metadata_path, "application/json", None),
            (folds_path, table_media, len(data.source)),
            (trials_path, "text/csv", len(trials_frame)),
            (history_path, "text/html", None),
            (progress_path, "image/png", None),
            (cv_path, table_media, len(data.source)),
            (full_path, table_media, len(data.source)),
            (performance_path, "image/png", None),
            (partial_dependence_path, "image/png", None),
            (shap_path, table_media, len(data.source)),
            (report_path, "text/html", None),
            (model_path, "application/json", None),
            (log_path, "text/plain", None),
        ]
        inventory.extend((asset, "image/png", None) for asset in report_assets)
        if optuna_database:
            inventory.append((optuna_database, "application/vnd.sqlite3", None))
        manifest_path = write_manifest(run_dir, inventory, run_id)
        logger.info(
            "run_completed total_duration=%.1fs artifacts=%d",
            time.perf_counter() - run_started_perf, len(inventory),
        )
        close_logger(logger)
        logger_closed = True
        artifacts = AutoMLArtifacts(
            manifest=manifest_path,
            resolved_config=config_path,
            run_metadata=metadata_path,
            folds=folds_path,
            trials=trials_path,
            optimization_history=history_path,
            optimization_progress=progress_path,
            cv_predictions=cv_path,
            full_data_predictions=full_path,
            performance_plot=performance_path,
            partial_dependence_plot=partial_dependence_path,
            shap_values=shap_path,
            shap_report=report_path,
            model=model_path,
            model_metadata=model_metadata_path,
            optuna_database=optuna_database,
            log=log_path,
        )
        return AutoMLResult(
            run_id, run_dir, model, outcome.study, score, config.scorer, metric.direction,
            "xgboost", outcome.best_parameters, outcome.best_trial_number, artifacts,
            selected_features=list(selected_features) if selected_features is not None else None,
        )
    except Exception as exc:
        if not logger_closed:
            logger.exception(
                "run_failed duration=%.1fs", time.perf_counter() - run_started_perf,
                extra={"run_id": run_id},
            )
            atomic_json(
                run_dir / "failure.json",
                {"status": "failed", "run_id": run_id, "failed_at": utc_now(), "error_type": type(exc).__name__, "error": str(exc)},
            )
            close_logger(logger)
        if isinstance(exc, AutoMLError):
            raise
        raise TrainingError(f"AutoML run failed: {exc}") from exc


def _prediction_frame(
    data: PreparedData, prediction: np.ndarray, task: str, include_fold: bool
) -> pd.DataFrame:
    frame = pd.DataFrame({"row_id": data.row_ids, "actual": data.target})
    if include_fold:
        frame["fold"] = data.fold_values
    if task == "regression":
        frame["prediction"] = prediction
    else:
        assert data.class_levels is not None
        negative, positive = data.class_levels
        frame["predicted_label"] = np.where(prediction >= 0.5, positive, negative)
        frame[f"probability_{_safe_suffix(str(negative))}"] = 1.0 - prediction
        frame[f"probability_{_safe_suffix(str(positive))}"] = prediction
    return frame


def _safe_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-") or "class"


def _run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"


def _versions() -> dict[str, str]:
    result = {}
    for name in ("automl-xgboost-optuna", "xgboost", "shap", "optuna", "scikit-learn", "pandas", "numpy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _slice_data(data: PreparedData, selected_features: list[str]) -> PreparedData:
    return PreparedData(
        source=data.source,
        features=data.features[selected_features].copy(),
        target=data.target,
        encoded_target=data.encoded_target,
        row_ids=data.row_ids,
        fold_values=data.fold_values,
        fold_codes=data.fold_codes,
        fold_mapping=data.fold_mapping,
        class_levels=data.class_levels,
        category_levels={k: v for k, v in data.category_levels.items() if k in set(selected_features)},
        missing_counts={k: v for k, v in data.missing_counts.items() if k in set(selected_features)},
    )
