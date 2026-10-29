"""CLI for config-file or flag-driven XGBoost optimization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml
from pydantic import ValidationError

from .api import run_automl
from .config import AutoMLConfig, load_config_file
from .exceptions import AutoMLError, ConfigurationError
from .scoring import score_saved_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="automl-xgboost",
        description="Tune XGBoost with Optuna and generate predictions plus SHAP artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the complete pipeline")
    run.add_argument("--config", type=Path)
    run.add_argument("--dataset", type=Path)
    run.add_argument("--target")
    run.add_argument("--task", choices=("regression", "classification"))
    run.add_argument("--scorer")
    run.add_argument("--predictors", nargs="+", metavar="COLUMN")
    run.add_argument("--fold-column")
    run.add_argument("--algorithms", nargs="+", choices=("xgboost",))
    run.add_argument("--monotonicity-constraints", help="inline JSON or YAML/JSON mapping path")
    run.add_argument("--positive-class")
    run.add_argument("--n-folds", type=int)
    run.add_argument("--n-trials", type=int)
    run.add_argument("--timeout-seconds", type=float)
    run.add_argument("--iterations", type=int)
    run.add_argument("--refine-factor", type=float)
    run.add_argument("--feature-selection", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--feature-selection-trials", type=int)
    run.add_argument("--max-selected-features", type=int)
    run.add_argument("--round1b-iterations", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--output-format", choices=("parquet", "csv"))
    run.add_argument("--explain-sample-size", type=int)
    run.add_argument("--n-jobs", type=int)
    run.add_argument("--device")
    run.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--optuna-storage")
    run.add_argument("--optuna-study-name")
    score = subparsers.add_parser("score", help="score new data with a saved successful run")
    score.add_argument("--run-dir", type=Path, required=True, help="directory containing model.json and manifest.json")
    score.add_argument("--dataset", type=Path, required=True, help="CSV or Parquet data to score")
    score.add_argument("--output", type=Path, required=True, help="prediction output ending in .csv or .parquet")
    return parser


def resolve_cli_config(namespace: argparse.Namespace) -> AutoMLConfig:
    values: dict[str, Any] = load_config_file(namespace.config) if namespace.config else {}
    for key in (
        "dataset", "target", "task", "scorer", "predictors", "fold_column", "algorithms",
        "positive_class", "n_folds", "n_trials", "timeout_seconds", "iterations",
        "refine_factor", "feature_selection", "feature_selection_trials",
        "max_selected_features", "round1b_iterations", "seed", "output_dir",
        "output_format", "explain_sample_size", "n_jobs", "device", "overwrite",
    ):
        value = getattr(namespace, key, None)
        if value is not None:
            values[key] = value
    if namespace.monotonicity_constraints is not None:
        values["monotonicity_constraints"] = _mapping(namespace.monotonicity_constraints)
    optuna = dict(values.get("optuna") or {})
    if namespace.optuna_storage is not None:
        optuna["storage"] = namespace.optuna_storage
    if namespace.optuna_study_name is not None:
        optuna["study_name"] = namespace.optuna_study_name
    if optuna:
        values["optuna"] = optuna
    return AutoMLConfig.model_validate(values)


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    try:
        if namespace.command == "score":
            result = score_saved_model(namespace.run_dir, namespace.dataset, namespace.output)
            print(json.dumps({
                "status": "success", "run_dir": str(result.run_dir),
                "rows_scored": len(result.predictions), "output": str(result.output_path),
            }, sort_keys=True))
            return 0
        config = resolve_cli_config(namespace)
        print("Starting XGBoost/Optuna/SHAP AutoML run...", file=sys.stderr)
        result = run_automl(config)
        print(json.dumps({
            "status": "success", "run_id": result.run_id,
            "output_dir": str(result.output_dir), "best_score": result.best_score,
            "scorer": result.scorer, "manifest": str(result.artifacts.manifest),
        }, sort_keys=True))
        return 0
    except (ValidationError, ConfigurationError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except AutoMLError as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1


def _mapping(value: str) -> dict[str, Any]:
    path = Path(value).expanduser()
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else json.loads(value)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"could not parse constraints: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigurationError("constraints must be a mapping")
    return loaded


if __name__ == "__main__":
    raise SystemExit(main())
