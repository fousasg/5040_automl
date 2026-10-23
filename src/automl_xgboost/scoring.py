"""Load a successful run and score new data with its saved XGBoost model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import sha256_file, write_dataframe
from .data import load_dataset
from .exceptions import ArtifactError, DataValidationError
from .result import ScoringResult


def score_saved_model(
    run_dir: str | Path,
    dataset: str | Path | pd.DataFrame,
    output_path: str | Path | None = None,
) -> ScoringResult:
    """Validate and score new rows using `model.json` from a successful run.

    Predictor nulls are preserved. Categorical values must belong to the domains
    recorded during training; unknown non-null levels fail rather than silently
    becoming missing values.
    """

    resolved_run = Path(run_dir).expanduser().resolve()
    metadata_path = resolved_run / "model_metadata.json"
    model_path = resolved_run / "model.json"
    manifest_path = resolved_run / "manifest.json"
    for path in (metadata_path, model_path, manifest_path):
        if not path.is_file():
            raise ArtifactError(f"saved run is missing required file: {path}")
    _verify_manifest_files(manifest_path, resolved_run, (model_path, metadata_path))
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"could not read model metadata: {exc}") from exc

    task = metadata.get("task")
    predictors = metadata.get("predictors")
    if task not in {"regression", "classification"} or not isinstance(predictors, list):
        raise ArtifactError("model metadata has invalid task or predictors")
    source = load_dataset(dataset if isinstance(dataset, pd.DataFrame) else Path(dataset))
    missing = sorted(set(predictors) - set(source.columns))
    if missing:
        raise DataValidationError(f"scoring dataset is missing predictors: {missing}")
    features = _prepare_features(source, predictors, metadata.get("category_levels") or {})

    from xgboost import XGBClassifier, XGBRegressor

    model = XGBRegressor() if task == "regression" else XGBClassifier()
    try:
        model.load_model(model_path)
        if task == "regression":
            prediction = np.asarray(model.predict(features), dtype=float)
            output = pd.DataFrame(
                {"row_id": np.arange(len(features), dtype=np.int64), "prediction": prediction}
            )
        else:
            class_levels = metadata.get("class_levels")
            if not isinstance(class_levels, list) or len(class_levels) != 2:
                raise ArtifactError("classification model metadata must contain two class levels")
            probability = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
            negative, positive = class_levels
            output = pd.DataFrame(
                {
                    "row_id": np.arange(len(features), dtype=np.int64),
                    "predicted_label": np.where(probability >= 0.5, positive, negative),
                    f"probability_{_safe_suffix(str(negative))}": 1.0 - probability,
                    f"probability_{_safe_suffix(str(positive))}": probability,
                }
            )
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"could not load or score the XGBoost model: {exc}") from exc
    if len(output) != len(source) or output.drop(columns=["predicted_label"], errors="ignore").isna().any().any():
        raise ArtifactError("saved model did not produce one complete prediction per input row")

    written = _write_optional_output(output, output_path)
    return ScoringResult(resolved_run, output, written, model)


def _prepare_features(
    source: pd.DataFrame, predictors: list[str], category_levels: dict[str, list[str]]
) -> pd.DataFrame:
    features = source.loc[:, predictors].copy()
    for column in predictors:
        if column in category_levels:
            string_values = features[column].astype("string")
            levels = [str(value) for value in category_levels[column]]
            unknown = sorted(set(string_values.dropna().unique()) - set(levels))
            if unknown:
                raise DataValidationError(
                    f"scoring predictor {column!r} contains unseen categories: {unknown}"
                )
            features[column] = string_values.astype(pd.CategoricalDtype(categories=levels))
        else:
            try:
                features[column] = pd.to_numeric(features[column], errors="raise")
            except (TypeError, ValueError) as exc:
                raise DataValidationError(f"scoring predictor {column!r} must be numeric") from exc
            values = features[column].dropna().to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise DataValidationError(f"scoring predictor {column!r} contains infinity")
    return features


def _verify_manifest_files(
    manifest_path: Path, run_dir: Path, paths: tuple[Path, ...]
) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = {entry["path"]: entry for entry in manifest["artifacts"]}
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid run manifest: {exc}") from exc
    if manifest.get("status") != "success":
        raise ArtifactError("only a successful run can be used for scoring")
    for path in paths:
        relative = str(path.relative_to(run_dir))
        entry = entries.get(relative)
        if not entry or entry.get("sha256") != sha256_file(path):
            raise ArtifactError(f"manifest checksum validation failed for {relative}")


def _write_optional_output(frame: pd.DataFrame, output_path: str | Path | None) -> Path | None:
    if output_path is None:
        return None
    output = Path(output_path).expanduser().resolve()
    suffix = output.suffix.lower()
    if suffix not in {".csv", ".parquet", ".pq"}:
        raise DataValidationError("scoring output path must end in .csv, .parquet, or .pq")
    output.parent.mkdir(parents=True, exist_ok=True)
    output_format = "csv" if suffix == ".csv" else "parquet"
    written = write_dataframe(frame, output.with_suffix(""), output_format)
    if suffix == ".pq" and written != output:
        written.replace(output)
        return output
    return written


def _safe_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-") or "class"

