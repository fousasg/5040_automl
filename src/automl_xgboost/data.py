"""Input validation and XGBoost-native feature preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AutoMLConfig
from .exceptions import DataValidationError
from .folds import generate_folds


@dataclass(slots=True)
class PreparedData:
    source: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    encoded_target: pd.Series
    row_ids: pd.Series
    fold_values: pd.Series
    fold_codes: pd.Series
    fold_mapping: dict[str, int]
    class_levels: list[Any] | None
    category_levels: dict[str, list[str]]
    missing_counts: dict[str, int]


def load_dataset(dataset: Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(dataset, pd.DataFrame):
        return dataset.copy(deep=True)
    path = Path(dataset).expanduser().resolve()
    if not path.is_file():
        raise DataValidationError(f"dataset does not exist: {path}")
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
    except Exception as exc:
        raise DataValidationError(f"could not read dataset: {exc}") from exc
    raise DataValidationError("dataset must be CSV or Parquet")


def prepare_data(frame: pd.DataFrame, config: AutoMLConfig) -> PreparedData:
    if frame.empty:
        raise DataValidationError("dataset must contain at least one row")
    if frame.columns.has_duplicates:
        raise DataValidationError("dataset contains duplicate column names")
    required = set(config.predictors) | {config.target}
    if config.fold_column:
        required.add(config.fold_column)
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise DataValidationError(f"dataset is missing columns: {missing_columns}")

    target = frame[config.target]
    if target.isna().any():
        raise DataValidationError("target contains null values; only predictor nulls are supported")
    class_levels: list[Any] | None = None
    if config.task == "regression":
        if not pd.api.types.is_numeric_dtype(target):
            raise DataValidationError("regression target must be numeric")
        if not np.isfinite(target.to_numpy(dtype=float)).all():
            raise DataValidationError("regression target must be finite")
        encoded_target = target.astype(float)
    else:
        values = list(pd.unique(target))
        if len(values) != 2:
            raise DataValidationError(f"classification requires exactly two classes; found {len(values)}")
        if config.positive_class is not None:
            matches = _matching_classes(values, config.positive_class)
            if len(matches) != 1:
                raise DataValidationError(f"positive_class {config.positive_class!r} does not identify one target class")
            positive = matches[0]
            negative = next(value for value in values if value != positive)
            class_levels = [negative, positive]
        else:
            class_levels = sorted(values, key=lambda value: str(value))
        mapping = {class_levels[0]: 0, class_levels[1]: 1}
        encoded_target = target.map(mapping).astype(np.int8)

    features = frame.loc[:, config.predictors].copy()
    category_levels: dict[str, list[str]] = {}
    for column in features:
        series = features[column]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            non_null = series.dropna().to_numpy(dtype=float)
            if not np.isfinite(non_null).all():
                raise DataValidationError(f"numeric predictor contains infinity: {column}")
            features[column] = pd.to_numeric(series, errors="raise")
        else:
            string_series = series.astype("string")
            levels = sorted(string_series.dropna().unique().tolist())
            if not levels:
                raise DataValidationError(f"categorical predictor has no observed levels: {column}")
            features[column] = string_series.astype(pd.CategoricalDtype(categories=levels))
            category_levels[column] = levels

    for column in config.monotonicity_constraints:
        if column in category_levels:
            raise DataValidationError(f"monotonicity constraint requires numeric predictor: {column}")

    if config.fold_column:
        supplied = frame[config.fold_column]
        if supplied.isna().any() or supplied.nunique() < 2:
            raise DataValidationError("fold column must have no nulls and at least two values")
        codes, unique = pd.factorize(supplied, sort=True)
        fold_codes = pd.Series(codes, dtype=np.int32, name="fold_code")
        fold_values = supplied.reset_index(drop=True).rename("fold")
        fold_mapping = {str(value): index for index, value in enumerate(unique)}
    else:
        fold_codes = generate_folds(target.reset_index(drop=True), config.task, config.n_folds, config.seed)
        fold_values = fold_codes.copy().rename("fold")
        fold_mapping = {str(value): int(value) for value in sorted(fold_codes.unique())}

    if config.task == "classification":
        for fold in sorted(fold_codes.unique()):
            training_classes = encoded_target[fold_codes != fold].nunique()
            if training_classes != 2:
                raise DataValidationError(f"training partition excluding fold {fold} does not contain both classes")

    missing_counts = {column: int(features[column].isna().sum()) for column in features}
    return PreparedData(
        source=frame.reset_index(drop=True),
        features=features.reset_index(drop=True),
        target=target.reset_index(drop=True),
        encoded_target=encoded_target.reset_index(drop=True),
        row_ids=pd.Series(np.arange(len(frame), dtype=np.int64), name="row_id"),
        fold_values=fold_values,
        fold_codes=fold_codes,
        fold_mapping=fold_mapping,
        class_levels=class_levels,
        category_levels=category_levels,
        missing_counts=missing_counts,
    )


def _matching_classes(values: list[Any], requested: Any) -> list[Any]:
    """Match a class label, including PyYAML's legacy yes/no boolean coercion."""

    direct = [value for value in values if value == requested or str(value) == str(requested)]
    if direct or not isinstance(requested, bool):
        return direct
    aliases = {"true", "yes", "y", "1"} if requested else {"false", "no", "n", "0"}
    return [value for value in values if str(value).strip().casefold() in aliases]
