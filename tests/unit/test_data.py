from pathlib import Path

import pandas as pd
import pytest

from automl_xgboost.config import AutoMLConfig
from automl_xgboost.data import load_dataset, prepare_data
from automl_xgboost.exceptions import DataValidationError


ROOT = Path(__file__).parents[2]


def test_regression_dataset_preserves_nulls_and_rows():
    path = ROOT / "data" / "test_regression_with_nulls.csv"
    source = load_dataset(path)
    config = AutoMLConfig(
        dataset=path, target="target", task="regression", scorer="rmse",
        predictors=["age", "income", "tenure", "region"], n_folds=3,
    )
    prepared = prepare_data(source, config)
    assert len(prepared.features) == len(source)
    assert len(source) > 100
    assert sum(prepared.missing_counts.values()) > 0
    assert prepared.features["region"].dtype.name == "category"
    assert prepared.features.isna().any(axis=1).sum() > 0


def test_classification_target_encoding_and_positive_class():
    path = ROOT / "data" / "test_classification_with_nulls.csv"
    source = load_dataset(path)
    config = AutoMLConfig(
        dataset=path, target="churned", task="classification", scorer="aucpr",
        positive_class="yes",
        predictors=["age", "monthly_spend", "tenure", "support_calls", "region"],
        n_folds=3,
    )
    prepared = prepare_data(source, config)
    assert prepared.class_levels == ["no", "yes"]
    assert set(prepared.encoded_target) == {0, 1}
    assert prepared.features.isna().any(axis=1).sum() == 28


def test_yaml_boolean_true_matches_yes_class():
    path = ROOT / "data" / "test_classification_with_nulls.csv"
    source = load_dataset(path)
    config = AutoMLConfig(
        dataset=path, target="churned", task="classification", scorer="aucpr",
        positive_class=True,
        predictors=["age", "monthly_spend", "tenure", "support_calls", "region"],
        n_folds=3,
    )
    assert prepare_data(source, config).class_levels == ["no", "yes"]


def test_target_nulls_and_categorical_constraints_are_rejected():
    frame = pd.DataFrame({"x": [1, 2], "category": ["a", None], "y": [1.0, None]})
    config = AutoMLConfig(
        dataset=frame, target="y", task="regression", scorer="rmse", predictors=["x", "category"], n_folds=2,
    )
    with pytest.raises(DataValidationError, match="target contains null"):
        prepare_data(frame, config)
    frame["y"] = [1.0, 2.0]
    constrained = config.model_copy(update={"monotonicity_constraints": {"category": 1}})
    with pytest.raises(DataValidationError, match="requires numeric"):
        prepare_data(frame, constrained)
