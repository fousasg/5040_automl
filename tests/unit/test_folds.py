import pandas as pd
import pytest

from automl_xgboost.exceptions import DataValidationError
from automl_xgboost.folds import generate_folds


def test_regression_folds_are_deterministic_and_balanced():
    target = pd.Series(range(23))
    first = generate_folds(target, "regression", 5, 42)
    assert first.equals(generate_folds(target, "regression", 5, 42))
    assert first.value_counts().max() - first.value_counts().min() <= 1


def test_classification_folds_are_stratified():
    target = pd.Series(["no"] * 12 + ["yes"] * 9)
    folds = generate_folds(target, "classification", 3, 4)
    table = pd.crosstab(target, folds)
    assert table.shape == (2, 3)
    assert (table > 0).all().all()


def test_class_must_have_enough_rows():
    with pytest.raises(DataValidationError, match="fewer rows"):
        generate_folds(pd.Series([0, 0, 1, 1, 1]), "classification", 3, 1)

