import json

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from automl_xgboost.artifacts import atomic_json, write_manifest
from automl_xgboost.exceptions import DataValidationError
from automl_xgboost.scoring import score_saved_model


def saved_regression_run(tmp_path):
    training = pd.DataFrame(
        {
            "number": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "category": pd.Series(["a", "b", "a", None, "b", "a"], dtype="category"),
        }
    )
    target = pd.Series([2.0, 4.0, 3.0, 8.0, 9.0, 11.0])
    model = XGBRegressor(n_estimators=3, max_depth=2, tree_method="hist", enable_categorical=True)
    model.fit(training, target)
    model_path = tmp_path / "model.json"
    model.save_model(model_path)
    metadata = atomic_json(
        tmp_path / "model_metadata.json",
        {
            "task": "regression",
            "predictors": ["number", "category"],
            "category_levels": {"category": ["a", "b"]},
            "class_levels": None,
        },
    )
    write_manifest(
        tmp_path,
        [(model_path, "application/json", None), (metadata, "application/json", None)],
        "test-run",
    )
    return training


def test_scores_saved_model_and_preserves_null_rows(tmp_path):
    data = saved_regression_run(tmp_path)
    output = tmp_path / "predictions.csv"
    result = score_saved_model(tmp_path, data, output)
    assert result.output_path == output
    assert len(result.predictions) == len(data)
    assert result.predictions["prediction"].notna().all()


def test_rejects_unseen_categorical_levels(tmp_path):
    data = saved_regression_run(tmp_path)
    data["category"] = data["category"].astype("string")
    data.loc[0, "category"] = "unseen"
    with pytest.raises(DataValidationError, match="unseen categories"):
        score_saved_model(tmp_path, data)

