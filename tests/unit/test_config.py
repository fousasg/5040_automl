import pandas as pd
import pytest
from pydantic import ValidationError

from automl_xgboost.config import AutoMLConfig


def make_config(**overrides):
    values = dict(
        dataset=pd.DataFrame({"x": [1, 2], "y": [2, 3]}),
        target="y", task="regression", scorer="RMSE", predictors=["x"],
    )
    values.update(overrides)
    return AutoMLConfig(**values)


def test_normalizes_and_accepts_only_xgboost():
    config = make_config(task="REGRESSION", algorithms=["XGBoost"], monotonicity_constraints={"x": 0})
    assert config.task == "regression"
    assert config.scorer == "rmse"
    assert config.algorithms == ["xgboost"]
    assert config.monotonicity_constraints == {}
    with pytest.raises(ValidationError, match="only algorithms"):
        make_config(algorithms=["gbm"])


def test_rejects_unknown_fields_and_overlapping_columns():
    with pytest.raises(ValidationError, match="Extra inputs"):
        make_config(unknown=True)
    with pytest.raises(ValidationError, match="target must not"):
        make_config(predictors=["x", "y"])


def test_redacted_in_memory_configuration():
    config = make_config(optuna={"storage": "postgresql://me:secret@host/db"})
    dumped = config.redacted_dict()
    assert dumped["dataset"].startswith("<in-memory DataFrame")
    assert "secret" not in dumped["optuna"]["storage"]

