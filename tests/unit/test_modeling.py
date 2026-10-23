import pandas as pd

from automl_xgboost.config import AutoMLConfig
from automl_xgboost.modeling import create_model, suggest_parameters


class Trial:
    def suggest_int(self, name, low, high):
        return (low + high) // 2

    def suggest_float(self, name, low, high, **kwargs):
        return (low + high) / 2


def test_search_space_has_required_xgboost_parameters():
    params = suggest_parameters(Trial())
    assert 50 <= params["n_estimators"] <= 500
    assert 2 <= params["max_depth"] <= 12
    assert {"learning_rate", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha"} <= params.keys()


def test_monotonicity_constraints_reach_xgboost():
    frame = pd.DataFrame({"x": [1, 2], "y": [2, 3]})
    config = AutoMLConfig(
        dataset=frame, target="y", task="regression", scorer="rmse",
        predictors=["x"], monotonicity_constraints={"x": 1}, n_folds=2,
    )
    model = create_model(config, {"n_estimators": 2, "max_depth": 2})
    assert model.get_params()["monotone_constraints"] == {"x": 1}
