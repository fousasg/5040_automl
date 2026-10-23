import logging
import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")

from automl_xgboost.config import AutoMLConfig
from automl_xgboost.data import prepare_data
from automl_xgboost.metrics import get_metric
from automl_xgboost.optimization import run_feature_selection


def test_feature_selection_considers_only_new_shared_study_trials(monkeypatch):
    source = pd.DataFrame(
        {
            "keep": [0, 1, 2, 3, 4, 5],
            "drop": [5, 4, 3, 2, 1, 0],
            "target": [0, 1, 2, 3, 4, 5],
        }
    )
    config = AutoMLConfig(
        dataset=source,
        target="target",
        task="regression",
        scorer="rmse",
        predictors=["keep", "drop"],
        n_folds=2,
        n_trials=1,
        feature_selection_trials=1,
    )
    data = prepare_data(source, config)
    metric = get_metric("regression", "rmse")
    study = optuna.create_study(direction="minimize")
    study.add_trial(
        optuna.trial.create_trial(
            params={"feat__000": 0, "feat__001": 1},
            distributions={
                "feat__000": optuna.distributions.IntDistribution(0, 1),
                "feat__001": optuna.distributions.IntDistribution(0, 1),
            },
            value=1.0,
            user_attrs={"selected_features": ["drop"], "cap_violation": 0.0},
        )
    )

    def fake_cv(data, config, parameters, selected_features=None):
        assert selected_features == ["keep"]
        return data.encoded_target.to_numpy(dtype=float)

    monkeypatch.setattr("automl_xgboost.optimization.cross_validated_predictions", fake_cv)

    selected, score = run_feature_selection(
        data,
        config,
        metric,
        best_parameters={},
        base_score=2.0,
        all_features=["keep", "drop"],
        active_features=["keep"],
        iteration=2,
        logger=logging.getLogger("test_feature_selection"),
        shared_study=study,
    )

    assert selected == ["keep"]
    assert score == 2.0
