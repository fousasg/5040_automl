import logging
import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")

from automl_xgboost.config import AutoMLConfig
from automl_xgboost.data import prepare_data
from automl_xgboost.metrics import get_metric
from automl_xgboost.optimization import run_importance_based_feature_selection


def _ranking_model_factory(importances):
    """Return a fake ranking model with fixed feature_importances_."""

    class _FakeModel:
        def __init__(self):
            self.feature_importances_ = importances

        def fit(self, *args, **kwargs):
            return self

    return _FakeModel()


def test_importance_based_elimination_drops_low_importance_feature(monkeypatch):
    # `drop` is deliberately less important than `keep`; the ranking should put
    # `drop` last, and removing it must be accepted when the score is the same.
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
    )
    data = prepare_data(source, config)
    metric = get_metric("regression", "rmse")

    # Force the ranking model to rank `keep` above `drop`.
    fake_model = _ranking_model_factory(importances=__import__("numpy").array([0.9, 0.1]))
    monkeypatch.setattr(
        "automl_xgboost.optimization.create_model", lambda *a, **k: fake_model
    )

    def fake_cv(data, config, parameters, selected_features=None):
        assert selected_features == ["keep"], selected_features
        return data.encoded_target.to_numpy(dtype=float)

    monkeypatch.setattr("automl_xgboost.optimization.cross_validated_predictions", fake_cv)

    selected, score = run_importance_based_feature_selection(
        data,
        config,
        metric,
        best_parameters={},
        base_score=2.0,
        active_features=["keep", "drop"],
        iteration=2,
        logger=logging.getLogger("test_elimination"),
    )

    # Surviving features are returned in their original active_features order
    # (importance ranking only decides removal order).
    assert selected == ["keep"]
    assert score == 0.0  # fake_cv returns the target itself -> rmse 0 <= baseline 2.0


def test_importance_based_elimination_keeps_feature_when_score_degrades(monkeypatch):
    # Removing the least-important feature makes the score worse: the loop must
    # stop and keep the set that still includes the feature we tried to remove.
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
    )
    data = prepare_data(source, config)
    metric = get_metric("regression", "rmse")

    fake_model = _ranking_model_factory(importances=__import__("numpy").array([0.9, 0.1]))
    monkeypatch.setattr(
        "automl_xgboost.optimization.create_model", lambda *a, **k: fake_model
    )

    def fake_cv(data, config, parameters, selected_features=None):
        assert selected_features == ["keep"], selected_features
        # Worse than the baseline of 2.0 -> removal must be rejected.
        return __import__("numpy").full(len(data.encoded_target), 100.0)

    monkeypatch.setattr("automl_xgboost.optimization.cross_validated_predictions", fake_cv)

    selected, score = run_importance_based_feature_selection(
        data,
        config,
        metric,
        best_parameters={},
        base_score=2.0,
        active_features=["keep", "drop"],
        iteration=1,
        logger=logging.getLogger("test_elimination_keep"),
    )

    # `drop` is kept because removing it degraded the score; order is preserved.
    assert selected == ["keep", "drop"]
    assert score == 2.0  # baseline unchanged
