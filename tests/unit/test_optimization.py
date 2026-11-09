import logging
import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")

from automl_xgboost.config import AutoMLConfig
from automl_xgboost.data import prepare_data
from automl_xgboost.metrics import get_metric
from automl_xgboost.optimization import (
    run_round1b_elimination,
    run_bottom_up_reduction,
    _build_variable_groups,
    run_multimodel_variable_reduction,
)


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

    selected, score = run_round1b_elimination(
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

    selected, score = run_round1b_elimination(
        data,
        config,
        metric,
        best_parameters={},
        base_score=2.0,
        active_features=["keep", "drop"],
        iteration=1,
        logger=logging.getLogger("test_elimination_keep"),
    )

    # `drop` is kept because removing it degraded the score; with only two
    # features the search exhausts the candidate list before reaching the
    # 3-failure patience. Order is preserved.
    assert selected == ["keep", "drop"]
    assert score == 2.0  # baseline unchanged


def test_importance_based_elimination_skips_failure_and_continues(monkeypatch):
    # Three features ranked a > b > c (c least important). Removing c degrades
    # the score, but removing b (the next least-important) improves it. With the
    # patience budget the search must skip the failed c and still drop b.
    source = pd.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1, 0],
            "c": [1, 0, 5, 4, 3, 2],
            "target": [0, 1, 2, 3, 4, 5],
        }
    )
    config = AutoMLConfig(
        dataset=source,
        target="target",
        task="regression",
        scorer="rmse",
        predictors=["a", "b", "c"],
        n_folds=2,
        n_trials=1,
    )
    data = prepare_data(source, config)
    metric = get_metric("regression", "rmse")

    # Ranked a (most) > b > c (least). Removal order: c, b, a.
    fake_model = _ranking_model_factory(
        importances=__import__("numpy").array([0.9, 0.5, 0.1])
    )
    monkeypatch.setattr(
        "automl_xgboost.optimization.create_model", lambda *a, **k: fake_model
    )

    def fake_cv(data, config, parameters, selected_features=None):
        # Map each candidate subset to a prediction array whose rmse vs the target
        # is the intended score. Removing c (keep [a, b]) degrades; removing b
        # (keep [a, c]) improves to rmse 0; removing a (keep [c]) degrades again.
        target = data.encoded_target.to_numpy(dtype=float)
        n = len(target)
        if selected_features == ["a", "b"]:
            return __import__("numpy").full(n, 100.0)
        if selected_features == ["a", "c"]:
            return target  # rmse 0 -> improvement
        if selected_features == ["c"]:
            return __import__("numpy").full(n, 100.0)
        raise AssertionError(selected_features)

    monkeypatch.setattr("automl_xgboost.optimization.cross_validated_predictions", fake_cv)

    selected, score = run_round1b_elimination(
        data,
        config,
        metric,
        best_parameters={},
        base_score=2.0,
        active_features=["a", "b", "c"],
        iteration=1,
        logger=logging.getLogger("test_elimination_skip"),
    )

    # c was skipped (failed removal), b was dropped (improved), then a failed and
    # the loop continued until the candidate list / patience ended. Survivors are
    # returned in the original active_features order: a, c.
    assert selected == ["a", "c"]
    assert score == 0.0


def test_importance_based_elimination_stops_after_three_consecutive_failures(monkeypatch):
    # Four features ranked a > b > c > d. Every removal degrades the score, so
    # the search must stop after ELIMINATION_FAILURE_PATIENCE (3) consecutive
    # failures and keep all four features.
    source = pd.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1, 0],
            "c": [1, 0, 5, 4, 3, 2],
            "d": [2, 3, 0, 5, 4, 1],
            "target": [0, 1, 2, 3, 4, 5],
        }
    )
    config = AutoMLConfig(
        dataset=source,
        target="target",
        task="regression",
        scorer="rmse",
        predictors=["a", "b", "c", "d"],
        n_folds=2,
        n_trials=1,
    )
    data = prepare_data(source, config)
    metric = get_metric("regression", "rmse")

    fake_model = _ranking_model_factory(
        importances=__import__("numpy").array([0.9, 0.5, 0.3, 0.1])
    )
    monkeypatch.setattr(
        "automl_xgboost.optimization.create_model", lambda *a, **k: fake_model
    )

    def fake_cv(data, config, parameters, selected_features=None):
        # Every candidate subset is worse than the baseline.
        return __import__("numpy").full(len(data.encoded_target), 100.0)

    monkeypatch.setattr("automl_xgboost.optimization.cross_validated_predictions", fake_cv)

    selected, score = run_round1b_elimination(
        data,
        config,
        metric,
        best_parameters={},
        base_score=2.0,
        active_features=["a", "b", "c", "d"],
        iteration=1,
        logger=logging.getLogger("test_elimination_patience"),
    )

    # Three consecutive failures (d, c, b) stop the search before trying a;
    # all four features are kept, baseline unchanged.
    assert selected == ["a", "b", "c", "d"]
    assert score == 2.0


def test_bottom_up_reduction_drops_low_importance_features(monkeypatch):
    # Features a, b above the threshold; c below. The reduction must drop c
    # and rebuild via CV on [a, b] (fake_cv returns the target -> rmse 0).
    source = pd.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1, 0],
            "c": [1, 0, 5, 4, 3, 2],
            "target": [0, 1, 2, 3, 4, 5],
        }
    )
    config = AutoMLConfig(
        dataset=source,
        target="target",
        task="regression",
        scorer="rmse",
        predictors=["a", "b", "c"],
        n_folds=2,
        n_trials=1,
        bottom_up_variable_reduction=True,
        bottom_up_importance_threshold=0.3,
    )
    data = prepare_data(source, config)
    metric = get_metric("regression", "rmse")

    fake_model = _ranking_model_factory(
        importances=__import__("numpy").array([0.5, 0.4, 0.02])
    )
    monkeypatch.setattr(
        "automl_xgboost.optimization.create_model", lambda *a, **k: fake_model
    )

    def fake_cv(data, config, parameters, selected_features=None):
        assert selected_features == ["a", "b"], selected_features
        return data.encoded_target.to_numpy(dtype=float)

    monkeypatch.setattr("automl_xgboost.optimization.cross_validated_predictions", fake_cv)

    selected, score = run_bottom_up_reduction(
        data,
        config,
        metric,
        best_parameters={},
        active_features=["a", "b", "c"],
        base_score=2.0,
        logger=logging.getLogger("test_bottom_up_drop"),
    )

    # c (importance 0.02 < 0.3) is dropped; survivors returned in original order.
    assert selected == ["a", "b"]
    assert score == 0.0  # fake_cv returns target -> rmse 0


def test_bottom_up_reduction_keeps_all_when_all_below_threshold(monkeypatch):
    # Every feature below the threshold -> reduction is a no-op (keep all, baseline).
    source = pd.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1, 0],
            "target": [0, 1, 2, 3, 4, 5],
        }
    )
    config = AutoMLConfig(
        dataset=source,
        target="target",
        task="regression",
        scorer="rmse",
        predictors=["a", "b"],
        n_folds=2,
        n_trials=1,
        bottom_up_variable_reduction=True,
        bottom_up_importance_threshold=0.9,
    )
    data = prepare_data(source, config)
    metric = get_metric("regression", "rmse")

    fake_model = _ranking_model_factory(
        importances=__import__("numpy").array([0.4, 0.3])
    )
    monkeypatch.setattr(
        "automl_xgboost.optimization.create_model", lambda *a, **k: fake_model
    )

    def fake_cv(data, config, parameters, selected_features=None):
        raise AssertionError("CV should not run when no feature is dropped")

    monkeypatch.setattr("automl_xgboost.optimization.cross_validated_predictions", fake_cv)

    selected, score = run_bottom_up_reduction(
        data,
        config,
        metric,
        best_parameters={},
        active_features=["a", "b"],
        base_score=2.0,
        logger=logging.getLogger("test_bottom_up_keep"),
    )

    # Both below 0.9 -> keep all, baseline unchanged.
    assert selected == ["a", "b"]
    assert score == 2.0


def test_build_variable_groups_balances_appearances_and_group_size():
    # 10 models, each of 100 predictors appears 3 times -> group size = 100*3/10 = 30.
    predictors = [f"p{i}" for i in range(100)]
    groups = _build_variable_groups(predictors, n_models=10, appearances=3, seed=42)
    assert len(groups) == 10
    assert all(len(g) == 30 for g in groups)
    # Each predictor appears exactly 3 times.
    from collections import Counter
    counts = Counter(feature for group in groups for feature in group)
    assert all(counts[p] == 3 for p in predictors)
    # Every predictor is covered.
    assert set(counts) == set(predictors)


def test_build_variable_groups_handles_unbalanced_counts():
    # 7 predictors * 3 appearances = 21 slots across 10 models -> 1 group of 3
    # and 9 groups of 2 (sizes vary by ±1); each predictor appears 3 times.
    from collections import Counter
    predictors = [f"p{i}" for i in range(7)]
    groups = _build_variable_groups(predictors, n_models=10, appearances=3, seed=42)
    assert len(groups) == 10
    sizes = sorted(len(g) for g in groups)
    assert sizes == [2] * 9 + [3]  # 21 slots total
    counts = Counter(feature for group in groups for feature in group)
    assert all(counts[p] == 3 for p in predictors)  # each appears 3 times
    assert all(len(g) == len(set(g)) for g in groups)  # no dups within a group


def test_build_variable_groups_rejects_too_many_models():
    # n_models > n_predictors * appearances is impossible (not enough slots).
    predictors = [f"p{i}" for i in range(3)]
    with pytest.raises(ValueError, match="cannot exceed"):
        _build_variable_groups(predictors, n_models=10, appearances=1, seed=42)


def test_multimodel_variable_reduction_excludes_bottom_and_splits(monkeypatch, tmp_path):
    # 4 predictors, 2 models, each appears once -> group size 2. Fake SHAP
    # importances make [a, b] more important than [c, d]. Exclude bottom 1
    # (count), survivors [a, b, c], top_k=2 -> [a, b] for iteration1, rest=[c].
    source = pd.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1, 0],
            "c": [1, 0, 5, 4, 3, 2],
            "d": [2, 3, 0, 5, 4, 1],
            "target": [0, 1, 2, 3, 4, 5],
        }
    )
    config = AutoMLConfig(
        dataset=source,
        target="target",
        task="regression",
        scorer="rmse",
        predictors=["a", "b", "c", "d"],
        n_folds=2,
        n_trials=1,
        multimodel_variable_reduction=True,
        multimodel_n_models=2,
        multimodel_variable_appearances=1,
        multimodel_exclude_bottom=1,  # exclude 1 predictor (count)
        multimodel_top_k=2,
    )
    data = prepare_data(source, config)

    # Fake model: its feature_importances_ ranks a > b > c > d.
    fake_model = _ranking_model_factory(
        importances=__import__("numpy").array([0.5, 0.3, 0.15, 0.05])
    )
    monkeypatch.setattr(
        "automl_xgboost.optimization.create_model", lambda *a, **k: fake_model
    )
    monkeypatch.setattr(
        "automl_xgboost.optimization._compute_shap_importance",
        lambda model, data, group, config: {feature: 0.4 - 0.1 * i for i, feature in enumerate(group)},
    )

    top_k, rest, importance = run_multimodel_variable_reduction(
        data,
        config,
        metric=get_metric("regression", "rmse"),
        run_dir=tmp_path,
        run_id="test",
        logger=logging.getLogger("test_multimodel"),
    )

    # Groups (seed 42) are [d, b] and [c, a]; fake per-group importance
    # makes c and d tie on average 0.2 (a and b tie on 0.15). Ranked
    # [c, d, a, b] (tie-break by name). Exclude bottom 1 -> b.
    # survivors [c, d, a], top_k=2 -> [c, d], rest=[a].
    assert top_k == ["c", "d"]
    assert rest == ["a"]
    # a, b, c, d averaged across 2 models. b excluded (bottom 1).
    assert set(importance) == {"a", "b", "c", "d"}
    # With appearances=1 every predictor is in exactly one group, so each
    # has a non-zero averaged importance (c, d=0.2; a, b=0.15).
    assert importance["c"] == pytest.approx(0.2)
    assert importance["a"] == pytest.approx(0.15)


def test_multimodel_exclude_bottom_accepts_fraction(monkeypatch):
    # Float 0.25 over 4 predictors -> exclude bottom 1 (round(0.25*4)=1).
    source = pd.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1, 0],
            "c": [1, 0, 5, 4, 3, 2],
            "d": [2, 3, 0, 5, 4, 1],
            "target": [0, 1, 2, 3, 4, 5],
        }
    )
    config = AutoMLConfig(
        dataset=source,
        target="target",
        task="regression",
        scorer="rmse",
        predictors=["a", "b", "c", "d"],
        n_folds=2,
        n_trials=1,
        multimodel_variable_reduction=True,
        multimodel_n_models=2,
        multimodel_variable_appearances=1,
        multimodel_exclude_bottom=0.25,  # bottom 25% -> 1 of 4
        multimodel_top_k=3,
    )
    assert config._multimodel_exclude_count(4) == 1
