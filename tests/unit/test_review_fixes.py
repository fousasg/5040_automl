"""Regression tests for defects found during the full-codebase review.

Each test pins a specific bug so it cannot silently come back:

* the automatic Gamma shift must lift the minimum label to 1 (not add 1),
* ``OptimizationOutcome.best_score`` must be the raw score while
  ``best_penalized_score`` carries the Optuna-comparison score,
* ``multimodel_importance.csv`` must be an owned artifact,
* the config validators must reject impossible multimodel settings and
  mismatched objectives,
* ``TrialRecorder`` trial numbers must be unique across rounds,
* slicing features must not drop the target transform chain,
* scoring must honour legacy metadata and caller row identifiers.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from automl_xgboost.artifacts import OWNED_FILES
from automl_xgboost.config import AutoMLConfig
from automl_xgboost.data import prepare_data
from automl_xgboost.metrics import get_metric
from automl_xgboost.optimization import OptimizationOutcome, feature_penalty
from automl_xgboost.recording import TrialRecorder
from automl_xgboost.scoring import _recorded_transforms, _row_ids
from automl_xgboost.transforms import (
    apply_transform_chain,
    build_target_transform_chain,
    gamma_shift_amount,
    invert_transform_chain,
    validate_chain,
)
from automl_xgboost.exceptions import DataValidationError


def _regression_config(**overrides) -> AutoMLConfig:
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0, 4.0]})
    values = {
        "dataset": frame,
        "target": "y",
        "task": "regression",
        "scorer": "rmse",
        "predictors": ["x"],
        "n_folds": 2,
    }
    values.update(overrides)
    return AutoMLConfig(**values)


# --------------------------------------------------------------------------- #
# Gamma target shift
# --------------------------------------------------------------------------- #


def test_gamma_shift_lifts_minimum_to_one_for_negative_targets():
    chain = build_target_transform_chain("none", "reg:gamma", -10.9589)
    assert chain == ["gamma_shift:11.9589"]
    shifted = apply_transform_chain(np.array([-10.9589, 0.0, 5.0]), chain)
    assert shifted.min() == pytest.approx(1.0)
    assert invert_transform_chain(shifted, chain) == pytest.approx([-10.9589, 0.0, 5.0])


def test_gamma_shift_is_plain_one_for_zero_minimum():
    chain = build_target_transform_chain("none", "reg:gamma", 0.0)
    assert chain == ["gamma_shift"]
    assert gamma_shift_amount(chain[0]) == 1.0


def test_gamma_shift_is_skipped_for_strictly_positive_targets():
    assert build_target_transform_chain("none", "reg:gamma", 0.5) == []
    assert build_target_transform_chain("log", "reg:squarederror", -3.0) == ["log"]


def test_gamma_shift_serialisation_round_trips_precisely():
    chain = build_target_transform_chain("none", "reg:gamma", -0.1)
    values = np.array([-0.1, 0.0, 2.5])
    recovered = invert_transform_chain(apply_transform_chain(values, chain), chain)
    assert np.allclose(recovered, values, rtol=0.0, atol=1e-12)


def test_validate_chain_reports_the_offending_step():
    with pytest.raises(DataValidationError, match="log"):
        validate_chain(np.array([-1.0, 2.0]), ["log"])


def test_validate_chain_accepts_shift_before_log():
    # The Gamma shift makes a negative target non-negative, so a following log
    # step is legitimate; validation must apply steps progressively.
    chain = build_target_transform_chain("log", "reg:gamma", -4.0)  # ["gamma_shift:5.0", "log"]
    validate_chain(np.array([-4.0, 0.0, 3.0]), chain)


# --------------------------------------------------------------------------- #
# Raw vs penalized score contract
# --------------------------------------------------------------------------- #


def test_feature_penalty_is_identity_without_cap():
    assert feature_penalty(0.5, 10, None, "minimize") == 0.5
    assert feature_penalty(0.5, 3, 3, "minimize") == 0.5


def test_feature_penalty_scales_by_excess_feature_count():
    # 6 active features against a cap of 2 => excess 4 => multiplier 5.
    assert feature_penalty(0.5, 6, 2, "minimize") == pytest.approx(2.5)
    assert feature_penalty(0.5, 6, 2, "maximize") == pytest.approx(0.1)


def test_optimization_outcome_separates_raw_and_penalized_scores():
    outcome = OptimizationOutcome(
        study=None,
        best_parameters={},
        best_score=0.869,
        best_trial_number=7,
        best_penalized_score=1.738,
    )
    assert outcome.best_score == 0.869
    assert outcome.best_penalized_score == 1.738


# --------------------------------------------------------------------------- #
# Artifact ownership
# --------------------------------------------------------------------------- #


def test_multimodel_importance_is_an_owned_artifact():
    assert "multimodel_importance.csv" in OWNED_FILES


# --------------------------------------------------------------------------- #
# Configuration validators
# --------------------------------------------------------------------------- #


def test_config_rejects_more_appearances_than_models():
    with pytest.raises(ValueError, match="multimodel_variable_appearances"):
        _regression_config(
            multimodel_variable_reduction=True,
            multimodel_n_models=2,
            multimodel_variable_appearances=3,
            multimodel_top_k=1,
        )


def test_config_accepts_integral_float_exclude_bottom():
    config = _regression_config(multimodel_exclude_bottom=2.0)
    assert config.multimodel_exclude_bottom == 2
    assert isinstance(config.multimodel_exclude_bottom, int)
    assert config._multimodel_exclude_count(10) == 2


def test_config_rejects_classification_objective_for_regression():
    with pytest.raises(ValueError, match="not a regression objective"):
        _regression_config(objective="binary:logistic")


def test_config_rejects_regression_objective_for_classification():
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="not a classification objective"):
        AutoMLConfig(
            dataset=frame,
            target="y",
            task="classification",
            scorer="roc_auc",
            predictors=["x"],
            n_folds=2,
            objective="reg:squarederror",
        )


def test_config_strips_objective_whitespace():
    assert _regression_config(objective="  reg:gamma  ").objective == "reg:gamma"


# --------------------------------------------------------------------------- #
# Trial recording
# --------------------------------------------------------------------------- #


def test_trial_recorder_numbers_trials_uniquely_across_rounds():
    config = _regression_config(iterations=2)
    recorder = TrialRecorder(config)
    for step in range(3):
        recorder.add(
            SimpleNamespace(number=step),
            round_name="round1b",
            iteration=1,
            raw_score=1.0,
            penalized_score=1.0,
            selected_features=None,
        )
    for number in range(2):
        recorder.add(
            SimpleNamespace(number=number),
            round_name="round1",
            iteration=2,
            raw_score=2.0,
            penalized_score=2.0,
            selected_features=None,
        )
    frame = recorder.dataframe()
    assert frame["trial_number"].tolist() == [0, 1, 2, 3, 4]
    assert frame["trial_number"].is_unique


# --------------------------------------------------------------------------- #
# Feature slicing keeps the transform chain
# --------------------------------------------------------------------------- #


def test_slice_data_keeps_target_transforms():
    from automl_xgboost.api import _slice_data

    source = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "z": [1.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0, 4.0]})
    config = _regression_config(predictors=["x", "z"], objective="reg:gamma", target_transform="log")
    data = prepare_data(source, config)
    assert data.target_transforms, "chain should not be empty for a Gamma objective"
    sliced = _slice_data(data, ["x"])
    assert sliced.target_transforms == data.target_transforms
    assert list(sliced.features.columns) == ["x"]


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #


def test_recorded_transforms_prefers_chain_and_falls_back_to_legacy_key():
    assert _recorded_transforms({"target_transforms": ["gamma_shift:5", "log"]}) == ["gamma_shift:5", "log"]
    assert _recorded_transforms({"target_transforms": None, "target_transform": "log"}) == ["log"]
    assert _recorded_transforms({"target_transform": "none"}) == []
    assert _recorded_transforms({}) == []


def test_row_ids_uses_input_column_when_unique():
    source = pd.DataFrame({"row_id": [10, 11, 12], "x": [1.0, 2.0, 3.0]})
    assert _row_ids(source).tolist() == [10, 11, 12]
    duplicated = pd.DataFrame({"row_id": [1, 1, 2], "x": [1.0, 2.0, 3.0]})
    assert _row_ids(duplicated).tolist() == [0, 1, 2]
    assert _row_ids(pd.DataFrame({"x": [1.0, 2.0]})).tolist() == [0, 1]
