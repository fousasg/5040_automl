import numpy as np
import pytest

from automl_xgboost.exceptions import ConfigurationError
from automl_xgboost.metrics import get_metric, running_best, score_predictions


def test_directions_and_running_best():
    assert get_metric("classification", "aucpr").direction == "maximize"
    assert get_metric("regression", "rmse").direction == "minimize"
    assert running_best([3, 4, 2], "minimize") == [3, 3, 2]
    assert running_best([.5, .4, .7], "maximize") == [.5, .5, .7]
    with pytest.raises(ConfigurationError):
        get_metric("regression", "auc")


def test_combined_metrics_return_expected_values():
    actual = np.array([0, 0, 1, 1])
    prediction = np.array([.1, .2, .8, .9])
    assert score_predictions(get_metric("classification", "auc"), actual, prediction) == 1.0
    assert score_predictions(get_metric("regression", "mse"), np.array([1, 2]), np.array([1, 4])) == 2.0

