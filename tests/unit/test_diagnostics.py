import numpy as np
import pandas as pd

from automl_xgboost import AutoMLConfig
from automl_xgboost.data import prepare_data
from automl_xgboost.diagnostics import (
    write_partial_dependence_plot,
    write_performance_plot,
)


class FakeRegressor:
    feature_importances_ = np.asarray([0.2, 0.8])

    def predict(self, features):
        category = features["category"].cat.codes.replace(-1, 0).to_numpy(dtype=float)
        numeric = features["numeric"].fillna(0).to_numpy(dtype=float)
        return numeric + category


def test_writes_regression_performance_and_ranked_pdp(tmp_path):
    source = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "category": ["a", "b", "a", None, "b", "a"],
            "target": [1.5, 3.0, 2.0, 5.0, 6.0, 7.0],
        }
    )
    config = AutoMLConfig(
        dataset=source,
        target="target",
        task="regression",
        scorer="rmse",
        predictors=["numeric", "category"],
        n_folds=2,
        n_trials=1,
    )
    data = prepare_data(source, config)

    performance = write_performance_plot(
        data, np.asarray([1.4, 2.9, 2.2, 5.1, 5.8, 7.1]), config, tmp_path
    )
    pdp, ranked = write_partial_dependence_plot(
        FakeRegressor(), data, config, tmp_path, grid_points=4, sample_size=4
    )

    assert performance.name == "actual_vs_predicted.png"
    assert performance.stat().st_size > 0
    assert pdp.name == "partial_dependence_top10.png"
    assert pdp.stat().st_size > 0
    assert ranked == ["category", "numeric"]


def test_writes_classification_confusion_matrix(tmp_path):
    source = pd.DataFrame(
        {
            "value": list(range(8)),
            "target": ["no", "yes", "no", "yes", "no", "yes", "no", "yes"],
        }
    )
    config = AutoMLConfig(
        dataset=source,
        target="target",
        task="classification",
        scorer="auc",
        predictors=["value"],
        positive_class="yes",
        n_folds=2,
        n_trials=1,
    )
    data = prepare_data(source, config)

    path = write_performance_plot(
        data,
        np.asarray([0.1, 0.8, 0.7, 0.6, 0.2, 0.4, 0.3, 0.9]),
        config,
        tmp_path,
    )

    assert path.name == "confusion_matrix.png"
    assert path.stat().st_size > 0
