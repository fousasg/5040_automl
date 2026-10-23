import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl_xgboost import AutoMLConfig, run_automl, score_saved_model
from automl_xgboost.data import prepare_data


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_XGBOOST_INTEGRATION") != "1",
        reason="set RUN_XGBOOST_INTEGRATION=1 to run integration tests",
    ),
]
ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    ("filename", "target", "task", "scorer", "predictors", "positive_class"),
    [
        ("test_regression_with_nulls.csv", "target", "regression", "rmse", ["age", "income", "tenure", "region", "noise"], None),
        ("test_classification_with_nulls.csv", "churned", "classification", "aucpr", ["age", "monthly_spend", "tenure", "support_calls", "region"], "yes"),
    ],
)
def test_real_pipeline_with_nulls(tmp_path, filename, target, task, scorer, predictors, positive_class):
    dataset = ROOT / "data" / filename
    config = AutoMLConfig(
        dataset=dataset, target=target, task=task, scorer=scorer,
        predictors=predictors, positive_class=positive_class,
        algorithms=["xgboost"], n_folds=2, n_trials=1,
        output_dir=tmp_path / task, output_format="csv", explain_sample_size=25, n_jobs=2,
    )
    result = run_automl(config)
    source = pd.read_csv(dataset)
    manifest = json.loads(result.artifacts.manifest.read_text())
    assert manifest["status"] == "success"
    for path in (result.artifacts.cv_predictions, result.artifacts.full_data_predictions, result.artifacts.shap_values):
        assert len(pd.read_csv(path)) == len(source)
    assert source[predictors].isna().any(axis=1).sum() > 0
    assert result.artifacts.shap_report.is_file()
    assert result.artifacts.performance_plot.is_file()
    assert result.artifacts.partial_dependence_plot.is_file()
    assert result.artifacts.variable_importance.is_file()
    assert result.artifacts.variable_importance_plot.is_file()
    expected_performance_name = "actual_vs_predicted.png" if task == "regression" else "confusion_matrix.png"
    assert result.artifacts.performance_plot.name == expected_performance_name
    manifest_paths = {artifact["path"] for artifact in manifest["artifacts"]}
    assert expected_performance_name in manifest_paths
    assert "partial_dependence_top10.png" in manifest_paths
    assert f"variable_importance.{config.output_format}" in manifest_paths
    assert "variable_importance.png" in manifest_paths

    scored = score_saved_model(
        result.output_dir,
        dataset,
        tmp_path / f"{task}_rescored.csv",
    )
    assert len(scored.predictions) == len(source)

    import xgboost as xgb

    loaded = xgb.XGBRegressor() if task == "regression" else xgb.XGBClassifier()
    loaded.load_model(result.artifacts.model)
    prepared = prepare_data(source, config)
    if task == "regression":
        reproduced = loaded.predict(prepared.features)
        saved = pd.read_csv(result.artifacts.full_data_predictions)["prediction"]
        assert np.allclose(scored.predictions["prediction"], saved)
    else:
        reproduced = loaded.predict_proba(prepared.features)[:, 1]
        saved = pd.read_csv(result.artifacts.full_data_predictions)["probability_yes"]
        assert np.allclose(scored.predictions["probability_yes"], saved)
    assert np.allclose(reproduced, saved)


@pytest.mark.parametrize(
    ("filename", "target", "task", "scorer", "predictors", "positive_class"),
    [
        ("test_regression_with_nulls.csv", "target", "regression", "rmse", ["age", "income", "tenure", "region", "noise"], None),
        ("test_classification_with_nulls.csv", "churned", "classification", "aucpr", ["age", "monthly_spend", "tenure", "support_calls", "region"], "yes"),
    ],
)
def test_feature_selection_pipeline(tmp_path, filename, target, task, scorer, predictors, positive_class):
    dataset = ROOT / "data" / filename
    config = AutoMLConfig(
        dataset=dataset, target=target, task=task, scorer=scorer,
        predictors=predictors, positive_class=positive_class,
        algorithms=["xgboost"], n_folds=2, n_trials=2,
        iterations=2, feature_selection=True, feature_selection_trials=3,
        output_dir=tmp_path / task, output_format="csv", explain_sample_size=25, n_jobs=2,
    )
    result = run_automl(config)
    source = pd.read_csv(dataset)
    manifest = json.loads(result.artifacts.manifest.read_text())
    assert manifest["status"] == "success"
    assert result.selected_features is not None
    assert len(result.selected_features) >= 1
    assert set(result.selected_features).issubset(set(predictors))
    for path in (result.artifacts.cv_predictions, result.artifacts.full_data_predictions):
        assert len(pd.read_csv(path)) == len(source)

    metadata = json.loads(result.artifacts.model_metadata.read_text())
    assert metadata["predictors"] == result.selected_features
    assert metadata["all_predictors"] == predictors
    assert metadata["selected_features"] == result.selected_features
