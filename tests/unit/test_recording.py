import pandas as pd
import pytest

pytest.importorskip("plotly")

from automl_xgboost.recording import write_trials_artifacts


def test_writes_log_optimization_progress(tmp_path):
    trials = pd.DataFrame(
        {
            "trial_number": [0, 1, 2],
            "round": ["single", "single", "single"],
            "iteration": [1, 1, 1],
            "trial_number_within_round": [0, 1, 2],
            "state": ["COMPLETE", "COMPLETE", "COMPLETE"],
            "penalized_score": [10.0, 5.0, 2.5],
            "raw_score": [10.0, 5.0, 2.5],
        }
    )

    trials_path, history, progress = write_trials_artifacts(
        trials, "rmse", "minimize", tmp_path
    )

    assert trials_path.name == "trials.csv"
    assert history.name == "optimization_history.html"
    assert progress.name == "optimization_progress.png"
    assert progress.stat().st_size > 0
