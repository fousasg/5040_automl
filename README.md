# XGBoost + Optuna + SHAP AutoML pipeline

This is the H2O-free implementation. It uses only XGBoost for modeling, Optuna for hyperparameter optimization, and SHAP for per-row contributions and the HTML explanation report.

## Behavior

- Regression and binary classification
- One shared deterministic fold assignment for every Optuna trial
- Best model selected from combined out-of-fold predictions
- Optional caller-provided folds and numeric monotonicity constraints
- **Iterative optimization with automatic variable selection**: each iteration runs two sub-rounds — one for hyper-parameters (Round 1) and one for feature selection (Round 2) — and carries the winning configuration forward
- Native XGBoost missing-value routing: predictor nulls are not imputed and rows are not dropped
- Native categorical features with stable pandas category domains
- Out-of-fold predictions, full-data in-sample predictions, full-row SHAP values, optimization graphs, JSON model, metadata, and checksummed manifest
- Out-of-fold actual-versus-predicted or confusion-matrix diagnostics and top-10 partial-dependence plots
- Python API and CLI with YAML/JSON configuration

Targets and fold columns may not contain nulls. Numeric and categorical predictors may contain nulls. The included datasets deliberately exercise both cases.

## Install and run with Conda

No Java installation is required.

```bash
cd gpt_56_sol_noh2o

conda create --name automl-xgboost --channel conda-forge python=3.11 pip --yes
conda activate automl-xgboost

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --editable . --no-deps
```

Verify and run the examples:

```bash
automl-xgboost --help
automl-xgboost run --config examples/regression.yaml
automl-xgboost run --config examples/classification.yaml
```

Results are written under `runs/`. Each example uses two trials for a practical smoke test; increase `n_trials` for real model selection.

Reactivate or remove the environment later:

```bash
conda activate automl-xgboost

# When no longer needed:
conda deactivate
conda env remove --name automl-xgboost
```

## Python API

```python
from automl_xgboost import AutoMLConfig, run_automl

result = run_automl(AutoMLConfig(
    dataset="data/test_regression_with_nulls.csv",
    target="target",
    task="regression",
    scorer="rmse",
    predictors=["age", "income", "tenure", "region"],
    algorithms=["xgboost"],
    monotonicity_constraints={"income": 1},
    n_folds=3,
    n_trials=10,
    output_dir="runs/python_example",
))

print(result.best_score)
print(result.artifacts.manifest)
```

## Iterative optimization and variable selection

By default the pipeline runs a single Optuna study that tunes XGBoost hyper-parameters using all provided predictors. Enabling feature selection adds a second optimization dimension: the pipeline alternates between tuning hyper-parameters and selecting which predictors to keep.

Each iteration consists of two sub-rounds:

1. **Round 1 (hyper-parameters)** — Optuna TPE searches over XGBoost hyper-parameters while holding the active feature set fixed. When `iterations > 1`, the search space is progressively narrowed around the best values using `refine_factor` (the Cyprus-Uni search-space refinement approach).
2. **Round 2 (feature selection)** — Fixes hyper-parameters to Round 1's best and uses Optuna binary flags (one per predictor) to search over feature subsets. The first trial is an anchor that preserves the Round 1 result as a baseline, so feature selection can never regress below the hyper-parameter-only result.

After both rounds, the iteration picks whichever round produced the better CV score and carries that feature set and parameters forward to the next iteration.

### Configuration

Five fields control the behavior:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `feature_selection` | bool | `false` | Enable the Round 2 feature-selection sub-run |
| `iterations` | int | `1` | Number of refinement iterations |
| `refine_factor` | float | `0.5` | Fraction to narrow the Round 1 hyper-parameter search space each iteration (0 < value <= 1) |
| `feature_selection_trials` | int | `n_trials` | Optuna trials for the feature-selection round (defaults to `n_trials` when omitted) |
| `max_selected_features` | int | none | Optional cap on how many features Round 2 can select |

With the defaults (`iterations: 1`, `feature_selection: false`), behavior is identical to the original single-pass pipeline.

### YAML example

```yaml
dataset: data/my_dataset.csv
target: target
task: regression
scorer: rmse
predictors: [x1, x2, x3, x4, x5, x6, x7, x8]
algorithms: [xgboost]
n_folds: 5
n_trials: 30
iterations: 3
feature_selection: true
feature_selection_trials: 30
max_selected_features: 5
output_dir: runs/with_feature_selection
```

### Python API

```python
from automl_xgboost import AutoMLConfig, run_automl

result = run_automl(AutoMLConfig(
    dataset="data/test_regression_with_nulls.csv",
    target="target",
    task="regression",
    scorer="rmse",
    predictors=["age", "income", "tenure", "region"],
    algorithms=["xgboost"],
    n_folds=3,
    n_trials=20,
    iterations=2,
    feature_selection=True,
    feature_selection_trials=20,
    max_selected_features=3,
    output_dir="runs/feature_selection_example",
))

print(result.best_score)
print(result.selected_features)  # subset of predictors chosen by the pipeline
```

The `result.selected_features` field contains the final subset of predictors. When feature selection is disabled it is `None`. The saved `model_metadata.json` records both `predictors` (the features the model uses) and `all_predictors` (the original candidate list), so scoring with `automl-xgboost score` works without changes.

## Score new data with a saved model

Point `--run-dir` at the exact successful run directory containing `manifest.json`, `model.json`, and `model_metadata.json`. The scoring dataset needs the saved predictors but does not need the target column. Write predictions outside the immutable training run directory so its manifest remains authoritative.

```bash
automl-xgboost score \
  --run-dir runs/classification \
  --dataset data/test_classification_with_nulls.csv \
  --output scored/classification_predictions.csv
```

If a later run created a timestamped child directory, use that exact child:

```bash
automl-xgboost score \
  --run-dir runs/classification/20260827T171105Z_e647a98b \
  --dataset data/new_customers.parquet \
  --output scored/new_customer_predictions.parquet
```

Python API:

```python
from automl_xgboost import score_saved_model

scored = score_saved_model(
    run_dir="runs/regression",
    dataset="data/new_rows.csv",  # may also be a pandas DataFrame
    output_path="scored/regression_predictions.csv",  # optional
)
print(scored.predictions)
```

Before loading, the scorer verifies model and metadata checksums against the run manifest. Predictor nulls remain null for XGBoost's native missing-value routing. Unknown categorical levels fail with an actionable error rather than being silently treated as missing.

## Null-value handling

The pipeline preserves predictor nulls as `NaN`/missing categories. XGBoost learns a default branch for missing values at each split. The run metadata and SHAP report record missing counts, and the prediction/SHAP outputs retain exactly one row for every input row.

## Artifacts

- `cv_predictions.*`: combined out-of-fold predictions
- `full_data_predictions.*`: in-sample predictions from the model fitted on all rows
- `shap_values.*`: one contribution per predictor plus `BiasTerm`, on raw-margin scale
- `shap_report.html` and `shap_assets/`: global importance, beeswarm, and local waterfall views
- `optimization_history.html`, `optimization_progress.png`, and `trials.csv`
- `actual_vs_predicted.png` for regression or `confusion_matrix.png` for classification, calculated from out-of-fold predictions
- `partial_dependence_top10.png`: one-way PDP panels for up to 10 predictors ranked by fitted XGBoost importance; curves average over a deterministic sample of at most 1,000 rows
- `model.json` and `model_metadata.json`, including categorical/class domains
- `resolved_config.yaml`, `run_metadata.json`, `folds.*`, `run.log`, `optuna.db`, and `manifest.json`

## Tests

```bash
pytest
RUN_XGBOOST_INTEGRATION=1 pytest -m integration
```
