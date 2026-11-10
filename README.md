# XGBoost + Optuna + SHAP AutoML pipeline

This is the H2O-free implementation. It uses only XGBoost for modeling, Optuna for hyperparameter optimization, and SHAP for per-row contributions and the HTML explanation report.

## Behavior

- Regression and binary classification
- One shared deterministic fold assignment for every Optuna trial
- Best model selected from combined out-of-fold predictions
- Optional caller-provided folds and numeric monotonicity constraints
- Optional regression target transformation (`log`, `sqrt`, `cbrt`) with predictions inverted back to the original scale when saved and when scoring new data
- Optional XGBoost `objective` override (e.g., `reg:gamma`); when the target contains zeros a +1 shift is applied automatically and inverted on predictions
- **Iterative optimization with automatic variable selection**: each iteration can run up to three sub-rounds — Round 1 (hyper-parameters), an optional Round 1b importance-based backward elimination, and Round 2 (Optuna feature selection) — and carries the winning configuration forward. An optional **bottom-up variable reduction** step (iteration 1 only) drops low-importance predictors and seeds Round 2 from the survivor model. An optional **multimodel variable reduction** pre-iteration stage splits predictors into random low-overlap groups, fits one model per group, averages SHAP-based importance, excludes the bottom predictors, and releases the top survivors across iteration 1 and iteration 2.
- Native XGBoost missing-value routing: predictor nulls are not imputed and rows are not dropped
- Native categorical features with stable pandas category domains
- Out-of-fold predictions, full-data in-sample predictions, full-row SHAP values, optimization graphs, JSON model, metadata, and checksummed manifest
- Out-of-fold actual-versus-predicted or confusion-matrix diagnostics and top-10 partial-dependence plots
- Python API and CLI with YAML/JSON configuration

Targets and fold columns may not contain nulls. Numeric and categorical predictors may contain nulls. The included datasets deliberately exercise both cases.

## Install and run with Conda

No Java installation is required.

```bash
cd 5040_automl

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

Results are written under `runs/`. The bundled example configs keep trial counts low enough for smoke tests; increase `n_trials` and `feature_selection_trials` for real model selection.

### Verbose / console logging

By default the pipeline writes structured progress INFO messages only to `run.log`, so the terminal shows Optuna's own per-trial output and the final JSON status line. Set `verbose: true` (or pass `--verbose`) to mirror the same INFO messages — `run_started`, config/data summary, iteration/round progress, feature-selection trials, bottom-up reduction, refinement bounds, optimization/refit/SHAP/run-completed summaries — to stderr while the run is in progress.

```bash
automl-xgboost run --config examples/regression.yaml --verbose
```

```yaml
verbose: true
```

```python
result = run_automl(AutoMLConfig(..., verbose=True))
```

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

Each iteration can consist of up to three sub-rounds:

1. **Round 1 (hyper-parameters)** — Optuna TPE searches over XGBoost hyper-parameters while holding the active feature set fixed. When `iterations > 1`, the search space is progressively narrowed around the best values using `refine_factor` (the Cyprus-Uni search-space refinement approach).
2. **Round 1b (importance-based backward elimination)** — Optional, runs only in the first `round1b_iterations` iterations. Fixes hyper-parameters to Round 1's best, fits a model, ranks predictors by XGBoost variable importance, and repeatedly drops the least-important survivor. A drop that degrades the CV score does not stop the search; the next least-important predictor is tried instead, and the search stops only after 3 consecutive failures. Every predictor dropped by Round1b is **permanently removed** from the Round 2 candidate pool in all later iterations.
3. **Round 2 (feature selection)** — Fixes hyper-parameters to Round 1's best and uses Optuna binary flags (one per predictor) to search over feature subsets. The first trial is an anchor that preserves the Round 1/Round1b result as a baseline, so feature selection can never regress below the hyper-parameter-only result. The candidate pool is all original predictors except those permanently eliminated by Round1b.

After the rounds, the iteration picks whichever round produced the better CV score and carries that feature set and parameters forward to the next iteration.

### Bottom-up variable reduction (iteration 1 only)

When `bottom_up_variable_reduction` is enabled, an extra reduction step runs in **iteration 1 only**, after Round 1 (or Round 1b if it ran). It fits the best model, ranks predictors by XGBoost variable importance, drops every predictor whose importance is below `bottom_up_importance_threshold`, and rebuilds a model via cross-validation on the survivors. That survivor model becomes the **starting point (anchor) of Round 2**: Round 2's anchor trial starts with only the survivors selected and the dropped predictors unselected, but the dropped predictors **remain available** in Round 2's candidate pool, so Round 2 can re-select them if doing so improves the score. Round 1 of iteration 1 uses `bottom_up_n_trials` trials (falls back to `n_trials` when unset) so a stronger importance ranking feeds the reduction.

### Multimodel variable reduction (pre-iteration stage)

When `multimodel_variable_reduction` is enabled, a variable-reduction stage runs **once at the very start of the pipeline**, before iteration 1. The predictors are split into `multimodel_n_models` random, low-overlap groups (each predictor appears in approximately `multimodel_variable_appearances` groups, so each model runs with about `len(predictors) * multimodel_variable_appearances / multimodel_n_models` predictors; when the totals don't divide evenly, group sizes vary by ±1). One XGBoost model is fit per group, SHAP-based variable importance is computed and saved for each model (`multimodel_importance.csv`), and the importance is averaged across the models. The bottom `multimodel_exclude_bottom` predictors are then excluded (an int count or a float bottom fraction) and **permanently removed** from the pipeline. The surviving predictors are split in two: the top `multimodel_top_k` are made available from iteration 1, and the rest are held out of iteration 1 and **introduced in the Round 2 candidate pool of iteration 2** (so Round 2 of iteration 2 can select them). Round 1 of iteration 1 runs with `bottom_up_n_trials` trials when that is set. This is independent of the Round 1b and bottom-up reductions; all three can be enabled together.

### Configuration

The fields controlling the behavior:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `feature_selection` | bool | `false` | Enable the Round 2 feature-selection sub-run |
| `iterations` | int | `1` | Number of refinement iterations |
| `refine_factor` | float | `0.5` | Fraction to narrow the Round 1 hyper-parameter search space each iteration (0 < value <= 1) |
| `feature_selection_trials` | int | `n_trials` | Optuna trials for the feature-selection round (defaults to `n_trials` when omitted) |
| `max_selected_features` | int | none | Optional cap on how many features Round 2 can select |
| `max_run_seconds` | float | none | Optional wall-clock budget for the whole iterative optimization. After each iteration completes, the elapsed run time is compared against this value and the remaining iterations are skipped (the current best is kept). `none` means no run-time limit. |
| `target_transform` | str | `none` | Regression target transformation applied before fitting: `none`, `log` (log1p, requires a non-negative target), `sqrt` (requires non-negative), or `cbrt` (cube root, handles any real value). Predictions are inverted back to the original scale when saved and when scoring new data. Ignored for classification. |
| `objective` | str | none | XGBoost objective overriding the regression default `reg:squarederror` or the classification default `binary:logistic`. When set to `reg:gamma` and the target contains zeros, a +1 shift is automatically applied to the target so it is strictly positive, and predictions are shifted back (-1) when saved and when scoring. Ignored when `target_transform` is set explicitly. |
| `round1b_iterations` | int | `1` | Number of iterations (from iteration 1) in which the importance-based backward elimination (round1b) runs. `0` disables it; `1` runs it only in the first iteration; values > 1 run it in the first N iterations. Features eliminated by round1b are permanently removed from the Round 2 candidate pool in every later iteration. |
| `bottom_up_variable_reduction` | bool | `false` | Enable the bottom-up importance-threshold reduction in iteration 1 |
| `bottom_up_n_trials` | int | none | Optuna trials for Round 1 of iteration 1 when bottom-up reduction is active (falls back to `n_trials` when unset) |
| `bottom_up_importance_threshold` | float | none | XGBoost variable-importance threshold below which predictors are dropped by the bottom-up reduction (importances normalized to sum to 1). When unset, the threshold is computed dynamically from the number of predictors as `(1 / n_predictors) / 2` (half the average importance); set an explicit float to use a fixed threshold instead. |
| `multimodel_variable_reduction` | bool | `false` | Enable the multimodel variable-reduction pre-iteration stage |
| `multimodel_n_models` | int | `10` | Number of XGBoost models fit in the multimodel stage |
| `multimodel_variable_appearances` | int | `3` | Number of groups each predictor appears in (so each model runs with about `len(predictors) * multimodel_variable_appearances / multimodel_n_models` predictors; when totals don't divide evenly, group sizes vary by ±1) |
| `multimodel_exclude_bottom` | int / float | `0.1` | Bottom predictors to exclude: an int ≥ 1 (a count) or a float 0 < x < 1 (a bottom fraction, e.g. `0.1` = bottom 10%) |
| `multimodel_top_k` | int | `10` | Number of surviving predictors made available from iteration 1; the rest are introduced in iteration 2 Round 2 |

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
max_run_seconds: 3600            # stop iterations after 1 hour wall-clock
target_transform: log              # log-transform the regression target
round1b_iterations: 1          # run round1b only in iteration 1
bottom_up_variable_reduction: true   # importance-threshold reduction in iteration 1
bottom_up_n_trials: 60           # more Round1 trials in iteration 1
bottom_up_importance_threshold: 0.01
multimodel_variable_reduction: true    # pre-iteration multimodel stage
multimodel_n_models: 10
multimodel_variable_appearances: 3
multimodel_exclude_bottom: 0.1         # exclude bottom 10%
multimodel_top_k: 15                     # 15 predictors from iteration 1
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
    max_run_seconds=3600,
    target_transform="log",
    round1b_iterations=1,
    bottom_up_variable_reduction=True,
    bottom_up_n_trials=40,
    bottom_up_importance_threshold=0.01,
    multimodel_variable_reduction=True,
    multimodel_n_models=10,
    multimodel_variable_appearances=3,
    multimodel_exclude_bottom=0.1,
    multimodel_top_k=15,
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
- `optimization_history.html`, `optimization_progress.png` (log-scale y axis), and `trials.csv`
- `actual_vs_predicted.png` for regression or `confusion_matrix.png` for classification, calculated from out-of-fold predictions
- `partial_dependence_top10.png`: one-way PDP panels for up to 10 predictors ranked by fitted XGBoost importance; curves average over a deterministic sample of at most 1,000 rows
- `variable_importance.*` and `variable_importance.png`: fitted-model feature importance values and a ranked bar chart
- `model.json` and `model_metadata.json`, including categorical/class domains
- `resolved_config.yaml`, `run_metadata.json`, `folds.*`, `run.log`, `optuna.db`, and `manifest.json`

## Tests

```bash
pytest
RUN_XGBOOST_INTEGRATION=1 pytest -m integration
```
