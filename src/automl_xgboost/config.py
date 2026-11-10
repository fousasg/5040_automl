"""Strict public configuration and YAML/JSON loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .exceptions import ConfigurationError


class OptunaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study_name: str | None = None
    storage: str | None = None
    load_if_exists: bool = False


class AutoMLConfig(BaseModel):
    """Configuration for one XGBoost optimization run."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    dataset: Path | pd.DataFrame
    target: str
    task: Literal["regression", "classification"]
    scorer: str
    predictors: list[str]
    fold_column: str | None = None
    algorithms: list[str] | None = None
    monotonicity_constraints: dict[str, Literal[-1, 0, 1]] = Field(default_factory=dict)
    positive_class: str | int | float | bool | None = None
    # Optional XGBoost objective overriding the regression default ``reg:squarederror``
    # or the classification default ``binary:logistic``. When set to
    # ``reg:gamma`` and the target contains zeros, a +1 shift is automatically
    # applied to the target so it is strictly positive, and predictions are
    # shifted back (-1) when saved and when scoring. Ignored when ``target_transform`` is
    # set explicitly.
    objective: str | None = None
    # Optional transformation applied to a continuous (regression) target
    # before fitting: ``none`` (default), ``log`` (log1p, requires a non-negative
    # target), ``sqrt`` (requires non-negative), or ``cbrt`` (cube root,
    # handles any real values). Predictions are inverted back to the original
    # scale when saved and when scoring new data. Ignored for classification.
    target_transform: Literal["none", "log", "sqrt", "cbrt"] = "none"

    n_folds: int = Field(default=5, ge=2)
    n_trials: int = Field(default=50, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
    # Optional wall-clock budget for the whole iterative optimization. When set,
    # after each iteration completes the elapsed run time is compared against
    # this value and the remaining iterations are skipped (the current best
    # is kept). None (default) means no run-time limit.
    max_run_seconds: float | None = Field(default=None, gt=0)
    iterations: int = Field(default=1, ge=1)
    refine_factor: float = Field(default=0.5, gt=0.0, le=1.0)
    feature_selection: bool = False
    feature_selection_trials: int | None = Field(default=None, ge=1)
    max_selected_features: int | None = Field(default=None, ge=1)
    # Number of iterations (counting from iteration 1) in which the
    # importance-based backward elimination (round1b) runs. 0 disables it
    # entirely; 1 (default) runs it only in the first iteration; values > 1
    # run it in the first N iterations. Features eliminated by round1b are
    # permanently removed from the Round 2 candidate pool in every later
    # iteration regardless of this setting.
    round1b_iterations: int = Field(default=1, ge=0)
    # Bottom-up variable reduction: after Round 1 (or Round 1b if enabled) of
    # iteration 1, fit the best model, rank predictors by XGBoost variable
    # importance, drop those below ``bottom_up_importance_threshold``, rebuild via
    # cross-validation on the survivors, and treat that model as the best from
    # this point onward. The survivor set also seeds the Round 2 Optuna study.
    bottom_up_variable_reduction: bool = False
    # Number of Optuna trials for Round 1 of iteration 1 when bottom-up variable
    # reduction is active. When None (default), falls back to ``n_trials``.
    bottom_up_n_trials: int | None = Field(default=None, ge=1)
    # XGBoost variable-importance threshold below which predictors are dropped
    # by the bottom-up reduction (importances are normalized to sum to 1).
    # When None (default), the threshold is computed dynamically from
    # the number of predictors the bottom-up model is built on as
    # ``(1 / n_predictors) / 2`` (half the average importance); set an
    # explicit float to use a fixed threshold instead.
    bottom_up_importance_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    # Multimodel variable reduction: a pre-iteration stage that splits the
    # predictors into ``multimodel_n_models`` random, low-overlap groups
    # (each predictor appears in ``multimodel_variable_appearances`` groups,
    # so each model runs with
    # ``len(predictors) * multimodel_variable_appearances / multimodel_n_models``
    # predictors), fits one XGBoost model per group, computes SHAP-based
    # variable importance for each, averages importance across the models,
    # excludes the bottom ``multimodel_exclude_bottom`` predictors (an int
    # count, or a float 0 < x < 1 bottom fraction), then splits the
    # survivors into the top ``multimodel_top_k`` (available from iteration 1)
    # and the rest (held out of iteration 1 and introduced in the Round 2
    # candidate pool of iteration 2). The excluded bottom predictors are
    # permanently removed from the pipeline.
    multimodel_variable_reduction: bool = False
    multimodel_n_models: int = Field(default=10, ge=1)
    multimodel_variable_appearances: int = Field(default=3, ge=1)
    # Bottom predictors to exclude: an int >= 1 (a count) or a float
    # 0 < x < 1 (a bottom fraction, e.g. 0.1 = bottom 10%).
    multimodel_exclude_bottom: int | float = Field(default=0.1)
    # Number of surviving predictors (after excluding the bottom ones) made
    # available from iteration 1; the rest are introduced in iteration 2.
    multimodel_top_k: int = Field(default=10, ge=1)
    # Mirror the run.log INFO messages to the console (stderr) while the run
    # is in progress. When False (default), the application logger only
    # writes to run.log and the terminal shows Optuna's own per-trial output;
    # when True, the same structured INFO lines (run_started, config/data
    # summary, iteration/round progress, feature-selection trials, bottom-up
    # reduction, refinement bounds, optimization/refit/shap/run_completed) are
    # also printed to stderr.
    verbose: bool = False
    seed: int = 42
    output_dir: Path = Path("automl_runs")
    overwrite: bool = False
    output_format: Literal["parquet", "csv"] = "parquet"
    explain_sample_size: int | None = Field(default=10_000, ge=1)
    n_jobs: int = -1
    device: str = "cpu"
    refit_score_relative_tolerance: float = Field(default=1e-10, ge=0)
    refit_score_absolute_tolerance: float = Field(default=1e-12, ge=0)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)

    @field_validator("task", "scorer", mode="before")
    @classmethod
    def normalize_string_enum(cls, value: Any) -> Any:
        return value.lower().strip() if isinstance(value, str) else value

    @field_validator("target", "fold_column")
    @classmethod
    def non_blank_column(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("column names cannot be blank")
        return value

    @field_validator("n_jobs")
    @classmethod
    def nonzero_jobs(cls, value: int) -> int:
        if value == 0:
            raise ValueError("n_jobs cannot be zero")
        return value

    @field_validator("multimodel_exclude_bottom")
    @classmethod
    def multimodel_exclude_bottom_kind(cls, value: int | float) -> int | float:
        if isinstance(value, bool):  # bool is an int subclass; reject explicitly
            raise ValueError("multimodel_exclude_bottom must be an int count or a float fraction")
        if isinstance(value, int):
            if value < 1:
                raise ValueError("int multimodel_exclude_bottom must be >= 1")
            return value
        if isinstance(value, float):
            if not 0.0 < value < 1.0:
                raise ValueError("float multimodel_exclude_bottom must be between 0 and 1 (exclusive)")
            return value
        raise ValueError("multimodel_exclude_bottom must be an int count or a float fraction")

    @field_validator("predictors")
    @classmethod
    def valid_predictors(cls, value: list[str]) -> list[str]:
        normalized = [str(item).strip() for item in value]
        if not normalized or any(not item for item in normalized):
            raise ValueError("predictors must be a non-empty list of names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("predictors must be unique")
        return normalized

    @field_validator("algorithms", mode="before")
    @classmethod
    def only_xgboost(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        normalized = list(dict.fromkeys(str(item).lower().strip() for item in value))
        if normalized != ["xgboost"]:
            raise ValueError("this implementation supports only algorithms: [xgboost]")
        return normalized

    @field_validator("monotonicity_constraints")
    @classmethod
    def remove_zero_constraints(
        cls, value: dict[str, Literal[-1, 0, 1]]
    ) -> dict[str, Literal[-1, 1]]:
        return {str(key): direction for key, direction in value.items() if direction != 0}

    @model_validator(mode="after")
    def column_relationships(self) -> "AutoMLConfig":
        if self.target in self.predictors:
            raise ValueError("target must not be included in predictors")
        if self.fold_column == self.target or self.fold_column in self.predictors:
            raise ValueError("fold_column must differ from target and predictors")
        unknown = set(self.monotonicity_constraints) - set(self.predictors)
        if unknown:
            raise ValueError(f"constrained columns must be predictors: {sorted(unknown)}")
        if self.task != "regression" and self.target_transform != "none":
            raise ValueError(
                "target_transform is only supported for regression tasks"
            )
        # Each predictor appears in `multimodel_variable_appearances` groups across
        # `multimodel_n_models` models, so the average per-model group size is
        # len(predictors) * appearances / n_models. The totals do not have to
        # divide evenly: when they don't, group sizes vary by ±1 and each
        # predictor appears approximately `appearances` times (within ±1),
        # which is the "as stratified as possible" given the counts.
        if self.multimodel_variable_reduction:
            if self.multimodel_n_models > len(self.predictors) * self.multimodel_variable_appearances:
                raise ValueError(
                    "multimodel_n_models cannot exceed the number of "
                    "predictor appearances"
                )
            survivors_upper = len(self.predictors) - self._multimodel_exclude_count(len(self.predictors))
            if self.multimodel_top_k > survivors_upper:
                raise ValueError(
                    f"multimodel_top_k({self.multimodel_top_k}) is larger than the "
                    f"surviving predictors after exclusion ({survivors_upper})"
                )
        return self

    def _multimodel_exclude_count(self, n_predictors: int) -> int:
        """Resolve `multimodel_exclude_bottom` to an integer count."""
        value = self.multimodel_exclude_bottom
        if isinstance(value, int):
            count = value
        else:  # float fraction 0 < x < 1
            count = int(round(value * n_predictors))
        return max(0, min(count, n_predictors))

    def redacted_dict(self) -> dict[str, Any]:
        dumped = self.model_dump(mode="python")
        if isinstance(dumped["dataset"], pd.DataFrame):
            frame = dumped["dataset"]
            dumped["dataset"] = f"<in-memory DataFrame rows={len(frame)} columns={len(frame.columns)}>"
        else:
            dumped["dataset"] = str(dumped["dataset"])
        dumped["output_dir"] = str(dumped["output_dir"])
        if dumped["optuna"].get("storage"):
            dumped["optuna"]["storage"] = redact_url(dumped["optuna"]["storage"])
        return dumped


def redact_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        if not parts.password:
            return url
        user = f"{parts.username}:***@" if parts.username else "***@"
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        return urlunsplit((parts.scheme, f"{user}{host}{port}", parts.path, parts.query, parts.fragment))
    except ValueError:
        return "<redacted-url>"


def load_config_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file() or config_path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise ConfigurationError(f"configuration must be an existing YAML/JSON file: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"could not load configuration: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("configuration root must be a mapping")
    for key in ("dataset", "output_dir"):
        value = raw.get(key)
        if value and not Path(value).expanduser().is_absolute():
            raw[key] = str((config_path.parent / value).resolve())
    return raw
