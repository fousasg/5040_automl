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

    n_folds: int = Field(default=5, ge=2)
    n_trials: int = Field(default=50, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
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
        return self

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
