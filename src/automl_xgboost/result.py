from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class AutoMLArtifacts:
    manifest: Path
    resolved_config: Path
    run_metadata: Path
    folds: Path
    trials: Path
    optimization_history: Path
    optimization_progress: Path
    cv_predictions: Path
    full_data_predictions: Path
    performance_plot: Path
    partial_dependence_plot: Path
    variable_importance: Path
    variable_importance_plot: Path
    shap_values: Path
    shap_report: Path
    model: Path
    model_metadata: Path
    optuna_database: Path | None
    log: Path


@dataclass(frozen=True, slots=True)
class AutoMLResult:
    run_id: str
    output_dir: Path
    best_model: Any
    study: Any
    best_score: float
    scorer: str
    direction: Literal["minimize", "maximize"]
    best_algorithm: str
    best_params: dict[str, Any]
    best_trial_number: int
    artifacts: AutoMLArtifacts
    selected_features: list[str] | None = None


@dataclass(frozen=True, slots=True)
class ScoringResult:
    run_dir: Path
    predictions: Any
    output_path: Path | None
    model: Any
