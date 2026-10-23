"""Public API for the XGBoost/Optuna/SHAP AutoML pipeline."""

from .api import run_automl
from .config import AutoMLConfig, OptunaConfig
from .result import AutoMLArtifacts, AutoMLResult, ScoringResult
from .scoring import score_saved_model

__all__ = [
    "AutoMLArtifacts", "AutoMLConfig", "AutoMLResult", "OptunaConfig",
    "ScoringResult", "run_automl", "score_saved_model",
]
__version__ = "0.1.0"
