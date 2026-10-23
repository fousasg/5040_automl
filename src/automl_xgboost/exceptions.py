class AutoMLError(Exception):
    """Base pipeline exception."""


class ConfigurationError(AutoMLError, ValueError):
    """Invalid configuration."""


class DataValidationError(ConfigurationError):
    """Invalid input data."""


class TrainingError(AutoMLError):
    """Optimization or model fitting failed."""


class TrialTrainingError(Exception):
    """A recoverable failure of one Optuna trial."""


class ArtifactError(AutoMLError):
    """A required output could not be produced."""

