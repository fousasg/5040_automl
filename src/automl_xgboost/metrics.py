"""Explicit combined out-of-fold scorer registry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .exceptions import ConfigurationError, TrainingError

Direction = Literal["minimize", "maximize"]


@dataclass(frozen=True, slots=True)
class MetricSpec:
    name: str
    task: str
    direction: Direction


_SPECS = [
    MetricSpec("rmse", "regression", "minimize"),
    MetricSpec("mse", "regression", "minimize"),
    MetricSpec("mae", "regression", "minimize"),
    MetricSpec("rmsle", "regression", "minimize"),
    MetricSpec("mean_residual_deviance", "regression", "minimize"),
    MetricSpec("auc", "classification", "maximize"),
    MetricSpec("aucpr", "classification", "maximize"),
    MetricSpec("logloss", "classification", "minimize"),
    MetricSpec("mean_per_class_error", "classification", "minimize"),
    MetricSpec("rmse", "classification", "minimize"),
    MetricSpec("mse", "classification", "minimize"),
]
METRICS = {(spec.task, spec.name): spec for spec in _SPECS}


def get_metric(task: str, scorer: str) -> MetricSpec:
    try:
        return METRICS[(task, scorer)]
    except KeyError as exc:
        choices = sorted(spec.name for spec in _SPECS if spec.task == task)
        raise ConfigurationError(f"scorer {scorer!r} is invalid for {task}; choose from {choices}") from exc


def score_predictions(spec: MetricSpec, actual: np.ndarray, prediction: np.ndarray) -> float:
    from sklearn.metrics import (
        average_precision_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        mean_squared_log_error,
        roc_auc_score,
    )

    if spec.task == "regression":
        if spec.name == "mae":
            value = mean_absolute_error(actual, prediction)
        elif spec.name == "rmsle":
            if (actual < 0).any() or (prediction < 0).any():
                raise TrainingError("rmsle requires non-negative actual values and predictions")
            value = math.sqrt(mean_squared_log_error(actual, prediction))
        else:
            mse = mean_squared_error(actual, prediction)
            value = math.sqrt(mse) if spec.name == "rmse" else mse
    else:
        if spec.name == "auc":
            value = roc_auc_score(actual, prediction)
        elif spec.name == "aucpr":
            value = average_precision_score(actual, prediction)
        elif spec.name == "logloss":
            value = log_loss(actual, prediction, labels=[0, 1])
        elif spec.name == "mean_per_class_error":
            labels = prediction >= 0.5
            errors = [np.mean(labels[actual == klass] != klass) for klass in (0, 1)]
            value = float(np.mean(errors))
        else:
            mse = mean_squared_error(actual, prediction)
            value = math.sqrt(mse) if spec.name == "rmse" else mse
    scalar = float(value)
    if not math.isfinite(scalar):
        raise TrainingError(f"scorer produced a non-finite value: {scalar}")
    return scalar


def worst_score(direction: Direction) -> float:
    return float("inf") if direction == "minimize" else float("-inf")


def is_better(new: float | None, old: float | None, direction: Direction) -> bool:
    if new is None:
        return False
    if old is None:
        return True
    return new > old if direction == "maximize" else new < old


def running_best(values: list[float], direction: Direction) -> list[float]:
    result: list[float] = []
    if not values:
        return result
    incumbent = values[0]
    choose = max if direction == "maximize" else min
    for value in values:
        incumbent = choose(incumbent, value)
        result.append(incumbent)
    return result

