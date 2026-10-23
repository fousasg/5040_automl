"""XGBoost estimator construction and prediction helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import AutoMLConfig


def create_model(
    config: AutoMLConfig,
    parameters: dict[str, Any],
    selected_features: list[str] | None = None,
) -> Any:
    from xgboost import XGBClassifier, XGBRegressor

    constraints = dict(config.monotonicity_constraints)
    if selected_features is not None:
        feature_set = set(selected_features)
        constraints = {k: v for k, v in constraints.items() if k in feature_set}

    common = {
        **parameters,
        "tree_method": "hist",
        "enable_categorical": True,
        "missing": np.nan,
        "random_state": config.seed,
        "n_jobs": config.n_jobs,
        "device": config.device,
        "verbosity": 0,
        "validate_parameters": True,
        "monotone_constraints": constraints or None,
    }
    if config.task == "regression":
        return XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)


def predict_for_score(model: Any, features: pd.DataFrame, task: str) -> np.ndarray:
    if task == "classification":
        return np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    return np.asarray(model.predict(features), dtype=float)


def suggest_parameters(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 100.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
    }


SEARCH_SPACE_DEFAULTS: dict[str, dict[str, Any]] = {
    "n_estimators": {"kind": "int", "low": 50, "high": 500, "step": 1, "log": False},
    "max_depth": {"kind": "int", "low": 2, "high": 12, "step": 1, "log": False},
    "learning_rate": {"kind": "float", "low": 1e-3, "high": 0.3, "log": True},
    "min_child_weight": {"kind": "float", "low": 1e-2, "high": 100.0, "log": True},
    "subsample": {"kind": "float", "low": 0.5, "high": 1.0, "log": False},
    "colsample_bytree": {"kind": "float", "low": 0.5, "high": 1.0, "log": False},
    "reg_lambda": {"kind": "float", "low": 1e-8, "high": 100.0, "log": True},
    "reg_alpha": {"kind": "float", "low": 1e-8, "high": 100.0, "log": True},
    "gamma": {"kind": "float", "low": 0.0, "high": 10.0, "log": False},
}


def build_search_state() -> dict[str, dict[str, Any]]:
    import copy
    state: dict[str, dict[str, Any]] = {}
    for key, spec in SEARCH_SPACE_DEFAULTS.items():
        entry = copy.deepcopy(spec)
        entry["origin_low"] = entry["low"]
        entry["origin_high"] = entry["high"]
        state[key] = entry
    return state


def suggest_from_state(trial: Any, search_state: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, spec in search_state.items():
        kind = spec["kind"]
        low = spec["low"]
        high = spec["high"]
        if kind == "int":
            step = spec.get("step", 1)
            params[key] = trial.suggest_int(key, int(low), int(high), step=step)
        else:
            params[key] = trial.suggest_float(key, float(low), float(high), log=spec.get("log", False))
    return params


def refine_search_state(
    search_state: dict[str, dict[str, Any]],
    best_params: dict[str, Any],
    refine_factor: float,
) -> None:
    refine_factor = max(0.05, float(refine_factor))

    for key, spec in search_state.items():
        if key not in best_params:
            continue
        kind = spec["kind"]
        if kind not in {"int", "float"}:
            continue

        best_value = float(best_params[key])
        low = float(spec["low"])
        high = float(spec["high"])
        origin_low = float(spec.get("origin_low", low))
        origin_high = float(spec.get("origin_high", high))

        width = max(high - low, 1e-9)
        half_window = width * refine_factor / 2.0

        new_low = max(origin_low, best_value - half_window)
        new_high = min(origin_high, best_value + half_window)

        if kind == "int":
            step = int(spec.get("step", 1))
            new_low = int(round(new_low / step) * step)
            new_high = int(round(new_high / step) * step)
            if new_low == new_high:
                new_low = max(int(origin_low), new_low - step)
                new_high = min(int(origin_high), new_high + step)
            if new_low > new_high:
                new_low, new_high = new_high, new_low
            spec["low"] = float(new_low)
            spec["high"] = float(new_high)
        else:
            if new_low == new_high:
                epsilon = max(1e-6, width * 0.01)
                new_low = max(origin_low, new_low - epsilon)
                new_high = min(origin_high, new_high + epsilon)
            if new_low > new_high:
                new_low, new_high = new_high, new_low
            spec["low"] = float(new_low)
            spec["high"] = float(new_high)

