"""Deterministic fold assignment shared by every trial."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .exceptions import DataValidationError


def generate_folds(target: pd.Series, task: str, n_folds: int, seed: int) -> pd.Series:
    row_count = len(target)
    if not 2 <= n_folds <= row_count:
        raise DataValidationError(f"n_folds must be between 2 and row count ({row_count})")
    rng = np.random.default_rng(seed)
    folds = np.full(row_count, -1, dtype=np.int32)
    if task == "classification":
        for value in pd.unique(target):
            indices = np.flatnonzero(target.to_numpy() == value)
            if len(indices) < n_folds:
                raise DataValidationError(
                    f"class {value!r} has fewer rows ({len(indices)}) than n_folds ({n_folds})"
                )
            shuffled = rng.permutation(indices)
            folds[shuffled] = np.arange(len(shuffled), dtype=np.int32) % n_folds
    else:
        shuffled = rng.permutation(row_count)
        folds[shuffled] = np.arange(row_count, dtype=np.int32) % n_folds
    if (folds < 0).any() or len(np.unique(folds)) != n_folds:
        raise DataValidationError("failed to generate non-empty folds")
    return pd.Series(folds, name="fold")

