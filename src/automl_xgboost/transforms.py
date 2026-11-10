"""Target transformations for regression and their inverses.

Transformations are applied to the regression target before fitting so the model
learns in the transformed space; predictions are inverted back to the
original scale when saved and when scoring new data. The Optuna CV score is
computed in the original-actuals space (predictions are inverted before
scoring).

Transforms may be chained (e.g. ``gamma_shift`` then ``log``); the chain is
applied forward in order and inverted in reverse order.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

TargetTransform = Literal["none", "log", "sqrt", "cbrt", "gamma_shift"]

_NON_NEGATIVE_TRANSFORMS = {"log", "sqrt"}
# Transforms that are only added automatically (not user-selectable via
# ``target_transform``); ``gamma_shift`` is prepended when ``objective`` is
# ``reg:gamma`` and the target contains zeros.
_AUTOMATIC_TRANSFORMS = {"gamma_shift"}


def transform_target(values: np.ndarray, transform: TargetTransform) -> np.ndarray:
    """Forward-transform a target array."""
    array = np.asarray(values, dtype=float)
    if transform == "none":
        return array
    if transform == "log":
        return np.log1p(array)
    if transform == "sqrt":
        return np.sqrt(array)
    if transform == "cbrt":
        return np.cbrt(array)
    if transform == "gamma_shift":
        return array + 1.0
    raise ValueError(f"unknown target transform: {transform!r}")


def invert_target(values: np.ndarray, transform: TargetTransform) -> np.ndarray:
    """Invert a transformed target array back to the original scale."""
    array = np.asarray(values, dtype=float)
    if transform == "none":
        return array
    if transform == "log":
        return np.expm1(array)
    if transform == "sqrt":
        return array ** 2
    if transform == "cbrt":
        return array ** 3
    if transform == "gamma_shift":
        return array - 1.0
    raise ValueError(f"unknown target transform: {transform!r}")


def apply_transform_chain(values: np.ndarray, chain: list[str]) -> np.ndarray:
    """Apply a chain of transforms forward (in order)."""
    result = np.asarray(values, dtype=float)
    for transform in chain:
        result = transform_target(result, transform)
    return result


def invert_transform_chain(values: np.ndarray, chain: list[str]) -> np.ndarray:
    """Invert a chain of transforms back to the original scale (reverse order)."""
    result = np.asarray(values, dtype=float)
    for transform in reversed(chain):
        result = invert_target(result, transform)
    return result


def build_target_transform_chain(
    target_transform: str, objective: str | None, target_has_zeros: bool
) -> list[str]:
    """Build the ordered transform chain applied to a regression target.

    The user's ``target_transform`` is appended after any automatic
    transform. When ``objective`` is ``reg:gamma`` and the target contains
    zeros, a ``gamma_shift`` (+1) is prepended so the target is strictly
    positive for the Gamma objective.
    """
    chain: list[str] = []
    if objective == "reg:gamma" and target_has_zeros:
        chain.append("gamma_shift")
    if target_transform != "none":
        chain.append(target_transform)
    return chain


def validate_target(values: np.ndarray, transform: TargetTransform) -> None:
    """Raise DataValidationError if the target cannot be transformed."""
    from .exceptions import DataValidationError

    if transform == "none":
        return
    if transform in _NON_NEGATIVE_TRANSFORMS:
        non_negative = np.all(values >= 0)
        if not non_negative:
            min_value = float(np.min(values))
            raise DataValidationError(
                f"target_transform {transform!r} requires a non-negative target; "
                f"minimum value is {min_value}"
            )
    if transform == "log":
        # log1p is defined for non-negative values; no upper bound needed.
        return

def validate_chain(values: np.ndarray, chain: list[str]) -> None:
    """"Raise DataValidationError if the target cannot be transformed by the chain."""
    from .exceptions import DataValidationError

    for transform in chain:
        if transform == "none":
            continue
        if transform in _NON_NEGATIVE_TRANSFORMS:
            # The gamma_shift happens before non-negative transforms, so validate the
            # already-shifted values (they are >= 1 when the shift was applied).
            if transform == "log":
                continue
            non_negative = np.all(values >= 0)
            if not non_negative:
                min_value = float(np.min(values))
                raise DataValidationError(
                    f"target_transform {transform!r} requires a non-negative target; "
                    f"minimum value is {min_value}"
                )
        if transform == "gamma_shift":
            # gamma_shift only requires that the shift makes the target positive,
            # which is guaranteed by build_target_transform_chain when the
            # original target has zeros; nothing to validate here.
            continue