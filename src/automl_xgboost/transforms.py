"""Target transformations for regression and their inverses.

Transformations are applied to the regression target before fitting so the model
learns in the transformed space; predictions are inverted back to the
original scale when saved and when scoring new data. The Optuna CV score is
computed in the original-actuals space (predictions are inverted before
scoring).

Transforms may be chained (e.g. ``gamma_shift`` then ``log``); the chain is
applied forward in order and inverted in reverse order.

The automatic ``gamma_shift`` entry carries the shift magnitude, e.g.
``"gamma_shift:12.5"``, because the Gamma objective requires strictly positive
labels and a +1 shift is not always enough. A bare ``"gamma_shift"`` is still
accepted and means +1 (the historical encoding).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

TargetTransform = Literal["none", "log", "sqrt", "cbrt", "gamma_shift"]

_NON_NEGATIVE_TRANSFORMS = {"log", "sqrt"}
# Transforms that are only added automatically (not user-selectable via
# ``target_transform``); ``gamma_shift`` is prepended when ``objective`` is
# ``reg:gamma`` and the target is not strictly positive.
_AUTOMATIC_TRANSFORMS = {"gamma_shift"}
_GAMMA_SHIFT_PREFIX = "gamma_shift"
# Smallest label value guaranteed after the automatic Gamma shift; the Gamma
# objective requires strictly positive labels.
GAMMA_MIN_LABEL = 1.0


def gamma_shift_amount(transform: str) -> float | None:
    """Return the shift carried by a ``gamma_shift`` chain entry, else ``None``."""
    if transform == _GAMMA_SHIFT_PREFIX:
        return 1.0
    if transform.startswith(f"{_GAMMA_SHIFT_PREFIX}:"):
        _, _, raw = transform.partition(":")
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"invalid target transform: {transform!r}") from exc
    return None


def transform_target(values: np.ndarray, transform: TargetTransform | str) -> np.ndarray:
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
    shift = gamma_shift_amount(transform)
    if shift is not None:
        return array + shift
    raise ValueError(f"unknown target transform: {transform!r}")


def invert_target(values: np.ndarray, transform: TargetTransform | str) -> np.ndarray:
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
    shift = gamma_shift_amount(transform)
    if shift is not None:
        return array - shift
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
    target_transform: str, objective: str | None, target_min: float
) -> list[str]:
    """Build the ordered transform chain applied to a regression target.

    The user's ``target_transform`` is appended after any automatic transform.
    When ``objective`` is ``reg:gamma`` the target must be strictly positive, so
    a ``gamma_shift`` is prepended that lifts the smallest label to
    :data:`GAMMA_MIN_LABEL` (a +1 shift when the minimum is zero, a larger shift
    when the target contains negative values).
    """
    chain: list[str] = []
    if objective == "reg:gamma" and float(target_min) <= 0.0:
        shift = GAMMA_MIN_LABEL - float(target_min)
        # repr round-trips exactly, so inversion subtracts precisely the value
        # that was added even after the chain is serialized by the manifest.
        chain.append("gamma_shift" if shift == 1.0 else f"gamma_shift:{shift!r}")
    if target_transform != "none":
        chain.append(target_transform)
    return chain


def validate_target(values: np.ndarray, transform: TargetTransform) -> None:
    """Raise DataValidationError if the target cannot be transformed."""
    from .exceptions import DataValidationError

    if transform == "none":
        return
    if transform in _NON_NEGATIVE_TRANSFORMS:
        array = np.asarray(values, dtype=float)
        if not np.all(array >= 0):
            min_value = float(np.min(array))
            raise DataValidationError(
                f"target_transform {transform!r} requires a non-negative target; "
                f"minimum value is {min_value}"
            )


def validate_chain(values: np.ndarray, chain: list[str]) -> None:
    """Raise DataValidationError if the chain cannot transform the target.

    Transforms are applied progressively so each step is validated against the
    values it will actually receive (e.g. ``log`` after a ``gamma_shift`` sees
    the shifted, strictly positive labels).
    """
    from .exceptions import DataValidationError

    current = np.asarray(values, dtype=float)
    for transform in chain:
        if transform == "none":
            continue
        if transform in _NON_NEGATIVE_TRANSFORMS:
            min_value = float(np.min(current))
            if min_value < 0:
                raise DataValidationError(
                    f"target_transform {transform!r} requires a non-negative target; "
                    f"minimum value is {min_value}"
                )
        current = transform_target(current, transform)
        if not np.all(np.isfinite(current)):
            raise DataValidationError(
                f"target_transform {transform!r} produced non-finite target values"
            )

            continue