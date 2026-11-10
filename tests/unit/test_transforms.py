import numpy as np
import pytest

from automl_xgboost.transforms import (
    transform_target,
    invert_target,
    validate_target,
)
from automl_xgboost.exceptions import DataValidationError


def test_transform_invert_roundtrip():
    values = np.array([0.0, 1.0, 10.0, 100.0])
    for transform in ("none", "log", "sqrt", "cbrt"):
        transformed = transform_target(values, transform)
        recovered = invert_target(transformed, transform)
        assert np.allclose(recovered, values)


def test_log_is_log1p():
    values = np.array([0.0, 1.0, np.e])
    assert np.allclose(transform_target(values, "log"), np.log1p(values))


def test_invert_log_is_expm1():
    values = np.array([0.0, 1.0, 2.0])
    assert np.allclose(invert_target(values, "log"), np.expm1(values))


def test_sqrt_and_cbrt():
    values = np.array([0.0, 4.0, 9.0])
    assert np.allclose(transform_target(values, "sqrt"), np.sqrt(values))
    assert np.allclose(invert_target(np.sqrt(values), "sqrt"), values)
    negatives = np.array([-8.0, 0.0, 8.0])
    assert np.allclose(transform_target(negatives, "cbrt"), np.cbrt(negatives))
    assert np.allclose(invert_target(np.cbrt(negatives), "cbrt"), negatives)


def test_validate_target_rejects_negative_for_log():
    with pytest.raises(DataValidationError, match="non-negative"):
        validate_target(np.array([-1.0, 0.0, 1.0]), "log")


def test_validate_target_rejects_negative_for_sqrt():
    with pytest.raises(DataValidationError, match="non-negative"):
        validate_target(np.array([0.0, -1.0]), "sqrt")


def test_validate_target_accepts_cbrt_with_negatives():
    # cbrt is defined for all real values, including negatives.
    validate_target(np.array([-8.0, 0.0, 8.0]), "cbrt")


def test_validate_target_none_is_noop():
    validate_target(np.array([-1.0, 0.0, 1.0]), "none")


def test_unknown_transform_raises():
    with pytest.raises(ValueError, match="unknown target transform"):
        transform_target(np.array([1.0]), "bogus")
