from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

QuaternionArray: TypeAlias = NDArray[np.float64]
FloatArray: TypeAlias = NDArray[np.float64]
ImagAxis: TypeAlias = str | Sequence[float] | NDArray[np.float64]

COMPONENT_LABELS = ("Re(q)", "i(q)", "j(q)", "k(q)")
SHORT_COMPONENT_LABELS = ("r", "i", "j", "k")

_AXES = {
    "i": np.array([1.0, 0.0, 0.0]),
    "j": np.array([0.0, 1.0, 0.0]),
    "k": np.array([0.0, 0.0, 1.0]),
    "equal": np.array([1.0, 1.0, 1.0]),
    "ijk": np.array([1.0, 1.0, 1.0]),
}


def as_quaternion(values: ArrayLike) -> QuaternionArray:
    """Return values as arrays with final component axis ``[r, i, j, k]``.

    Real scalars and real arrays become real-valued quaternions. Complex arrays
    are embedded in the ``1 + i`` plane. Arrays whose last axis has length four
    are treated as already storing quaternion components.
    """
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        complex_values = np.asarray(values, dtype=np.complex128)
        out = np.zeros(complex_values.shape + (4,), dtype=np.float64)
        out[..., 0] = complex_values.real
        out[..., 1] = complex_values.imag
        return out

    array = np.asarray(values, dtype=np.float64)
    if array.shape != () and array.shape[-1] == 4:
        return array

    out = np.zeros(array.shape + (4,), dtype=np.float64)
    out[..., 0] = array
    return out


def quaternion_norm(values: ArrayLike) -> FloatArray:
    """Return ``||q||`` for each quaternion in ``values``."""
    q = as_quaternion(values)
    return np.sqrt(np.sum(q * q, axis=-1))


def imaginary_norm(values: ArrayLike) -> FloatArray:
    """Return ``||Im(q)||`` for each quaternion in ``values``."""
    q = as_quaternion(values)
    return np.sqrt(np.sum(q[..., 1:] * q[..., 1:], axis=-1))


def geometric_phase(values: ArrayLike) -> FloatArray:
    """Return the quaternion phase ``atan2(||Im(q)||, Re(q))`` in ``[0, pi]``."""
    q = as_quaternion(values)
    return np.arctan2(imaginary_norm(q), q[..., 0])


def signed_phase(values: ArrayLike, imag_axis: ImagAxis = "i") -> FloatArray:
    """Return a complex-like signed phase around a chosen imaginary axis."""
    q = as_quaternion(values)
    axis = normalize_imag_axis(imag_axis)
    projected_imag = np.tensordot(q[..., 1:], axis, axes=([-1], [0]))
    return np.arctan2(projected_imag, q[..., 0])


def normalize_imag_axis(axis: ImagAxis = "i") -> FloatArray:
    """Return a unit imaginary direction from ``'i'``, ``'j'``, ``'k'`` or a vector."""
    if isinstance(axis, str):
        try:
            vector = _AXES[axis.lower()]
        except KeyError as exc:
            known = ", ".join(sorted(_AXES))
            raise ValueError(f"imag_axis must be one of {known} or a length-3 vector") from exc
    else:
        vector = np.asarray(axis, dtype=np.float64)

    if vector.shape != (3,):
        raise ValueError("imag_axis vectors must have shape (3,)")

    length = float(np.linalg.norm(vector))
    if length == 0.0 or not np.isfinite(length):
        raise ValueError("imag_axis must have a finite, non-zero length")
    return vector / length


def quaternion_slice(real: ArrayLike, imag: ArrayLike, imag_axis: ImagAxis = "i") -> QuaternionArray:
    """Build ``q = real + imag * axis`` from broadcastable arrays."""
    real_values, imag_values = np.broadcast_arrays(
        np.asarray(real, dtype=np.float64),
        np.asarray(imag, dtype=np.float64),
    )
    axis = normalize_imag_axis(imag_axis)
    q = np.zeros(real_values.shape + (4,), dtype=np.float64)
    q[..., 0] = real_values
    q[..., 1:] = imag_values[..., np.newaxis] * axis
    return q
