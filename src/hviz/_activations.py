from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._quaternion import as_quaternion, imaginary_norm


def quaternion_tanh(values: ArrayLike) -> NDArray[np.float64]:
    """Return the analytic quaternion hyperbolic tangent.

    For ``q = r + v`` with imaginary norm ``b = ||v||``:
    ``tanh(q) = (sinh(2r) + v / b * sin(2b)) / (cosh(2r) + cos(2b))``.
    """
    q = as_quaternion(values)
    real = q[..., 0]
    imag = q[..., 1:]
    imag_length = imaginary_norm(q)

    denominator = np.cosh(2.0 * real) + np.cos(2.0 * imag_length)
    out = np.empty_like(q)
    with np.errstate(all="ignore"):
        out[..., 0] = np.sinh(2.0 * real) / denominator
        imag_scale = np.divide(
            np.sin(2.0 * imag_length),
            imag_length * denominator,
            out=np.zeros_like(imag_length),
            where=imag_length > 0.0,
        )
    out[..., 1:] = imag * imag_scale[..., np.newaxis]
    return out
