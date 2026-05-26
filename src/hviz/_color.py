from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._quaternion import ImagAxis, as_quaternion, geometric_phase, quaternion_norm, signed_phase

FloatArray = NDArray[np.floating]
PhaseMode = Literal["signed", "geometric"]


def default_abs_scaling(abs_values: ArrayLike) -> NDArray[np.float64]:
    """Map magnitudes to a bounded lightness scale."""
    values = np.asarray(abs_values, dtype=float)
    return values / (values + 1.0)


def get_srgb1(
    values: ArrayLike,
    *,
    phase_mode: PhaseMode = "signed",
    imag_axis: ImagAxis = "i",
    abs_scaling: Callable[[ArrayLike], ArrayLike] | None = None,
    saturation_adjustment: float = 1.0,
    contours_abs: float | None = None,
    contours_phase: Sequence[float] | None = None,
    emphasize_abs_contour_1: bool = True,
    contour_width: float = 0.045,
    nan_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> NDArray[np.float64]:
    """Return RGB triples in ``[0, 1]`` for quaternion input values."""
    return domain_color(
        values,
        phase_mode=phase_mode,
        imag_axis=imag_axis,
        abs_scaling=abs_scaling,
        saturation_adjustment=saturation_adjustment,
        contours_abs=contours_abs,
        contours_phase=contours_phase,
        emphasize_abs_contour_1=emphasize_abs_contour_1,
        contour_width=contour_width,
        nan_color=nan_color,
        alpha=False,
    )


def domain_color(
    values: ArrayLike,
    *,
    phase_mode: PhaseMode = "signed",
    imag_axis: ImagAxis = "i",
    abs_scaling: Callable[[ArrayLike], ArrayLike] | None = None,
    saturation_adjustment: float = 1.0,
    contours_abs: float | None = 2.0,
    contours_phase: Sequence[float] | None = None,
    emphasize_abs_contour_1: bool = True,
    contour_width: float = 0.045,
    nan_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    alpha: bool = True,
) -> NDArray[np.float64]:
    """Map quaternion values to RGB or RGBA image data.

    ``phase_mode='signed'`` behaves like complex domain coloring on a selected
    imaginary axis. ``phase_mode='geometric'`` colors the quaternion phase
    ``atan2(||Im(q)||, Re(q))`` used by quaternion activation visualizations.
    """
    q = as_quaternion(values)
    abs_values = quaternion_norm(q)
    finite = np.all(np.isfinite(q), axis=-1) & np.isfinite(abs_values)

    scaler = abs_scaling or default_abs_scaling
    with np.errstate(all="ignore"):
        scaled_abs = np.asarray(scaler(abs_values), dtype=float)
    scaled_abs = np.where(np.isfinite(scaled_abs), scaled_abs, 1.0)
    scaled_abs = np.clip(scaled_abs, 0.0, 1.0)

    phase, hue = _phase_and_hue(q, phase_mode=phase_mode, imag_axis=imag_axis)
    lightness = 0.35 + 0.53 * scaled_abs
    chroma = np.clip(0.16 * saturation_adjustment, 0.0, 0.32)
    rgb = _oklch_to_srgb(lightness, chroma, hue)

    contour_strength = _contour_strength(
        abs_values,
        phase,
        phase_mode=phase_mode,
        contours_abs=contours_abs,
        contours_phase=contours_phase,
        emphasize_abs_contour_1=emphasize_abs_contour_1,
        contour_width=contour_width,
    )
    if np.any(contour_strength):
        rgb *= 1.0 - 0.48 * contour_strength[..., np.newaxis]

    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.where(finite[..., np.newaxis], rgb, np.asarray(nan_color, dtype=float))

    if not alpha:
        return rgb

    alpha_channel = finite.astype(float)[..., np.newaxis]
    return np.concatenate([rgb, alpha_channel], axis=-1)


def phase_color(
    values: ArrayLike,
    *,
    phase_mode: PhaseMode = "geometric",
    imag_axis: ImagAxis = "i",
    saturation_adjustment: float = 1.0,
    nan_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    alpha: bool = True,
) -> NDArray[np.float64]:
    """Color only phase, with constant lightness for phase-map panels."""
    q = as_quaternion(values)
    finite = np.all(np.isfinite(q), axis=-1)
    _, hue = _phase_and_hue(q, phase_mode=phase_mode, imag_axis=imag_axis)
    lightness = np.full(q.shape[:-1], 0.72, dtype=float)
    chroma = np.clip(0.17 * saturation_adjustment, 0.0, 0.32)
    rgb = np.clip(_oklch_to_srgb(lightness, chroma, hue), 0.0, 1.0)
    rgb = np.where(finite[..., np.newaxis], rgb, np.asarray(nan_color, dtype=float))
    if not alpha:
        return rgb
    alpha_channel = finite.astype(float)[..., np.newaxis]
    return np.concatenate([rgb, alpha_channel], axis=-1)


def _phase_and_hue(
    values: ArrayLike,
    *,
    phase_mode: PhaseMode,
    imag_axis: ImagAxis,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if phase_mode == "signed":
        phase = signed_phase(values, imag_axis=imag_axis)
        hue = (145.0 + np.degrees(phase)) % 360.0
        return phase, hue
    if phase_mode == "geometric":
        phase = geometric_phase(values)
        hue = (145.0 + 2.0 * np.degrees(phase)) % 360.0
        return phase, hue
    raise ValueError("phase_mode must be 'signed' or 'geometric'")


def _contour_strength(
    abs_values: NDArray[np.float64],
    phase: NDArray[np.float64],
    *,
    phase_mode: PhaseMode,
    contours_abs: float | None,
    contours_phase: Sequence[float] | None,
    emphasize_abs_contour_1: bool,
    contour_width: float,
) -> NDArray[np.float64]:
    strength = np.zeros(abs_values.shape, dtype=float)

    if contours_abs is not None:
        if contours_abs <= 1.0:
            raise ValueError("contours_abs must be greater than 1.0")

        with np.errstate(divide="ignore", invalid="ignore"):
            log_abs = np.log(abs_values) / math.log(contours_abs)
        finite_log = np.isfinite(log_abs)
        distance = np.full(abs_values.shape, np.inf, dtype=float)
        distance[finite_log] = np.abs(log_abs[finite_log] - np.rint(log_abs[finite_log]))
        abs_strength = _soft_line(distance, contour_width)
        strength = np.maximum(strength, abs_strength * 0.72)

        if emphasize_abs_contour_1:
            unit_distance = np.full(abs_values.shape, np.inf, dtype=float)
            unit_distance[finite_log] = np.abs(log_abs[finite_log])
            unit_strength = _soft_line(unit_distance, contour_width * 1.6)
            strength = np.maximum(strength, unit_strength)

    if contours_phase:
        phase_strength = np.zeros(abs_values.shape, dtype=float)
        for contour in contours_phase:
            target = float(contour)
            if phase_mode == "signed":
                distance = np.abs(_wrapped_angle_distance(phase, target))
                width = contour_width * math.pi
            else:
                distance = np.abs(phase - target)
                width = contour_width * math.pi
            phase_strength = np.maximum(phase_strength, _soft_line(distance, width))
        strength = np.maximum(strength, phase_strength * 0.65)

    return np.clip(strength, 0.0, 1.0)


def _soft_line(distance: NDArray[np.float64], width: float) -> NDArray[np.float64]:
    if width <= 0.0:
        raise ValueError("contour_width must be positive")
    return np.exp(-((distance / width) ** 2))


def _wrapped_angle_distance(values: NDArray[np.float64], target: float) -> NDArray[np.float64]:
    return (values - target + math.pi) % (2.0 * math.pi) - math.pi


def _oklch_to_srgb(
    lightness: NDArray[np.float64],
    chroma: float,
    hue_degrees: NDArray[np.float64],
) -> NDArray[np.float64]:
    hue_radians = np.radians(hue_degrees)
    a = chroma * np.cos(hue_radians)
    b = chroma * np.sin(hue_radians)

    l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    s_ = lightness - 0.0894841775 * a - 1.2914855480 * b

    l = l_**3
    m = m_**3
    s = s_**3

    linear_rgb = np.stack(
        [
            +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        ],
        axis=-1,
    )
    return _linear_srgb_to_srgb(linear_rgb)


def _linear_srgb_to_srgb(values: FloatArray) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=float)
    return np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.power(np.maximum(values, 0.0), 1.0 / 2.4) - 0.055,
    )
