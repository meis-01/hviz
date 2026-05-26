from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._color import PhaseMode, domain_color, phase_color
from ._quaternion import (
    COMPONENT_LABELS,
    ImagAxis,
    as_quaternion,
    normalize_imag_axis,
    quaternion_norm,
    quaternion_slice,
)

AxisRange = tuple[float, float, int]
QuaternionFunction = Callable[[NDArray[np.float64]], ArrayLike]


@dataclass(frozen=True)
class QuaternionSliceGrid:
    """A 2D quaternion slice and its rendered domain-color image."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    q: NDArray[np.float64]
    values: NDArray[np.float64]
    image: NDArray[np.float64]
    imag_axis: NDArray[np.float64]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (float(self.x[0]), float(self.y[0]), float(self.x[-1]), float(self.y[-1]))


@dataclass(frozen=True)
class QuaternionNormGrid:
    """A ``Re(q)`` by ``||Im(q)||`` grid used for activation diagnostics."""

    real: NDArray[np.float64]
    imag_norm: NDArray[np.float64]
    q: NDArray[np.float64]
    values: NDArray[np.float64]
    phase_image: NDArray[np.float64]
    imag_axis: NDArray[np.float64]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            float(self.real[0]),
            float(self.imag_norm[0]),
            float(self.real[-1]),
            float(self.imag_norm[-1]),
        )


def sample_slice(
    function: QuaternionFunction,
    x_range: AxisRange = (-2.0, 2.0, 400),
    y_range: AxisRange = (-2.0, 2.0, 400),
    *,
    imag_axis: ImagAxis = "i",
    phase_mode: PhaseMode = "signed",
    abs_scaling: Callable[[ArrayLike], ArrayLike] | None = None,
    contours_abs: float | None = 2.0,
    contours_phase: Sequence[float] | None = None,
    emphasize_abs_contour_1: bool = True,
    saturation_adjustment: float = 1.0,
    contour_width: float = 0.045,
    nan_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> QuaternionSliceGrid:
    """Evaluate a quaternion function on ``q = x + y * imag_axis``."""
    x_min, x_max, x_count = _parse_axis_range(x_range, "x_range")
    y_min, y_max, y_count = _parse_axis_range(y_range, "y_range")

    x = np.linspace(x_min, x_max, x_count)
    y = np.linspace(y_min, y_max, y_count)
    real, imag = np.meshgrid(x, y)
    axis = normalize_imag_axis(imag_axis)
    q = quaternion_slice(real, imag, axis)

    values = _evaluate_function(function, q)
    image = domain_color(
        values,
        phase_mode=phase_mode,
        imag_axis=axis,
        abs_scaling=abs_scaling,
        contours_abs=contours_abs,
        contours_phase=contours_phase,
        emphasize_abs_contour_1=emphasize_abs_contour_1,
        saturation_adjustment=saturation_adjustment,
        contour_width=contour_width,
        nan_color=nan_color,
        alpha=True,
    )
    return QuaternionSliceGrid(x=x, y=y, q=q, values=values, image=image, imag_axis=axis)


def sample_re_im_norm(
    function: QuaternionFunction,
    real_range: AxisRange = (-2.0, 2.0, 240),
    imag_norm_range: AxisRange = (0.0, 2.0, 240),
    *,
    imag_axis: ImagAxis = "equal",
    saturation_adjustment: float = 1.0,
    nan_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> QuaternionNormGrid:
    """Evaluate a function on the literature-style ``Re(q)``/``||Im(q)||`` grid."""
    real_min, real_max, real_count = _parse_axis_range(real_range, "real_range")
    imag_min, imag_max, imag_count = _parse_axis_range(imag_norm_range, "imag_norm_range")
    if imag_min < 0.0:
        raise ValueError("imag_norm_range minimum must be non-negative")

    real = np.linspace(real_min, real_max, real_count)
    imag_norm = np.linspace(imag_min, imag_max, imag_count)
    real_grid, imag_norm_grid = np.meshgrid(real, imag_norm)
    axis = normalize_imag_axis(imag_axis)
    q = quaternion_slice(real_grid, imag_norm_grid, axis)

    values = _evaluate_function(function, q)
    image = phase_color(
        values,
        phase_mode="geometric",
        imag_axis=axis,
        saturation_adjustment=saturation_adjustment,
        nan_color=nan_color,
        alpha=True,
    )
    return QuaternionNormGrid(
        real=real,
        imag_norm=imag_norm,
        q=q,
        values=values,
        phase_image=image,
        imag_axis=axis,
    )


def plot_slice(
    function: QuaternionFunction,
    x_range: AxisRange = (-2.0, 2.0, 400),
    y_range: AxisRange = (-2.0, 2.0, 400),
    *,
    imag_axis: ImagAxis = "i",
    phase_mode: PhaseMode = "signed",
    abs_scaling: Callable[[ArrayLike], ArrayLike] | None = None,
    contours_abs: float | None = 2.0,
    contours_phase: Sequence[float] | None = None,
    emphasize_abs_contour_1: bool = True,
    saturation_adjustment: float = 1.0,
    contour_width: float = 0.045,
    nan_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    title: str | None = None,
    width: int = 700,
    height: int = 700,
    add_axes_labels: bool = True,
    tools: Sequence[str] = ("hover", "wheel_zoom", "pan", "reset", "save"),
    active_tools: Sequence[str] = ("wheel_zoom",),
    **opts: Any,
):
    """Return a Bokeh-backed HoloViews RGB plot for a quaternion slice."""
    hv = _holoviews("bokeh")
    grid = sample_slice(
        function,
        x_range,
        y_range,
        imag_axis=imag_axis,
        phase_mode=phase_mode,
        abs_scaling=abs_scaling,
        contours_abs=contours_abs,
        contours_phase=contours_phase,
        emphasize_abs_contour_1=emphasize_abs_contour_1,
        saturation_adjustment=saturation_adjustment,
        contour_width=contour_width,
        nan_color=nan_color,
    )
    kdims = ["Re(q)", _slice_axis_label(grid.imag_axis)] if add_axes_labels else ["x", "y"]
    image = hv.RGB(grid.image, bounds=grid.bounds, kdims=kdims)
    options = {
        "data_aspect": 1,
        "frame_width": width,
        "frame_height": height,
        "tools": list(tools),
        "active_tools": list(active_tools),
        "title": title or "",
    }
    options.update(opts)
    return image.opts(**options)


def plot_components(
    function: QuaternionFunction,
    x_range: AxisRange = (-2.0, 2.0, 300),
    y_range: AxisRange = (-2.0, 2.0, 300),
    *,
    imag_axis: ImagAxis = "equal",
    title: str | None = None,
    width: int = 320,
    height: int = 320,
    cmap: str = "RdBu_r",
    **opts: Any,
):
    """Return Bokeh heatmaps for the four output components ``r, i, j, k``."""
    hv = _holoviews("bokeh")
    grid = sample_slice(function, x_range, y_range, imag_axis=imag_axis)
    kdims = ["Re(q)", _slice_axis_label(grid.imag_axis)]
    images = []
    for index, label in enumerate(COMPONENT_LABELS):
        component = grid.values[..., index]
        clim = _symmetric_clim(component)
        image = hv.Image(component, bounds=grid.bounds, kdims=kdims, vdims=[label]).opts(
            cmap=cmap,
            colorbar=True,
            clim=clim,
            data_aspect=1,
            frame_width=width,
            frame_height=height,
            title=f"{title + ' - ' if title else ''}{label}",
            **opts,
        )
        images.append(image)
    return hv.Layout(images).cols(2)


def plot_magnitude_phase(
    function: QuaternionFunction,
    real_range: AxisRange = (-2.0, 2.0, 240),
    imag_norm_range: AxisRange = (0.0, 2.0, 240),
    *,
    imag_axis: ImagAxis = "equal",
    title: str | None = None,
    width: int = 420,
    height: int = 420,
    cmap: str = "Viridis",
    **opts: Any,
):
    """Return Bokeh 2D panels for output magnitude and quaternion phase."""
    hv = _holoviews("bokeh")
    grid = sample_re_im_norm(function, real_range, imag_norm_range, imag_axis=imag_axis)
    magnitude = quaternion_norm(grid.values)
    kdims = ["Re(q)", "||Im(q)||"]

    magnitude_image = hv.Image(
        magnitude,
        bounds=grid.bounds,
        kdims=kdims,
        vdims=["||f(q)||"],
    ).opts(
        cmap=cmap,
        colorbar=True,
        data_aspect=1,
        frame_width=width,
        frame_height=height,
        title=f"{title + ' - ' if title else ''}Magnitude",
        **opts,
    )
    phase_image = hv.RGB(grid.phase_image, bounds=grid.bounds, kdims=kdims).opts(
        data_aspect=1,
        frame_width=width,
        frame_height=height,
        title=f"{title + ' - ' if title else ''}Phase",
    )
    return (magnitude_image + phase_image).cols(2)


def plot_surface(
    function: QuaternionFunction,
    real_range: AxisRange = (-2.0, 2.0, 120),
    imag_norm_range: AxisRange = (0.0, 2.0, 120),
    *,
    imag_axis: ImagAxis = "equal",
    title: str | None = None,
    width: int = 760,
    height: int = 620,
    cmap: str = "Viridis",
    **opts: Any,
):
    """Return a Plotly-backed HoloViews 3D surface of ``||f(q)||``."""
    hv = _holoviews("plotly")
    grid = sample_re_im_norm(function, real_range, imag_norm_range, imag_axis=imag_axis)
    magnitude = quaternion_norm(grid.values)
    surface = hv.Surface(
        (grid.real, grid.imag_norm, magnitude),
        kdims=["Re(q)", "||Im(q)||"],
        vdims=["||f(q)||"],
    )
    options = {
        "width": width,
        "height": height,
        "cmap": cmap,
        "colorbar": True,
        "title": title or "",
    }
    options.update(opts)
    return surface.opts(**options)


plot = plot_slice


def save(obj: Any, path: str | Path, *, backend: str = "bokeh", **kwargs: Any) -> None:
    """Save a HoloViews object, defaulting to the Bokeh backend."""
    hv = _holoviews(backend)
    hv.save(obj, str(path), backend=backend, **kwargs)


def _parse_axis_range(axis_range: AxisRange, name: str) -> AxisRange:
    if len(axis_range) != 3:
        raise ValueError(f"{name} must be a tuple of (min, max, count)")

    axis_min, axis_max, count = axis_range
    axis_min = float(axis_min)
    axis_max = float(axis_max)
    count = int(count)

    if not np.isfinite(axis_min) or not np.isfinite(axis_max):
        raise ValueError(f"{name} bounds must be finite")
    if axis_min == axis_max:
        raise ValueError(f"{name} min and max must differ")
    if count < 2:
        raise ValueError(f"{name} count must be at least 2")

    return (axis_min, axis_max, count)


def _evaluate_function(function: QuaternionFunction, q: NDArray[np.float64]) -> NDArray[np.float64]:
    with np.errstate(all="ignore"):
        values = as_quaternion(function(q))

    if values.shape == (4,):
        values = np.broadcast_to(values, q.shape).astype(np.float64)
    elif values.shape != q.shape:
        try:
            values = np.broadcast_to(values, q.shape).astype(np.float64)
        except ValueError as exc:
            msg = (
                "function must return a scalar, a quaternion component array matching "
                "the grid shape, or a value broadcastable to the grid shape"
            )
            raise ValueError(msg) from exc

    return values


def _slice_axis_label(axis: NDArray[np.float64]) -> str:
    rounded = np.round(axis, 12)
    known = {
        (1.0, 0.0, 0.0): "Im_i(q)",
        (0.0, 1.0, 0.0): "Im_j(q)",
        (0.0, 0.0, 1.0): "Im_k(q)",
    }
    key = tuple(float(v) for v in rounded)
    if key in known:
        return known[key]
    return "Signed imaginary slice"


def _symmetric_clim(values: NDArray[np.float64]) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0)
    limit = float(np.nanmax(np.abs(finite)))
    if limit == 0.0:
        limit = 1.0
    return (-limit, limit)


def _holoviews(backend: str = "bokeh"):
    import holoviews as hv

    hv.extension(backend)
    return hv
