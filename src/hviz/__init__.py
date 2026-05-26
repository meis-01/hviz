"""Quaternion activation visualization with HoloViews, Bokeh, and Plotly."""

from ._activations import quaternion_tanh
from ._color import domain_color, get_srgb1, phase_color
from ._plot import (
    AxisRange,
    QuaternionNormGrid,
    QuaternionSliceGrid,
    plot,
    plot_components,
    plot_magnitude_phase,
    plot_slice,
    plot_surface,
    sample_re_im_norm,
    sample_slice,
    save,
)
from ._quaternion import (
    as_quaternion,
    geometric_phase,
    imaginary_norm,
    normalize_imag_axis,
    quaternion_norm,
    quaternion_slice,
    signed_phase,
)

__all__ = [
    "AxisRange",
    "QuaternionNormGrid",
    "QuaternionSliceGrid",
    "as_quaternion",
    "domain_color",
    "geometric_phase",
    "get_srgb1",
    "imaginary_norm",
    "normalize_imag_axis",
    "phase_color",
    "plot",
    "plot_components",
    "plot_magnitude_phase",
    "plot_slice",
    "plot_surface",
    "quaternion_norm",
    "quaternion_slice",
    "quaternion_tanh",
    "sample_re_im_norm",
    "sample_slice",
    "save",
    "signed_phase",
]
