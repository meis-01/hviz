import numpy as np
import pytest

import hviz


def identity(q):
    return q


def test_sample_slice_shapes_match_grid():
    grid = hviz.sample_slice(identity, (-1.0, 1.0, 7), (-2.0, 2.0, 5))

    assert grid.q.shape == (5, 7, 4)
    assert grid.values.shape == (5, 7, 4)
    assert grid.image.shape == (5, 7, 4)
    assert grid.bounds == (-1.0, -2.0, 1.0, 2.0)


def test_sample_accepts_scalar_quaternion_result():
    grid = hviz.sample_slice(lambda q: [2.0, 3.0, 4.0, 5.0], (-1.0, 1.0, 3), (-1.0, 1.0, 4))

    assert grid.values.shape == (4, 3, 4)
    assert np.all(grid.values == np.array([2.0, 3.0, 4.0, 5.0]))


def test_sample_re_im_norm_rejects_negative_norm_range():
    with pytest.raises(ValueError, match="non-negative"):
        hviz.sample_re_im_norm(identity, (-1.0, 1.0, 4), (-1.0, 1.0, 4))


def test_plot_slice_returns_holoviews_rgb_element():
    hv_plot = hviz.plot_slice(identity, (-1.0, 1.0, 8), (-1.0, 1.0, 6), width=320, height=240)

    assert hv_plot.__class__.__name__ == "RGB"
    assert hv_plot.bounds.lbrt() == (-1.0, -1.0, 1.0, 1.0)


def test_plot_components_returns_layout():
    layout = hviz.plot_components(identity, (-1.0, 1.0, 6), (-1.0, 1.0, 5))

    assert layout.__class__.__name__ == "Layout"
    assert len(layout) == 4


def test_plot_magnitude_phase_returns_two_panel_layout():
    layout = hviz.plot_magnitude_phase(identity, (-1.0, 1.0, 6), (0.0, 1.0, 5))

    assert layout.__class__.__name__ == "Layout"
    assert len(layout) == 2


def test_plot_surface_returns_surface_element():
    surface = hviz.plot_surface(identity, (-1.0, 1.0, 6), (0.0, 1.0, 5))

    assert surface.__class__.__name__ == "Surface"


def test_axis_range_requires_at_least_two_samples():
    with pytest.raises(ValueError, match="count"):
        hviz.sample_slice(identity, (-1.0, 1.0, 1), (-1.0, 1.0, 2))
