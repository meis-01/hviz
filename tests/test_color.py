import numpy as np
import pytest

import hviz


def test_get_srgb1_scalar_returns_rgb_triple():
    rgb = hviz.get_srgb1([1.0, 0.0, 0.0, 0.0])

    assert rgb.shape == (3,)
    assert np.all((0.0 <= rgb) & (rgb <= 1.0))


def test_domain_color_returns_rgba_for_arrays_by_default():
    values = np.array(
        [
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0], [np.inf, 0.0, 0.0, 0.0]],
        ]
    )

    image = hviz.domain_color(values)

    assert image.shape == (2, 2, 4)
    assert np.all((0.0 <= image) & (image <= 1.0))
    assert image[1, 1, 3] == 0.0


def test_phase_color_can_return_rgb_only():
    rgb = hviz.phase_color([1.0, 0.0, 0.0, 0.0], alpha=False)

    assert rgb.shape == (3,)


def test_contours_abs_validates_base():
    with pytest.raises(ValueError, match="contours_abs"):
        hviz.domain_color([[1.0, 0.0, 0.0, 0.0]], contours_abs=1.0)


def test_contour_width_validates_positive_value():
    with pytest.raises(ValueError, match="contour_width"):
        hviz.domain_color([[1.0, 0.0, 0.0, 0.0]], contour_width=0.0)
