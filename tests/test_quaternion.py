import numpy as np
import pytest

import hviz


def test_as_quaternion_embeds_complex_values():
    q = hviz.as_quaternion(np.array([1 + 2j, 3 - 4j]))

    assert q.shape == (2, 4)
    np.testing.assert_allclose(q[:, 0], [1.0, 3.0])
    np.testing.assert_allclose(q[:, 1], [2.0, -4.0])
    np.testing.assert_allclose(q[:, 2:], 0.0)


def test_geometric_phase_is_between_zero_and_pi():
    q = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])

    phase = hviz.geometric_phase(q)

    assert np.all((0.0 <= phase) & (phase <= np.pi))
    assert phase[0] == 0.0
    assert phase[1] == np.pi


def test_signed_phase_uses_selected_axis():
    q = hviz.quaternion_slice([1.0, 1.0], [1.0, -1.0], "j")

    phase = hviz.signed_phase(q, imag_axis="j")

    assert phase[0] > 0.0
    assert phase[1] < 0.0


def test_normalize_imag_axis_rejects_zero_axis():
    with pytest.raises(ValueError, match="non-zero"):
        hviz.normalize_imag_axis([0.0, 0.0, 0.0])
