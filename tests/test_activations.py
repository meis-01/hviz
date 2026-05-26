import numpy as np

import hviz


def test_quaternion_tanh_matches_real_tanh_on_real_axis():
    values = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    q = hviz.as_quaternion(values)

    out = hviz.quaternion_tanh(q)

    np.testing.assert_allclose(out[..., 0], np.tanh(values))
    np.testing.assert_allclose(out[..., 1:], 0.0)


def test_quaternion_tanh_matches_complex_tanh_in_i_plane():
    z = np.array([0.25 + 0.5j, -1.0 + 0.25j, 0.5 - 0.75j])
    q = hviz.as_quaternion(z)

    out = hviz.quaternion_tanh(q)
    expected = np.tanh(z)

    np.testing.assert_allclose(out[..., 0], expected.real)
    np.testing.assert_allclose(out[..., 1], expected.imag)
    np.testing.assert_allclose(out[..., 2:], 0.0, atol=1e-14)
