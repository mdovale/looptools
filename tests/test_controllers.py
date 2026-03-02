"""Unit tests for loopkit controller components."""

import copy
import numpy as np
import pytest

from loopkit.components import PIControllerComponent


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

SPS = 20e6  # Sample rate (Hz) - Simulink PLL fLoop


# -----------------------------------------------------------------------------
# PIControllerComponent gain_scale tests
# -----------------------------------------------------------------------------


class TestPIControllerGainScale:
    """Tests for PIControllerComponent gain_scale (log2 vs linear)."""

    def test_pi_controller_linear_gains(self):
        """Kp=5000, Ki=4000 with gain_scale='linear' produces correct TF."""
        pi = PIControllerComponent(
            "PI",
            sps=SPS,
            Kp=5000.0,
            Ki=4000.0,
            gain_scale="linear",
        )
        assert pi.Kp == pytest.approx(5000.0)
        assert pi.Ki == pytest.approx(4000.0)
        assert pi.gain_scale == "linear"
        # PI TF from control.parallel: (Kp - (Kp-Ki)*z^-1)/(1-z^-1)
        expected_nume = np.array([5000.0, -1000.0])
        expected_deno = np.array([1.0, -1.0])
        np.testing.assert_array_almost_equal(pi.nume, expected_nume)
        np.testing.assert_array_almost_equal(pi.deno, expected_deno)

    def test_pi_controller_log2_gains_unchanged(self):
        """Default gain_scale='log2' preserves existing behavior."""
        # log2(5000) ≈ 12.29, log2(4000) ≈ 11.97
        pi = PIControllerComponent(
            "PI",
            sps=SPS,
            Kp=np.log2(5000),
            Ki=np.log2(4000),
        )
        assert pi.Kp == pytest.approx(5000.0, rel=1e-10)
        assert pi.Ki == pytest.approx(4000.0, rel=1e-10)
        assert pi.gain_scale == "log2"
        expected_nume = np.array([5000.0, -1000.0])
        expected_deno = np.array([1.0, -1.0])
        np.testing.assert_array_almost_equal(pi.nume, expected_nume)
        np.testing.assert_array_almost_equal(pi.deno, expected_deno)

    def test_pi_controller_linear_equals_log2_same_effective_gains(self):
        """Linear and log2 produce identical TF when effective gains match."""
        pi_linear = PIControllerComponent(
            "PI", sps=SPS, Kp=5000, Ki=4000, gain_scale="linear"
        )
        pi_log2 = PIControllerComponent(
            "PI", sps=SPS, Kp=np.log2(5000), Ki=np.log2(4000)
        )
        np.testing.assert_array_almost_equal(pi_linear.nume, pi_log2.nume)
        np.testing.assert_array_almost_equal(pi_linear.deno, pi_log2.deno)

    def test_pi_controller_linear_setter(self):
        """Kp/Ki setters with gain_scale='linear' accept direct values."""
        pi = PIControllerComponent(
            "PI", sps=SPS, Kp=100, Ki=50, gain_scale="linear"
        )
        pi.Kp = 200
        pi.Ki = 100
        assert pi.Kp == pytest.approx(200.0)
        assert pi.Ki == pytest.approx(100.0)
        # PI TF: (Kp - (Kp-Ki)*z^-1)/(1-z^-1) -> [200, -100]
        expected_nume = np.array([200.0, -100.0])
        np.testing.assert_array_almost_equal(pi.nume, expected_nume)

    def test_pi_controller_log2_setter(self):
        """Kp/Ki setters with gain_scale='log2' accept log2 exponents."""
        pi = PIControllerComponent("PI", sps=SPS, Kp=4, Ki=2)  # 2^4=16, 2^2=4
        pi.Kp = 5  # 2^5 = 32
        pi.Ki = 3  # 2^3 = 8
        assert pi.Kp == pytest.approx(32.0)
        assert pi.Ki == pytest.approx(8.0)

    def test_pi_controller_gain_scale_invalid_raises(self):
        """Invalid gain_scale raises ValueError."""
        with pytest.raises(ValueError, match="gain_scale must be"):
            PIControllerComponent(
                "PI", sps=SPS, Kp=1, Ki=1, gain_scale="invalid"
            )

    def test_pi_controller_deepcopy_preserves_gain_scale(self):
        """Deep copy preserves gain_scale and effective gains."""
        pi = PIControllerComponent(
            "PI", sps=SPS, Kp=5000, Ki=4000, gain_scale="linear"
        )
        pi_copy = copy.deepcopy(pi)
        assert pi_copy.gain_scale == "linear"
        assert pi_copy.Kp == pytest.approx(5000.0)
        assert pi_copy.Ki == pytest.approx(4000.0)
        np.testing.assert_array_almost_equal(pi.nume, pi_copy.nume)
        np.testing.assert_array_almost_equal(pi.deno, pi_copy.deno)
