"""Unit tests for looptools.component."""

import numpy as np
import control
import pytest

from looptools.component import Component, transfer_function
from looptools import dimension as dim


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

SPS = 1000.0  # Sample rate (Hz) for tests


@pytest.fixture
def gain_component():
    """Simple gain component (nume/deno construction)."""
    return Component("Gain", SPS, nume=[2.0], deno=[1.0])


@pytest.fixture
def integrator_component():
    """Discrete integrator 0.01/(1 - z^-1)."""
    return Component("I", SPS, tf="0.01/(1 - z**-1)", domain="z")


@pytest.fixture
def frequency_array():
    """Log-spaced frequency array for Bode/transfer function evaluation."""
    return np.logspace(-2, 2, 50)


# -----------------------------------------------------------------------------
# Construction tests
# -----------------------------------------------------------------------------


class TestComponentConstruction:
    """Tests for Component construction from various input formats."""

    def test_construction_from_nume_deno(self):
        """Component can be constructed from numerator and denominator arrays."""
        c = Component("G", SPS, nume=[1.0, 0.5], deno=[1.0, -0.5])
        assert c.name == "G"
        assert c.sps == SPS
        np.testing.assert_array_almost_equal(c.nume, [1.0, 0.5])
        np.testing.assert_array_almost_equal(c.deno, [1.0, -0.5])
        assert c.TE is not None
        assert c.TF is not None

    def test_construction_from_tf_string_z_domain_gain(self):
        """Component from string in z-domain: constant gain."""
        c = Component("G", SPS, tf="1.5", domain="z")
        np.testing.assert_array_almost_equal(c.nume, [1.5])
        np.testing.assert_array_almost_equal(c.deno, [1.0])

    def test_construction_from_tf_string_z_domain_integrator(self):
        """Component from string in z-domain: discrete integrator."""
        c = Component("I", SPS, tf="0.01/(1 - z**-1)", domain="z")
        # 0.01/(1 - z^-1) = 0.01*z/(z-1) -> nume=[0.01, 0], deno=[1, -1]
        np.testing.assert_array_almost_equal(c.deno, [1.0, -1.0])
        assert c.nume[0] == pytest.approx(0.01)

    def test_construction_from_tf_string_s_domain(self):
        """Component from string in s-domain is discretized via bilinear transform."""
        c = Component("H", SPS, tf="1/(s + 1)", domain="s")
        assert c.TE is not None
        assert c.TF is not None
        # Discretized system should have finite coefficients
        assert np.all(np.isfinite(c.nume))
        assert np.all(np.isfinite(c.deno))

    def test_construction_from_tf_string_s_domain_requires_sps(self):
        """domain='s' requires sps to be set (sps is required in __init__)."""
        # sps is a required positional arg, so this always has it
        c = Component("H", 1000.0, tf="1/(s+1)", domain="s")
        assert c.sps == 1000.0

    def test_construction_from_tf_scalar(self):
        """Component from numeric scalar (gain)."""
        c = Component("G", SPS, tf=3.14)
        np.testing.assert_array_almost_equal(c.nume, [3.14])
        np.testing.assert_array_almost_equal(c.deno, [1.0])

    def test_construction_from_tf_transfer_function(self):
        """Component from control.TransferFunction object."""
        tf_ctrl = control.tf([1.0, 0.5], [1.0, -0.3], 1 / SPS)
        c = Component("G", SPS, tf=tf_ctrl)
        np.testing.assert_array_almost_equal(c.nume, [1.0, 0.5])
        np.testing.assert_array_almost_equal(c.deno, [1.0, -0.3])

    def test_construction_from_tf_tuple(self):
        """Component from (nume, deno) tuple."""
        c = Component("G", SPS, tf=([2.0, 1.0], [1.0, -0.5]))
        np.testing.assert_array_almost_equal(c.nume, [2.0, 1.0])
        np.testing.assert_array_almost_equal(c.deno, [1.0, -0.5])

    def test_construction_invalid_tf_format_raises(self):
        """Unsupported tf format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported tf format"):
            Component("G", SPS, tf={"invalid": "dict"})

    def test_construction_invalid_domain_raises(self):
        """Invalid domain raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized domain"):
            Component("G", SPS, tf="1", domain="invalid")

    def test_construction_wrong_symbol_in_expression_raises(self):
        """Using 's' in expression with domain='z' raises ValueError."""
        with pytest.raises(ValueError, match="TF expression used symbol"):
            Component("G", SPS, tf="1/(s+1)", domain="z")

    def test_construction_unit_default_dimensionless(self):
        """Default unit is dimensionless."""
        c = Component("G", SPS, nume=[1.0], deno=[1.0])
        assert c.unit.numes == ()
        assert c.unit.denos == ()

    def test_construction_unit_custom(self):
        """Custom unit can be passed."""
        unit = dim.Dimension(numes=["m"], denos=["s"])
        c = Component("G", SPS, nume=[1.0], deno=[1.0], unit=unit)
        assert c.unit.numes == ("m",)
        assert c.unit.denos == ("s",)


# -----------------------------------------------------------------------------
# Arithmetic tests
# -----------------------------------------------------------------------------


class TestComponentArithmetic:
    """Tests for Component __add__ and __mul__."""

    def test_add_parallel(self, gain_component):
        """Parallel addition creates combined component."""
        c2 = Component("G2", SPS, nume=[1.0], deno=[1.0])
        combined = gain_component + c2
        assert combined.name == "Gain+G2"
        assert combined.sps == SPS
        assert combined.TE is not None
        assert combined.TF is not None
        # Parallel: G1 + G2 = 2 + 1 = 3 at DC
        f = np.array([0.001])  # Near DC
        val = combined.TF(f=f)
        np.testing.assert_allclose(np.abs(val), 3.0, rtol=1e-5)

    def test_mul_series(self, gain_component):
        """Series multiplication creates combined component."""
        c2 = Component("G2", SPS, nume=[1.5], deno=[1.0])
        combined = gain_component * c2
        assert combined.name == "Gain*G2"
        assert combined.sps == SPS
        # Series: 2 * 1.5 = 3 at DC
        f = np.array([0.001])
        val = combined.TF(f=f)
        np.testing.assert_allclose(np.abs(val), 3.0, rtol=1e-5)

    def test_mul_units_combined(self):
        """Series multiplication combines units."""
        u1 = dim.Dimension(numes=["m"], denos=["s"])
        u2 = dim.Dimension(numes=["s"], denos=[])
        c1 = Component("C1", SPS, nume=[1.0], deno=[1.0], unit=u1)
        c2 = Component("C2", SPS, nume=[1.0], deno=[1.0], unit=u2)
        combined = c1 * c2
        # m/s * s = m (s cancels)
        assert "m" in combined.unit.numes
        assert "s" not in combined.unit.numes and "s" not in combined.unit.denos


# -----------------------------------------------------------------------------
# Method tests
# -----------------------------------------------------------------------------


class TestComponentMethods:
    """Tests for Component methods: modify, update, extrapolate_tf, group_delay, bode."""

    def test_modify_numerator(self, gain_component):
        """modify() updates numerator coefficients."""
        gain_component.modify(5.0)
        np.testing.assert_array_almost_equal(gain_component.nume, [5.0])
        f = np.array([1.0])
        val = gain_component.TF(f=f)
        np.testing.assert_allclose(np.abs(val), 5.0, rtol=1e-5)

    def test_modify_denominator(self):
        """modify() can update denominator when new_deno is provided."""
        c = Component("G", SPS, nume=[1.0], deno=[1.0, -0.5])
        c.modify(2.0, new_deno=[1.0, -0.8])
        np.testing.assert_array_almost_equal(c.nume, [2.0])
        np.testing.assert_array_almost_equal(c.deno, [1.0, -0.8])

    def test_update_refreshes_tf(self, gain_component):
        """update() refreshes TE and TF."""
        gain_component.nume = np.array([10.0])
        gain_component.update()
        f = np.array([1.0])
        val = gain_component.TF(f=f)
        np.testing.assert_allclose(np.abs(val), 10.0, rtol=1e-5)

    def test_extrapolate_tf_sets_callable(self, gain_component, frequency_array):
        """extrapolate_tf() sets TF with extrapolation; callable still works."""
        gain_component.extrapolate_tf(f_trans=1.0, power=-2)
        val = gain_component.TF(f=frequency_array)
        assert val.shape == frequency_array.shape
        assert np.all(np.isfinite(val))

    def test_group_delay_returns_array(self, gain_component):
        """group_delay() returns array of delays."""
        omega = 2 * np.pi * np.array([1.0, 10.0, 100.0])
        delay = gain_component.group_delay(omega)
        assert delay.shape == omega.shape
        assert np.all(np.isfinite(delay))

    def test_bode_returns_mag_and_phase(self, gain_component, frequency_array):
        """bode() returns magnitude and phase arrays."""
        mag, phase = gain_component.bode(frequency_array, dB=False, deg=True)
        assert mag.shape == frequency_array.shape
        assert phase.shape == frequency_array.shape
        np.testing.assert_allclose(mag, 2.0, rtol=1e-5)  # gain = 2
        np.testing.assert_allclose(phase, 0.0, atol=1e-5)  # real gain, zero phase

    def test_bode_dB_mode(self, gain_component, frequency_array):
        """bode() with dB=True returns magnitude in dB."""
        mag, _ = gain_component.bode(frequency_array, dB=True)
        expected_dB = 20 * np.log10(2.0)
        np.testing.assert_allclose(mag, expected_dB, rtol=1e-5)

    def test_bode_radians_mode(self, gain_component, frequency_array):
        """bode() with deg=False returns phase in radians."""
        _, phase = gain_component.bode(frequency_array, deg=False)
        assert np.all(np.abs(phase) <= np.pi + 0.01)

    def test_bode_plot_returns_axes(self, gain_component, frequency_array):
        """bode_plot() returns magnitude and phase axes."""
        ax_mag, ax_phase = gain_component.bode_plot(frequency_array)
        assert ax_mag is not None
        assert ax_phase is not None
        assert ax_mag.figure is ax_phase.figure


# -----------------------------------------------------------------------------
# transfer_function function tests
# -----------------------------------------------------------------------------


class TestTransferFunction:
    """Tests for the standalone transfer_function() function."""

    def test_transfer_function_basic(self, gain_component, frequency_array):
        """transfer_function evaluates component TF at given frequencies."""
        val = transfer_function(frequency_array, gain_component)
        assert val.shape == frequency_array.shape
        np.testing.assert_allclose(np.abs(val), 2.0, rtol=1e-5)

    def test_transfer_function_with_extrapolation(self, gain_component, frequency_array):
        """transfer_function with extrapolate=True applies extrapolation."""
        val = transfer_function(
            frequency_array,
            gain_component,
            extrapolate=True,
            f_trans=1.0,
            power=-2,
        )
        assert val.shape == frequency_array.shape
        assert np.all(np.isfinite(val))
