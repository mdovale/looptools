"""Unit tests for loopkit filter components."""

import numpy as np
import pytest

from loopkit.component import Component
from loopkit.components import IIRFilterComponent, iir_from_sos


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

SPS = 20e6  # Sample rate for IIR tests (Simulink PLL fLoop)
SOS_SCALED_26 = [
    16777216,
    33554432,
    16777216,
    16777216,
    -33181752,
    16408629,
]


# -----------------------------------------------------------------------------
# iir_from_sos tests
# -----------------------------------------------------------------------------


class TestIirFromSos:
    """Tests for iir_from_sos helper."""

    def test_basic_sos_conversion(self):
        """iir_from_sos returns correct (nume, deno) for single section."""
        nume, deno = iir_from_sos(SOS_SCALED_26)
        # Normalized: b/a0 = [1, 2, 1], a/a0 = [1, a1/a0, a2/a0]
        np.testing.assert_array_almost_equal(deno, [1.0, -1.9778, 0.9780], decimal=3)
        np.testing.assert_array_almost_equal(nume, [1.0, 2.0, 1.0], decimal=5)

    def test_sos_with_scaling(self):
        """iir_from_sos applies input_scale and output_scale correctly."""
        input_scale = 2**-24
        output_scale = 2**-14
        nume, deno = iir_from_sos(SOS_SCALED_26, input_scale, output_scale)
        expected_scale = input_scale * output_scale
        np.testing.assert_array_almost_equal(
            nume,
            np.array([1.0, 2.0, 1.0]) * expected_scale,
            decimal=15,
        )
        np.testing.assert_array_almost_equal(deno, [1.0, -1.9778, 0.9780], decimal=3)

    def test_sos_too_short_raises(self):
        """iir_from_sos raises if sos has fewer than 6 elements."""
        with pytest.raises(ValueError, match="at least 6 elements"):
            iir_from_sos([1, 2, 3, 4, 5])

    def test_sos_zero_a0_raises(self):
        """iir_from_sos raises if a0 is zero."""
        sos_bad = [1, 2, 1, 0, -2, 1]  # a0=0
        with pytest.raises(ValueError, match="a0.*non-zero"):
            iir_from_sos(sos_bad)


# -----------------------------------------------------------------------------
# IIRFilterComponent tests
# -----------------------------------------------------------------------------


class TestIIRFilterComponent:
    """Tests for IIRFilterComponent."""

    def test_basic_construction(self):
        """IIRFilterComponent constructs with (b, a) and no scaling."""
        nume = np.array([1.0, 0.5])
        deno = np.array([1.0, -0.5])
        iir = IIRFilterComponent("IIR", SPS, nume, deno)
        assert iir.name == "IIR"
        assert iir.sps == SPS
        np.testing.assert_array_almost_equal(iir.nume, [1.0, 0.5])
        np.testing.assert_array_almost_equal(iir.deno, [1.0, -0.5])
        assert iir.input_scale == 1.0
        assert iir.output_scale == 1.0
        assert iir.TE is not None
        assert iir.TF is not None

    def test_construction_with_scaling(self):
        """IIRFilterComponent applies input_scale and output_scale."""
        input_scale = 2**-24
        output_scale = 2**-14
        nume = np.array([1.0, 2.0, 1.0])
        deno = np.array([1.0, -1.9778, 0.9780])
        iir = IIRFilterComponent(
            "IIR", SPS, nume, deno,
            input_scale=input_scale,
            output_scale=output_scale,
        )
        expected = nume * input_scale * output_scale
        np.testing.assert_array_almost_equal(iir.nume, expected, decimal=15)
        np.testing.assert_array_almost_equal(iir.deno, deno, decimal=10)

    def test_deno_normalization(self):
        """IIRFilterComponent normalizes denominator so a0=1."""
        nume = np.array([2.0, 4.0, 2.0])
        deno = np.array([4.0, -2.0, 1.0])  # a0=4
        iir = IIRFilterComponent("IIR", SPS, nume, deno)
        np.testing.assert_array_almost_equal(iir.deno, [1.0, -0.5, 0.25])
        np.testing.assert_array_almost_equal(iir.nume, [0.5, 1.0, 0.5])

    def test_matches_manual_component(self):
        """IIRFilterComponent matches manual Component for sosScaled26."""
        nume, deno = iir_from_sos(SOS_SCALED_26, 2**-24, 2**-14)
        iir = IIRFilterComponent("IIR", SPS, nume, deno)
        manual = Component("Manual", SPS, nume=nume, deno=deno)
        np.testing.assert_array_almost_equal(iir.nume, manual.nume)
        np.testing.assert_array_almost_equal(iir.deno, manual.deno)

    def test_from_sos_class_method(self):
        """IIRFilterComponent.from_sos creates correct filter."""
        iir = IIRFilterComponent.from_sos(
            "IIR",
            SPS,
            SOS_SCALED_26,
            input_scale=2**-24,
            output_scale=2**-14,
        )
        nume, deno = iir_from_sos(SOS_SCALED_26, 2**-24, 2**-14)
        np.testing.assert_array_almost_equal(iir.nume, nume)
        np.testing.assert_array_almost_equal(iir.deno, deno)

    def test_bode_response(self):
        """IIRFilterComponent produces expected Bode shape (low-pass)."""
        nume = np.array([1.0, 2.0, 1.0])
        deno = np.array([1.0, -1.9778, 0.9780])
        iir = IIRFilterComponent("IIR", SPS, nume, deno)
        # Stay well below Nyquist (10 MHz) to avoid log10(0) = -inf at zeros
        frfr = np.logspace(0, 6, 100)
        mag, phase = iir.bode(frfr, dB=True)
        # Low-pass: high freq should roll off
        assert mag[0] > mag[-1]
        assert np.all(np.isfinite(mag))
        assert np.all(np.isfinite(phase))

    def test_empty_nume_raises(self):
        """IIRFilterComponent raises on empty nume."""
        with pytest.raises(ValueError, match="nume must be non-empty"):
            IIRFilterComponent("IIR", SPS, [], [1.0])

    def test_empty_deno_raises(self):
        """IIRFilterComponent raises on empty deno."""
        with pytest.raises(ValueError, match="deno must be non-empty"):
            IIRFilterComponent("IIR", SPS, [1.0], [])

    def test_zero_deno_leading_raises(self):
        """IIRFilterComponent raises when deno[0]=0."""
        with pytest.raises(ValueError, match="deno.*leading.*non-zero"):
            IIRFilterComponent("IIR", SPS, [1.0], [0.0, 1.0])

    def test_deepcopy(self):
        """IIRFilterComponent deepcopy preserves state."""
        iir = IIRFilterComponent(
            "IIR", SPS,
            np.array([1.0, 2.0, 1.0]),
            np.array([1.0, -1.98, 0.98]),
            input_scale=2**-24,
            output_scale=2**-14,
        )
        import copy
        iir2 = copy.deepcopy(iir)
        assert iir2.name == iir.name
        assert iir2.sps == iir.sps
        np.testing.assert_array_almost_equal(iir2.nume, iir.nume)
        np.testing.assert_array_almost_equal(iir2.deno, iir.deno)
        assert iir2.input_scale == iir.input_scale
        assert iir2.output_scale == iir.output_scale

    def test_input_scale_setter(self):
        """IIRFilterComponent input_scale setter updates transfer function."""
        iir = IIRFilterComponent("IIR", SPS, [1.0], [1.0], input_scale=1.0)
        mag_before, _ = iir.bode([1e3], dB=True)
        iir.input_scale = 0.5
        mag_after, _ = iir.bode([1e3], dB=True)
        # Gain should halve -> -6 dB
        assert mag_after[0] == pytest.approx(mag_before[0] - 6.0, rel=0.01)
