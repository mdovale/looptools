"""Unit tests for loopkit rate transition components."""

import copy

import numpy as np
import pytest

from loopkit.component import Component
from loopkit.components import DownsampleComponent, RateTransitionComponent


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

SPS = 40e6  # Input sample rate (e.g. Simulink PLL ADC)


# -----------------------------------------------------------------------------
# DownsampleComponent tests
# -----------------------------------------------------------------------------


class TestDownsampleComponent:
    """Tests for DownsampleComponent."""

    def test_basic_construction(self):
        """DownsampleComponent constructs with M factor."""
        ds = DownsampleComponent("DS", SPS, M=2)
        assert ds.name == "DS"
        assert ds.sps == SPS
        assert ds.M == 2
        assert ds.sps_out == SPS / 2
        assert ds.include_delay is True
        assert ds.TE is not None
        assert ds.TF is not None

    def test_M1_no_downsampling(self):
        """M=1 yields unity gain, no delay."""
        ds = DownsampleComponent("DS", SPS, M=1)
        np.testing.assert_array_equal(ds.nume, [1.0])
        np.testing.assert_array_equal(ds.deno, [1.0])
        mag, _ = ds.bode([1e3], dB=True)
        assert mag[0] == pytest.approx(0.0, abs=0.01)  # 0 dB

    def test_M2_with_delay(self):
        """M=2 with include_delay yields 1-sample delay."""
        ds = DownsampleComponent("DS", SPS, M=2, include_delay=True)
        np.testing.assert_array_equal(ds.nume, [1.0])
        np.testing.assert_array_equal(ds.deno, [1.0, 0.0])
        # H(z)=1/z: at f=sps/4, phase = -90 deg
        f_90deg = SPS / 4
        mag, phase = ds.bode([f_90deg], dB=True)
        assert mag[0] == pytest.approx(0.0, abs=0.01)  # 0 dB
        assert phase[0] == pytest.approx(-90.0, abs=1.0)

    def test_M4_delay(self):
        """M=4 yields 3-sample delay."""
        ds = DownsampleComponent("DS", SPS, M=4, include_delay=True)
        assert ds.deno.size == 4  # [1, 0, 0, 0]
        np.testing.assert_array_equal(ds.deno, [1.0, 0.0, 0.0, 0.0])

    def test_no_delay_mode(self):
        """include_delay=False yields pure unity gain."""
        ds = DownsampleComponent("DS", SPS, M=4, include_delay=False)
        np.testing.assert_array_equal(ds.nume, [1.0])
        np.testing.assert_array_equal(ds.deno, [1.0])
        mag, _ = ds.bode([1e3], dB=True)
        assert mag[0] == pytest.approx(0.0, abs=0.01)

    def test_dc_gain_unity(self):
        """DC gain is unity (0 dB) for all M."""
        for M in [1, 2, 4, 8]:
            ds = DownsampleComponent("DS", SPS, M=M)
            mag, _ = ds.bode([1.0], dB=True)
            assert mag[0] == pytest.approx(0.0, abs=0.01)

    def test_M_setter_updates_tf(self):
        """M setter updates transfer function."""
        ds = DownsampleComponent("DS", SPS, M=2)
        ds.M = 4
        assert ds.M == 4
        assert ds.deno.size == 4
        assert ds.sps_out == SPS / 4

    def test_include_delay_setter(self):
        """include_delay setter updates transfer function."""
        ds = DownsampleComponent("DS", SPS, M=4, include_delay=True)
        assert ds.deno.size == 4
        ds.include_delay = False
        np.testing.assert_array_equal(ds.deno, [1.0])

    def test_deepcopy(self):
        """Deepcopy preserves state."""
        ds = DownsampleComponent("DS", SPS, M=4, include_delay=True)
        ds2 = copy.deepcopy(ds)
        assert ds2.name == ds.name
        assert ds2.sps == ds.sps
        assert ds2.M == ds.M
        assert ds2.include_delay == ds.include_delay
        np.testing.assert_array_equal(ds2.nume, ds.nume)
        np.testing.assert_array_equal(ds2.deno, ds.deno)

    def test_invalid_M_raises(self):
        """M < 1 raises ValueError."""
        with pytest.raises(ValueError, match="M must be >= 1"):
            DownsampleComponent("DS", SPS, M=0)
        with pytest.raises(ValueError, match="non-negative|>= 1"):
            DownsampleComponent("DS", SPS, M=-1)

    def test_series_with_component(self):
        """DownsampleComponent can be composed with other components."""
        ds = DownsampleComponent("DS", SPS, M=2)
        gain = Component("G", SPS, nume=[2.0], deno=[1.0])
        cascaded = ds * gain
        assert cascaded.TE is not None
        mag, _ = cascaded.bode([1e3], dB=True)
        # 0 dB (ds) + 6 dB (gain=2) = 6 dB
        assert mag[0] == pytest.approx(6.0, abs=0.1)


# -----------------------------------------------------------------------------
# RateTransitionComponent tests
# -----------------------------------------------------------------------------


class TestRateTransitionComponent:
    """Tests for RateTransitionComponent."""

    def test_downsample_40_to_20_mhz(self):
        """40 MHz to 20 MHz transition (Simulink PLL case)."""
        rt = RateTransitionComponent("RT", sps_in=40e6, sps_out=20e6)
        assert rt.sps_in == 40e6
        assert rt.sps_out == 20e6
        assert rt.M == 2
        assert rt.sps == 40e6  # Component uses input rate
        np.testing.assert_array_equal(rt.deno, [1.0, 0.0])

    def test_downsample_40_to_10_mhz(self):
        """40 MHz to 10 MHz: M=4."""
        rt = RateTransitionComponent("RT", sps_in=40e6, sps_out=10e6)
        assert rt.M == 4
        assert rt.deno.size == 4

    def test_same_rate_no_change(self):
        """sps_in == sps_out: M=1, unity gain."""
        rt = RateTransitionComponent("RT", sps_in=20e6, sps_out=20e6)
        assert rt.M == 1
        np.testing.assert_array_equal(rt.nume, [1.0])
        np.testing.assert_array_equal(rt.deno, [1.0])

    def test_non_integer_ratio_raises(self):
        """sps_in/sps_out non-integer raises ValueError."""
        with pytest.raises(ValueError, match="must be an integer"):
            RateTransitionComponent("RT", sps_in=40e6, sps_out=15e6)

    def test_upsample_20_to_40_mhz(self):
        """20 MHz to 40 MHz upsampling (M=2): DC gain 1, component sps = output rate."""
        rt = RateTransitionComponent("RT", sps_in=20e6, sps_out=40e6)
        assert rt.sps_in == 20e6
        assert rt.sps_out == 40e6
        assert rt.M == 2
        assert rt.sps == 40e6  # Component uses output rate for upsampling
        mag, phase = rt.bode([1.0], dB=True)
        assert mag[0] == pytest.approx(0.0, abs=0.01)  # DC gain 1 (0 dB)
        # At Nyquist (20 MHz): ZOH has sinc roll-off
        mag_nyq, phase_nyq = rt.bode([20e6], dB=True)
        assert np.isfinite(mag_nyq[0])
        assert np.isfinite(phase_nyq[0])

    def test_upsample_integer_ratio_required(self):
        """sps_out/sps_in must be integer for upsampling."""
        with pytest.raises(ValueError, match="must be an integer"):
            RateTransitionComponent("RT", sps_in=20e6, sps_out=30e6)

    def test_rate_transition_bidirectional(self):
        """40→20 and 20→40 both work."""
        rt_down = RateTransitionComponent("RT", sps_in=40e6, sps_out=20e6)
        rt_up = RateTransitionComponent("RT", sps_in=20e6, sps_out=40e6)
        assert rt_down.M == 2
        assert rt_up.M == 2
        assert rt_down.sps == 40e6
        assert rt_up.sps == 40e6
        frfr = np.logspace(2, 6, 50)
        mag_d, _ = rt_down.bode(frfr, dB=True)
        mag_u, _ = rt_up.bode(frfr, dB=True)
        assert np.all(np.isfinite(mag_d))
        assert np.all(np.isfinite(mag_u))

    def test_upsample_equivalent_to_downsample_inverse(self):
        """Cascade down then up gives approximately unity (down 40→20, up 20→40)."""
        rt_down = RateTransitionComponent("RT", sps_in=40e6, sps_out=20e6)
        rt_up = RateTransitionComponent("RT", sps_in=20e6, sps_out=40e6)
        frfr = np.logspace(2, 5, 100)  # Stay below Nyquist at 20 MHz
        tf_down = rt_down.TF(f=frfr)
        tf_up = rt_up.TF(f=frfr)
        cascade = tf_down * tf_up
        mag = np.abs(cascade)
        # Cascade should be ~1 at low freq (ZOH up has sinc, down has delay; product ~1)
        assert mag[0] == pytest.approx(1.0, abs=0.1)

    def test_include_delay_false(self):
        """include_delay=False yields pure gain."""
        rt = RateTransitionComponent(
            "RT", sps_in=40e6, sps_out=20e6, include_delay=False
        )
        np.testing.assert_array_equal(rt.nume, [1.0])
        np.testing.assert_array_equal(rt.deno, [1.0])

    def test_equivalent_to_downsample(self):
        """RateTransitionComponent matches DownsampleComponent for same M."""
        rt = RateTransitionComponent("RT", sps_in=40e6, sps_out=20e6)
        ds = DownsampleComponent("DS", sps=40e6, M=2)
        np.testing.assert_array_equal(rt.nume, ds.nume)
        np.testing.assert_array_equal(rt.deno, ds.deno)
        frfr = np.logspace(2, 6, 50)
        mag_rt, phase_rt = rt.bode(frfr, dB=True)
        mag_ds, phase_ds = ds.bode(frfr, dB=True)
        np.testing.assert_array_almost_equal(mag_rt, mag_ds)
        np.testing.assert_array_almost_equal(phase_rt, phase_ds)

    def test_sps_in_setter(self):
        """sps_in setter updates M and TF."""
        rt = RateTransitionComponent("RT", sps_in=40e6, sps_out=20e6)
        rt.sps_in = 80e6
        assert rt.M == 4
        assert rt.deno.size == 4

    def test_sps_out_setter(self):
        """sps_out setter updates M and TF."""
        rt = RateTransitionComponent("RT", sps_in=40e6, sps_out=20e6)
        rt.sps_out = 10e6
        assert rt.M == 4

    def test_deepcopy(self):
        """Deepcopy preserves state."""
        rt = RateTransitionComponent("RT", sps_in=40e6, sps_out=20e6)
        rt2 = copy.deepcopy(rt)
        assert rt2.sps_in == rt.sps_in
        assert rt2.sps_out == rt.sps_out
        assert rt2.M == rt.M
        np.testing.assert_array_equal(rt2.nume, rt.nume)
        np.testing.assert_array_equal(rt2.deno, rt.deno)

    def test_bode_finite(self):
        """Bode response is finite over frequency range."""
        rt = RateTransitionComponent("RT", sps_in=40e6, sps_out=20e6)
        frfr = np.logspace(0, 6, 100)
        mag, phase = rt.bode(frfr, dB=True)
        assert np.all(np.isfinite(mag))
        assert np.all(np.isfinite(phase))
