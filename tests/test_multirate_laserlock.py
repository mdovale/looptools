"""Unit tests for loopkit MultiRateLaserLock loop class."""

import copy

import numpy as np
import pytest

from loopkit.component import Component
from loopkit.dimension import Dimension
from loopkit.loops import MultiRateLaserLock


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

SPS_LOOP = 20e6  # Loop rate (fLoop) from Simulink PLL
SPS_ADC = 40e6   # ADC rate (fADC)
AMP = 0.5
KP = 5000.0
KI = 4000.0


@pytest.fixture
def multirate_default():
    """MultiRateLaserLock single-rate (sps_adc = sps_loop)."""
    return MultiRateLaserLock(sps_loop=SPS_LOOP, Amp=AMP, Kp=KP, Ki=KI)


@pytest.fixture
def multirate_40_20():
    """MultiRateLaserLock multi-rate (40 MHz ADC, 20 MHz loop)."""
    return MultiRateLaserLock(
        sps_loop=SPS_LOOP, Amp=AMP, Kp=KP, Ki=KI,
        sps_adc=SPS_ADC,
    )


@pytest.fixture
def multirate_minimal():
    """MultiRateLaserLock minimal chain (exclude Laser, PA, LUT)."""
    return MultiRateLaserLock(
        sps_loop=SPS_LOOP, Amp=AMP, Kp=KP, Ki=KI,
        off=["Laser", "PA", "LUT"],
    )


@pytest.fixture
def multirate_vco_40():
    """MultiRateLaserLock with VCO at 40 MHz (20→40 upsampling)."""
    return MultiRateLaserLock(
        sps_loop=SPS_LOOP, Amp=AMP, Kp=KP, Ki=KI,
        vco_sps=40e6,
        off=["Laser"],  # Keep PA, LUT to verify they run at vco_sps
    )


@pytest.fixture
def frequency_array():
    """Log-spaced frequency array."""
    return np.logspace(2, 6, 50)


# -----------------------------------------------------------------------------
# Validation tests
# -----------------------------------------------------------------------------


class TestMultiRateLaserLockValidation:
    """Tests for MultiRateLaserLock input validation."""

    def test_invalid_sps_loop_raises(self):
        """Invalid sps_loop raises ValueError."""
        with pytest.raises(ValueError, match="sps_loop must be positive"):
            MultiRateLaserLock(sps_loop=0, Amp=AMP, Kp=KP, Ki=KI)
        with pytest.raises(ValueError, match="sps_loop must be positive"):
            MultiRateLaserLock(sps_loop=-1, Amp=AMP, Kp=KP, Ki=KI)

    def test_invalid_amp_raises(self):
        """Invalid Amp raises ValueError."""
        with pytest.raises(ValueError, match="Amp must be positive"):
            MultiRateLaserLock(sps_loop=SPS_LOOP, Amp=0, Kp=KP, Ki=KI)

    def test_invalid_n_reg_raises(self):
        """Invalid n_reg raises ValueError."""
        with pytest.raises(ValueError, match="n_reg must be non-negative"):
            MultiRateLaserLock(sps_loop=SPS_LOOP, Amp=AMP, Kp=KP, Ki=KI, n_reg=-1)

    def test_invalid_off_raises(self):
        """Invalid off component name raises ValueError."""
        with pytest.raises(ValueError, match="off must contain only"):
            MultiRateLaserLock(sps_loop=SPS_LOOP, Amp=AMP, Kp=KP, Ki=KI, off=["INVALID"])

    def test_sps_adc_lt_sps_loop_raises(self):
        """sps_adc < sps_loop raises ValueError."""
        with pytest.raises(ValueError, match="sps_adc must be >= sps_loop"):
            MultiRateLaserLock(sps_loop=40e6, Amp=AMP, Kp=KP, Ki=KI, sps_adc=20e6)

    def test_non_integer_ratio_raises(self):
        """sps_adc/sps_loop non-integer raises ValueError."""
        with pytest.raises(ValueError, match="must be an integer"):
            MultiRateLaserLock(sps_loop=20e6, Amp=AMP, Kp=KP, Ki=KI, sps_adc=30e6)

    def test_vco_sps_lt_sps_loop_raises(self):
        """vco_sps < sps_loop raises ValueError."""
        with pytest.raises(ValueError, match="vco_sps must be >= sps_loop"):
            MultiRateLaserLock(
                sps_loop=SPS_LOOP, Amp=AMP, Kp=KP, Ki=KI,
                vco_sps=10e6,
            )

    def test_vco_sps_non_integer_ratio_raises(self):
        """vco_sps/sps_loop non-integer raises ValueError."""
        with pytest.raises(ValueError, match="vco_sps/sps_loop must be an integer"):
            MultiRateLaserLock(
                sps_loop=20e6, Amp=AMP, Kp=KP, Ki=KI,
                vco_sps=30e6,
            )


# -----------------------------------------------------------------------------
# Construction tests
# -----------------------------------------------------------------------------


class TestMultiRateLaserLockConstruction:
    """Tests for MultiRateLaserLock construction."""

    def test_construction_default_name(self):
        """MultiRateLaserLock uses default name when name=None."""
        pll = MultiRateLaserLock(sps_loop=SPS_LOOP, Amp=AMP, Kp=KP, Ki=KI, off=["Laser", "PA", "LUT"])
        assert pll.name == "MultiRateLaserLock"

    def test_construction_custom_name(self):
        """MultiRateLaserLock accepts custom name."""
        pll = MultiRateLaserLock(
            sps_loop=SPS_LOOP, Amp=AMP, Kp=KP, Ki=KI,
            off=["Laser", "PA", "LUT"],
            name="MyMultiRateLaserLock",
        )
        assert pll.name == "MyMultiRateLaserLock"

    def test_construction_single_rate_chain(self, multirate_default):
        """Single-rate: PD, IIR, PIII, Delay, DAC, Laser, PreGain, PA, LUT (no RateTransition)."""
        names = set(multirate_default.components_dict.keys())
        expected = {"PD", "IIR", "PIII", "Delay", "DAC", "Laser", "PreGain", "PA", "LUT"}
        assert names == expected
        assert not multirate_default.multirate

    def test_construction_multi_rate_includes_rate_transition(self, multirate_40_20):
        """Multi-rate: includes RateTransition component."""
        names = set(multirate_40_20.components_dict.keys())
        assert "RateTransition" in names
        assert multirate_40_20.multirate
        assert multirate_40_20.sps_adc == SPS_ADC
        assert multirate_40_20.sps_loop == SPS_LOOP

    def test_construction_with_off(self, multirate_minimal):
        """MultiRateLaserLock excludes components in off."""
        names = set(multirate_minimal.components_dict.keys())
        assert "Laser" not in names
        assert "PA" not in names
        assert "LUT" not in names
        assert "PD" in names
        assert "IIR" in names
        assert "PIII" in names

    def test_construction_vco_sps_includes_upsample(self, multirate_vco_40):
        """When vco_sps > sps_loop, includes Upsample and PA/LUT at vco_sps."""
        names = set(multirate_vco_40.components_dict.keys())
        assert "Upsample" in names
        assert multirate_vco_40.vco_sps == 40e6
        assert multirate_vco_40.multirate
        upsample = multirate_vco_40.components_dict["Upsample"]
        assert upsample.sps_in == SPS_LOOP
        assert upsample.sps_out == 40e6
        pa = multirate_vco_40.components_dict["PA"]
        assert pa.sps == 40e6

    def test_construction_vco_sps_eq_sps_loop_skips_upsample(self):
        """When vco_sps == sps_loop, loop is single-rate and skips Upsample."""
        pll = MultiRateLaserLock(
            sps_loop=SPS_LOOP, Amp=AMP, Kp=KP, Ki=KI,
            vco_sps=SPS_LOOP,
            off=["Laser"],
        )
        names = set(pll.components_dict.keys())
        assert "Upsample" not in names
        assert pll.vco_sps == SPS_LOOP
        assert not pll.multirate
        pa = pll.components_dict["PA"]
        assert pa.sps == SPS_LOOP

    def test_construction_stores_parameters(self, multirate_default):
        """MultiRateLaserLock stores constructor parameters."""
        assert multirate_default.sps == SPS_LOOP
        assert multirate_default.Amp == AMP
        assert multirate_default.Kp == KP
        assert multirate_default.Ki == KI
        assert multirate_default.n_reg == 1
        assert multirate_default.dac_gain == pytest.approx(2 / 2**16)
        assert multirate_default.pre_gain == pytest.approx(0.1)

    def test_construction_custom_plant(self):
        """MultiRateLaserLock accepts custom Plant component."""
        custom_plant = Component(
            "Laser", SPS_LOOP,
            nume=np.array([1.0]),
            deno=np.array([1.0, -0.5]),
            unit=Dimension(dimensionless=True),
        )
        pll = MultiRateLaserLock(
            sps_loop=SPS_LOOP, Amp=AMP, Kp=KP, Ki=KI,
            Plant=custom_plant,
        )
        assert "Laser" in pll.components_dict
        laser = pll.components_dict["Laser"]
        np.testing.assert_array_equal(laser.nume, [1.0])
        np.testing.assert_array_equal(laser.deno, [1.0, -0.5])


# -----------------------------------------------------------------------------
# Transfer function tests
# -----------------------------------------------------------------------------


class TestMultiRateLaserLockTransferFunctions:
    """Tests for MultiRateLaserLock transfer functions."""

    def test_Gf_open_loop(self, multirate_default, frequency_array):
        """Gf returns open-loop transfer function."""
        G = multirate_default.Gf(f=frequency_array)
        assert G.shape == frequency_array.shape
        assert np.all(np.isfinite(G))
        assert np.issubdtype(G.dtype, np.complexfloating)

    def test_Gf_multi_rate(self, multirate_40_20, frequency_array):
        """Gf works for multi-rate configuration."""
        G = multirate_40_20.Gf(f=frequency_array)
        assert G.shape == frequency_array.shape
        assert np.all(np.isfinite(G))
        assert np.abs(G[0]) > 0

    def test_Hf_closed_loop(self, multirate_default, frequency_array):
        """Hf returns closed-loop transfer function."""
        H = multirate_default.Hf(f=frequency_array)
        assert H.shape == frequency_array.shape
        assert np.all(np.isfinite(H))

    def test_Ef_error_transfer(self, multirate_default, frequency_array):
        """Ef returns error transfer function."""
        E = multirate_default.Ef(f=frequency_array)
        assert E.shape == frequency_array.shape
        assert np.all(np.isfinite(E))

    def test_point_to_point_pd_to_piii(self, multirate_default, frequency_array):
        """point_to_point_tf from PD to PIII works."""
        tf = multirate_default.point_to_point_tf(
            frequency_array, _from="PD", _to="PIII",
        )
        assert tf.shape == frequency_array.shape
        assert np.all(np.isfinite(tf))

    def test_Gf_vco_sps(self, multirate_vco_40, frequency_array):
        """Gf works for vco_sps configuration (Simulink-faithful VCO path)."""
        G = multirate_vco_40.Gf(f=frequency_array)
        assert G.shape == frequency_array.shape
        assert np.all(np.isfinite(G))
        assert np.abs(G[0]) > 0


# -----------------------------------------------------------------------------
# Deepcopy tests
# -----------------------------------------------------------------------------


class TestMultiRateLaserLockDeepCopy:
    """Tests for MultiRateLaserLock deepcopy."""

    def test_deepcopy_creates_independent_instance(self, multirate_default):
        """Deepcopy creates independent instance."""
        pll2 = copy.deepcopy(multirate_default)
        assert pll2 is not multirate_default
        assert pll2.sps == multirate_default.sps
        assert pll2.Amp == multirate_default.Amp

    def test_deepcopy_preserves_parameters(self, multirate_default):
        """Deepcopy preserves constructor parameters."""
        pll2 = copy.deepcopy(multirate_default)
        assert pll2.Kp == multirate_default.Kp
        assert pll2.Ki == multirate_default.Ki
        assert pll2.n_reg == multirate_default.n_reg
        assert pll2.dac_gain == multirate_default.dac_gain
        assert pll2.pre_gain == multirate_default.pre_gain

    def test_deepcopy_preserves_vco_sps(self, multirate_vco_40):
        """Deepcopy preserves vco_sps and Upsample component."""
        pll2 = copy.deepcopy(multirate_vco_40)
        assert pll2.vco_sps == multirate_vco_40.vco_sps
        assert "Upsample" in pll2.components_dict

    def test_deepcopy_same_tf(self, multirate_default, frequency_array):
        """Deepcopy produces same transfer functions."""
        pll2 = copy.deepcopy(multirate_default)
        G1 = multirate_default.Gf(f=frequency_array)
        G2 = pll2.Gf(f=frequency_array)
        np.testing.assert_array_almost_equal(G1, G2)


# -----------------------------------------------------------------------------
# Simulink PLL compatibility
# -----------------------------------------------------------------------------


class TestMultiRateLaserLockSimulinkCompatibility:
    """Tests for Simulink PLL parameter compatibility."""

    def test_simulink_default_params(self):
        """MultiRateLaserLock with Simulink default params builds and runs."""
        pll = MultiRateLaserLock(
            sps_loop=20e6,
            Amp=0.5,
            Kp=5000,
            Ki=4000,
            dac_gain=2 / 2**16,
            pre_gain=0.1,
            n_reg=1,
        )
        frfr = np.logspace(2, 6, 50)
        G = pll.Gf(f=frfr)
        assert np.all(np.isfinite(G))
        assert np.abs(G[0]) > 0

    def test_simulink_multi_rate_40_20(self):
        """Multi-rate 40 MHz ADC, 20 MHz loop (Simulink topology)."""
        pll = MultiRateLaserLock(
            sps_loop=20e6,
            Amp=0.5,
            Kp=5000,
            Ki=4000,
            sps_adc=40e6,
        )
        assert "RateTransition" in pll.components_dict
        frfr = np.logspace(2, 6, 50)
        G = pll.Gf(f=frfr)
        assert np.all(np.isfinite(G))

    def test_custom_sos(self):
        """MultiRateLaserLock accepts custom SOS coefficients."""
        sos = [16777216, 33554432, 16777216, 16777216, -33181752, 16408629]
        pll = MultiRateLaserLock(
            sps_loop=SPS_LOOP, Amp=AMP, Kp=KP, Ki=KI,
            sos=sos,
            off=["Laser", "PA", "LUT"],
        )
        assert "IIR" in pll.components_dict
        frfr = np.logspace(2, 5, 20)
        G = pll.Gf(f=frfr)
        assert np.all(np.isfinite(G))

    def test_simulink_vco_40_mhz(self):
        """Simulink-faithful: loop at 20 MHz, VCO (PA+LUT) at 40 MHz."""
        pll = MultiRateLaserLock(
            sps_loop=20e6,
            Amp=0.5,
            Kp=5000,
            Ki=4000,
            vco_sps=40e6,
            off=["Laser"],
        )
        assert "Upsample" in pll.components_dict
        assert pll.components_dict["PA"].sps == 40e6
        assert pll.components_dict["LUT"].sps == 40e6
        frfr = np.logspace(2, 6, 50)
        G = pll.Gf(f=frfr)
        assert np.all(np.isfinite(G))
