"""Unit tests for looptools.pll."""

import copy
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from looptools.loopmath import get_margin
from looptools.loops import PLL


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

SPS = 1000.0  # Sample rate (Hz) for tests

GOLDEN_DIR = Path(__file__).parent / "golden"


@pytest.fixture
def pll_default():
    """PLL with default parameters (full component chain)."""
    return PLL(
        sps=SPS,
        Amp=0.5,
        Cshift=4,
        Klf=0.5,
        Kp=2,
        Ki=3,
    )


@pytest.fixture
def pll_single_stage():
    """PLL with single-stage LPF (twostages=False)."""
    return PLL(
        sps=SPS,
        Amp=0.5,
        Cshift=4,
        Klf=0.5,
        Kp=2,
        Ki=3,
        twostages=False,
    )


@pytest.fixture
def pll_with_exclusions():
    """PLL with some components excluded via but."""
    return PLL(
        sps=SPS,
        Amp=0.5,
        Cshift=4,
        Klf=0.5,
        Kp=2,
        Ki=3,
        but=["LUT", "DSP"],
    )


@pytest.fixture
def frequency_array():
    """Log-spaced frequency array."""
    return np.logspace(-2, 2, 50)


# -----------------------------------------------------------------------------
# Construction tests
# -----------------------------------------------------------------------------


class TestPLLConstruction:
    """Tests for PLL construction and initialization."""

    def test_construction_default_name(self):
        """PLL uses default name 'PLL' when name=None."""
        pll = PLL(SPS, Amp=0.5, Cshift=4, Klf=0.5, Kp=2, Ki=3)
        assert pll.name == "PLL"

    def test_construction_custom_name(self):
        """PLL accepts custom name."""
        pll = PLL(
            SPS, Amp=0.5, Cshift=4, Klf=0.5, Kp=2, Ki=3, name="MyPLL"
        )
        assert pll.name == "MyPLL"

    def test_construction_stores_parameters(self, pll_default):
        """PLL stores all constructor parameters."""
        assert pll_default.sps == SPS
        assert pll_default.Amp == 0.5
        assert pll_default.Cshift == 4
        assert pll_default.Klf == 0.5
        assert pll_default.Kp == 2
        assert pll_default.Ki == 3
        assert pll_default.twostages is True
        assert pll_default.n_reg == 10
        assert pll_default.but == [None]

    def test_construction_full_component_chain(self, pll_default):
        """PLL with but=[None] includes all standard components."""
        expected = ["PD", "LPF", "Gain", "PI", "PA", "LUT", "DSP"]
        for name in expected:
            assert name in pll_default.components_dict

    def test_construction_twostages_uses_two_stage_lpf(self, pll_default):
        """twostages=True uses TwoStageLPFComponent."""
        lpf = pll_default.components_dict["LPF"]
        assert lpf.__class__.__name__ == "TwoStageLPFComponent"

    def test_construction_single_stage_uses_lpf(self, pll_single_stage):
        """twostages=False uses LPFComponent."""
        lpf = pll_single_stage.components_dict["LPF"]
        assert lpf.__class__.__name__ == "LPFComponent"

    def test_construction_excludes_components_via_but(self, pll_with_exclusions):
        """but parameter excludes specified components."""
        assert "LUT" not in pll_with_exclusions.components_dict
        assert "DSP" not in pll_with_exclusions.components_dict
        assert "PD" in pll_with_exclusions.components_dict
        assert "LPF" in pll_with_exclusions.components_dict

    def test_construction_exclusion_warning(self, caplog):
        """Excluding components logs a warning."""
        caplog.set_level(logging.WARNING)
        PLL(SPS, Amp=0.5, Cshift=4, Klf=0.5, Kp=2, Ki=3, but=["LUT"])
        assert "not included in the loop" in caplog.text
        assert "LUT" in caplog.text

    def test_construction_no_warning_when_but_none(self, caplog):
        """but=[None] does not log warning."""
        caplog.set_level(logging.WARNING)
        PLL(SPS, Amp=0.5, Cshift=4, Klf=0.5, Kp=2, Ki=3)
        assert "not included in the loop" not in caplog.text

    def test_construction_custom_n_reg(self):
        """PLL accepts custom n_reg."""
        pll = PLL(
            SPS, Amp=0.5, Cshift=4, Klf=0.5, Kp=2, Ki=3, n_reg=5
        )
        assert pll.n_reg == 5
        dsp = pll.components_dict["DSP"]
        assert dsp.n_reg == 5

    def test_construction_update_sets_transfer_elements(self, pll_default):
        """Construction calls update and sets Gc, Hc, Ec."""
        assert pll_default.Gc is not None
        assert pll_default.Hc is not None
        assert pll_default.Ec is not None
        assert callable(pll_default.Gf)
        assert callable(pll_default.Hf)
        assert callable(pll_default.Ef)


# -----------------------------------------------------------------------------
# __deepcopy__ tests
# -----------------------------------------------------------------------------


class TestPLLDeepCopy:
    """Tests for PLL.__deepcopy__."""

    def test_deepcopy_creates_independent_instance(self, pll_default):
        """Deep copy is independent of original."""
        pll_copy = copy.deepcopy(pll_default)
        assert pll_copy is not pll_default
        assert pll_copy.sps == pll_default.sps
        assert pll_copy.Amp == pll_default.Amp
        assert pll_copy.components_dict is not pll_default.components_dict

    def test_deepcopy_preserves_parameters(self, pll_with_exclusions):
        """Deep copy preserves all PLL parameters including but."""
        pll_copy = copy.deepcopy(pll_with_exclusions)
        assert pll_copy.but == pll_with_exclusions.but
        assert pll_copy.twostages == pll_with_exclusions.twostages
        assert pll_copy.n_reg == pll_with_exclusions.n_reg

    def test_deepcopy_preserves_callbacks(self, pll_default):
        """Deep copy preserves callbacks."""
        called = []

        def cb():
            called.append(1)

        pll_default.register_callback(cb)
        pll_copy = copy.deepcopy(pll_default)
        pll_copy.notify_callbacks()
        assert called == [1]


# -----------------------------------------------------------------------------
# Transfer function tests (inherited from LOOP)
# -----------------------------------------------------------------------------


class TestPLLTransferFunctions:
    """Tests for PLL transfer functions (Gf, Hf, Ef)."""

    def test_Gf_open_loop(self, pll_default, frequency_array):
        """Gf returns open-loop transfer function."""
        val = pll_default.Gf(f=frequency_array)
        assert val.shape == frequency_array.shape
        assert np.all(np.isfinite(val))

    def test_Hf_closed_loop(self, pll_default, frequency_array):
        """Hf returns closed-loop transfer function."""
        val = pll_default.Hf(f=frequency_array)
        assert val.shape == frequency_array.shape
        assert np.all(np.isfinite(val))

    def test_Ef_error_transfer(self, pll_default, frequency_array):
        """Ef returns error transfer function."""
        val = pll_default.Ef(f=frequency_array)
        assert val.shape == frequency_array.shape
        assert np.all(np.isfinite(val))


# -----------------------------------------------------------------------------
# point_to_point tests
# -----------------------------------------------------------------------------


class TestPLLPointToPoint:
    """Tests for point_to_point_component and point_to_point_tf."""

    def test_point_to_point_component_pd_to_lpf(self, pll_default):
        """point_to_point_component returns TE for PD to LPF path."""
        comp = pll_default.point_to_point_component(
            _from="PD", _to="LPF", suppression=False
        )
        assert comp is not None
        assert comp.TE is not None

    def test_point_to_point_component_with_suppression(self, pll_default):
        """point_to_point_component with suppression includes E."""
        comp = pll_default.point_to_point_component(
            _from="PD", _to=None, suppression=True
        )
        assert comp is not None

    def test_point_to_point_tf(self, pll_default, frequency_array):
        """point_to_point_tf returns frequency response of path."""
        tf = pll_default.point_to_point_tf(
            frequency_array, _from="PD", _to="LPF", suppression=False
        )
        assert tf.shape == frequency_array.shape
        assert np.all(np.isfinite(tf))

    def test_point_to_point_tf_full_loop(self, pll_default, frequency_array):
        """point_to_point_tf with _from==_to gives full loop."""
        tf = pll_default.point_to_point_tf(
            frequency_array, _from="PD", _to="PD", suppression=False
        )
        assert tf.shape == frequency_array.shape
        assert np.all(np.isfinite(tf))


# -----------------------------------------------------------------------------
# show_all_te smoke test
# -----------------------------------------------------------------------------


class TestPLLShowAllTE:
    """Tests for show_all_te."""

    def test_show_all_te_runs_without_error(self, pll_default, capsys):
        """show_all_te prints and does not raise."""
        pll_default.show_all_te()
        captured = capsys.readouterr()
        assert "transfer function" in captured.out
        assert "PD" in captured.out or "G" in captured.out


# -----------------------------------------------------------------------------
# Golden-file regression tests
# -----------------------------------------------------------------------------


@pytest.fixture
def golden_data():
    """Load golden reference data. Skip if golden files missing."""
    npz_path = GOLDEN_DIR / "pll_golden.npz"
    params_path = GOLDEN_DIR / "pll_golden_params.json"
    if not npz_path.exists() or not params_path.exists():
        pytest.skip(
            "Golden files not found. Run: python tests/generate_pll_golden.py"
        )
    data = dict(np.load(npz_path, allow_pickle=True))
    with open(params_path) as f:
        data["params"] = json.load(f)
    return data


@pytest.fixture
def pll_golden(golden_data):
    """PLL instance built with golden parameters."""
    p = golden_data["params"]
    return PLL(
        sps=p["sps"],
        Amp=p["Amp"],
        Cshift=p["Cshift"],
        Klf=p["Klf"],
        Kp=p["Kp"],
        Ki=p["Ki"],
        twostages=p["twostages"],
        n_reg=p["n_reg"],
    )


class TestPLLRegressionGolden:
    """Regression tests against golden reference data.

    Golden data is generated from notebooks/0.1_pll-demo.ipynb parameters.
    To regenerate after intentional PLL changes: python tests/generate_pll_golden.py
    """

    def test_Gf_matches_golden(self, pll_golden, golden_data):
        """Open-loop transfer function Gf matches golden reference."""
        frfr = golden_data["frfr"]
        Gf = pll_golden.Gf(f=frfr)
        np.testing.assert_allclose(
            Gf, golden_data["Gf"], rtol=1e-10, atol=1e-14
        )

    def test_Hf_matches_golden(self, pll_golden, golden_data):
        """Closed-loop transfer function Hf matches golden reference."""
        frfr = golden_data["frfr"]
        Hf = pll_golden.Hf(f=frfr)
        np.testing.assert_allclose(
            Hf, golden_data["Hf"], rtol=1e-10, atol=1e-14
        )

    def test_Ef_matches_golden(self, pll_golden, golden_data):
        """Error transfer function Ef matches golden reference."""
        frfr = golden_data["frfr"]
        Ef = pll_golden.Ef(f=frfr)
        np.testing.assert_allclose(
            Ef, golden_data["Ef"], rtol=1e-10, atol=1e-14
        )

    def test_ugf_phase_margin_match_golden(self, pll_golden, golden_data):
        """Unity gain frequency and phase margin match golden reference."""
        frfr = golden_data["frfr"]
        Gf = pll_golden.Gf(f=frfr)
        ugf, phase_margin = get_margin(Gf, frfr, deg=True)
        np.testing.assert_allclose(ugf, float(golden_data["ugf"]), rtol=1e-6)
        np.testing.assert_allclose(
            phase_margin, float(golden_data["phase_margin"]), rtol=1e-6
        )

    def test_point_to_point_pd_to_lpf_matches_golden(
        self, pll_golden, golden_data
    ):
        """point_to_point_tf(PD, LPF) matches golden reference."""
        frfr = golden_data["frfr"]
        tf = pll_golden.point_to_point_tf(
            frfr, _from="PD", _to="LPF", suppression=False
        )
        np.testing.assert_allclose(
            tf, golden_data["tf_pd_to_lpf"], rtol=1e-10, atol=1e-14
        )

    def test_point_to_point_pd_suppressed_matches_golden(
        self, pll_golden, golden_data
    ):
        """point_to_point_tf(PD, None, suppression=True) matches golden."""
        frfr = golden_data["frfr"]
        tf = pll_golden.point_to_point_tf(
            frfr, _from="PD", _to=None, suppression=True
        )
        np.testing.assert_allclose(
            tf, golden_data["tf_pd_suppressed"], rtol=1e-10, atol=1e-14
        )
