"""Unit tests for looptools.loopmath."""

import numpy as np
import pytest

from looptools.loopmath import (
    loop_crossover,
    wrap_phase,
    tf_group_delay,
    polynomial_conversion_s_to_z,
    get_margin,
    tf_power_fitting,
    tf_power_solver,
    tf_power_extrapolate,
    add_transfer_function,
    mul_transfer_function,
    gain_for_crossover_frequency,
    Klf_from_cutoff,
    log2_gain_to_db,
    db_to_log2_gain,
    linear_to_log2_gain,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

SPS = 1000.0


class MockLoop:
    """Mock loop with Gf(f) returning complex values."""

    def __init__(self, mag_func):
        self._mag_func = mag_func

    def Gf(self, f=None):
        f = np.atleast_1d(f)
        mag = self._mag_func(f)
        return mag * np.exp(1j * np.zeros_like(f))  # Real, positive


@pytest.fixture
def frequency_array():
    """Log-spaced frequency array."""
    return np.logspace(-2, 2, 50)


# -----------------------------------------------------------------------------
# loop_crossover tests
# -----------------------------------------------------------------------------


class TestLoopCrossover:
    """Tests for loop_crossover."""

    def test_crossover_detected(self):
        """Finds first frequency where |G1| crosses |G2|."""
        # Loop1: gain 2 at low f, drops to 0.5 at high f
        # Loop2: gain 0.5 at low f, rises to 2 at high f
        # They cross somewhere in the middle
        frfr = np.logspace(0, 2, 100)
        loop1 = MockLoop(lambda f: 2.0 / (1 + f / 50))
        loop2 = MockLoop(lambda f: 0.5 * (1 + f / 50))
        cross = loop_crossover(loop1, loop2, frfr)
        assert np.isfinite(cross)
        assert cross >= frfr[0]
        assert cross <= frfr[-1]

    def test_no_crossover_returns_nan(self):
        """Returns nan when loops never cross."""
        frfr = np.logspace(0, 2, 50)
        loop1 = MockLoop(lambda f: np.ones_like(f) * 2.0)  # Always 2
        loop2 = MockLoop(lambda f: np.ones_like(f) * 0.5)  # Always 0.5
        cross = loop_crossover(loop1, loop2, frfr)
        assert np.isnan(cross)

    def test_exact_crossing(self):
        """Detects crossover when magnitudes are equal at a point."""
        frfr = np.array([1.0, 10.0, 100.0])
        # At index 1, both have mag 1
        loop1 = MockLoop(lambda f: np.array([2.0, 1.0, 0.5])[: len(f)])
        loop2 = MockLoop(lambda f: np.array([0.5, 1.0, 2.0])[: len(f)])
        cross = loop_crossover(loop1, loop2, frfr)
        assert np.isfinite(cross)


# -----------------------------------------------------------------------------
# wrap_phase tests
# -----------------------------------------------------------------------------


class TestWrapPhase:
    """Tests for wrap_phase."""

    def test_wrap_radians_in_range(self):
        """Values in [-π, π) stay unchanged."""
        phase = np.array([0.0, np.pi / 2, -np.pi / 2])
        out = wrap_phase(phase, deg=False)
        np.testing.assert_allclose(out, phase)

    def test_wrap_radians_overflow(self):
        """Phase > π wraps to negative."""
        out = wrap_phase(3 * np.pi / 2, deg=False)
        assert out == pytest.approx(-np.pi / 2)

    def test_wrap_degrees_in_range(self):
        """Values in [-180, 180) stay unchanged."""
        phase = np.array([0.0, 90.0, -90.0])
        out = wrap_phase(phase, deg=True)
        np.testing.assert_allclose(out, phase)

    def test_wrap_degrees_overflow(self):
        """Phase 270° wraps to -90°."""
        out = wrap_phase(270.0, deg=True)
        assert out == pytest.approx(-90.0)

    def test_wrap_scalar(self):
        """Accepts scalar input."""
        out = wrap_phase(400.0, deg=True)
        assert out == pytest.approx(40.0)

    def test_wrap_array(self):
        """Accepts array input. 180° wraps to -180° (interval is [-180, 180))."""
        phase = np.array([0, 360, -360, 180])
        out = wrap_phase(phase, deg=True)
        np.testing.assert_allclose(out, [0, 0, 0, -180])


# -----------------------------------------------------------------------------
# tf_group_delay tests
# -----------------------------------------------------------------------------


class TestTfGroupDelay:
    """Tests for tf_group_delay."""

    def test_pure_delay(self):
        """Constant group delay for pure delay TF H(f) = exp(-j*2*pi*f*tau)."""
        f = np.linspace(1, 100, 50)
        tau = 0.01  # 10 ms
        tf = np.exp(-1j * 2 * np.pi * f * tau)
        gd = tf_group_delay(f, tf)
        np.testing.assert_allclose(gd, tau, rtol=1e-2)

    def test_shape_matches_input(self):
        """Output shape matches frequency array."""
        f = np.logspace(0, 2, 30)
        tf = 1.0 / (1 + 1j * f / 10)
        gd = tf_group_delay(f, tf)
        assert gd.shape == f.shape

    def test_with_nan_preserves_positions(self):
        """NaN in tf yields NaN at same position in output."""
        f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tf = np.array([1.0, 1.0 + 1j, np.nan, 1.0 - 1j, 1.0])
        gd = tf_group_delay(f, tf)
        assert gd.shape == f.shape
        assert np.isnan(gd[2])


# -----------------------------------------------------------------------------
# polynomial_conversion_s_to_z tests
# -----------------------------------------------------------------------------


class TestPolynomialConversionSToZ:
    """Tests for polynomial_conversion_s_to_z."""

    def test_constant_polynomial(self):
        """s^0 -> z^0, no change in structure."""
        s_coeffs = np.array([1.0])
        z = polynomial_conversion_s_to_z(s_coeffs, sps=1000.0)
        assert z.shape == (1,)
        assert z[0] != 0

    def test_first_order(self):
        """s + a0 maps to z-domain polynomial."""
        s_coeffs = np.array([1.0, 1.0])  # s + 1
        z = polynomial_conversion_s_to_z(s_coeffs, sps=100.0)
        assert z.shape == (2,)
        assert np.all(np.isfinite(z))

    def test_output_shape(self):
        """Output length equals input length."""
        s_coeffs = np.array([1.0, 2.0, 3.0])
        z = polynomial_conversion_s_to_z(s_coeffs, sps=1000.0)
        assert len(z) == len(s_coeffs)


# -----------------------------------------------------------------------------
# get_margin tests (list/array input)
# -----------------------------------------------------------------------------


class TestGetMarginList:
    """Tests for get_margin with [magnitude, phase] list input."""

    def test_simple_margin_linear(self):
        """Phase margin from mag/phase arrays, linear magnitude."""
        f = np.logspace(0, 2, 100)
        # Unity gain at some index, phase = -135° there -> PM = 45°
        idx_ugf = 40
        mag = np.ones_like(f) * 2
        mag[idx_ugf] = 1.0
        mag[:idx_ugf] = 1.5
        mag[idx_ugf + 1 :] = 0.8
        phase = np.ones_like(f) * (-135.0)  # degrees (deg=True)
        ugf, margin = get_margin([mag, phase], f, dB=False, deg=True)
        assert np.isfinite(ugf)
        assert margin == pytest.approx(45.0, abs=1.0)

    def test_simple_margin_dB(self):
        """Phase margin with magnitude in dB."""
        f = np.array([1.0, 10.0, 100.0])
        mag_dB = np.array([10.0, 0.0, -10.0])  # 0 dB at index 1
        phase = np.array([-90.0, -120.0, -180.0])  # degrees (deg=True)
        ugf, margin = get_margin([mag_dB, phase], f, dB=True, deg=True)
        assert ugf == pytest.approx(10.0)
        assert margin == pytest.approx(60.0, abs=1.0)


class TestGetMarginArray:
    """Tests for get_margin with complex ndarray input."""

    def test_complex_tf_crosses_unity(self):
        """Phase margin from complex TF that crosses unity gain."""
        f = np.logspace(0, 2, 100)
        # TF with gain > 1 at low f, < 1 at high f so it crosses unity
        f0 = 10.0
        tf = 2.0 / (1 + 1j * f / f0)  # |TF| = 2 at DC, crosses 1 at f0*sqrt(3)
        ugf, margin = get_margin(tf, f, deg=True)
        assert np.isfinite(ugf)
        assert np.isfinite(margin)
        # UGF for 2/(1+j*f/f0) is at f = f0*sqrt(3) ≈ 17.32
        expected_ugf = f0 * np.sqrt(3)
        assert ugf == pytest.approx(expected_ugf, rel=0.1)
        # Phase margin is positive (stable) for this TF
        assert margin > 0
        assert margin < 180

    def test_gain_never_crosses_unity_returns_nan(self):
        """Returns nan when magnitude never crosses 1."""
        f = np.array([1.0, 10.0, 100.0])
        tf = np.array([0.1, 0.2, 0.3])  # All < 1
        ugf, margin = get_margin(tf, f, deg=True)
        assert np.isnan(ugf)
        assert np.isnan(margin)

    def test_interpolate_false_uses_nearest(self):
        """interpolate=False uses nearest point to unity."""
        f = np.array([1.0, 10.0, 100.0])
        tf = np.array([2.0, 1.1, 0.5])  # Closest to 1 at index 1
        ugf, margin = get_margin(tf, f, deg=True, interpolate=False)
        assert ugf == 10.0


# -----------------------------------------------------------------------------
# tf_power_fitting tests
# -----------------------------------------------------------------------------


class TestTfPowerFitting:
    """Tests for tf_power_fitting."""

    def test_power_law_extrapolation(self):
        """Extrapolates 1/f^2 behavior."""
        f = np.array([10.0, 20.0, 50.0])
        tf = 1.0 / (1j * 2 * np.pi * f) ** 2  # 1/(j*omega)^2
        fnew = np.array([5.0, 100.0])
        tfnew = tf_power_fitting(f, tf, fnew, power=-2)
        assert tfnew.shape == fnew.shape
        assert np.all(np.isfinite(tfnew))
        # Magnitude should scale as 1/f^2
        mag_ratio = np.abs(tfnew[1]) / np.abs(tfnew[0])
        freq_ratio = (fnew[0] / fnew[1]) ** 2
        assert mag_ratio == pytest.approx(freq_ratio, rel=0.5)


# -----------------------------------------------------------------------------
# tf_power_solver tests
# -----------------------------------------------------------------------------


class TestTfPowerSolver:
    """Tests for tf_power_solver."""

    def test_solver_extrapolation(self):
        """Extrapolates from single point."""
        f = 10.0
        tf = 1.0 / (1j * 2 * np.pi * f) ** 2
        fnew = np.array([5.0, 20.0])
        tfnew = tf_power_solver(f, tf, fnew, power=-2)
        assert tfnew.shape == fnew.shape
        # At f=5, mag should be (10/5)^2 = 4 times larger than at f=10
        mag_at_5 = np.abs(tfnew[0])
        mag_at_20 = np.abs(tfnew[1])
        expected_ratio = (20.0 / 5.0) ** 2
        assert mag_at_5 / mag_at_20 == pytest.approx(expected_ratio, rel=1e-5)

    def test_raises_on_non_float_f(self):
        """Raises ValueError if f is not a scalar."""
        with pytest.raises(ValueError, match="scalar"):
            tf_power_solver(np.array([10.0]), 1.0 + 0j, np.array([5.0]), power=-2)


# -----------------------------------------------------------------------------
# tf_power_extrapolate tests
# -----------------------------------------------------------------------------


class TestTfPowerExtrapolate:
    """Tests for tf_power_extrapolate."""

    def test_left_side_extrapolation(self):
        """side='left' extrapolates below f_trans, keeps data above."""
        f = np.logspace(-1, 2, 50)
        tf = 1.0 / (1j * 2 * np.pi * f)
        f_trans = 1.0
        tfnew = tf_power_extrapolate(f, tf, f_trans, power=-1, side="left")
        assert tfnew.shape == f.shape
        np.testing.assert_array_equal(tfnew[f > f_trans], tf[f > f_trans])

    def test_right_side_extrapolation(self):
        """side='right' extrapolates above f_trans, keeps data below."""
        f = np.logspace(-1, 2, 50)
        tf = 1.0 / (1j * 2 * np.pi * f)
        f_trans = 10.0
        tfnew = tf_power_extrapolate(f, tf, f_trans, power=-1, side="right")
        assert tfnew.shape == f.shape
        np.testing.assert_array_equal(tfnew[f < f_trans], tf[f < f_trans])


# -----------------------------------------------------------------------------
# add_transfer_function tests
# -----------------------------------------------------------------------------


class TestAddTransferFunction:
    """Tests for add_transfer_function."""

    def test_simple_sum(self):
        """Sum of two TFs without extrapolation."""
        f = np.array([1.0, 10.0, 100.0])
        tf1 = lambda freq: np.ones_like(freq, dtype=complex)
        tf2 = lambda freq: 2.0 * np.ones_like(freq, dtype=complex)
        result = add_transfer_function(f, tf1, tf2, extrapolate=False)
        np.testing.assert_allclose(result, 3.0)

    def test_with_extrapolation(self):
        """Sum with extrapolation runs without error."""
        f = np.logspace(-2, 2, 50)
        tf1 = lambda freq: 1.0 / (1 + 1j * freq / 10)
        tf2 = lambda freq: 0.5 / (1 + 1j * freq / 5)
        result = add_transfer_function(
            f, tf1, tf2, extrapolate=True, f_trans=0.1, power=-2
        )
        assert result.shape == f.shape
        assert np.all(np.isfinite(result))


# -----------------------------------------------------------------------------
# mul_transfer_function tests
# -----------------------------------------------------------------------------


class TestMulTransferFunction:
    """Tests for mul_transfer_function."""

    def test_simple_product(self):
        """Product of two TFs without extrapolation."""
        f = np.array([1.0, 10.0, 100.0])
        tf1 = lambda freq: 2.0 * np.ones_like(freq, dtype=complex)
        tf2 = lambda freq: 3.0 * np.ones_like(freq, dtype=complex)
        result = mul_transfer_function(f, tf1, tf2, extrapolate=False)
        np.testing.assert_allclose(result, 6.0)

    def test_with_extrapolation(self):
        """Product with extrapolation runs without error."""
        f = np.logspace(-2, 2, 50)
        tf1 = lambda freq: 1.0 / (1 + 1j * freq / 10)
        tf2 = lambda freq: 1.0 / (1 + 1j * freq / 5)
        result = mul_transfer_function(
            f, tf1, tf2, extrapolate=True, f_trans=0.1, power=-2
        )
        assert result.shape == f.shape
        assert np.all(np.isfinite(result))


# -----------------------------------------------------------------------------
# gain_for_crossover_frequency tests
# -----------------------------------------------------------------------------


class TestGainForCrossoverFrequency:
    """Tests for gain_for_crossover_frequency."""

    def test_kind_I_returns_log2_gain(self):
        """kind='I' returns single log2 gain."""
        Kp_log2 = 0.0  # Kp = 1
        f_cross = 10.0
        result = gain_for_crossover_frequency(Kp_log2, SPS, f_cross, kind="I")
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

    def test_kind_D_returns_log2_gain(self):
        """kind='D' returns single log2 gain."""
        Kp_log2 = 0.0
        f_cross = 10.0
        result = gain_for_crossover_frequency(Kp_log2, SPS, f_cross, kind="D")
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

    def test_kind_II_returns_tuple(self):
        """kind='II' returns (Ki_log2, Kii_log2)."""
        Kp_log2 = 0.0
        f_cross = (5.0, 20.0)
        result = gain_for_crossover_frequency(Kp_log2, SPS, f_cross, kind="II")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(np.isfinite(r) for r in result)

    def test_invalid_kind_raises(self):
        """Invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="kind must be"):
            gain_for_crossover_frequency(0.0, SPS, 10.0, kind="X")

    def test_II_requires_tuple_f_cross(self):
        """kind='II' with non-tuple f_cross raises."""
        with pytest.raises(ValueError, match="f_cross must be a tuple"):
            gain_for_crossover_frequency(0.0, SPS, 10.0, kind="II")


# -----------------------------------------------------------------------------
# Klf_from_cutoff tests
# -----------------------------------------------------------------------------


class TestKlfFromCutoff:
    """Tests for Klf_from_cutoff."""

    def test_returns_finite_log2_gain(self):
        """Returns finite Klf for valid cutoff."""
        f_c = 10.0
        Klf = Klf_from_cutoff(f_c, SPS, n=1)
        assert np.isfinite(Klf)
        assert isinstance(Klf, (float, np.floating))

    def test_higher_cutoff_gives_different_Klf(self):
        """Different cutoff frequencies give different Klf."""
        Klf_10 = Klf_from_cutoff(10.0, SPS, n=1)
        Klf_100 = Klf_from_cutoff(100.0, SPS, n=1)
        assert Klf_10 != Klf_100

    def test_n_stages_affects_result(self):
        """n=2 gives different result than n=1."""
        Klf_1 = Klf_from_cutoff(10.0, SPS, n=1)
        Klf_2 = Klf_from_cutoff(10.0, SPS, n=2)
        assert Klf_1 != Klf_2


# -----------------------------------------------------------------------------
# log2_gain_to_db, db_to_log2_gain, linear_to_log2_gain tests
# -----------------------------------------------------------------------------


class TestGainConversions:
    """Tests for gain conversion utilities."""

    def test_log2_to_db_example(self):
        """log2_gain_to_db(3.3219) ≈ 20 dB."""
        assert log2_gain_to_db(3.3219) == pytest.approx(20.0, rel=1e-3)

    def test_db_to_log2_example(self):
        """db_to_log2_gain(20) ≈ 3.3219."""
        assert db_to_log2_gain(20.0) == pytest.approx(3.3219, rel=1e-3)

    def test_linear_to_log2_example(self):
        """linear_to_log2_gain(10) ≈ 3.3219."""
        assert linear_to_log2_gain(10.0) == pytest.approx(3.3219, rel=1e-3)

    def test_log2_db_roundtrip(self):
        """log2_gain_to_db and db_to_log2_gain are inverses."""
        log2_val = 2.5
        db_val = log2_gain_to_db(log2_val)
        back = db_to_log2_gain(db_val)
        assert back == pytest.approx(log2_val, rel=1e-10)

    def test_linear_log2_roundtrip(self):
        """linear_to_log2_gain and 2**x are inverses."""
        linear = 7.0
        log2_val = linear_to_log2_gain(linear)
        assert 2**log2_val == pytest.approx(linear, rel=1e-10)

    def test_zero_db(self):
        """0 dB corresponds to log2(1) = 0."""
        assert db_to_log2_gain(0.0) == pytest.approx(0.0)
        assert log2_gain_to_db(0.0) == pytest.approx(0.0)
