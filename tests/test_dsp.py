"""Unit tests for loopkit.dsp."""

import numpy as np
import pytest

from loopkit.dsp import (
    crop_data,
    index_of_the_nearest,
    integral_rms,
    nan_checker,
)


# -----------------------------------------------------------------------------
# index_of_the_nearest tests
# -----------------------------------------------------------------------------


class TestIndexOfTheNearest:
    """Tests for index_of_the_nearest."""

    def test_exact_match(self):
        """Exact value returns its index."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        assert index_of_the_nearest(data, 3.0) == 2

    def test_closest_below(self):
        """Target below midpoint returns lower index."""
        data = np.array([0.0, 1.0, 2.0, 3.0])
        assert index_of_the_nearest(data, 1.4) == 1

    def test_closest_above(self):
        """Target above midpoint returns higher index."""
        data = np.array([0.0, 1.0, 2.0, 3.0])
        assert index_of_the_nearest(data, 1.6) == 2

    def test_equidistant_returns_first(self):
        """When equidistant, returns first occurrence (argmin behavior)."""
        data = np.array([1.0, 3.0, 5.0])
        # 4 is equidistant from 3 and 5; argmin returns first
        idx = index_of_the_nearest(data, 4.0)
        assert idx in (1, 2)

    def test_single_element(self):
        """Single-element array returns 0."""
        data = np.array([42.0])
        assert index_of_the_nearest(data, 100.0) == 0

    def test_list_input(self):
        """Accepts list (converted via np.array)."""
        data = [10.0, 20.0, 30.0]
        assert index_of_the_nearest(data, 26.0) == 2  # 30 is closer than 20

    def test_negative_values(self):
        """Works with negative values."""
        data = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
        assert index_of_the_nearest(data, -0.5) == 2  # 0 is closer than -2

    def test_empty_array_raises(self):
        """Empty array raises ValueError."""
        with pytest.raises(ValueError, match="empty array"):
            index_of_the_nearest(np.array([]), 1.0)


# -----------------------------------------------------------------------------
# crop_data tests
# -----------------------------------------------------------------------------


class TestCropData:
    """Tests for crop_data."""

    def test_full_range(self):
        """Full range returns all data."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        x_out, y_out = crop_data(x, y, 1.0, 5.0)
        np.testing.assert_array_equal(x_out, x)
        np.testing.assert_array_equal(y_out, y)

    def test_partial_crop(self):
        """Partial range crops correctly."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        x_out, y_out = crop_data(x, y, 2.0, 4.0)
        np.testing.assert_array_equal(x_out, [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(y_out, [20.0, 30.0, 40.0])

    def test_single_point(self):
        """Single point in range."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        x_out, y_out = crop_data(x, y, 2.0, 2.0)
        np.testing.assert_array_equal(x_out, [2.0])
        np.testing.assert_array_equal(y_out, [20.0])

    def test_empty_result(self):
        """Range outside data returns empty arrays."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        x_out, y_out = crop_data(x, y, 10.0, 20.0)
        assert len(x_out) == 0
        assert len(y_out) == 0

    def test_inclusive_bounds(self):
        """Bounds are inclusive (xmin <= x <= xmax)."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        x_out, y_out = crop_data(x, y, 1.0, 3.0)
        np.testing.assert_array_equal(x_out, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(y_out, [10.0, 20.0, 30.0])

    def test_preserves_pairing(self):
        """x and y remain correctly paired after crop."""
        x = np.array([0.5, 1.5, 2.5, 3.5])
        y = np.array([100.0, 200.0, 300.0, 400.0])
        x_out, y_out = crop_data(x, y, 1.0, 3.0)
        assert len(x_out) == len(y_out)
        for i in range(len(x_out)):
            orig_idx = np.where(x == x_out[i])[0][0]
            assert y_out[i] == y[orig_idx]

    def test_mismatched_lengths_raise(self):
        """x and y with different lengths raise ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0])
        with pytest.raises(ValueError, match="same length"):
            crop_data(x, y, 1.0, 3.0)

    def test_xmin_gt_xmax_raises(self):
        """xmin > xmax raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        with pytest.raises(ValueError, match="xmin must be <= xmax"):
            crop_data(x, y, 5.0, 1.0)

    def test_list_input(self):
        """Accepts list inputs (converted to ndarray)."""
        x = [1.0, 2.0, 3.0]
        y = [10.0, 20.0, 30.0]
        x_out, y_out = crop_data(x, y, 1.5, 2.5)
        np.testing.assert_array_equal(x_out, [2.0])
        np.testing.assert_array_equal(y_out, [20.0])


# -----------------------------------------------------------------------------
# integral_rms tests
# -----------------------------------------------------------------------------


class TestIntegralRms:
    """Tests for integral_rms."""

    def test_constant_asd_full_range(self):
        """Constant ASD over full range: RMS = ASD * sqrt(bandwidth)."""
        f = np.linspace(1.0, 100.0, 100)
        asd = np.ones_like(f) * 0.1  # 0.1 m/√Hz
        rms = integral_rms(f, asd)
        # ∫(ASD²) df from 1 to 100 = 0.01 * 99 = 0.99, sqrt = ~0.995
        expected = 0.1 * np.sqrt(99.0)
        assert rms == pytest.approx(expected, rel=1e-4)

    def test_with_pass_band(self):
        """Pass band restricts integration range."""
        f = np.linspace(0.0, 100.0, 101)
        asd = np.ones_like(f) * 0.1
        rms_full = integral_rms(f, asd)
        rms_band = integral_rms(f, asd, pass_band=[10.0, 50.0])
        assert rms_band < rms_full
        expected = 0.1 * np.sqrt(40.0)  # 50 - 10 = 40 Hz
        assert rms_band == pytest.approx(expected, rel=1e-3)

    def test_pass_band_none_uses_full_range(self):
        """pass_band=None uses full frequency range."""
        f = np.array([1.0, 2.0, 3.0])
        asd = np.array([1.0, 1.0, 1.0])
        rms = integral_rms(f, asd, pass_band=None)
        expected = np.sqrt(2.0)  # ∫1² df from 1 to 3 = 2
        assert rms == pytest.approx(expected, rel=1e-5)

    def test_zero_asd_yields_zero_rms(self):
        """Zero ASD yields zero RMS."""
        f = np.linspace(1.0, 10.0, 10)
        asd = np.zeros_like(f)
        rms = integral_rms(f, asd)
        assert rms == 0.0

    def test_single_point_asd(self):
        """Single-point ASD yields zero (no integration interval)."""
        f = np.array([10.0])
        asd = np.array([1.0])
        rms = integral_rms(f, asd)
        assert rms == 0.0

    def test_empty_passband_overlap_returns_zero(self):
        """Pass band outside frequency range returns 0.0."""
        f = np.linspace(1.0, 10.0, 10)
        asd = np.ones_like(f)
        rms = integral_rms(f, asd, pass_band=[100.0, 200.0])
        assert rms == 0.0

    def test_mismatched_lengths_raise(self):
        """fourier_freq and asd with different lengths raise ValueError."""
        f = np.array([1.0, 2.0, 3.0])
        asd = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="same length"):
            integral_rms(f, asd)

    def test_invalid_pass_band_length_raises(self):
        """pass_band with wrong number of elements raises ValueError."""
        f = np.array([1.0, 2.0, 3.0])
        asd = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="exactly 2 elements"):
            integral_rms(f, asd, pass_band=[1.0, 2.0, 3.0])

    def test_invalid_pass_band_order_raises(self):
        """pass_band with f_min > f_max raises ValueError."""
        f = np.array([1.0, 2.0, 3.0])
        asd = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="f_min must be <= f_max"):
            integral_rms(f, asd, pass_band=[10.0, 1.0])


# -----------------------------------------------------------------------------
# nan_checker tests
# -----------------------------------------------------------------------------


class TestNanChecker:
    """Tests for nan_checker."""

    def test_no_nans_returns_unchanged(self):
        """Array without NaNs returns input and all-False mask."""
        x = np.array([1.0, 2.0, 3.0])
        xnew, nanarray = nan_checker(x)
        np.testing.assert_array_equal(xnew, x)
        np.testing.assert_array_equal(nanarray, [False, False, False])

    def test_with_nans_removes_and_marks(self):
        """Array with NaNs: NaNs removed, mask marks original positions."""
        x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        xnew, nanarray = nan_checker(x)
        np.testing.assert_array_equal(xnew, [1.0, 3.0, 5.0])
        np.testing.assert_array_equal(nanarray, [False, True, False, True, False])

    def test_all_nans(self):
        """All-NaN array returns empty xnew and all-True mask."""
        x = np.array([np.nan, np.nan])
        xnew, nanarray = nan_checker(x)
        assert len(xnew) == 0
        np.testing.assert_array_equal(nanarray, [True, True])

    def test_single_nan(self):
        """Single NaN in array."""
        x = np.array([1.0, np.nan, 2.0])
        xnew, nanarray = nan_checker(x)
        np.testing.assert_array_equal(xnew, [1.0, 2.0])
        np.testing.assert_array_equal(nanarray, [False, True, False])

    def test_warns_on_nan(self, caplog):
        """Logs warning when NaNs are detected."""
        import logging

        caplog.set_level(logging.WARNING)
        x = np.array([1.0, np.nan, 2.0])
        nan_checker(x)
        assert "NaN was detected" in caplog.text

    def test_no_warning_without_nan(self, caplog):
        """No warning when no NaNs present."""
        import logging

        caplog.set_level(logging.WARNING)
        x = np.array([1.0, 2.0, 3.0])
        nan_checker(x)
        assert "NaN was detected" not in caplog.text

    def test_integer_array_returns_unchanged(self):
        """Integer arrays cannot contain NaN; returns input and all-False mask."""
        x = np.array([1, 2, 3])
        xnew, nanarray = nan_checker(x)
        np.testing.assert_array_equal(xnew, x)
        np.testing.assert_array_equal(nanarray, [False, False, False])
