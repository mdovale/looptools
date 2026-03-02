"""Unit tests for looptools.dimension."""

import pytest

from looptools.dimension import Dimension


# -----------------------------------------------------------------------------
# Construction tests
# -----------------------------------------------------------------------------


class TestDimensionConstruction:
    """Tests for Dimension construction."""

    def test_construction_empty(self):
        """Empty construction yields dimensionless."""
        d = Dimension()
        assert d.numes == []
        assert d.denos == []

    def test_construction_dimensionless_flag(self):
        """dimensionless=True yields empty numes and denos."""
        d = Dimension(dimensionless=True)
        assert d.numes == []
        assert d.denos == []
        assert d.unit_string() == ""

    def test_construction_simple(self):
        """Simple m/s dimension."""
        d = Dimension(["m"], ["s"])
        assert d.numes == ["m"]
        assert d.denos == ["s"]
        assert d.unit_string() == "m/s"

    def test_construction_multiple_units(self):
        """Multiple units in num and den."""
        d = Dimension(["m", "kg"], ["s", "s"])
        assert sorted(d.numes) == ["kg", "m"]
        assert d.denos == ["s", "s"]
        assert d.unit_string() in ("m*kg/s*s", "kg*m/s*s")  # order may vary

    def test_construction_reduces_on_init(self):
        """Common terms are reduced on construction."""
        d = Dimension(["m", "s"], ["s"])
        assert d.numes == ["m"]
        assert d.denos == []

    def test_construction_no_input_mutation(self):
        """Constructor does not mutate caller's lists."""
        numes, denos = ["m", "s"], ["s"]
        Dimension(numes, denos)
        assert numes == ["m", "s"]
        assert denos == ["s"]


# -----------------------------------------------------------------------------
# Multiplication tests
# -----------------------------------------------------------------------------


class TestDimensionMultiplication:
    """Tests for Dimension.__mul__."""

    def test_mul_simple_cancellation(self):
        """m/s * s = m."""
        d1 = Dimension(["m"], ["s"])
        d2 = Dimension(["s"], [])
        result = d1 * d2
        assert result.numes == ["m"]
        assert result.denos == []
        assert result.unit_string() == "m"

    def test_mul_dimensionless_left(self):
        """1 * m/s = m/s."""
        d1 = Dimension(dimensionless=True)
        d2 = Dimension(["m"], ["s"])
        result = d1 * d2
        assert result.numes == ["m"]
        assert result.denos == ["s"]

    def test_mul_dimensionless_right(self):
        """m/s * 1 = m/s."""
        d1 = Dimension(["m"], ["s"])
        d2 = Dimension(dimensionless=True)
        result = d1 * d2
        assert result.numes == ["m"]
        assert result.denos == ["s"]

    def test_mul_dimensionless_both(self):
        """1 * 1 = 1."""
        d1 = Dimension(dimensionless=True)
        d2 = Dimension(dimensionless=True)
        result = d1 * d2
        assert result.numes == []
        assert result.denos == []

    def test_mul_complex(self):
        """m²/s * s*kg/m = m*kg."""
        d1 = Dimension(["m", "m"], ["s"])
        d2 = Dimension(["s", "kg"], ["m"])
        result = d1 * d2
        assert "m" in result.numes and "kg" in result.numes
        assert len(result.numes) == 2
        assert result.denos == []


# -----------------------------------------------------------------------------
# Division tests
# -----------------------------------------------------------------------------


class TestDimensionDivision:
    """Tests for Dimension.__truediv__."""

    def test_truediv_simple(self):
        """m / s = m/s."""
        d1 = Dimension(["m"], [])
        d2 = Dimension(["s"], [])
        result = d1 / d2
        assert result.numes == ["m"]
        assert result.denos == ["s"]

    def test_truediv_with_cancellation(self):
        """m/s / s = m/s²."""
        d1 = Dimension(["m"], ["s"])
        d2 = Dimension(["s"], [])
        result = d1 / d2
        assert result.numes == ["m"]
        assert result.denos == ["s", "s"]

    def test_truediv_inverse_of_mul(self):
        """(m/s * s) / s = m/s."""
        d = Dimension(["m"], ["s"]) * Dimension(["s"], []) / Dimension(["s"], [])
        assert d.numes == ["m"]
        assert d.denos == ["s"]


# -----------------------------------------------------------------------------
# Equality tests
# -----------------------------------------------------------------------------


class TestDimensionEquality:
    """Tests for Dimension.__eq__."""

    def test_eq_same(self):
        """Identical dimensions are equal."""
        d1 = Dimension(["m"], ["s"])
        d2 = Dimension(["m"], ["s"])
        assert d1 == d2

    def test_eq_order_independent(self):
        """Order of units does not affect equality."""
        d1 = Dimension(["m", "kg"], ["s"])
        d2 = Dimension(["kg", "m"], ["s"])
        assert d1 == d2

    def test_eq_dimensionless(self):
        """Dimensionless dimensions are equal."""
        d1 = Dimension(dimensionless=True)
        d2 = Dimension([], [])
        assert d1 == d2

    def test_eq_different_numes(self):
        """Different numes are not equal."""
        d1 = Dimension(["m"], ["s"])
        d2 = Dimension(["kg"], ["s"])
        assert d1 != d2

    def test_eq_different_denos(self):
        """Different denos are not equal."""
        d1 = Dimension(["m"], ["s"])
        d2 = Dimension(["m"], ["kg"])
        assert d1 != d2

    def test_eq_multiplicity_matters(self):
        """Multiplicity of units matters (m² != m)."""
        d1 = Dimension(["m", "m"], [])
        d2 = Dimension(["m"], [])
        assert d1 != d2

    def test_eq_false_for_non_dimension(self):
        """Comparison with non-Dimension returns False (NotImplemented -> False)."""
        d = Dimension(["m"], ["s"])
        assert (d == "m/s") is False
        assert (d == 42) is False


# -----------------------------------------------------------------------------
# unit_string and repr tests
# -----------------------------------------------------------------------------


class TestDimensionStringRepresentation:
    """Tests for unit_string, __repr__, and show."""

    def test_unit_string_dimensionless(self):
        """Dimensionless yields empty string."""
        d = Dimension(dimensionless=True)
        assert d.unit_string() == ""

    def test_unit_string_numerator_only(self):
        """Numerator only: m."""
        d = Dimension(["m"], [])
        assert d.unit_string() == "m"

    def test_unit_string_denominator_only(self):
        """Denominator only: 1/s."""
        d = Dimension([], ["s"])
        assert d.unit_string() == "1/s"

    def test_unit_string_both(self):
        """Both: m/s."""
        d = Dimension(["m"], ["s"])
        assert d.unit_string() == "m/s"

    def test_repr_dimensionless(self):
        """repr for dimensionless."""
        d = Dimension(dimensionless=True)
        assert repr(d) == "Dimension(dimensionless=True)"

    def test_repr_with_units(self):
        """repr includes unit string."""
        d = Dimension(["m"], ["s"])
        assert repr(d) == "Dimension('m/s')"

    def test_show_prints_bracketed(self, capsys):
        """show() prints [unit_string]."""
        d = Dimension(["m"], ["s"])
        d.show()
        captured = capsys.readouterr()
        assert captured.out.strip() == "[m/s]"


# -----------------------------------------------------------------------------
# reduction tests (internal behavior)
# -----------------------------------------------------------------------------


class TestDimensionReduction:
    """Tests for reduction logic."""

    def test_reduction_full_cancel(self):
        """Fully cancelling terms yields dimensionless."""
        d = Dimension(["m", "s"], ["m", "s"])
        assert d.numes == []
        assert d.denos == []

    def test_reduction_partial_cancel(self):
        """Partial cancellation."""
        d = Dimension(["m", "m", "s"], ["m", "s"])
        assert d.numes == ["m"]
        assert d.denos == []
