"""Unit tests for loopkit.loop."""

import numpy as np
import pytest

from loopkit.loop import LOOP
from loopkit.component import Component


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

SPS = 1000.0  # Sample rate (Hz) for tests


@pytest.fixture
def gain_component():
    """Simple gain component."""
    return Component("G", SPS, nume=[2.0], deno=[1.0])


@pytest.fixture
def integrator_component():
    """Discrete integrator."""
    return Component("I", SPS, tf="0.01/(1 - z**-1)", domain="z")


@pytest.fixture
def loop_single(gain_component):
    """LOOP with a single gain component."""
    return LOOP(SPS, [gain_component], name="TestLoop")


@pytest.fixture
def loop_two_components(gain_component, integrator_component):
    """LOOP with gain and integrator."""
    return LOOP(SPS, [gain_component, integrator_component], name="TestLoop")


@pytest.fixture
def frequency_array():
    """Log-spaced frequency array."""
    return np.logspace(-2, 2, 50)


# -----------------------------------------------------------------------------
# Construction tests
# -----------------------------------------------------------------------------


class TestLOOPConstruction:
    """Tests for LOOP construction and initialization."""

    def test_construction_empty(self):
        """LOOP can be constructed with no components."""
        loop = LOOP(SPS)
        assert loop.sps == SPS
        assert loop.name == "Loop"
        assert loop.components_dict == {}
        assert loop.property_list == []
        assert loop.callbacks == []
        assert loop.Gc is None
        assert loop.Hc is None
        assert loop.Ec is None

    def test_construction_with_name(self):
        """LOOP accepts custom name."""
        loop = LOOP(SPS, name="MyLoop")
        assert loop.name == "MyLoop"

    def test_construction_invalid_sps_raises(self):
        """LOOP with invalid sps raises ValueError."""
        with pytest.raises(ValueError, match="sps must be a positive number"):
            LOOP(0)
        with pytest.raises(ValueError, match="sps must be a positive number"):
            LOOP(-1.0)

    def test_construction_empty_name_raises(self):
        """LOOP with empty name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            LOOP(SPS, name="")

    def test_construction_with_component_list(self, gain_component):
        """LOOP with component_list adds components and calls update."""
        loop = LOOP(SPS, [gain_component])
        assert "G" in loop.components_dict
        assert loop.components_dict["G"] is gain_component
        assert gain_component._loop is loop
        assert loop.Gc is not None
        assert loop.Gf is not None
        assert loop.Hf is not None
        assert loop.Ef is not None

    def test_construction_with_multiple_components(
        self, gain_component, integrator_component
    ):
        """LOOP with multiple components builds series product."""
        loop = LOOP(SPS, [gain_component, integrator_component])
        assert "G" in loop.components_dict
        assert "I" in loop.components_dict
        assert loop.Gc is not None
        # At low frequency, integrator dominates: gain * 1/(j*2*pi*f*Ts) -> large
        f = np.array([0.01])
        val = loop.Gf(f=f)
        assert np.abs(val) > 1


# -----------------------------------------------------------------------------
# Component management tests
# -----------------------------------------------------------------------------


class TestLOOPComponentManagement:
    """Tests for add_component, remove_component, replace_component, update_component."""

    def test_add_component(self, loop_single, integrator_component):
        """add_component adds a new component."""
        loop = LOOP(SPS)
        loop.add_component(Component("G", SPS, nume=[1.0], deno=[1.0]))
        loop.update()
        assert "G" in loop.components_dict
        loop.add_component(integrator_component, loop_update=True)
        assert "I" in loop.components_dict
        assert integrator_component._loop is loop

    def test_add_component_unnamed_raises(self, loop_single):
        """Adding component with empty name raises ValueError."""
        loop = loop_single
        unnamed = Component("", SPS, nume=[1.0], deno=[1.0])
        unnamed.name = ""
        with pytest.raises(ValueError, match="unnamed"):
            loop.add_component(unnamed)

    def test_add_component_duplicate_name_does_not_add(self, loop_single, caplog):
        """Adding component with existing name logs error and does not add."""
        import logging

        caplog.set_level(logging.ERROR)
        dup = Component("G", SPS, nume=[5.0], deno=[1.0])
        loop_single.add_component(dup)
        assert "G" in loop_single.components_dict
        # Original component should remain (not replaced)
        assert loop_single.components_dict["G"].nume[0] == pytest.approx(2.0)

    def test_remove_component(self, loop_two_components):
        """remove_component removes component by name."""
        loop = loop_two_components
        assert "G" in loop.components_dict
        loop.remove_component("G", loop_update=True)
        assert "G" not in loop.components_dict
        assert "I" in loop.components_dict

    def test_remove_component_nonexistent_raises(self, loop_single):
        """Removing nonexistent component raises ValueError."""
        with pytest.raises(ValueError, match="inexistent"):
            loop_single.remove_component("Nonexistent")

    def test_replace_component(self, loop_single):
        """replace_component swaps component by name."""
        loop = loop_single
        assert loop.components_dict["G"].nume[0] == pytest.approx(2.0)
        new_gain = Component("G", SPS, nume=[10.0], deno=[1.0])
        loop.replace_component("G", new_gain, loop_update=True)
        assert loop.components_dict["G"] is new_gain
        assert loop.components_dict["G"].nume[0] == pytest.approx(10.0)

    def test_replace_component_nonexistent_raises(self, loop_single):
        """Replacing nonexistent component raises ValueError."""
        new_comp = Component("X", SPS, nume=[1.0], deno=[1.0])
        with pytest.raises(ValueError, match="inexistent"):
            loop_single.replace_component("Nonexistent", new_comp)


# -----------------------------------------------------------------------------
# Update and transfer function tests
# -----------------------------------------------------------------------------


class TestLOOPUpdateAndTransferFunctions:
    """Tests for update(), Gf, Hf, Ef, tf_series."""

    def test_update_requires_components(self):
        """update() fails when no components (np.prod of empty -> float, no TE)."""
        loop = LOOP(SPS)
        with pytest.raises(AttributeError):
            loop.update()

    def test_update_sets_Gc_Hc_Ec(self, loop_single):
        """update() sets Gc, Hc, Ec and callable Gf, Hf, Ef."""
        loop = loop_single
        assert loop.Gc is not None
        assert loop.Hc is not None
        assert loop.Ec is not None
        assert callable(loop.Gf)
        assert callable(loop.Hf)
        assert callable(loop.Ef)

    def test_Gf_open_loop(self, loop_single, frequency_array):
        """Gf returns open-loop transfer function (product of components)."""
        val = loop_single.Gf(f=frequency_array)
        assert val.shape == frequency_array.shape
        # Gain of 2 at DC
        f_dc = np.array([0.001])
        g_dc = loop_single.Gf(f=f_dc)
        assert np.abs(g_dc[0]) == pytest.approx(2.0, rel=1e-4)

    def test_Hf_closed_loop(self, loop_single, frequency_array):
        """Hf returns closed-loop (complementary sensitivity) G/(1+G)."""
        val = loop_single.Hf(f=frequency_array)
        assert val.shape == frequency_array.shape
        # At DC with G=2: H = 2/(1+2) = 2/3
        f_dc = np.array([0.001])
        h_dc = loop_single.Hf(f=f_dc)
        assert np.abs(h_dc[0]) == pytest.approx(2.0 / 3.0, rel=1e-4)

    def test_Ef_error_transfer(self, loop_single, frequency_array):
        """Ef returns error transfer 1/(1+G)."""
        val = loop_single.Ef(f=frequency_array)
        assert val.shape == frequency_array.shape
        # At DC with G=2: E = 1/(1+2) = 1/3
        f_dc = np.array([0.001])
        e_dc = loop_single.Ef(f=f_dc)
        assert np.abs(e_dc[0]) == pytest.approx(1.0 / 3.0, rel=1e-4)

    def test_tf_series_mode_none(self, loop_single, frequency_array):
        """tf_series with mode=None returns open-loop product."""
        val = loop_single.tf_series(f=frequency_array, mode=None)
        expected = loop_single.Gf(f=frequency_array)
        np.testing.assert_allclose(val, expected, rtol=1e-10)

    def test_tf_series_mode_H(self, loop_single, frequency_array):
        """tf_series with mode='H' returns closed-loop."""
        val = loop_single.tf_series(f=frequency_array, mode="H")
        expected = loop_single.Hf(f=frequency_array)
        np.testing.assert_allclose(val, expected, rtol=1e-10)

    def test_tf_series_mode_E(self, loop_single, frequency_array):
        """tf_series with mode='E' returns error transfer."""
        val = loop_single.tf_series(f=frequency_array, mode="E")
        expected = loop_single.Ef(f=frequency_array)
        np.testing.assert_allclose(val, expected, rtol=1e-10)

    def test_tf_series_invalid_mode_raises(self, loop_single, frequency_array):
        """tf_series with invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="invalid mode"):
            loop_single.tf_series(f=frequency_array, mode="X")

    def test_mag_phase_lambdas(self, loop_single, frequency_array):
        """mag, phase, mag_dB, phase_deg are callable and return correct shapes."""
        frfr = frequency_array
        assert loop_single.mag(frfr).shape == frfr.shape
        assert loop_single.phase(frfr).shape == frfr.shape
        assert loop_single.mag_dB(frfr).shape == frfr.shape
        assert loop_single.phase_deg(frfr).shape == frfr.shape
        assert loop_single.phase_unwrapped(frfr).shape == frfr.shape
        assert loop_single.phase_deg_unwrapped(frfr).shape == frfr.shape


# -----------------------------------------------------------------------------
# Callback tests
# -----------------------------------------------------------------------------


class TestLOOPCallbacks:
    """Tests for register_callback and notify_callbacks."""

    def test_register_and_notify_callback(self, loop_single):
        """Registered callback is executed by notify_callbacks."""
        called = []

        def cb():
            called.append(1)

        loop_single.register_callback(cb)
        assert len(loop_single.callbacks) == 1
        loop_single.notify_callbacks()
        assert called == [1]

    def test_callback_with_args(self, loop_single):
        """Callback receives *args and **kwargs."""
        result = {}

        def cb(a, b, c=None):
            result["a"] = a
            result["b"] = b
            result["c"] = c

        loop_single.register_callback(cb, 10, 20, c=30)
        loop_single.notify_callbacks()
        assert result["a"] == 10
        assert result["b"] == 20
        assert result["c"] == 30

    def test_multiple_callbacks(self, loop_single):
        """Multiple callbacks are all executed."""
        order = []

        def cb1():
            order.append(1)

        def cb2():
            order.append(2)

        loop_single.register_callback(cb1)
        loop_single.register_callback(cb2)
        loop_single.notify_callbacks()
        assert order == [1, 2]


# -----------------------------------------------------------------------------
# collect_components and point-to-point tests
# -----------------------------------------------------------------------------


class TestLOOPCollectComponents:
    """Tests for collect_components."""

    def test_collect_components_empty_when_to_none(self, loop_two_components):
        """collect_components with _to=None returns empty list."""
        compo_list, path = loop_two_components.collect_components(_from="G", _to=None)
        assert compo_list == []
        assert path == ""

    def test_collect_components_forward_path(self, loop_two_components):
        """collect_components returns components from _from to _to (exclusive)."""
        compo_list, path = loop_two_components.collect_components(_from="G", _to="I")
        assert len(compo_list) == 1
        assert compo_list[0].name == "G"
        assert "G" in path

    def test_collect_components_wraparound(self, loop_two_components):
        """collect_components wraps around loop when start > end."""
        compo_list, path = loop_two_components.collect_components(_from="I", _to="G")
        assert len(compo_list) == 1
        assert compo_list[0].name == "I"

    def test_collect_components_full_loop(self, loop_two_components):
        """collect_components _from==_to returns full loop."""
        compo_list, path = loop_two_components.collect_components(_from="G", _to="G")
        assert len(compo_list) == 2
        names = [c.name for c in compo_list]
        assert "G" in names and "I" in names

    def test_collect_components_nonexistent_from_raises(self, loop_two_components):
        """collect_components with nonexistent _from raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            loop_two_components.collect_components(_from="X", _to="G")

    def test_collect_components_nonexistent_to_raises(self, loop_two_components):
        """collect_components with nonexistent _to raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            loop_two_components.collect_components(_from="G", _to="X")


# -----------------------------------------------------------------------------
# point_to_point_component and point_to_point_tf tests
# -----------------------------------------------------------------------------


class TestLOOPPointToPoint:
    """Tests for point_to_point_component and point_to_point_tf."""

    def test_point_to_point_component_open_loop(
        self, loop_two_components
    ):
        """point_to_point_component returns product along path."""
        comp = loop_two_components.point_to_point_component(
            _from="G", _to="I", closed=False
        )
        assert comp is not None
        assert comp.TE is not None
        assert comp.TF is not None

    def test_point_to_point_component_closed_loop(
        self, loop_two_components
    ):
        """point_to_point_component with closed=True includes E."""
        comp = loop_two_components.point_to_point_component(
            _from="G", _to=None, closed=True
        )
        assert comp is not None

    def test_point_to_point_tf(self, loop_two_components, frequency_array):
        """point_to_point_tf returns frequency response of path."""
        tf = loop_two_components.point_to_point_tf(
            frequency_array, _from="G", _to="I", closed=False
        )
        assert tf.shape == frequency_array.shape
        assert np.all(np.isfinite(tf))


# -----------------------------------------------------------------------------
# noise_propagation tests
# -----------------------------------------------------------------------------


class TestLOOPNoisePropagation:
    """Tests for noise_propagation_t and noise_propagation_asd."""

    def test_noise_propagation_t(self, loop_single):
        """noise_propagation_t propagates time-domain noise through path."""
        # Time steps must be multiples of sampling time (1/SPS = 0.001 s)
        n = 100
        tau = np.arange(n) / SPS
        noise = np.random.randn(n)
        noise_prop, unit_prop = loop_single.noise_propagation_t(
            tau, noise, _from="G", _to=None
        )
        assert noise_prop.shape == noise.shape
        assert unit_prop is not None

    def test_noise_propagation_asd(self, loop_single, frequency_array):
        """noise_propagation_asd propagates ASD through path."""
        asd = np.ones_like(frequency_array) * 0.1
        asd_prop, unit_prop, bode, rms = loop_single.noise_propagation_asd(
            frequency_array, asd, _from="G", _to=None
        )
        assert asd_prop.shape == frequency_array.shape
        assert "f" in bode and "mag" in bode and "phase" in bode
        assert rms >= 0
        assert np.all(np.isfinite(asd_prop))


# -----------------------------------------------------------------------------
# Plotting tests (smoke tests - just ensure they run)
# -----------------------------------------------------------------------------


class TestLOOPPlots:
    """Smoke tests for plotting methods."""

    def test_magnitude_plot(self, loop_single, frequency_array):
        """magnitude_plot runs without error."""
        ax = loop_single.magnitude_plot(
            frequency_array, which="G", figsize=(4, 3)
        )
        assert ax is not None
        assert ax.figure is not None

    def test_magnitude_plot_invalid_which_raises(
        self, loop_single, frequency_array
    ):
        """magnitude_plot with invalid which raises ValueError."""
        with pytest.raises(ValueError, match="Invalid transfer function key"):
            loop_single.magnitude_plot(
                frequency_array, which="X"
            )

    def test_bode_plot(self, loop_single, frequency_array):
        """bode_plot runs without error."""
        ax_mag, ax_phase = loop_single.bode_plot(
            frequency_array, which="G", figsize=(4, 4)
        )
        assert ax_mag is not None
        assert ax_phase is not None
        assert ax_mag.figure is ax_phase.figure

    def test_bode_plot_invalid_which_raises(
        self, loop_single, frequency_array
    ):
        """bode_plot with invalid which raises ValueError."""
        with pytest.raises(ValueError, match="Invalid transfer function key"):
            loop_single.bode_plot(frequency_array, which=["X"])

    def test_nyquist_plot(self, loop_single, frequency_array):
        """nyquist_plot runs without error."""
        ax = loop_single.nyquist_plot(
            frequency_array, which="G", figsize=(4, 4)
        )
        assert ax is not None
        assert ax.figure is not None

    def test_nyquist_plot_invalid_which_raises(
        self, loop_single, frequency_array
    ):
        """nyquist_plot with invalid which raises ValueError."""
        with pytest.raises(ValueError, match="Invalid transfer function key"):
            loop_single.nyquist_plot(frequency_array, which="X")


# -----------------------------------------------------------------------------
# Property delegator tests
# -----------------------------------------------------------------------------


class TestLOOPPropertyDelegator:
    """Tests for create_property_delegator and register_component_properties."""

    def test_create_property_delegator_requires_properties(self, loop_single):
        """create_property_delegator works when component has properties."""
        # Standard Component does not have a 'properties' dict.
        # We need a component with properties - e.g. one that defines them.
        # For now, test that register_component_properties runs without error
        # when components have no properties.
        loop_single.register_component_properties()
        # With no properties, property_list may stay empty
        assert isinstance(loop_single.property_list, list)
