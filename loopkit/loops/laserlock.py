# BSD 3-Clause License

# Copyright (c) 2025, Miguel Dovale

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting this software to
# foreign countries or providing access to foreign persons.
#
from __future__ import annotations

import logging
from functools import partial
from typing import Optional, Sequence, Tuple

import numpy as np

import loopkit.components as lc
import loopkit.loopmath as lm
from loopkit.component import Component
from loopkit.dimension import Dimension
from loopkit.loop import LOOP

logger = logging.getLogger(__name__)

# Default SOS coefficients from Simulink PLL (sosScaled26)
_DEFAULT_SOS = [16777216, 33554432, 16777216, 16777216, -33181752, 16408629]


def _default_laser_plant(sps: float) -> Component:
    """Create default laser dynamics plant from PLL-SPEC (lasTFb, lasTFa)."""
    nume = np.array([
        -52937.5288030121, 59309.27023733428,
        52938.33203142742, -59308.466786107,
    ])
    deno = np.array([
        1.0, -2.98154285463868, 2.9631583948738554, -0.9816155352742084,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])
    return Component(
        "Laser", sps, nume=nume, deno=deno,
        unit=Dimension(dimensionless=True),
    )


def _validate_multirate_params(
    sps_loop: float,
    sps_adc: float,
    amp: float,
    kp: float,
    ki: float,
    n_reg: int,
    off: Tuple[str, ...],
) -> None:
    """Validate MultiRateLaserLock constructor parameters."""
    if not isinstance(sps_loop, (int, float)) or sps_loop <= 0:
        raise ValueError(f"sps_loop must be positive, got {sps_loop!r}")
    if not isinstance(sps_adc, (int, float)) or sps_adc <= 0:
        raise ValueError(f"sps_adc must be positive, got {sps_adc!r}")
    if sps_adc < sps_loop:
        raise ValueError(
            f"sps_adc must be >= sps_loop for downsampling. "
            f"Got sps_adc={sps_adc}, sps_loop={sps_loop}"
        )
    ratio = sps_adc / sps_loop
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            f"sps_adc/sps_loop must be an integer. Got {ratio}"
        )
    if not isinstance(amp, (int, float)) or amp <= 0:
        raise ValueError(f"Amp must be positive, got {amp!r}")
    if not isinstance(kp, (int, float)):
        raise ValueError(f"Kp must be numeric, got {type(kp).__name__}")
    if not isinstance(ki, (int, float)):
        raise ValueError(f"Ki must be numeric, got {type(ki).__name__}")
    if not isinstance(n_reg, int) or n_reg < 0:
        raise ValueError(f"n_reg must be non-negative int, got {n_reg!r}")
    valid_names = frozenset((
        "PD", "IIR", "RateTransition", "PIII", "Delay", "DAC",
        "Laser", "PreGain", "PA", "LUT",
    ))
    for item in off:
        if item not in valid_names:
            raise ValueError(
                f"off must contain only {sorted(valid_names)}, got {item!r}"
            )


def _validate_laserlock_params(
    plant: Component,
    amp_reference: float,
    amp_input: float,
    lpf_cutoff: float,
    lpf_n: int,
    cshift: int,
    f_i: Optional[float],
    f_ii: Optional[float],
    n_reg: Optional[int],
    sps_mixer: float,
    nume_mixer: Sequence[float],
    f_trans: Optional[float],
) -> None:
    """Validate LaserLock constructor parameters. Raises ValueError on invalid input."""
    if not isinstance(plant, Component):
        raise TypeError(f"Plant must be a Component, got {type(plant).__name__}")
    if not (isinstance(amp_reference, (int, float)) and amp_reference > 0):
        raise ValueError(f"Amp_reference must be positive, got {amp_reference!r}")
    if not (isinstance(amp_input, (int, float)) and amp_input > 0):
        raise ValueError(f"Amp_input must be positive, got {amp_input!r}")
    if not (isinstance(lpf_cutoff, (int, float)) and lpf_cutoff > 0):
        raise ValueError(f"LPF_cutoff must be positive, got {lpf_cutoff!r}")
    if not (isinstance(lpf_n, int) and lpf_n >= 1):
        raise ValueError(f"LPF_n must be an int >= 1, got {lpf_n!r}")
    if not (isinstance(cshift, int) and cshift >= 0):
        raise ValueError(f"Cshift must be a non-negative int, got {cshift!r}")
    if f_i is not None and not (isinstance(f_i, (int, float)) and f_i > 0):
        raise ValueError(f"f_I must be positive when provided, got {f_i!r}")
    if f_ii is not None and not (isinstance(f_ii, (int, float)) and f_ii > 0):
        raise ValueError(f"f_II must be positive when provided, got {f_ii!r}")
    if n_reg is not None and not (isinstance(n_reg, int) and n_reg >= 0):
        raise ValueError(f"n_reg must be a non-negative int when provided, got {n_reg!r}")
    if not (isinstance(sps_mixer, (int, float)) and sps_mixer > 0):
        raise ValueError(f"sps_mixer must be positive, got {sps_mixer!r}")
    if f_trans is not None and not (isinstance(f_trans, (int, float)) and f_trans > 0):
        raise ValueError(f"f_trans must be positive when provided, got {f_trans!r}")
    if len(nume_mixer) == 0:
        raise ValueError("nume_mixer must be a non-empty sequence")


class LaserLock(LOOP):
    """
    Generic feedback loop model for a heterodyne phase-locking laser lock system.

    This base class defines the signal flow and control elements (Plant, Mixer, LPF,
    Gain, Servo, Delay) without baked-in sampling frequencies. Subclasses provide
    hardware-specific sps values.

    Parameters
    ----------
    sps : float
        Main loop sample rate (Hz), used for Servo and Delay components.
    Plant : Component
        loopkit Component specifying the plant.
    Amp_reference : float
        Mixer local oscillator amplitude (Vpp).
    Amp_input : float
        Beatnote amplitude (Vpp).
    LPF_cutoff : float
        Butterworth LPF cutoff frequency (Hz).
    LPF_n : int
        Butterworth LPF number of stages (must be >= 1).
    Cshift : int
        Gain reduction stage, number of bits for RightBitShift.
    Kp_db : float
        P-gain (dB).
    f_I : float, optional
        First integrator crossover frequency (Hz).
    f_II : float, optional
        Second integrator crossover frequency (Hz).
    n_reg : int, optional
        DSP delay component (number of registers). Defaults to 0.
    sps_mixer : float, optional
        Sample rate for Mixer, LPF, and Gain components (Hz). Defaults to sps.
    nume_mixer : sequence of float, optional
        Mixer transfer function numerator (rad/s). Defaults to computed value.
    off : sequence of str, optional
        Component names to exclude from the loop. Defaults to none excluded.
    f_trans : float, optional
        Transfer function extrapolation frequency for the servo.
    """

    def __init__(
        self,
        sps: float,
        Plant: Component,
        Amp_reference: float,
        Amp_input: float,
        LPF_cutoff: float,
        LPF_n: int,
        Cshift: int,
        Kp_db: float,
        f_I: Optional[float] = None,
        f_II: Optional[float] = None,
        n_reg: Optional[int] = None,
        sps_mixer: Optional[float] = None,
        nume_mixer: Optional[Sequence[float]] = None,
        off: Optional[Sequence[str]] = None,
        f_trans: Optional[float] = None,
    ) -> None:
        super().__init__(sps)

        _sps_mixer = float(sps) if sps_mixer is None else float(sps_mixer)
        _nume_mixer: Tuple[float, ...]
        if nume_mixer is None:
            _nume_mixer = (2 * np.pi * Amp_input * Amp_reference / _sps_mixer,)
        else:
            _nume_mixer = tuple(float(x) for x in nume_mixer)

        _validate_laserlock_params(
            Plant, Amp_reference, Amp_input, LPF_cutoff, LPF_n, Cshift,
            f_I, f_II, n_reg, _sps_mixer, _nume_mixer, f_trans,
        )

        _off: Tuple[str, ...] = () if off is None else tuple(str(x) for x in off)
        _n_reg: int = 0 if n_reg is None else n_reg

        self.Plant = Plant
        self.Amp_reference = float(Amp_reference)
        self.Amp_input = float(Amp_input)
        self.LPF_cutoff = float(LPF_cutoff)
        self.LPF_n = int(LPF_n)
        self.Cshift = int(Cshift)
        self.Kp_db = float(Kp_db)
        self.f_I = f_I if f_I is None else float(f_I)
        self.f_II = f_II if f_II is None else float(f_II)
        self.n_reg = _n_reg
        self.off = _off
        self.sps_mixer = _sps_mixer
        self.nume_mixer = _nume_mixer
        self.Kp_log2 = lm.db_to_log2_gain(Kp_db)

        if f_I is not None and f_II is None:
            self.Ki_log2 = lm.gain_for_crossover_frequency(
                self.Kp_log2, sps, f_I, kind="I"
            )
            self.Kii_log2 = None
        elif (f_I, f_II) != (None, None):
            self.Ki_log2, self.Kii_log2 = lm.gain_for_crossover_frequency(
                self.Kp_log2, sps, (f_I, f_II), kind="II"
            )
        else:
            self.Ki_log2 = None
            self.Kii_log2 = None

        if "Plant" not in _off:
            self.add_component(Plant)

        if "Mixer" not in _off:
            self.add_component(
                Component("Mixer", _sps_mixer, nume=list(_nume_mixer), deno=[1, -1])
            )

        if "LPF" not in _off:
            self.add_component(
                lc.ButterworthLPFComponent("LPF", _sps_mixer, LPF_cutoff, LPF_n)
            )

        if "Gain" not in _off:
            self.add_component(lc.RightBitShiftComponent("Gain", _sps_mixer, Cshift))

        if "Servo" not in _off:
            self.add_component(
                lc.MokuPIDController(
                    "Servo", sps, Kp_db, f_I, f_II, None, f_trans=f_trans
                )
            )

        if "Delay" not in _off:
            self.add_component(lc.DSPDelayComponent("Delay", sps, _n_reg))

        self.update()
        self.register_component_properties()


class MultiRateLaserLock(LOOP):
    """
    Multi-rate optical PLL for heterodyne laser locking (Simulink PLL topology).

    Supports different sample rates for the ADC path (sps_adc) and loop path (sps_loop).
    When sps_adc > sps_loop, a RateTransitionComponent models the 40→20 MHz downsampling.

    Topology: PD → IIR → [RateTransition] → PIII → Delay → DAC → Laser → PreGain → VCO (PA + LUT)

    Parameters
    ----------
    sps_loop : float
        Loop update rate in Hz (e.g. 20e6 for fLoop).
    sps_adc : float, optional
        ADC/sensor path rate in Hz (e.g. 40e6). Default: same as sps_loop (single-rate).
    Amp : float
        Phase detector amplitude (normalized).
    Kp : float
        Proportional gain (linear, e.g. 5000).
    Ki : float
        Integral gain (linear, e.g. 4000).
    sos : sequence of float, optional
        IIR filter SOS coefficients. Default: Simulink sosScaled26.
    Plant : Component, optional
        Laser dynamics. Default: discrete TF from PLL-SPEC.
    dac_gain : float, optional
        DAC gain. Default 2/2^16.
    pre_gain : float, optional
        Pre-loop gain. Default 0.1.
    n_reg : int, optional
        Pipeline delay (samples). Default 1.
    iir_input_scale : float, optional
        IIR input scaling. Default 2^-24.
    iir_output_scale : float, optional
        IIR output scaling. Default 2^-14.
    off : sequence of str, optional
        Component names to exclude.
    name : str, optional
        Loop name.
    """

    def __init__(
        self,
        sps_loop: float,
        Amp: float,
        Kp: float,
        Ki: float,
        *,
        sps_adc: Optional[float] = None,
        sos: Optional[Sequence[float]] = None,
        Plant: Optional[Component] = None,
        dac_gain: float = 2 / 2**16,
        pre_gain: float = 0.1,
        n_reg: int = 1,
        iir_input_scale: float = 2**-24,
        iir_output_scale: float = 2**-14,
        off: Optional[Sequence[str]] = None,
        name: Optional[str] = None,
    ) -> None:
        _sps_adc = float(sps_loop) if sps_adc is None else float(sps_adc)
        _off: Tuple[str, ...] = () if off is None else tuple(str(x) for x in off)
        _validate_multirate_params(
            sps_loop, _sps_adc, Amp, Kp, Ki, n_reg, _off,
        )

        super().__init__(sps_loop, name=name or "MultiRateLaserLock")

        self._sps_adc = _sps_adc
        self._sps_loop = float(sps_loop)
        self._amp = float(Amp)
        self._kp = float(Kp)
        self._ki = float(Ki)
        self._n_reg = int(n_reg)
        self._dac_gain = float(dac_gain)
        self._pre_gain = float(pre_gain)
        self._off = _off
        self._sos = list(sos) if sos is not None else _DEFAULT_SOS
        self._multirate = _sps_adc > sps_loop
        _plant = Plant if Plant is not None else _default_laser_plant(sps_loop)

        # High-rate path (sps_adc): PD, IIR, [RateTransition]
        if "PD" not in _off:
            self.add_component(lc.PDComponent("PD", _sps_adc, self._amp))

        if "IIR" not in _off:
            self.add_component(
                lc.IIRFilterComponent.from_sos(
                    "IIR", _sps_adc, self._sos,
                    input_scale=iir_input_scale,
                    output_scale=iir_output_scale,
                )
            )

        if self._multirate and "RateTransition" not in _off:
            self.add_component(
                lc.RateTransitionComponent(
                    "RateTransition", sps_in=_sps_adc, sps_out=sps_loop,
                )
            )

        # Low-rate path (sps_loop)
        if "PIII" not in _off:
            self.add_component(
                lc.PIControllerComponent(
                    "PIII", sps_loop, self._kp, self._ki, gain_scale="linear"
                )
            )

        if "Delay" not in _off:
            self.add_component(lc.DSPDelayComponent("Delay", sps_loop, self._n_reg))

        if "DAC" not in _off:
            self.add_component(
                lc.MultiplierComponent(
                    "DAC", sps_loop, self._dac_gain,
                    unit=Dimension(dimensionless=True),
                )
            )

        if "Laser" not in _off:
            self.add_component(_plant)

        if "PreGain" not in _off:
            self.add_component(
                lc.MultiplierComponent(
                    "PreGain", sps_loop, self._pre_gain,
                    unit=Dimension(dimensionless=True),
                )
            )

        if "PA" not in _off:
            self.add_component(lc.PAComponent("PA", sps_loop))
        if "LUT" not in _off:
            self.add_component(lc.LUTComponent("LUT", sps_loop))

        if _off:
            logger.warning("MultiRateLaserLock: excluded components: %s", _off)

        self.update()
        self.register_component_properties()

    @property
    def sps_adc(self) -> float:
        """ADC/sensor path sample rate."""
        return self._sps_adc

    @property
    def sps_loop(self) -> float:
        """Loop update rate."""
        return self._sps_loop

    @property
    def Amp(self) -> float:
        """Phase detector amplitude."""
        return self._amp

    @property
    def Kp(self) -> float:
        """Proportional gain (linear)."""
        return self._kp

    @property
    def Ki(self) -> float:
        """Integral gain (linear)."""
        return self._ki

    @property
    def n_reg(self) -> int:
        """Pipeline delay (samples)."""
        return self._n_reg

    @property
    def dac_gain(self) -> float:
        """DAC gain."""
        return self._dac_gain

    @property
    def pre_gain(self) -> float:
        """Pre-loop gain."""
        return self._pre_gain

    @property
    def off(self) -> Tuple[str, ...]:
        """Excluded component names."""
        return self._off

    @property
    def multirate(self) -> bool:
        """True if sps_adc != sps_loop."""
        return self._multirate

    def update(self) -> None:
        """Override: handle multi-rate by computing Gc numerically when needed."""
        import control
        if not self._multirate:
            super().update()
            return

        # Multi-rate: components have different sps; control.series fails.
        # Build Gc as a Component with custom TF that multiplies each component's TF.
        # Use a placeholder TE for compatibility; actual response from tf_series.
        import control
        placeholder = control.tf([1.0], [1.0], 1 / self.sps)
        self.Gc = Component("G", self.sps, tf=placeholder, unit=Dimension(dimensionless=True))
        self.Gc.TF = partial(self._multirate_tf, mode=None)

        H_TE = control.feedback(placeholder, 1)
        self.Hc = Component("H", self.sps, tf=H_TE, unit=self.Gc.unit)
        self.Hc.TF = partial(self._multirate_tf, mode="H")

        E_TE = control.feedback(1, placeholder)
        self.Ec = Component("E", self.sps, tf=E_TE, unit=self.Gc.unit)
        self.Ec.TF = partial(self._multirate_tf, mode="E")

        def _get_phase(tf_func, frfr, deg):
            return np.angle(tf_func(frfr), deg=deg)

        def _get_magnitude(tf_func, frfr, dB):
            mag = np.abs(tf_func(frfr))
            return control.mag2db(mag) if dB else mag

        self.Gf = partial(self.tf_series, mode=None)
        self.Hf = partial(self.tf_series, mode="H")
        self.Ef = partial(self.tf_series, mode="E")
        self.phase = lambda frfr: _get_phase(self.Gf, frfr, deg=False)
        self.phase_deg = lambda frfr: _get_phase(self.Gf, frfr, deg=True)
        self.mag = lambda frfr: _get_magnitude(self.Gf, frfr, dB=False)
        self.mag_dB = lambda frfr: _get_magnitude(self.Gf, frfr, dB=True)
        self.phase_unwrapped = lambda frfr: np.unwrap(self.phase(frfr))
        self.phase_deg_unwrapped = lambda frfr: np.unwrap(self.phase_deg(frfr), period=360)

        if getattr(self, "_loop", None) is None and hasattr(self, "callbacks"):
            for cb in self.callbacks:
                cb()

    def _multirate_tf(
        self,
        f: np.ndarray,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        """Compute multi-rate loop TF by multiplying each component at its own sps."""
        tf = self.tf_series(f=f, mode=mode)
        return tf

    def tf_series(
        self,
        f,
        components=None,
        mode=None,
        extrapolate=False,
        f_trans=1e-1,
        power=-2,
        size=2,
        solver=True,
    ):
        """Override: multiply each component's TF at its own sps for multi-rate."""
        from loopkit.loopmath import tf_power_extrapolate

        if components is None:
            comps = list(self.components_dict.values())
        else:
            comps = list(components)

        tf = np.ones(np.atleast_1d(f).shape, dtype=complex)
        for comp in comps:
            tf *= comp.TF(f=f)

        if mode is None:
            output = tf
        elif mode == "H":
            output = tf / (1 + tf)
        elif mode == "E":
            output = 1 / (1 + tf)
        else:
            raise ValueError(f"invalid mode {mode}")

        if extrapolate:
            output = tf_power_extrapolate(f, output, f_trans=f_trans, power=power, size=size, solver=solver)
        return output

    def point_to_point_component(
        self,
        _from: Optional[str] = None,
        _to: Optional[str] = None,
        closed: bool = False,
        view: bool = False,
    ) -> Component:
        """Override: use tf_series for multi-rate (avoids np.prod of mixed-rate components)."""
        import copy
        compo_list, propagation_path = self.collect_components(_from, _to)
        if closed:
            compo_list.append(self.Ec)
        if _from == _to and closed:
            return copy.deepcopy(self.Hc)
        if not compo_list:
            return Component("empty", self.sps, 1.0, 1.0)
        if len(compo_list) == 1:
            out = compo_list[0]
        else:
            comps = compo_list
            unit = comps[0].unit
            composite = Component("path", self.sps, np.array([1.0]), np.array([1.0]), unit=unit)
            def _tf(f, c=comps):
                val = np.ones(np.atleast_1d(f).shape, dtype=complex)
                for x in c:
                    val *= x.TF(f=f)
                return val
            composite.TF = _tf
            out = composite
        if view:
            print(f"propagation path: {propagation_path}")
        return out

    def __deepcopy__(self, memo: dict) -> "MultiRateLaserLock":
        laser = self.components_dict.get("Laser")
        plant = None
        if laser is not None:
            plant = Component(
                laser.name, self.sps,
                nume=np.array(laser.nume, copy=True),
                deno=np.array(laser.deno, copy=True),
                unit=laser.unit,
            )
        new_obj: MultiRateLaserLock = MultiRateLaserLock.__new__(MultiRateLaserLock)
        new_obj.__init__(
            self._sps_loop,
            self._amp,
            self._kp,
            self._ki,
            sps_adc=self._sps_adc,
            sos=self._sos,
            Plant=plant,
            dac_gain=self._dac_gain,
            pre_gain=self._pre_gain,
            n_reg=self._n_reg,
            off=self._off,
        )
        new_obj.callbacks = self.callbacks
        return new_obj
