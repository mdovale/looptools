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

from typing import Optional, Sequence, Tuple

import numpy as np

import loopkit.components as lc
import loopkit.loopmath as lm
from loopkit.component import Component
from loopkit.loop import LOOP


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
