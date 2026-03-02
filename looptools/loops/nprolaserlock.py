# BSD 3-Clause License
#
# Copyright (c) 2025, Miguel Dovale
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting this software to
# foreign countries or providing access to foreign persons.
#
from __future__ import annotations

import copy
import logging
from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from looptools.component import Component
from looptools.components import (
    ActuatorComponent,
    DoubleIntegratorComponent,
    DSPDelayComponent,
    ImplicitAccumulatorComponent,
    MultiplierComponent,
    OpAmp_dict,
    PIControllerComponent,
    RightBitShiftComponent,
)
from looptools.dimension import Dimension
from looptools.dsp import integral_rms
from looptools.loop import LOOP
from looptools.loopmath import add_transfer_function

logger = logging.getLogger(__name__)

# Type aliases
ExtrapolateFpll = Tuple[bool, float, float]
ExtrapolateII = Tuple[bool, float]
ExtrapolateDict = Dict[str, Union[ExtrapolateFpll, ExtrapolateII]]
Mode = Literal["frequency", "phase"]

_DEFAULT_EXTRAPOLATE: ExtrapolateDict = {
    "Fpll": (False, 1e2, 1.0),
    "p_II1": (False, 1e2),
    "t_II1": (False, 1e2),
}


def _validate_npro_params(
    sps: float,
    pll: Any,
    C1: int,
    C2: int,
    Ki1: float,
    Kii1: float,
    Kp2: float,
    Ki2: float,
    Kdac: float,
    Kc_pzt: float,
    Fc_pzt: float,
    Ka_pzt: float,
    Fa_pzt: float,
    Kc_temp: float,
    Fc_temp: float,
    Ka_temp: float,
    Fa_temp: float,
    Nreg1: int,
    mode: str,
    extrapolate: ExtrapolateDict,
) -> None:
    """Validate NPROLaserLock constructor parameters."""
    if not isinstance(sps, (int, float)) or sps <= 0:
        raise ValueError(f"sps must be positive, got {sps!r}")
    if not hasattr(pll, "point_to_point_component"):
        raise TypeError(
            f"pll must have point_to_point_component method, got {type(pll).__name__}"
        )
    if not isinstance(C1, int) or C1 < 0:
        raise ValueError(f"C1 must be a non-negative int, got {C1!r}")
    if not isinstance(C2, int) or C2 < 0:
        raise ValueError(f"C2 must be a non-negative int, got {C2!r}")
    for name, val in [
        ("Ki1", Ki1),
        ("Kii1", Kii1),
        ("Kdac", Kdac),
        ("Kc_pzt", Kc_pzt),
        ("Fc_pzt", Fc_pzt),
        ("Ka_pzt", Ka_pzt),
        ("Fa_pzt", Fa_pzt),
        ("Kc_temp", Kc_temp),
        ("Fc_temp", Fc_temp),
        ("Ka_temp", Ka_temp),
        ("Fa_temp", Fa_temp),
    ]:
        if not isinstance(val, (int, float)) or val <= 0:
            raise ValueError(f"{name} must be positive, got {val!r}")
    if not isinstance(Nreg1, int) or Nreg1 < 0:
        raise ValueError(f"Nreg1 must be a non-negative int, got {Nreg1!r}")
    if not isinstance(Kp2, (int, float)) or Kp2 < 0:
        raise ValueError(f"Kp2 must be non-negative, got {Kp2!r}")
    if not isinstance(Ki2, (int, float)) or Ki2 <= 0:
        raise ValueError(f"Ki2 must be positive, got {Ki2!r}")
    if mode not in ("phase", "frequency"):
        raise ValueError(f"mode must be 'phase' or 'frequency', got {mode!r}")
    for key in ("Fpll", "p_II1", "t_II1"):
        if key not in extrapolate:
            raise ValueError(f"extrapolate must contain key {key!r}")
        entry = extrapolate[key]
        if key == "Fpll":
            if len(entry) != 3:
                raise ValueError(f"extrapolate['Fpll'] must be [bool, f_trans, power], got {entry}")
        else:
            if len(entry) != 2:
                raise ValueError(f"extrapolate[{key!r}] must be [bool, f_trans], got {entry}")


def _validate_pzt_params(
    sps: float,
    C1: int,
    Ki1: float,
    Kii1: float,
    Kdac: float,
    Kc_pzt: float,
    Fc_pzt: float,
    Ka_pzt: float,
    Fa_pzt: float,
    Nreg1: int,
) -> None:
    """Validate LaserLockPZT constructor parameters."""
    if not isinstance(sps, (int, float)) or sps <= 0:
        raise ValueError(f"sps must be positive, got {sps!r}")
    if not isinstance(C1, int) or C1 < 0:
        raise ValueError(f"C1 must be a non-negative int, got {C1!r}")
    for name, val in [
        ("Ki1", Ki1),
        ("Kii1", Kii1),
        ("Kdac", Kdac),
        ("Kc_pzt", Kc_pzt),
        ("Fc_pzt", Fc_pzt),
        ("Ka_pzt", Ka_pzt),
        ("Fa_pzt", Fa_pzt),
    ]:
        if not isinstance(val, (int, float)) or val <= 0:
            raise ValueError(f"{name} must be positive, got {val!r}")
    if not isinstance(Nreg1, int) or Nreg1 < 0:
        raise ValueError(f"Nreg1 must be a non-negative int, got {Nreg1!r}")


def _validate_temp_params(
    sps: float,
    C1: int,
    C2: int,
    Ki1: float,
    Kii1: float,
    Kp2: float,
    Ki2: float,
    Kdac: float,
    Kc_temp: float,
    Fc_temp: float,
    Ka_temp: float,
    Fa_temp: float,
    Nreg1: int,
) -> None:
    """Validate LaserLockTemp constructor parameters."""
    _validate_pzt_params(sps, C1, Ki1, Kii1, Kdac, Kc_temp, Fc_temp, Ka_temp, Fa_temp, Nreg1)
    if not isinstance(C2, int) or C2 < 0:
        raise ValueError(f"C2 must be a non-negative int, got {C2!r}")
    if not isinstance(Kp2, (int, float)) or Kp2 < 0:
        raise ValueError(f"Kp2 must be non-negative, got {Kp2!r}")
    if not isinstance(Ki2, (int, float)) or Ki2 <= 0:
        raise ValueError(f"Ki2 must be positive, got {Ki2!r}")


def _to_extrapolate_dict(
    value: Optional[ExtrapolateDict],
) -> ExtrapolateDict:
    """Convert extrapolate to immutable dict, applying defaults."""
    if value is None:
        return dict(_DEFAULT_EXTRAPOLATE)
    result: ExtrapolateDict = {}
    for key in ("Fpll", "p_II1", "t_II1"):
        entry = value.get(key, _DEFAULT_EXTRAPOLATE[key])
        if key == "Fpll":
            result[key] = (
                bool(entry[0]),
                float(entry[1]),
                float(entry[2]),
            )
        else:
            result[key] = (bool(entry[0]), float(entry[1]))
    return result


def _to_off_tuple(off: Optional[Sequence[Optional[str]]]) -> Tuple[Optional[str], ...]:
    """Convert off to immutable tuple. [None] means no exclusions."""
    if off is None:
        return (None,)
    return tuple(off)


class NPROLaserLock:
    """
    Composite model of a dual-loop NPRO laser frequency (or phase) lock system.

    Combines a fast PZT control loop and a slower temperature loop to simulate
    the closed-loop frequency or phase locking of a Non-Planar Ring Oscillator
    (NPRO) laser. Wraps two LOOP-based subsystems (LaserLockPZT and LaserLockTemp)
    and manages their shared structure and synchronization.
    """

    def __init__(
        self,
        sps: float,
        pll: Component,
        C1: int,
        C2: int,
        Ki1: float,
        Kii1: float,
        Kp2: float,
        Ki2: float,
        Kdac: float,
        Kc_pzt: float,
        Fc_pzt: float,
        Ka_pzt: float,
        Fa_pzt: float,
        Kc_temp: float,
        Fc_temp: float,
        Ka_temp: float,
        Fa_temp: float,
        Nreg1: int,
        OPpzt: Optional[str] = None,
        OPtemp: Optional[str] = None,
        mode: Mode = "frequency",
        extrapolate: Optional[ExtrapolateDict] = None,
        off: Optional[Sequence[Optional[str]]] = None,
    ) -> None:
        """
        Parameters
        ----------
        sps : float
            System clock frequency in Hz.
        pll : Component
            Phase-locked loop model providing the input sensing transfer function.
        C1 : int
            Bit shift gain stage 1 (digital).
        C2 : int
            Bit shift gain stage 2 (digital, used only in temperature loop).
        Ki1 : float
            Integral gain of the first digital controller (common to both loops).
        Kii1 : float
            Double-integrator gain of the first digital controller (common).
        Kp2 : float
            Proportional gain of the second digital controller (temperature loop).
        Ki2 : float
            Integral gain of the second digital controller (temperature loop).
        Kdac : float
            DAC gain [V/count].
        Kc_pzt : float
            Analog servo gain in the PZT path.
        Fc_pzt : float
            Corner frequency of the analog PZT servo.
        Ka_pzt : float
            Actuator gain of the PZT tuning mechanism [Hz/V].
        Fa_pzt : float
            Bandwidth of the PZT actuator [Hz].
        Kc_temp : float
            Analog servo gain in the temperature path.
        Fc_temp : float
            Corner frequency of the analog temperature servo.
        Ka_temp : float
            Actuator gain of the temperature tuning mechanism [Hz/V].
        Fa_temp : float
            Bandwidth of the temperature actuator [Hz].
        Nreg1 : int
            Number of registers used to model shared DSP delay in the loop.
        OPpzt : str, optional
            Label of operational amplifier used in the PZT analog path.
        OPtemp : str, optional
            Label of operational amplifier used in the temperature analog path.
        mode : {'frequency', 'phase'}, default='frequency'
            Determines whether the loop stabilizes frequency or phase.
        extrapolate : dict, optional
            Dictionary configuring TF extrapolation. Keys: Fpll, p_II1, t_II1.
        off : sequence of str or None, optional
            Component names to exclude from loop construction. Default: none.
        """
        _extrapolate = _to_extrapolate_dict(extrapolate)
        _off = _to_off_tuple(off)

        _validate_npro_params(
            sps, pll, C1, C2, Ki1, Kii1, Kp2, Ki2, Kdac,
            Kc_pzt, Fc_pzt, Ka_pzt, Fa_pzt,
            Kc_temp, Fc_temp, Ka_temp, Fa_temp,
            Nreg1, mode, _extrapolate,
        )

        self.sps: float = float(sps)

        from_pll = "PD"
        to_pll = "PD" if mode == "phase" else "PA"
        pll_inst = pll.point_to_point_component(
            _from=from_pll, _to=to_pll, suppression=True
        )
        if mode == "frequency":
            fpll = _extrapolate["Fpll"]
            pll_inst.TF = partial(
                pll_inst.TF,
                extrapolate=fpll[0],
                f_trans=fpll[1],
                power=fpll[2],
            )
        pll_inst.name = "Fpll"

        self.pzt = LaserLockPZT(
            self.sps, pll_inst, C1, Ki1, Kii1, Kdac,
            Kc_pzt, Fc_pzt, Ka_pzt, Fa_pzt, Nreg1,
            OPpzt, _off, _extrapolate["p_II1"],
        )
        self.temp = LaserLockTemp(
            self.sps, pll_inst, C1, C2, Ki1, Kii1, Kp2, Ki2, Kdac,
            Kc_temp, Fc_temp, Ka_temp, Fa_temp, Nreg1,
            OPtemp, _off, _extrapolate["t_II1"],
        )

        for name, comp in self.pzt.components_dict.items():
            if name in self.temp.components_dict:
                self.pzt.register_callback(
                    self.temp.replace_component,
                    comp.name, comp, loop_update=True,
                )

        self.pzt.register_callback(self._update_Gf)

        for name, comp in self.temp.components_dict.items():
            if name in self.pzt.components_dict:
                self.temp.register_callback(
                    self.pzt.replace_component,
                    comp.name, comp, loop_update=True,
                )

        self.temp.register_callback(self._update_Gf)
        self._update_Gf()

    def _update_Gf(self) -> None:
        """Recompute combined forward transfer function from PZT and temp loops."""
        self.Gf: Callable[..., NDArray[np.complexfloating]] = partial(
            add_transfer_function,
            tf1=self.pzt.Gf,
            tf2=self.temp.Gf,
        )

    def __deepcopy__(self, memo: dict) -> NPROLaserLock:
        new_obj = NPROLaserLock.__new__(NPROLaserLock)
        new_obj.temp = copy.deepcopy(self.temp, memo)
        new_obj.pzt = copy.deepcopy(self.pzt, memo)
        new_obj.sps = self.sps
        new_obj._update_Gf()
        return new_obj

    def noise_propagation_asd(
        self,
        f: ArrayLike,
        asd: ArrayLike,
        unit: Optional[Dimension] = None,
        _from: str = "PD",
        _to: Optional[str] = None,
        view: bool = False,
        isTF: bool = True,
    ) -> Tuple[NDArray[np.floating], Dimension, Dict[str, Any], float]:
        """
        Propagate noise through the closed-loop sensitivity function.

        Parameters
        ----------
        f : array_like
            Fourier frequencies (Hz).
        asd : array_like
            Input amplitude spectral density.
        unit : Dimension, optional
            Unit of the input ASD. Default: dimensionless.
        _from : str, optional
            Unused; kept for API compatibility.
        _to : str, optional
            Unused; kept for API compatibility.
        view : bool, optional
            Unused; kept for API compatibility.
        isTF : bool, optional
            Unused; kept for API compatibility.

        Returns
        -------
        asd_prop : ndarray
            Propagated ASD.
        unit_prop : Dimension
            Unit of propagated ASD.
        bode : dict
            Bode data with keys 'f', 'mag', 'phase'.
        rms : float
            RMS value of propagated ASD over [0, inf) Hz.
        """
        if unit is None:
            unit = Dimension(dimensionless=True)
        TF = self.Gf(f=f)
        TF = 1 / (1 + TF)
        mag = np.abs(TF)
        phase = np.angle(TF, deg=False)
        bode: Dict[str, Any] = {"f": f, "mag": mag, "phase": phase}
        asd_prop = bode["mag"] * np.asarray(asd)
        rms = integral_rms(f, asd_prop, (0.0, np.inf))
        return asd_prop, unit, bode, rms


class LaserLockPZT(LOOP):
    """
    Fast PZT-based feedback loop for laser frequency control.

    Builds a control chain modeling the fast response of a piezoelectric
    actuator used to finely tune the laser frequency.
    """

    def __init__(
        self,
        sps: float,
        pll: Component,
        C1: int,
        Ki1: float,
        Kii1: float,
        Kdac: float,
        Kc_pzt: float,
        Fc_pzt: float,
        Ka_pzt: float,
        Fa_pzt: float,
        Nreg1: int,
        OPpzt: Optional[str],
        off: Tuple[Optional[str], ...],
        extrapolate: ExtrapolateII,
    ) -> None:
        super().__init__(sps)
        _validate_pzt_params(
            sps, C1, Ki1, Kii1, Kdac, Kc_pzt, Fc_pzt, Ka_pzt, Fa_pzt, Nreg1,
        )

        self.pll = pll
        self.C1 = C1
        self.Ki1 = Ki1
        self.Kii1 = Kii1
        self.Kdac = Kdac
        self.Kc_pzt = Kc_pzt
        self.Fc_pzt = Fc_pzt
        self.Ka_pzt = Ka_pzt
        self.Fa_pzt = Fa_pzt
        self.Nreg1 = Nreg1
        self.OPpzt = OPpzt
        self.off = off
        self.extrapolate = extrapolate

        self.add_component(pll)

        if "Fgain1" not in off:
            self.add_component(RightBitShiftComponent("Fgain1", self.sps, C1))

        if "Fctrl1" not in off:
            self.add_component(
                DoubleIntegratorComponent(
                    "Fctrl1", self.sps, Ki1, Kii1, list(extrapolate)
                )
            )

        if "Kdac" not in off:
            self.add_component(
                MultiplierComponent(
                    "Kdac", self.sps, Kdac, Dimension(dimensionless=True)
                )
            )

        if "pztFcond" not in off:
            self.add_component(
                ActuatorComponent(
                    "pztFcond", self.sps, Kc_pzt, Fc_pzt,
                    Dimension(dimensionless=True),
                )
            )
            if OPpzt is not None:
                gbp = OpAmp_dict[OPpzt]["GBP"] / Kc_pzt
                self.add_component(
                    ActuatorComponent(
                        "pztFop", self.sps, 1.0, gbp,
                        Dimension(dimensionless=True),
                    )
                )

        if "pztFplant" not in off:
            self.add_component(
                ActuatorComponent(
                    "Fplant", self.sps, Ka_pzt, Fa_pzt,
                    Dimension(["Hz"], ["V"]),
                )
            )

        if "Fnu2phi" not in off:
            self.add_component(ImplicitAccumulatorComponent("Fnu2phi", self.sps))

        if "DSP" not in off:
            self.add_component(DSPDelayComponent("DSP", self.sps, Nreg1))

        if off != (None,):
            logger.warning(
                "The following components are not included in the loop: %s", off
            )

        self.update()
        self.register_component_properties()

    def __deepcopy__(self, memo: dict) -> LaserLockPZT:
        new_obj = LaserLockPZT.__new__(LaserLockPZT)
        new_obj.__init__(
            self.sps, self.pll, self.C1, self.Ki1, self.Kii1, self.Kdac,
            self.Kc_pzt, self.Fc_pzt, self.Ka_pzt, self.Fa_pzt, self.Nreg1,
            self.OPpzt, self.off, self.extrapolate,
        )
        new_obj.callbacks = self.callbacks
        return new_obj


class LaserLockTemp(LOOP):
    """
    Slow, temperature-based feedback loop for laser frequency control.

    Models a thermal control path that compensates low-frequency drifts
    in the laser frequency.
    """

    def __init__(
        self,
        sps: float,
        pll: Component,
        C1: int,
        C2: int,
        Ki1: float,
        Kii1: float,
        Kp2: float,
        Ki2: float,
        Kdac: float,
        Kc_temp: float,
        Fc_temp: float,
        Ka_temp: float,
        Fa_temp: float,
        Nreg1: int,
        OPtemp: Optional[str],
        off: Tuple[Optional[str], ...],
        extrapolate: ExtrapolateII,
    ) -> None:
        super().__init__(sps)
        _validate_temp_params(
            sps, C1, C2, Ki1, Kii1, Kp2, Ki2, Kdac,
            Kc_temp, Fc_temp, Ka_temp, Fa_temp, Nreg1,
        )

        self.pll = pll
        self.C1 = C1
        self.C2 = C2
        self.Ki1 = Ki1
        self.Kii1 = Kii1
        self.Kp2 = Kp2
        self.Ki2 = Ki2
        self.Kdac = Kdac
        self.Kc_temp = Kc_temp
        self.Fc_temp = Fc_temp
        self.Ka_temp = Ka_temp
        self.Fa_temp = Fa_temp
        self.Nreg1 = Nreg1
        self.OPtemp = OPtemp
        self.off = off
        self.extrapolate = extrapolate

        self.add_component(pll)

        if "Fgain1" not in off:
            self.add_component(RightBitShiftComponent("Fgain1", self.sps, C1))

        self.add_component(
            DoubleIntegratorComponent(
                "Fctrl1", self.sps, Ki1, Kii1, list(extrapolate)
            )
        )

        if "Fgain2" not in off:
            self.add_component(RightBitShiftComponent("Fgain2", self.sps, C2))

        if "Fctrl2" not in off:
            self.add_component(PIControllerComponent("Fctrl2", self.sps, Kp2, Ki2))

        if "Kdac" not in off:
            self.add_component(
                MultiplierComponent(
                    "Kdac", self.sps, Kdac, Dimension(dimensionless=True)
                )
            )

        if "tempFcond" not in off:
            self.add_component(
                ActuatorComponent(
                    "tempFcond", self.sps, Kc_temp, Fc_temp,
                    Dimension(dimensionless=True),
                )
            )
            if OPtemp is not None:
                gbp = OpAmp_dict[OPtemp]["GBP"] / Kc_temp
                self.add_component(
                    ActuatorComponent(
                        "tempFop", self.sps, 1.0, gbp,
                        Dimension(dimensionless=True),
                    )
                )

        if "tempFplant" not in off:
            self.add_component(
                ActuatorComponent(
                    "tempFplant", self.sps, Ka_temp, Fa_temp,
                    Dimension(["Hz"], ["V"]),
                )
            )

        if "Fnu2phi" not in off:
            self.add_component(ImplicitAccumulatorComponent("Fnu2phi", self.sps))

        if "DSP" not in off:
            self.add_component(DSPDelayComponent("DSP", self.sps, Nreg1))

        if off != (None,):
            logger.warning(
                "The following components are not included in the loop: %s", off
            )

        self.update()
        self.register_component_properties()

    def __deepcopy__(self, memo: dict) -> LaserLockTemp:
        new_obj = LaserLockTemp.__new__(LaserLockTemp)
        new_obj.__init__(
            self.sps, self.pll, self.C1, self.C2, self.Ki1, self.Kii1,
            self.Kp2, self.Ki2, self.Kdac, self.Kc_temp, self.Fc_temp,
            self.Ka_temp, self.Fa_temp, self.Nreg1, self.OPtemp, self.off,
            self.extrapolate,
        )
        new_obj.callbacks = self.callbacks
        return new_obj
