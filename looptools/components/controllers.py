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
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting this software to
# foreign countries or providing access to foreign persons.
#
from __future__ import annotations

from functools import partial
from typing import Any, Optional, Tuple

import numpy as np

import looptools.loopmath as lm
from looptools.component import Component
from looptools.dimension import Dimension

from looptools.components._validation import (
    _validate_extrapolate,
    _validate_numeric,
    _validate_optional_positive,
    _validate_positive,
)


class PIControllerComponent(Component):
    """
    Proportional-Integral controller component, P+I.

    Combines proportional and integral actions into a PI controller with
    bit-shift-based tunable gain.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    Kp : float
        Proportional gain as log2(Kp).
    Ki : float
        Integral gain as log2(Ki).

    Attributes
    ----------
    Kp : float
        Proportional gain.
    Ki : float
        Integral gain.
    """

    def __init__(
        self,
        name: str,
        sps: float,
        Kp: float,
        Ki: float,
    ) -> None:
        sps_f = _validate_positive("sps", sps)
        self._Kp = 2 ** _validate_numeric("Kp", Kp)
        self._Ki = 2 ** _validate_numeric("Ki", Ki)
        p_comp = Component(
            "P",
            sps_f,
            np.array([self._Kp]),
            np.array([1.0]),
            unit=Dimension(["cycle"], ["s", "rad"]),
        )
        i_comp = Component(
            "I",
            sps_f,
            np.array([self._Ki]),
            np.array([1.0, -1.0]),
            unit=Dimension(["cycle"], ["s", "rad"]),
        )
        pi_comp = p_comp + i_comp
        super().__init__(name, sps_f, pi_comp.nume, pi_comp.deno, unit=pi_comp.unit)
        self.properties = {
            "Kp": (lambda: self.Kp, lambda value: setattr(self, "Kp", value)),
            "Ki": (lambda: self.Ki, lambda value: setattr(self, "Ki", value)),
        }

    def __deepcopy__(self, memo: dict[int, Any]) -> PIControllerComponent:
        new_obj = PIControllerComponent.__new__(PIControllerComponent)
        new_obj.__init__(self.name, self.sps, np.log2(self._Kp), np.log2(self._Ki))
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Kp(self) -> float:
        return self._Kp

    @Kp.setter
    def Kp(self, value: float) -> None:
        self._Kp = 2 ** _validate_numeric("Kp", value)
        self.update_component()

    @property
    def Ki(self) -> float:
        return self._Ki

    @Ki.setter
    def Ki(self, value: float) -> None:
        self._Ki = 2 ** _validate_numeric("Ki", value)
        self.update_component()

    def update_component(self) -> None:
        p_comp = Component(
            "P",
            self.sps,
            np.array([self._Kp]),
            np.array([1.0]),
            unit=Dimension(["cycle"], ["s", "rad"]),
        )
        i_comp = Component(
            "I",
            self.sps,
            np.array([self._Ki]),
            np.array([1.0, -1.0]),
            unit=Dimension(["cycle"], ["s", "rad"]),
        )
        pi_comp = p_comp + i_comp
        super().__init__(self.name, self.sps, pi_comp.nume, pi_comp.deno, unit=pi_comp.unit)


class DoubleIntegratorComponent(Component):
    """
    Second-order integrator, I+II.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    Ki : float
        Gain of first integrator (log2 scale).
    Kii : float
        Gain of second integrator (log2 scale).
    extrapolate : tuple(bool, float)
        (Enable extrapolation, transition frequency)

    Attributes
    ----------
    Ki : float
        Gain of the first integrator.
    Kii : float
        Gain of the second integrator.
    """

    def __init__(
        self,
        name: str,
        sps: float,
        Ki: float,
        Kii: float,
        extrapolate: Tuple[bool, float],
    ) -> None:
        sps_f = _validate_positive("sps", sps)
        self._extrapolate: Tuple[bool, float] = _validate_extrapolate(extrapolate)
        self._Ki = 2 ** _validate_numeric("Ki", Ki)
        self._Kii = 2 ** _validate_numeric("Kii", Kii)
        i_comp = Component(
            "I",
            sps_f,
            np.array([self._Ki]),
            np.array([1.0, -1.0]),
            unit=Dimension(dimensionless=True),
        )
        ii_comp = Component(
            "II",
            sps_f,
            np.array([self._Kii]),
            np.array([1.0, -2.0, 1.0]),
            unit=Dimension(dimensionless=True),
        )
        ii_comp.TF = partial(
            ii_comp.TF,
            extrapolate=self._extrapolate[0],
            f_trans=self._extrapolate[1],
            power=-2,
        )
        double_i = i_comp + ii_comp
        super().__init__(
            name, sps_f, double_i.nume, double_i.deno, unit=double_i.unit
        )
        self.TE = double_i.TE
        self.TE.name = name
        self.TF = partial(
            lm.add_transfer_function, tf1=i_comp.TF, tf2=ii_comp.TF
        )
        self.properties = {
            "Ki": (lambda: self.Ki, lambda value: setattr(self, "Ki", value)),
            "Kii": (lambda: self.Kii, lambda value: setattr(self, "Kii", value)),
        }

    @property
    def extrapolate(self) -> Tuple[bool, float]:
        """Immutable (enable, f_trans) tuple for double-integrator extrapolation."""
        return self._extrapolate

    def __deepcopy__(self, memo: dict[int, Any]) -> DoubleIntegratorComponent:
        new_obj = DoubleIntegratorComponent.__new__(DoubleIntegratorComponent)
        new_obj.__init__(
            self.name,
            self.sps,
            np.log2(self._Ki),
            np.log2(self._Kii),
            self._extrapolate,
        )
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Ki(self) -> float:
        return self._Ki

    @Ki.setter
    def Ki(self, value: float) -> None:
        self._Ki = 2 ** _validate_numeric("Ki", value)
        self.update_component()

    @property
    def Kii(self) -> float:
        return self._Kii

    @Kii.setter
    def Kii(self, value: float) -> None:
        self._Kii = 2 ** _validate_numeric("Kii", value)
        self.update_component()

    def update_component(self) -> None:
        i_comp = Component(
            "I",
            self.sps,
            np.array([self._Ki]),
            np.array([1.0, -1.0]),
            unit=Dimension(dimensionless=True),
        )
        ii_comp = Component(
            "II",
            self.sps,
            np.array([self._Kii]),
            np.array([1.0, -2.0, 1.0]),
            unit=Dimension(dimensionless=True),
        )
        ii_comp.TF = partial(
            ii_comp.TF,
            extrapolate=self._extrapolate[0],
            f_trans=self._extrapolate[1],
            power=-2,
        )
        double_i = i_comp + ii_comp
        super().__init__(
            self.name, self.sps, double_i.nume, double_i.deno, unit=double_i.unit
        )
        self.TE = double_i.TE
        self.TE.name = self.name
        self.TF = partial(
            lm.add_transfer_function, tf1=i_comp.TF, tf2=ii_comp.TF
        )


class PIIControllerComponent(Component):
    """
    Proportional + Integrator + Double Integrator controller component, P+I+II.

    This component models a control law consisting of:
        - A proportional term (P)
        - A first-order integrator (I)
        - A second-order integrator (II)

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    Kp : float
        Proportional gain (log₂ scale).
    Ki : float
        First integrator gain (log₂ scale).
    Kii : float
        Second integrator gain (log₂ scale).
    extrapolate : tuple(bool, float)
        Tuple (enable_extrapolation, transition_frequency) for the double integrator.

    Attributes
    ----------
    Kp : float
        Proportional gain.
    Ki : float
        First integrator gain.
    Kii : float
        Second integrator gain.
    """

    def __init__(
        self,
        name: str,
        sps: float,
        Kp: float,
        Ki: float,
        Kii: float,
        extrapolate: Tuple[bool, float] = (False, 1e2),
    ) -> None:
        sps_f = _validate_positive("sps", sps)
        self._extrapolate: Tuple[bool, float] = _validate_extrapolate(extrapolate)
        self._Kp = 2 ** _validate_numeric("Kp", Kp)
        self._Ki = 2 ** _validate_numeric("Ki", Ki)
        self._Kii = 2 ** _validate_numeric("Kii", Kii)

        p_comp = Component(
            "P",
            sps_f,
            np.array([self._Kp]),
            np.array([1.0]),
            unit=Dimension(["cycle"], ["s", "rad"]),
        )
        i_comp = Component(
            "I",
            sps_f,
            np.array([self._Ki]),
            np.array([1.0, -1.0]),
            unit=Dimension(["cycle"], ["s", "rad"]),
        )
        ii_comp = Component(
            "II",
            sps_f,
            np.array([self._Kii]),
            np.array([1.0, -2.0, 1.0]),
            unit=Dimension(["cycle"], ["s", "rad"]),
        )
        ii_comp.TF = partial(
            ii_comp.TF,
            extrapolate=self._extrapolate[0],
            f_trans=self._extrapolate[1],
            power=-2,
        )

        pii_comp = p_comp + i_comp + ii_comp
        super().__init__(
            name, sps_f, pii_comp.nume, pii_comp.deno, unit=pii_comp.unit
        )

        self.TE = pii_comp.TE
        self.TE.name = name
        self.TF = partial(
            lm.add_transfer_function,
            tf1=p_comp.TF,
            tf2=partial(
                lm.add_transfer_function,
                tf1=i_comp.TF,
                tf2=ii_comp.TF,
            ),
        )

        self.properties = {
            "Kp": (lambda: self.Kp, lambda value: setattr(self, "Kp", value)),
            "Ki": (lambda: self.Ki, lambda value: setattr(self, "Ki", value)),
            "Kii": (lambda: self.Kii, lambda value: setattr(self, "Kii", value)),
        }

    @property
    def extrapolate(self) -> Tuple[bool, float]:
        """Immutable (enable, f_trans) tuple for double-integrator extrapolation."""
        return self._extrapolate

    def __deepcopy__(self, memo: dict[int, Any]) -> PIIControllerComponent:
        new_obj = PIIControllerComponent.__new__(PIIControllerComponent)
        new_obj.__init__(
            self.name,
            self.sps,
            np.log2(self._Kp),
            np.log2(self._Ki),
            np.log2(self._Kii),
            self._extrapolate,
        )
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Kp(self) -> float:
        return self._Kp

    @Kp.setter
    def Kp(self, value: float) -> None:
        self._Kp = 2 ** _validate_numeric("Kp", value)
        self.update_component()

    @property
    def Ki(self) -> float:
        return self._Ki

    @Ki.setter
    def Ki(self, value: float) -> None:
        self._Ki = 2 ** _validate_numeric("Ki", value)
        self.update_component()

    @property
    def Kii(self) -> float:
        return self._Kii

    @Kii.setter
    def Kii(self, value: float) -> None:
        self._Kii = 2 ** _validate_numeric("Kii", value)
        self.update_component()

    def update_component(self) -> None:
        p_comp = Component(
            "P",
            self.sps,
            np.array([self._Kp]),
            np.array([1.0]),
            unit=Dimension(["cycle"], ["s", "rad"]),
        )
        i_comp = Component(
            "I",
            self.sps,
            np.array([self._Ki]),
            np.array([1.0, -1.0]),
            unit=Dimension(["cycle"], ["s", "rad"]),
        )
        ii_comp = Component(
            "II",
            self.sps,
            np.array([self._Kii]),
            np.array([1.0, -2.0, 1.0]),
            unit=Dimension(["cycle"], ["s", "rad"]),
        )
        ii_comp.TF = partial(
            ii_comp.TF,
            extrapolate=self._extrapolate[0],
            f_trans=self._extrapolate[1],
            power=-2,
        )
        pii_comp = p_comp + i_comp + ii_comp
        super().__init__(
            self.name,
            self.sps,
            pii_comp.nume,
            pii_comp.deno,
            unit=pii_comp.unit,
        )
        self.TE = pii_comp.TE
        self.TE.name = self.name
        self.TF = partial(
            lm.add_transfer_function,
            tf1=p_comp.TF,
            tf2=partial(
                lm.add_transfer_function,
                tf1=i_comp.TF,
                tf2=ii_comp.TF,
            ),
        )


class MokuPIDSymbolicController(Component):
    """
    Moku-style symbolic PID controller using P, I, II, and D terms.

    WARNING: II-term causes numerical instabilities at low frequencies, use MokuPIDController instead.

    This component constructs the transfer function symbolically using known corner frequencies
    and proportional gain in dB, as implemented by Liquid Instruments Moku devices.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate [Hz].
    Kp_dB : float
        Proportional gain in dB.
    Fc_i : float or None
        First integrator (I) crossover frequency [Hz].
    Fc_ii : float or None
        Second integrator (II) crossover frequency [Hz].
    Fc_d : float or None
        Derivative (D) crossover frequency [Hz].
    f_trans : float or None
        Transition frequency for regularization [Hz]. Used to improve numerical behavior of double integrator.

    Properties
    ----------
    Kp_dB, Ki_dB, Kii_dB, Kd_dB : float
        Gains in decibels.
    Fc_i, Fc_ii, Fc_d : float
        Crossover frequencies in Hz.
    """

    def __init__(
        self,
        name: str,
        sps: float,
        Kp_dB: float,
        Fc_i: Optional[float] = None,
        Fc_ii: Optional[float] = None,
        Fc_d: Optional[float] = None,
        f_trans: Optional[float] = None,
    ) -> None:
        self.name = name
        self.sps = _validate_positive("sps", sps)
        self.f_trans = _validate_optional_positive("f_trans", f_trans)

        self._Kp_dB = _validate_numeric("Kp_dB", Kp_dB)
        self._Fc_i = _validate_optional_positive("Fc_i", Fc_i)
        self._Fc_ii = _validate_optional_positive("Fc_ii", Fc_ii)
        self._Fc_d = _validate_optional_positive("Fc_d", Fc_d)

        self.update_component()

        self.properties = {
            "Kp_dB": (lambda: self.Kp_dB, lambda value: setattr(self, "Kp_dB", value)),
            "Ki_dB": (lambda: self.Ki_dB, lambda value: setattr(self, "Ki_dB", value)),
            "Kii_dB": (lambda: self.Kii_dB, lambda value: setattr(self, "Kii_dB", value)),
            "Kd_dB": (lambda: self.Kd_dB, lambda value: setattr(self, "Kd_dB", value)),
            "Fc_i": (lambda: self.Fc_i, lambda value: setattr(self, "Fc_i", value)),
            "Fc_ii": (lambda: self.Fc_ii, lambda value: setattr(self, "Fc_ii", value)),
            "Fc_d": (lambda: self.Fc_d, lambda value: setattr(self, "Fc_d", value)),
        }

    def update_component(self) -> None:
        tf_str = self.moku_pid_tf_string(
            self.sps,
            Kp_dB=self._Kp_dB,
            Ki_dB=None if self._Fc_i is not None else lm.log2_gain_to_db(np.log2(self._Ki)) if hasattr(self, "_Ki") else None,
            Kii_dB=None if self._Fc_ii is not None else lm.log2_gain_to_db(np.log2(self._Kii)) if hasattr(self, "_Kii") else None,
            Kd_dB=None if self._Fc_d is not None else lm.log2_gain_to_db(np.log2(self._Kd)) if hasattr(self, "_Kd") else None,
            Fc_i=self._Fc_i,
            Fc_ii=self._Fc_ii,
            Fc_d=self._Fc_d,
            f_trans=self.f_trans,
        )

        super().__init__(self.name, tf=tf_str, sps=self.sps, unit=Dimension(["cycle"], ["s", "rad"]))

        Kp_log2 = lm.db_to_log2_gain(self._Kp_dB)
        self._Kp = 2 ** Kp_log2
        self._Ki = 0.0 if self._Fc_i is None else 2 ** lm.gain_for_crossover_frequency(0.0, self.sps, self._Fc_i, kind="I")
        self._Kii = 0.0 if self._Fc_ii is None else 2 ** lm.gain_for_crossover_frequency(0.0, self.sps, self._Fc_ii, kind="II")

        if self._Fc_d is not None:
            omega_d = 2 * np.pi * self._Fc_d / self.sps
            mag = abs((1 - np.exp(-1j * omega_d)) / (1 + np.exp(-1j * omega_d)))
            self._Kd = self._Kp / mag
        elif hasattr(self, "_Kd"):
            pass
        else:
            self._Kd = 0.0

    @staticmethod
    def moku_pid_tf_string(
        sps: float,
        Kp_dB: float = 0.0,
        Ki_dB: Optional[float] = None,
        Kii_dB: Optional[float] = None,
        Kd_dB: Optional[float] = None,
        Fc_i: Optional[float] = None,
        Fc_ii: Optional[float] = None,
        Fc_d: Optional[float] = None,
        f_trans: Optional[float] = None,
    ) -> str:
        Kp_log2 = lm.db_to_log2_gain(Kp_dB)
        Kp = 2 ** Kp_log2

        if Fc_i is not None:
            Ki_log2 = lm.gain_for_crossover_frequency(0.0, sps, Fc_i, kind="I")
        elif Ki_dB is not None:
            Ki_log2 = lm.db_to_log2_gain(Ki_dB)
        else:
            Ki_log2 = float("-inf")

        if Fc_ii is not None:
            Kii_log2 = lm.gain_for_crossover_frequency(0.0, sps, Fc_ii, kind="II")
        elif Kii_dB is not None:
            Kii_log2 = lm.db_to_log2_gain(Kii_dB)
        else:
            Kii_log2 = float("-inf")

        if Fc_d is not None:
            Kd_log2 = lm.gain_for_crossover_frequency(0.0, sps, Fc_d, kind="D")
        elif Kd_dB is not None:
            Kd_log2 = lm.db_to_log2_gain(Kd_dB)
        else:
            Kd_log2 = float("-inf")

        Ki = 0.0 if not np.isfinite(Ki_log2) else 2 ** Ki_log2
        Kii = 0.0 if not np.isfinite(Kii_log2) else 2 ** Kii_log2
        Kd = 0.0 if not np.isfinite(Kd_log2) else 2 ** Kd_log2

        if f_trans is not None:
            delta = (2 * np.pi * f_trans / sps) ** 2
            Kii_term = f"({Kii:.6g})/((1 - z**-1)**2 + {delta:.3e})"
        else:
            Kii_term = f"({Kii:.6g})/(1 - z**-1)**2"

        return (
            f"{Kp:.6g} * (1"
            f" + ({Ki:.6g})/(1 - z**-1)"
            f" + {Kii_term}"
            f" + ({Kd:.6g})*(1 - z**-1)/(1 + z**-1))"
        )

    @property
    def Kp_dB(self) -> float:
        return self._Kp_dB

    @Kp_dB.setter
    def Kp_dB(self, value: float) -> None:
        self._Kp_dB = _validate_numeric("Kp_dB", value)
        self.update_component()

    @property
    def Ki_dB(self) -> Optional[float]:
        return (
            None
            if self._Ki == 0.0
            else lm.log2_gain_to_db(np.log2(self._Ki))
        )

    @Ki_dB.setter
    def Ki_dB(self, value: float) -> None:
        self._Fc_i = None
        self._Ki = 2 ** lm.db_to_log2_gain(_validate_numeric("Ki_dB", value))
        self.update_component()

    @property
    def Kii_dB(self) -> Optional[float]:
        return (
            None
            if self._Kii == 0.0
            else lm.log2_gain_to_db(np.log2(self._Kii))
        )

    @Kii_dB.setter
    def Kii_dB(self, value: float) -> None:
        self._Fc_ii = None
        self._Kii = 2 ** lm.db_to_log2_gain(_validate_numeric("Kii_dB", value))
        self.update_component()

    @property
    def Kd_dB(self) -> Optional[float]:
        return (
            None
            if self._Kd == 0.0
            else lm.log2_gain_to_db(np.log2(self._Kd))
        )

    @Kd_dB.setter
    def Kd_dB(self, value: float) -> None:
        self._Fc_d = None
        self._Kd = 2 ** lm.db_to_log2_gain(_validate_numeric("Kd_dB", value))
        self.update_component()

    @property
    def Fc_i(self) -> Optional[float]:
        return self._Fc_i

    @Fc_i.setter
    def Fc_i(self, value: Optional[float]) -> None:
        self._Fc_i = _validate_optional_positive("Fc_i", value)
        self.update_component()

    @property
    def Fc_ii(self) -> Optional[float]:
        return self._Fc_ii

    @Fc_ii.setter
    def Fc_ii(self, value: Optional[float]) -> None:
        self._Fc_ii = _validate_optional_positive("Fc_ii", value)
        self.update_component()

    @property
    def Fc_d(self) -> Optional[float]:
        return self._Fc_d

    @Fc_d.setter
    def Fc_d(self, value: Optional[float]) -> None:
        self._Fc_d = _validate_optional_positive("Fc_d", value)
        self.update_component()

    def __deepcopy__(self, memo: dict[int, Any]) -> MokuPIDSymbolicController:
        return MokuPIDSymbolicController(
            name=self.name,
            sps=self.sps,
            Kp_dB=self._Kp_dB,
            Fc_i=self._Fc_i,
            Fc_ii=self._Fc_ii,
            Fc_d=self._Fc_d,
            f_trans=self.f_trans,
        )


class MokuPIDController(Component):
    """
    Moku-style PID controller with P, optional I, II, and D terms using symbolic structure
    and extrapolated low-frequency behavior for numerical stability.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate [Hz].
    Kp_dB : float
        Proportional gain in dB.
    Fc_i : float or None
        First integrator crossover frequency [Hz]. If None, I is omitted.
    Fc_ii : float or None
        Second integrator crossover frequency [Hz]. If None, II is omitted.
    Fc_d : float or None
        Derivative crossover frequency [Hz]. If None, D is omitted.
    f_trans : float
        Transition frequency below which extrapolation is applied [Hz].
    """

    def __init__(
        self,
        name: str,
        sps: float,
        Kp_dB: float,
        Fc_i: Optional[float] = None,
        Fc_ii: Optional[float] = None,
        Fc_d: Optional[float] = None,
        f_trans: Optional[float] = None,
    ) -> None:
        self.name = name
        self.sps = _validate_positive("sps", sps)
        self.f_trans = _validate_optional_positive("f_trans", f_trans)
        self._Kp_dB = _validate_numeric("Kp_dB", Kp_dB)
        self._Fc_i = _validate_optional_positive("Fc_i", Fc_i)
        self._Fc_ii = _validate_optional_positive("Fc_ii", Fc_ii)
        self._Fc_d = _validate_optional_positive("Fc_d", Fc_d)

        self.update_component()

        self.properties = {
            "Kp_dB": (lambda: self.Kp_dB, lambda value: setattr(self, "Kp_dB", value)),
            "Ki_dB": (lambda: self.Ki_dB, lambda value: setattr(self, "Ki_dB", value)),
            "Kii_dB": (lambda: self.Kii_dB, lambda value: setattr(self, "Kii_dB", value)),
            "Kd_dB": (lambda: self.Kd_dB, lambda value: setattr(self, "Kd_dB", value)),
            "Fc_i": (lambda: self.Fc_i, lambda value: setattr(self, "Fc_i", value)),
            "Fc_ii": (lambda: self.Fc_ii, lambda value: setattr(self, "Fc_ii", value)),
            "Fc_d": (lambda: self.Fc_d, lambda value: setattr(self, "Fc_d", value)),
        }

    def update_component(self) -> None:
        Kp_log2 = lm.db_to_log2_gain(self._Kp_dB)
        self._Kp = 2 ** Kp_log2

        p_comp = Component(
            "P",
            self.sps,
            np.array([self._Kp]),
            np.array([1.0]),
            unit=Dimension(["cycle"], ["s", "rad"]),
        )
        components: list[Component] = [p_comp]

        if self._Fc_i is not None and self._Fc_ii is None:
            self._Ki = 2 ** lm.gain_for_crossover_frequency(
                Kp_log2, self.sps, self._Fc_i, kind="I"
            )
            i_comp = Component(
                "I",
                self.sps,
                np.array([self._Ki]),
                np.array([1.0, -1.0]),
                unit=p_comp.unit,
            )
            components.append(i_comp)
        else:
            self._Ki = None

        if self._Fc_ii is not None and self._Fc_i is not None:
            i_log2, ii_log2 = lm.gain_for_crossover_frequency(
                Kp_log2, self.sps, [self._Fc_i, self._Fc_ii], kind="II"
            )
            self._Ki, self._Kii = 2 ** i_log2, 2 ** ii_log2
            i_comp = Component(
                "I",
                self.sps,
                np.array([self._Ki]),
                np.array([1.0, -1.0]),
                unit=p_comp.unit,
            )
            components.append(i_comp)

            ii_comp = Component(
                "II",
                self.sps,
                np.array([self._Kii]),
                np.array([1.0, -2.0, 1.0]),
                unit=p_comp.unit,
            )
            if self.f_trans is not None:
                ii_comp.TF = partial(
                    ii_comp.TF,
                    extrapolate=True,
                    f_trans=self.f_trans,
                    power=-2,
                )
            components.append(ii_comp)
        else:
            self._Ki = None
            self._Kii = None

        if self._Fc_d is not None:
            self._Kd = 2 ** lm.gain_for_crossover_frequency(
                Kp_log2, self.sps, self._Fc_d, kind="D"
            )
            d_comp = Component(
                "D",
                self.sps,
                np.array([self._Kd, -self._Kd]),
                np.array([1.0, 0.0, 1.0]),
                unit=p_comp.unit,
            )
            components.append(d_comp)
        else:
            self._Kd = None

        pid_comp = components[0]
        for comp in components[1:]:
            pid_comp = pid_comp + comp

        super().__init__(
            self.name,
            self.sps,
            pid_comp.nume,
            pid_comp.deno,
            unit=pid_comp.unit,
        )
        self.TF = pid_comp.TF
        self.TE = pid_comp.TE
        self.TE.name = self.name

    def __deepcopy__(self, memo: dict[int, Any]) -> MokuPIDController:
        return MokuPIDController(
            name=self.name,
            sps=self.sps,
            Kp_dB=self._Kp_dB,
            Fc_i=self._Fc_i,
            Fc_ii=self._Fc_ii,
            Fc_d=self._Fc_d,
            f_trans=self.f_trans,
        )

    @property
    def Kp_dB(self) -> float:
        return self._Kp_dB

    @Kp_dB.setter
    def Kp_dB(self, value: float) -> None:
        self._Kp_dB = _validate_numeric("Kp_dB", value)
        self.update_component()

    @property
    def Ki_dB(self) -> Optional[float]:
        return (
            None
            if self._Ki is None
            else lm.log2_gain_to_db(np.log2(self._Ki))
        )

    @Ki_dB.setter
    def Ki_dB(self, value: float) -> None:
        self._Fc_i = None
        self._Ki = 2 ** lm.db_to_log2_gain(_validate_numeric("Ki_dB", value))
        self.update_component()

    @property
    def Kii_dB(self) -> Optional[float]:
        return (
            None
            if self._Kii is None
            else lm.log2_gain_to_db(np.log2(self._Kii))
        )

    @Kii_dB.setter
    def Kii_dB(self, value: float) -> None:
        self._Fc_ii = None
        self._Kii = 2 ** lm.db_to_log2_gain(_validate_numeric("Kii_dB", value))
        self.update_component()

    @property
    def Kd_dB(self) -> Optional[float]:
        return (
            None
            if self._Kd is None
            else lm.log2_gain_to_db(np.log2(self._Kd))
        )

    @Kd_dB.setter
    def Kd_dB(self, value: float) -> None:
        self._Fc_d = None
        self._Kd = 2 ** lm.db_to_log2_gain(_validate_numeric("Kd_dB", value))
        self.update_component()

    @property
    def Fc_i(self) -> Optional[float]:
        return self._Fc_i

    @Fc_i.setter
    def Fc_i(self, value: Optional[float]) -> None:
        self._Fc_i = _validate_optional_positive("Fc_i", value)
        self.update_component()

    @property
    def Fc_ii(self) -> Optional[float]:
        return self._Fc_ii

    @Fc_ii.setter
    def Fc_ii(self, value: Optional[float]) -> None:
        self._Fc_ii = _validate_optional_positive("Fc_ii", value)
        self.update_component()

    @property
    def Fc_d(self) -> Optional[float]:
        return self._Fc_d

    @Fc_d.setter
    def Fc_d(self, value: Optional[float]) -> None:
        self._Fc_d = _validate_optional_positive("Fc_d", value)
        self.update_component()
