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

import numbers
from typing import Any, Dict

import numpy as np

import loopkit.loopmath as lm
from loopkit.component import Component
from loopkit.dimension import Dimension


def _validate_positive(name: str, value: float, *, strict: bool = True) -> float:
    """Validate and coerce to positive float. Raises TypeError or ValueError."""
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    v = float(value)
    if strict and v <= 0:
        raise ValueError(f"{name} must be positive, got {v}")
    return v


def _validate_non_negative(name: str, value: float) -> float:
    """Validate and coerce to non-negative float. Raises TypeError or ValueError."""
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    v = float(value)
    if v < 0:
        raise ValueError(f"{name} must be non-negative, got {v}")
    return v


def _validate_numeric(name: str, value: float) -> float:
    """Validate and coerce to float. Raises TypeError."""
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    return float(value)


class ActuatorComponent(Component):
    """
    PZT actuator model with gain and cutoff frequency.

    Converts s-domain coefficients into z-domain using polynomial_conversion_s_to_z.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz. Must be positive.
    Ka_pzt : float
        Actuator gain. Must be positive.
    Fa_pzt : float
        Actuator cutoff frequency (Hz). Must be positive.
    unit : Dimension
        Dimensional unit of the actuator.

    Attributes
    ----------
    Ka_pzt : float
        Gain.
    Fa_pzt : float
        Cutoff frequency.
    """

    def __init__(
        self,
        name: str,
        sps: float,
        Ka_pzt: float,
        Fa_pzt: float,
        unit: Dimension,
    ) -> None:
        sps_f = _validate_positive("sps", sps)
        self._Ka_pzt = _validate_positive("Ka_pzt", Ka_pzt)
        self._Fa_pzt = _validate_positive("Fa_pzt", Fa_pzt)
        if not isinstance(unit, Dimension):
            raise TypeError(f"unit must be a Dimension, got {type(unit).__name__}")

        nume = lm.polynomial_conversion_s_to_z(np.array([self._Ka_pzt]), sps_f)
        deno = lm.polynomial_conversion_s_to_z(
            np.array([1.0 / (2.0 * np.pi * self._Fa_pzt), 1.0]), sps_f
        )
        super().__init__(name, sps_f, nume, deno, unit=unit)
        self.properties = {
            "Ka_pzt": (lambda: self.Ka_pzt, lambda v: setattr(self, "Ka_pzt", v)),
            "Fa_pzt": (lambda: self.Fa_pzt, lambda v: setattr(self, "Fa_pzt", v)),
        }

    def __deepcopy__(self, memo: Dict[int, Any]) -> ActuatorComponent:
        new_obj = ActuatorComponent.__new__(ActuatorComponent)
        new_obj.__init__(
            self.name, self.sps, self._Ka_pzt, self._Fa_pzt, self.unit
        )
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Fa_pzt(self) -> float:
        return self._Fa_pzt

    @Fa_pzt.setter
    def Fa_pzt(self, value: float) -> None:
        self._Fa_pzt = _validate_positive("Fa_pzt", value)
        self.update_component()

    @property
    def Ka_pzt(self) -> float:
        return self._Ka_pzt

    @Ka_pzt.setter
    def Ka_pzt(self, value: float) -> None:
        self._Ka_pzt = _validate_positive("Ka_pzt", value)
        self.update_component()

    def update_component(self) -> None:
        nume = lm.polynomial_conversion_s_to_z(
            np.array([self._Ka_pzt]), self.sps
        )
        deno = lm.polynomial_conversion_s_to_z(
            np.array([1.0 / (2.0 * np.pi * self._Fa_pzt), 1.0]), self.sps
        )
        super().__init__(self.name, self.sps, nume, deno, unit=self.unit)


class ImplicitAccumulatorComponent(Component):
    """
    Continuous-time accumulator modeled in discrete-time domain.

    Approximates a pure integrator with scaling factor of 2pi.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz. Must be positive.
    """

    def __init__(self, name: str, sps: float) -> None:
        sps_f = _validate_positive("sps", sps)
        nume = lm.polynomial_conversion_s_to_z(np.array([2.0 * np.pi]), sps_f)
        deno = lm.polynomial_conversion_s_to_z(np.array([1.0, 0.0]), sps_f)
        super().__init__(
            name, sps_f, nume, deno, unit=Dimension(["rad"], ["Hz"])
        )

    def __deepcopy__(self, memo: Dict[int, Any]) -> ImplicitAccumulatorComponent:
        new_obj = ImplicitAccumulatorComponent.__new__(
            ImplicitAccumulatorComponent
        )
        new_obj.__init__(self.name, self.sps)
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj


class LeadLagComponent(Component):
    """
    Lead-Lag controller component.

    Implements a compensator of the form:

        G(s) = K * (s + wz) / (s + wp)

    where `wz = 2π*fz` is the zero frequency and `wp = 2π*fp` is the pole frequency.

    This component is useful for phase compensation: when fz < fp, it behaves like
    a phase lead; when fz > fp, it's a lag; and when fz ≈ fp, it is a gain shaper.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz. Must be positive.
    K : float
        Gain factor (may be negative for inverting).
    fz : float
        Zero frequency in Hz. Must be non-negative.
    fp : float
        Pole frequency in Hz. Must be non-negative.
    unit : Dimension, optional
        Dimensional unit of the signal. Defaults to dimensionless.

    Attributes
    ----------
    K : float
        Gain factor.
    fz : float
        Zero frequency.
    fp : float
        Pole frequency.
    """

    def __init__(
        self,
        name: str,
        sps: float,
        K: float,
        fz: float,
        fp: float,
        unit: Dimension = Dimension(dimensionless=True),
    ) -> None:
        sps_f = _validate_positive("sps", sps)
        self._K = _validate_numeric("K", K)
        self._fz = _validate_non_negative("fz", fz)
        self._fp = _validate_non_negative("fp", fp)
        if not isinstance(unit, Dimension):
            raise TypeError(f"unit must be a Dimension, got {type(unit).__name__}")

        w_z = 2 * np.pi * self._fz
        w_p = 2 * np.pi * self._fp
        nume = np.array([self._K, self._K * w_z])
        deno = np.array([1.0, w_p])
        super().__init__(name, sps_f, nume, deno, unit=unit)
        self.properties = {
            "K": (lambda: self.K, lambda v: setattr(self, "K", v)),
            "fz": (lambda: self.fz, lambda v: setattr(self, "fz", v)),
            "fp": (lambda: self.fp, lambda v: setattr(self, "fp", v)),
        }

    def __deepcopy__(self, memo: Dict[int, Any]) -> LeadLagComponent:
        new_obj = LeadLagComponent.__new__(LeadLagComponent)
        new_obj.__init__(
            self.name, self.sps, self._K, self._fz, self._fp, self.unit
        )
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def K(self) -> float:
        return self._K

    @K.setter
    def K(self, value: float) -> None:
        self._K = _validate_numeric("K", value)
        self.update_component()

    @property
    def fz(self) -> float:
        return self._fz

    @fz.setter
    def fz(self, value: float) -> None:
        self._fz = _validate_non_negative("fz", value)
        self.update_component()

    @property
    def fp(self) -> float:
        return self._fp

    @fp.setter
    def fp(self, value: float) -> None:
        self._fp = _validate_non_negative("fp", value)
        self.update_component()

    def update_component(self) -> None:
        w_z = 2 * np.pi * self._fz
        w_p = 2 * np.pi * self._fp
        nume = np.array([self._K, self._K * w_z])
        deno = np.array([1.0, w_p])
        super().__init__(self.name, self.sps, nume, deno, unit=self.unit)
