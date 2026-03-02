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
"""Gain and multiplier components for control loops."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from loopkit.component import Component
from loopkit.dimension import Dimension

from ._validation import (
    _validate_numeric,
    _validate_positive,
    _validate_str_non_empty,
)


class MultiplierComponent(Component):
    """
    Static gain multiplier component.

    Parameters
    ----------
    name : str
        Name of the component.
    sps : float
        Sample rate in Hz. Must be positive.
    gain : float
        Gain value.
    unit : Dimension
        Dimensional unit of the signal after multiplication.

    Attributes
    ----------
    gain : float
        Gain applied to the input signal.
    """

    def __init__(
        self,
        name: str,
        sps: float,
        gain: float,
        unit: Dimension,
    ) -> None:
        name_s = _validate_str_non_empty("name", name)
        sps_f = _validate_positive("sps", sps)
        self._gain = _validate_numeric("gain", gain)
        if not isinstance(unit, Dimension):
            raise TypeError(f"unit must be a Dimension, got {type(unit).__name__}")

        super().__init__(
            name_s, sps_f, np.array([self._gain]), np.array([1.0]), unit=unit
        )
        self.properties = {
            "gain": (lambda: self.gain, lambda v: setattr(self, "gain", v)),
        }

    def __deepcopy__(self, memo: Dict[int, Any]) -> MultiplierComponent:
        new_obj = MultiplierComponent.__new__(MultiplierComponent)
        new_obj.__init__(self.name, self.sps, self._gain, self.unit)
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def gain(self) -> float:
        """Gain applied to the input signal."""
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        self._gain = _validate_numeric("gain", value)
        self.update_component()

    def update_component(self) -> None:
        super().__init__(
            self.name, self.sps,
            np.array([self._gain]), np.array([1.0]),
            unit=self.unit,
        )


class RightBitShiftComponent(Component):
    """
    Simulates a right bit-shift operation (/2^Cshift).

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz. Must be positive.
    Cshift : int or float
        Shift value; actual gain is 2^(-Cshift).

    Attributes
    ----------
    Cshift : float
        Exponent of the power-of-two shift (gain = 2^(-Cshift)).
    """

    def __init__(self, name: str, sps: float, Cshift: float) -> None:
        name_s = _validate_str_non_empty("name", name)
        sps_f = _validate_positive("sps", sps)
        cshift_exp = _validate_numeric("Cshift", Cshift)
        self._Cshift = 2.0 ** float(-cshift_exp)

        super().__init__(
            name_s,
            sps_f,
            np.array([self._Cshift]),
            np.array([1.0]),
            unit=Dimension(dimensionless=True),
        )
        self.properties = {
            "Cshift": (lambda: self.Cshift, lambda v: setattr(self, "Cshift", v)),
        }

    def __deepcopy__(self, memo: Dict[int, Any]) -> RightBitShiftComponent:
        new_obj = RightBitShiftComponent.__new__(RightBitShiftComponent)
        # Cshift exponent = -log2(gain)
        exponent = -np.log2(self._Cshift)
        new_obj.__init__(self.name, self.sps, exponent)
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Cshift(self) -> float:
        """Effective gain (2^(-Cshift))."""
        return self._Cshift

    @Cshift.setter
    def Cshift(self, value: float) -> None:
        self._Cshift = 2.0 ** float(-_validate_numeric("Cshift", value))
        self.update_component()

    def update_component(self) -> None:
        super().__init__(
            self.name,
            self.sps,
            np.array([self._Cshift]),
            np.array([1.0]),
            unit=Dimension(dimensionless=True),
        )
