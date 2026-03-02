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
import numpy as np
from looptools.component import Component
from looptools.dimension import Dimension


class MultiplierComponent(Component):
    """
    Static gain multiplier component.

    Parameters
    ----------
    name : str
        Name of the component.
    sps : float
        Sample rate in Hz.
    gain : float
        Gain value.
    unit : Dimension
        Dimensional unit of the signal after multiplication.

    Attributes
    ----------
    gain : float
        Gain applied to the input signal.
    """

    def __init__(self, name, sps, gain, unit):
        self._gain = gain
        super().__init__(name, sps, np.array([self._gain]), np.array([1.0]), unit=unit)
        self.properties = {"gain": (lambda: self.gain, lambda value: setattr(self, "gain", value))}

    def __deepcopy__(self, memo):
        new_obj = MultiplierComponent.__new__(MultiplierComponent)
        new_obj.__init__(self.name, self.sps, self._gain, self.unit)
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        self._gain = value
        self.update_component()

    def update_component(self):
        super().__init__(self.name, self.sps, np.array([self._gain]), np.array([1.0]), unit=self.unit)


class RightBitShiftComponent(Component):
    """
    Simulates a right bit-shift operation (/2^Cshift).

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    Cshift : int or float
        Shift value; actual gain is 2^(-Cshift).

    Attributes
    ----------
    Cshift : float
        Exponent of the power-of-two shift.
    """

    def __init__(self, name, sps, Cshift):
        self._Cshift = 2.0 ** float(-Cshift)
        super().__init__(name, sps, np.array([self._Cshift]), np.array([1.0]), unit=Dimension(dimensionless=True))
        self.properties = {"Cshift": (lambda: self.Cshift, lambda value: setattr(self, "Cshift", value))}

    def __deepcopy__(self, memo):
        new_obj = RightBitShiftComponent.__new__(RightBitShiftComponent)
        new_obj.__init__(self.name, self.sps, np.log2(self._Cshift))
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Cshift(self):
        return self._Cshift

    @Cshift.setter
    def Cshift(self, value):
        self._Cshift = 2.0 ** float(-value)
        self.update_component()

    def update_component(self):
        super().__init__(self.name, self.sps, np.array([self._Cshift]), np.array([1.0]), unit=Dimension(dimensionless=True))
