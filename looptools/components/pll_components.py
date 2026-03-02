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


class PDComponent(Component):
    """
    Phase Detector (PD) component for phase-locked loop simulations.

    This component models the behavior of a digital phase detector, which mixes
    an input signal with a numerically controlled oscillator (NCO) in a PLL system.
    The output of the phase detector is proportional to the phase difference
    between its input and reference signals.

    The internal transfer function is a static gain element with a value derived
    from the amplitude (`Amp`), initialized as `Amp / 4.0`. This scaling factor
    reflects typical digital mixing gain behavior.

    Parameters
    ----------
    name : str
        Name of the component.
    sps : float
        Sample rate in Hz.
    Amp : float
        Peak amplitude of the input signal; determines phase detector gain.

    Attributes
    ----------
    Amp : float
        Input amplitude used to compute the transfer gain.
    ival : float
        Internal gain value, computed as Amp / 4.0.
    """

    def __init__(self, name, sps, Amp):
        self._Amp = Amp
        self._ival = Amp / 4.0
        super().__init__(name, sps, np.array([self._ival]), np.array([1.0]), unit=Dimension(dimensionless=True))
        self.properties = {
            "Amp": (lambda self=self: self.Amp, lambda value, self=self: setattr(self, "Amp", value)),
            "ival": (lambda self=self: self.ival, lambda value, self=self: setattr(self, "ival", value)),
        }

    def __deepcopy__(self, memo):
        new_obj = PDComponent.__new__(PDComponent)
        new_obj.__init__(self.name, self.sps, self._Amp)
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Amp(self):
        return self._Amp

    @Amp.setter
    def Amp(self, value):
        self._Amp = float(value)
        self._ival = self._Amp / 4.0
        self.update_component()

    @property
    def ival(self):
        return self._ival

    @ival.setter
    def ival(self, value):
        self._ival = float(value)
        self._Amp = 4.0 * self._ival
        self.update_component()

    def update_component(self):
        super().__init__(self.name, self.sps, np.array([self._ival]), np.array([1.0]), unit=Dimension(dimensionless=True))


class PAComponent(Component):
    """
    Phase accumulator.

    Implements a pure integrator (I(z) = 1 / (1 - z⁻¹)).

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    """

    def __init__(self, name, sps):
        super().__init__(name, sps, np.array([1.0]), np.array([1.0, -1.0]), unit=Dimension(["s"], []))

    def __deepcopy__(self, memo):
        new_obj = PAComponent.__new__(PAComponent)
        new_obj.__init__(self.name, self.sps)
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj


class LUTComponent(Component):
    """
    Lookup table phase converter.

    Converts digital phase to analog signal (rad ↔ cycle).

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    """

    def __init__(self, name, sps):
        super().__init__(name, sps, np.array([2.0 * np.pi]), np.array([1.0]), unit=Dimension(["rad"], ["cycle"]))

    def __deepcopy__(self, memo):
        new_obj = LUTComponent.__new__(LUTComponent)
        new_obj.__init__(self.name, self.sps)
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj
