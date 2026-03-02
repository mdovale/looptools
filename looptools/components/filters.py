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
from scipy.signal import butter
from looptools.component import Component
from looptools.dimension import Dimension


class LPFComponent(Component):
    """
    Low pass filter component (first-order or cascaded first-order IIR).

    Models a low-pass IIR filter with tunable gain. Higher-order filters can be
    simulated by cascading multiple identical first-order sections.

    Parameters
    ----------
    name : str
        Name of the component.
    sps : float
        Sample rate in Hz.
    Klf : float
        Log2 representation of loop gain (gain = 2^-Klf).
    n : int, optional
        Number of cascaded first-order sections (default is 1).

    Attributes
    ----------
    Klf : float
        Filter gain as 2^-Klf.
    n : int
        Filter order (number of cascaded stages).
    """

    def __init__(self, name, sps, Klf, n=1):
        self.n = int(n)
        self._Klf = 2.0 ** float(-Klf)
        num = np.array([self._Klf]) ** self.n
        den = np.poly1d([1.0, -(1.0 - self._Klf)]) ** self.n
        super().__init__(name, sps, num, den.coeffs, unit=Dimension(dimensionless=True))
        self.properties = {
            "Klf": (lambda: self.Klf, lambda value: setattr(self, "Klf", value)),
            "n": (lambda: self.n, lambda value: setattr(self, "n", int(value))),
        }

    def __deepcopy__(self, memo):
        new_obj = LPFComponent.__new__(LPFComponent)
        new_obj.__init__(self.name, self.sps, -np.log2(self._Klf), self.n)
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Klf(self):
        return self._Klf

    @Klf.setter
    def Klf(self, value):
        self._Klf = 2 ** float(-value)
        self.update_component()

    def update_component(self):
        num = np.array([self._Klf]) ** self.n
        den = np.poly1d([1.0, -(1.0 - self._Klf)]) ** self.n
        super().__init__(self.name, self.sps, num, den.coeffs, unit=Dimension(dimensionless=True))


class ButterworthLPFComponent(Component):
    """
    Digital Butterworth low-pass filter component (n-th order IIR).

    Uses scipy's butter() design to produce a maximally flat low-pass filter.

    Parameters
    ----------
    name : str
        Name of the component.
    sps : float
        Sample rate in Hz.
    f_c : float
        -3 dB cutoff frequency in Hz.
    order : int, optional
        Filter order (default: 1). Must be >= 1.

    Attributes
    ----------
    f_c : float
        Cutoff frequency in Hz.
    order : int
        Filter order.
    """

    def __init__(self, name, sps, f_c, order=1):
        if order < 1:
            raise ValueError("Butterworth filter order must be >= 1.")
        self.f_c = f_c
        self.order = int(order)

        norm_cutoff = f_c / (0.5 * sps)
        b, a = butter(N=order, Wn=norm_cutoff, btype="low", analog=False)

        super().__init__(name=name, sps=sps, nume=b, deno=a, unit=Dimension(dimensionless=True))

        self.properties = {
            "f_c": (lambda: self.f_c, self._set_fc),
            "order": (lambda: self.order, self._set_order),
        }

    def _set_fc(self, f_c):
        self.f_c = float(f_c)
        self._update_tf()

    def _set_order(self, order):
        self.order = int(order)
        self._update_tf()

    def _update_tf(self):
        norm_cutoff = self.f_c / (0.5 * self.sps)
        b, a = butter(N=self.order, Wn=norm_cutoff, btype="low", analog=False)
        self.nume = np.atleast_1d(b)
        self.deno = np.atleast_1d(a)


class TwoStageLPFComponent(Component):
    """
    Cascaded low pass filter with two identical first-order stages.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    Klf : float
        Log2 representation of gain.

    Attributes
    ----------
    Klf : float
        Effective loop filter gain (applied twice in series).
    """

    def __init__(self, name, sps, Klf):
        self._Klf = 2 ** float(-Klf)
        LF = Component(
            "LPF", sps, np.array([self._Klf]), np.array([1.0, -(1.0 - self._Klf)]), unit=Dimension(dimensionless=True)
        )
        LF = LF * LF
        super().__init__(name, sps, LF.nume, LF.deno, unit=LF.unit)
        self.TE = LF.TE
        self.TE.name = name
        self.TF = LF.TF
        self.properties = {"Klf": (lambda: self.Klf, lambda value: setattr(self, "Klf", value))}

    def __deepcopy__(self, memo):
        new_obj = TwoStageLPFComponent.__new__(TwoStageLPFComponent)
        new_obj.__init__(self.name, self.sps, -np.log2(self._Klf))
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Klf(self):
        return self._Klf

    @Klf.setter
    def Klf(self, value):
        self._Klf = 2 ** float(-value)
        self.update_component()

    def update_component(self):
        LF = Component(
            "LPF", self.sps, np.array([self._Klf]), np.array([1.0, -(1.0 - self._Klf)]), unit=Dimension(dimensionless=True)
        )
        LF = LF * LF
        super().__init__(self.name, self.sps, LF.nume, LF.deno, unit=LF.unit)
        self.TE = LF.TE
        self.TE.name = self.name
        self.TF = LF.TF
