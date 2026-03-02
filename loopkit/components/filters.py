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
"""Low-pass filter components for control loops."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.signal import butter

from loopkit.component import Component
from loopkit.dimension import Dimension

from ._validation import (
    _validate_int_positive,
    _validate_numeric,
    _validate_positive,
    _validate_str_non_empty,
)


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
        Sample rate in Hz. Must be positive.
    Klf : float
        Log2 representation of loop gain (gain = 2^-Klf).
    n : int, optional
        Number of cascaded first-order sections (default is 1). Must be >= 1.

    Attributes
    ----------
    Klf : float
        Filter gain as 2^-Klf.
    n : int
        Filter order (number of cascaded stages).
    """

    def __init__(
        self,
        name: str,
        sps: float,
        Klf: float,
        n: int = 1,
    ) -> None:
        name_s = _validate_str_non_empty("name", name)
        sps_f = _validate_positive("sps", sps)
        self._n = _validate_int_positive("n", n)
        klf_exp = _validate_numeric("Klf", Klf)
        self._Klf = 2.0 ** float(-klf_exp)

        num = np.array([self._Klf]) ** self._n
        den = np.poly1d([1.0, -(1.0 - self._Klf)]) ** self._n
        super().__init__(
            name_s, sps_f, num, den.coeffs, unit=Dimension(dimensionless=True)
        )
        self.properties = {
            "Klf": (lambda: self.Klf, lambda v: setattr(self, "Klf", v)),
            "n": (lambda: self.n, lambda v: setattr(self, "n", v)),
        }

    @property
    def n(self) -> int:
        """Number of cascaded first-order sections."""
        return self._n

    @n.setter
    def n(self, value: int) -> None:
        self._n = _validate_int_positive("n", value)
        self.update_component()

    def __deepcopy__(self, memo: Dict[int, Any]) -> LPFComponent:
        new_obj = LPFComponent.__new__(LPFComponent)
        new_obj.__init__(self.name, self.sps, -np.log2(self._Klf), self._n)
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Klf(self) -> float:
        """Filter gain (2^-Klf)."""
        return self._Klf

    @Klf.setter
    def Klf(self, value: float) -> None:
        self._Klf = 2.0 ** float(-_validate_numeric("Klf", value))
        self.update_component()

    def update_component(self) -> None:
        num = np.array([self._Klf]) ** self._n
        den = np.poly1d([1.0, -(1.0 - self._Klf)]) ** self._n
        super().__init__(
            self.name, self.sps, num, den.coeffs,
            unit=Dimension(dimensionless=True)
        )


class ButterworthLPFComponent(Component):
    """
    Digital Butterworth low-pass filter component (n-th order IIR).

    Uses scipy's butter() design to produce a maximally flat low-pass filter.

    Parameters
    ----------
    name : str
        Name of the component.
    sps : float
        Sample rate in Hz. Must be positive.
    f_c : float
        -3 dB cutoff frequency in Hz. Must be positive and < Nyquist (sps/2).
    order : int, optional
        Filter order (default: 1). Must be >= 1.

    Attributes
    ----------
    f_c : float
        Cutoff frequency in Hz.
    order : int
        Filter order.
    """

    def __init__(
        self,
        name: str,
        sps: float,
        f_c: float,
        order: int = 1,
    ) -> None:
        name_s = _validate_str_non_empty("name", name)
        sps_f = _validate_positive("sps", sps)
        self._f_c = _validate_positive("f_c", f_c)
        self._order = _validate_int_positive("order", order)

        nyquist = 0.5 * sps_f
        if self._f_c >= nyquist:
            raise ValueError(
                f"f_c must be < Nyquist (sps/2 = {nyquist} Hz), got {self._f_c}"
            )

        norm_cutoff = self._f_c / nyquist
        b, a = butter(
            N=self._order, Wn=norm_cutoff, btype="low", analog=False
        )

        super().__init__(
            name=name_s,
            sps=sps_f,
            nume=b,
            deno=a,
            unit=Dimension(dimensionless=True),
        )

        self.properties = {
            "f_c": (lambda: self.f_c, self._set_fc),
            "order": (lambda: self.order, self._set_order),
        }

    @property
    def f_c(self) -> float:
        """Cutoff frequency in Hz."""
        return self._f_c

    @property
    def order(self) -> int:
        """Filter order."""
        return self._order

    def _set_fc(self, f_c: float) -> None:
        f_c_f = _validate_positive("f_c", f_c)
        nyquist = 0.5 * self.sps
        if f_c_f >= nyquist:
            raise ValueError(
                f"f_c must be < Nyquist (sps/2 = {nyquist} Hz), got {f_c_f}"
            )
        self._f_c = f_c_f
        self._update_tf()

    def _set_order(self, order: int) -> None:
        self._order = _validate_int_positive("order", order)
        self._update_tf()

    def _update_tf(self) -> None:
        norm_cutoff = self._f_c / (0.5 * self.sps)
        b, a = butter(
            N=self._order, Wn=norm_cutoff, btype="low", analog=False
        )
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
        Sample rate in Hz. Must be positive.
    Klf : float
        Log2 representation of gain.

    Attributes
    ----------
    Klf : float
        Effective loop filter gain (applied twice in series).
    """

    def __init__(self, name: str, sps: float, Klf: float) -> None:
        name_s = _validate_str_non_empty("name", name)
        sps_f = _validate_positive("sps", sps)
        klf_exp = _validate_numeric("Klf", Klf)
        self._Klf = 2.0 ** float(-klf_exp)

        num = np.array([self._Klf])
        den = np.array([1.0, -(1.0 - self._Klf)])
        unit = Dimension(dimensionless=True)
        lf = Component(
            "LPF", sps_f, num, den, unit=unit
        )
        lf = lf * lf

        super().__init__(name_s, sps_f, lf.nume, lf.deno, unit=lf.unit)
        self.TE = lf.TE
        self.TE.name = name_s
        self.TF = lf.TF
        self.properties = {
            "Klf": (lambda: self.Klf, lambda v: setattr(self, "Klf", v)),
        }

    def __deepcopy__(self, memo: Dict[int, Any]) -> TwoStageLPFComponent:
        new_obj = TwoStageLPFComponent.__new__(TwoStageLPFComponent)
        new_obj.__init__(self.name, self.sps, -np.log2(self._Klf))
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Klf(self) -> float:
        """Filter gain (2^-Klf)."""
        return self._Klf

    @Klf.setter
    def Klf(self, value: float) -> None:
        self._Klf = 2.0 ** float(-_validate_numeric("Klf", value))
        self.update_component()

    def update_component(self) -> None:
        num = np.array([self._Klf])
        den = np.array([1.0, -(1.0 - self._Klf)])
        unit = Dimension(dimensionless=True)
        lf = Component("LPF", self.sps, num, den, unit=unit)
        lf = lf * lf
        super().__init__(self.name, self.sps, lf.nume, lf.deno, unit=lf.unit)
        self.TE = lf.TE
        self.TE.name = self.name
        self.TF = lf.TF
