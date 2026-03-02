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
from numpy.typing import ArrayLike
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


def iir_from_sos(
    sos: ArrayLike,
    input_scale: float = 1.0,
    output_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert SOS format [b0, b1, b2, a0, a1, a2] to (nume, deno) with scaling.

    Supports fixed-point DSP implementations (e.g. Simulink-style integer-scaled
    coefficients like sosScaled26). For a single second-order section, the format
    is [b0, b1, b2, a0, a1, a2]. The returned nume and deno are normalized so
    deno[0] = 1, and the numerator is scaled by input_scale * output_scale.

    Parameters
    ----------
    sos : array_like
        Second-order section coefficients [b0, b1, b2, a0, a1, a2].
    input_scale : float, optional
        Input scaling factor (e.g. 2^-24). Default 1.0.
    output_scale : float, optional
        Output scaling factor (e.g. 2^-14). Default 1.0.

    Returns
    -------
    nume : ndarray
        Numerator coefficients [b0, b1, b2] (normalized, scaled).
    deno : ndarray
        Denominator coefficients [1, a1, a2] (a0=1 convention).

    Examples
    --------
    >>> sos = [16777216, 33554432, 16777216, 16777216, -33181752, 16408629]
    >>> nume, deno = iir_from_sos(sos, input_scale=2**-24, output_scale=2**-14)
    """
    sos_arr = np.atleast_1d(np.asarray(sos, dtype=float))
    if sos_arr.size < 6:
        raise ValueError("sos must have at least 6 elements [b0,b1,b2,a0,a1,a2]")
    b0, b1, b2, a0, a1, a2 = sos_arr[:6]
    if a0 == 0:
        raise ValueError("sos a0 (denominator leading coefficient) must be non-zero")
    scale = float(input_scale) * float(output_scale)
    nume = np.array([b0 / a0, b1 / a0, b2 / a0]) * scale
    deno = np.array([1.0, a1 / a0, a2 / a0])
    return nume, deno


class IIRFilterComponent(Component):
    """
    IIR filter from direct numerator/denominator coefficients.

    Supports optional input/output scaling for fixed-point DSP implementations
    (e.g. Simulink-style integer-scaled coefficients).

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz. Must be positive.
    nume : array_like
        Numerator coefficients [b0, b1, b2, ...] (z^-1 convention).
    deno : array_like
        Denominator coefficients [a0, a1, a2, ...]. If a0 != 1, coefficients
        are normalized so a0=1.
    input_scale : float, optional
        Input scaling factor (e.g. 2^-24). Default 1.0.
    output_scale : float, optional
        Output scaling factor (e.g. 2^-14). Default 1.0.

    Attributes
    ----------
    nume : ndarray
        Numerator coefficients (normalized, scaled).
    deno : ndarray
        Denominator coefficients (a0=1).
    input_scale : float
        Input scaling factor.
    output_scale : float
        Output scaling factor.
    """

    def __init__(
        self,
        name: str,
        sps: float,
        nume: ArrayLike,
        deno: ArrayLike,
        input_scale: float = 1.0,
        output_scale: float = 1.0,
    ) -> None:
        name_s = _validate_str_non_empty("name", name)
        sps_f = _validate_positive("sps", sps)
        _validate_numeric("input_scale", input_scale)
        _validate_numeric("output_scale", output_scale)

        nume_arr = np.atleast_1d(np.asarray(nume, dtype=float))
        deno_arr = np.atleast_1d(np.asarray(deno, dtype=float))

        if nume_arr.size == 0:
            raise ValueError("nume must be non-empty")
        if deno_arr.size == 0:
            raise ValueError("deno must be non-empty")
        if deno_arr[0] == 0:
            raise ValueError("deno[0] (leading coefficient) must be non-zero")

        self._input_scale = float(input_scale)
        self._output_scale = float(output_scale)
        scale = self._input_scale * self._output_scale

        # Normalize: a0=1
        a0 = deno_arr[0]
        deno_norm = deno_arr / a0
        nume_norm = nume_arr / a0 * scale

        super().__init__(
            name_s,
            sps_f,
            nume=nume_norm,
            deno=deno_norm,
            unit=Dimension(dimensionless=True),
        )
        # Store original (pre-normalization, pre-scaling) for update_component
        self._nume = np.array(nume_arr, copy=True)
        self._deno = np.array(deno_arr, copy=True)
        self.properties = {
            "nume": (lambda: self.nume, self._set_nume),
            "deno": (lambda: self.deno, self._set_deno),
            "input_scale": (
                lambda: self.input_scale,
                lambda v: setattr(self, "input_scale", v),
            ),
            "output_scale": (
                lambda: self.output_scale,
                lambda v: setattr(self, "output_scale", v),
            ),
        }

    def _set_nume(self, v: ArrayLike) -> None:
        arr = np.atleast_1d(np.asarray(v, dtype=float))
        if arr.size == 0:
            raise ValueError("nume must be non-empty")
        self._nume = arr
        self.update_component()

    def _set_deno(self, v: ArrayLike) -> None:
        arr = np.atleast_1d(np.asarray(v, dtype=float))
        if arr.size == 0:
            raise ValueError("deno must be non-empty")
        if arr[0] == 0:
            raise ValueError("deno[0] must be non-zero")
        self._deno = arr
        self.update_component()

    def __deepcopy__(self, memo: Dict[int, Any]) -> IIRFilterComponent:
        new_obj = IIRFilterComponent.__new__(IIRFilterComponent)
        new_obj.__init__(
            self.name,
            self.sps,
            self._nume.copy(),
            self._deno.copy(),
            self._input_scale,
            self._output_scale,
        )
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def input_scale(self) -> float:
        """Input scaling factor."""
        return self._input_scale

    @input_scale.setter
    def input_scale(self, value: float) -> None:
        self._input_scale = _validate_numeric("input_scale", value)
        self.update_component()

    @property
    def output_scale(self) -> float:
        """Output scaling factor."""
        return self._output_scale

    @output_scale.setter
    def output_scale(self, value: float) -> None:
        self._output_scale = _validate_numeric("output_scale", value)
        self.update_component()

    def update_component(self) -> None:
        scale = self._input_scale * self._output_scale
        a0 = self._deno[0]
        if a0 == 0:
            raise ValueError("deno[0] must be non-zero")
        deno_norm = self._deno / a0
        nume_norm = self._nume / a0 * scale
        super().__init__(
            self.name,
            self.sps,
            nume=nume_norm,
            deno=deno_norm,
            unit=Dimension(dimensionless=True),
        )

    @classmethod
    def from_sos(
        cls,
        name: str,
        sps: float,
        sos: ArrayLike,
        input_scale: float = 1.0,
        output_scale: float = 1.0,
    ) -> "IIRFilterComponent":
        """
        Create IIRFilterComponent from SOS format [b0, b1, b2, a0, a1, a2].

        Parameters
        ----------
        name : str
            Component name.
        sps : float
            Sample rate in Hz.
        sos : array_like
            Second-order section [b0, b1, b2, a0, a1, a2].
        input_scale : float, optional
            Input scaling factor. Default 1.0.
        output_scale : float, optional
            Output scaling factor. Default 1.0.

        Returns
        -------
        IIRFilterComponent
        """
        nume, deno = iir_from_sos(sos, input_scale, output_scale)
        return cls(name, sps, nume, deno, input_scale=1.0, output_scale=1.0)
