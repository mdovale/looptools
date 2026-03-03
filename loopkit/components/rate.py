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
"""Rate transition components for multi-rate control loop modeling."""
from __future__ import annotations

from functools import partial
from typing import Any, Dict

import numpy as np

from loopkit.component import Component
from loopkit.dimension import Dimension

from ._validation import (
    _validate_int_positive,
    _validate_positive,
    _validate_str_non_empty,
)


def _tf_pure_delay(f: np.ndarray, n_delay: int, sps: float) -> np.ndarray:
    """Analytic delay TF: H(f) = exp(-j * 2π * f * n_delay / sps). Numerically stable."""
    f = np.asarray(f, dtype=float)
    return np.exp(-1j * 2 * np.pi * f * n_delay / sps)


class DownsampleComponent(Component):
    """
    Downsample component: y[n] = x[n*M] (keep every M-th sample).

    Models ideal downsampling by factor M. For loop analysis, approximated as
    unity gain at DC with optional causal delay of (M-1) samples at input rate.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Input sample rate in Hz. Output rate is sps/M.
    M : int
        Downsample factor. Must be >= 1 (M=1 means no downsampling).
    include_delay : bool, optional
        If True (default), model causal delay of (M-1) samples. If False,
        use pure gain H(z)=1 for ideal pass-through.

    Attributes
    ----------
    M : int
        Downsample factor.
    sps_out : float
        Output sample rate (sps/M).
    include_delay : bool
        Whether delay is included in the transfer function.
    """

    def __init__(
        self,
        name: str,
        sps: float,
        M: int,
        include_delay: bool = True,
    ) -> None:
        name_s = _validate_str_non_empty("name", name)
        sps_f = _validate_positive("sps", sps)
        self._M = _validate_int_positive("M", M)
        self._include_delay = bool(include_delay)

        nume = np.array([1.0])
        if self._include_delay and self._M > 1:
            # Delay of (M-1) samples: H(z) = z^(-(M-1))
            n_delay = self._M - 1
            deno = np.zeros(n_delay + 1)
            deno[0] = 1.0
        else:
            deno = np.array([1.0])

        super().__init__(
            name_s,
            sps_f,
            nume=nume,
            deno=deno,
            unit=Dimension(dimensionless=True),
        )
        self._set_tf_delay()
        self.properties = {
            "M": (lambda: self.M, lambda v: setattr(self, "M", v)),
            "include_delay": (
                lambda: self.include_delay,
                lambda v: setattr(self, "include_delay", v),
            ),
        }

    def _set_tf_delay(self) -> None:
        """Use analytic delay TF for numerical stability."""
        if self._include_delay and self._M > 1:
            n_delay = self._M - 1
            self.TF = partial(_tf_pure_delay, n_delay=n_delay, sps=self.sps)

    @property
    def M(self) -> int:
        """Downsample factor."""
        return self._M

    @M.setter
    def M(self, value: int) -> None:
        self._M = _validate_int_positive("M", value)
        self.update_component()

    @property
    def sps_out(self) -> float:
        """Output sample rate (sps/M)."""
        return self.sps / self._M

    @property
    def include_delay(self) -> bool:
        """Whether causal delay is included in the transfer function."""
        return self._include_delay

    @include_delay.setter
    def include_delay(self, value: bool) -> None:
        self._include_delay = bool(value)
        self.update_component()

    def __deepcopy__(self, memo: Dict[int, Any]) -> DownsampleComponent:
        new_obj = DownsampleComponent.__new__(DownsampleComponent)
        new_obj.__init__(self.name, self.sps, self._M, self._include_delay)
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    def update_component(self) -> None:
        nume = np.array([1.0])
        if self._include_delay and self._M > 1:
            n_delay = self._M - 1
            deno = np.zeros(n_delay + 1)
            deno[0] = 1.0
        else:
            deno = np.array([1.0])
        super().__init__(
            self.name,
            self.sps,
            nume=nume,
            deno=deno,
            unit=Dimension(dimensionless=True),
        )
        self._set_tf_delay()


class RateTransitionComponent(Component):
    """
    Rate transition component: changes sample rate from sps_in to sps_out.

    When sps_in > sps_out (downsampling): equivalent to DownsampleComponent with
    M = sps_in / sps_out (must be integer). When sps_in < sps_out (upsampling):
    not yet supported.

    Parameters
    ----------
    name : str
        Component name.
    sps_in : float
        Input sample rate in Hz.
    sps_out : float
        Output sample rate in Hz.
    include_delay : bool, optional
        If True (default), model causal delay. If False, use pure gain.

    Attributes
    ----------
    sps_in : float
        Input sample rate.
    sps_out : float
        Output sample rate.
    M : int
        Downsample factor (sps_in/sps_out) when downsampling.
    """

    def __init__(
        self,
        name: str,
        sps_in: float,
        sps_out: float,
        include_delay: bool = True,
    ) -> None:
        name_s = _validate_str_non_empty("name", name)
        sps_in_f = _validate_positive("sps_in", sps_in)
        sps_out_f = _validate_positive("sps_out", sps_out)

        ratio = sps_in_f / sps_out_f
        if ratio < 1.0:
            raise ValueError(
                f"Upsampling (sps_in < sps_out) not yet supported. "
                f"Got sps_in={sps_in_f}, sps_out={sps_out_f}."
            )
        M_int = int(round(ratio))
        if abs(ratio - M_int) > 1e-9:
            raise ValueError(
                f"sps_in/sps_out must be an integer for downsampling. "
                f"Got {ratio} (sps_in={sps_in_f}, sps_out={sps_out_f})."
            )

        self._sps_in = sps_in_f
        self._sps_out = sps_out_f
        self._M = M_int
        self._include_delay = bool(include_delay)

        # Delegate to DownsampleComponent logic; sps = input rate
        nume = np.array([1.0])
        if self._include_delay and self._M > 1:
            n_delay = self._M - 1
            deno = np.zeros(n_delay + 1)
            deno[0] = 1.0
        else:
            deno = np.array([1.0])

        super().__init__(
            name_s,
            sps_in_f,  # Component operates at input rate
            nume=nume,
            deno=deno,
            unit=Dimension(dimensionless=True),
        )
        self._set_tf_delay()
        self.properties = {
            "sps_in": (lambda: self.sps_in, lambda v: setattr(self, "sps_in", v)),
            "sps_out": (lambda: self.sps_out, lambda v: setattr(self, "sps_out", v)),
            "include_delay": (
                lambda: self.include_delay,
                lambda v: setattr(self, "include_delay", v),
            ),
        }

    def _set_tf_delay(self) -> None:
        """Use analytic delay TF for numerical stability (avoids polynomial eval)."""
        if self._include_delay and self._M > 1:
            n_delay = self._M - 1
            self.TF = partial(_tf_pure_delay, n_delay=n_delay, sps=self.sps)
        # else: keep default TF from parent (unity gain)

    @property
    def sps_in(self) -> float:
        """Input sample rate in Hz."""
        return self._sps_in

    @sps_in.setter
    def sps_in(self, value: float) -> None:
        self._sps_in = _validate_positive("sps_in", value)
        self._recompute_M()
        self.update_component()

    @property
    def sps_out(self) -> float:
        """Output sample rate in Hz."""
        return self._sps_out

    @sps_out.setter
    def sps_out(self, value: float) -> None:
        self._sps_out = _validate_positive("sps_out", value)
        self._recompute_M()
        self.update_component()

    def _recompute_M(self) -> None:
        ratio = self._sps_in / self._sps_out
        if ratio < 1.0:
            raise ValueError(
                "Upsampling not yet supported. sps_in must be >= sps_out."
            )
        M_int = int(round(ratio))
        if abs(ratio - M_int) > 1e-9:
            raise ValueError(
                f"sps_in/sps_out must be an integer. Got {ratio}."
            )
        self._M = M_int

    @property
    def M(self) -> int:
        """Downsample factor (sps_in/sps_out)."""
        return self._M

    @property
    def include_delay(self) -> bool:
        """Whether causal delay is included."""
        return self._include_delay

    @include_delay.setter
    def include_delay(self, value: bool) -> None:
        self._include_delay = bool(value)
        self.update_component()

    def __deepcopy__(self, memo: Dict[int, Any]) -> RateTransitionComponent:
        new_obj = RateTransitionComponent.__new__(RateTransitionComponent)
        new_obj.__init__(
            self.name,
            self._sps_in,
            self._sps_out,
            self._include_delay,
        )
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    def update_component(self) -> None:
        nume = np.array([1.0])
        if self._include_delay and self._M > 1:
            n_delay = self._M - 1
            deno = np.zeros(n_delay + 1)
            deno[0] = 1.0
        else:
            deno = np.array([1.0])
        super().__init__(
            self.name,
            self._sps_in,
            nume=nume,
            deno=deno,
            unit=Dimension(dimensionless=True),
        )
        self._set_tf_delay()
