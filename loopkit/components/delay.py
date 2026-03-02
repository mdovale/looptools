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

from typing import Any

import numpy as np

from loopkit.component import Component
from loopkit.dimension import Dimension

from loopkit.components._validation import _validate_int_non_negative, _validate_positive


class DSPDelayComponent(Component):
    """
    Discrete pipeline delay component.

    Implements delay through register depth `n_reg`.

    Parameters
    ----------
    name : str
        Name of the component.
    sps : float
        Sample rate in Hz. Must be positive.
    n_reg : int
        Number of DSP registers (delay in samples). Must be non-negative.

    Attributes
    ----------
    n_reg : int
        Length of the pipeline delay.
    """

    def __init__(self, name: str, sps: float, n_reg: int | float) -> None:
        _validate_positive("sps", sps)
        self._n_reg = _validate_int_non_negative("n_reg", n_reg)
        dsp_denom = np.zeros(self._n_reg + 1)
        dsp_denom[0] = 1.0
        super().__init__(
            name,
            sps,
            np.array([1.0]),
            dsp_denom,
            unit=Dimension(dimensionless=True),
        )
        self.properties = {
            "n_reg": (lambda: self.n_reg, lambda value: setattr(self, "n_reg", value)),
        }

    def __deepcopy__(self, memo: dict[int, Any]) -> DSPDelayComponent:
        new_obj = DSPDelayComponent.__new__(DSPDelayComponent)
        new_obj.__init__(self.name, self.sps, self._n_reg)
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def n_reg(self) -> int:
        """Number of DSP registers (delay in samples)."""
        return self._n_reg

    @n_reg.setter
    def n_reg(self, value: int | float) -> None:
        self._n_reg = _validate_int_non_negative("n_reg", value)
        self.update_component()

    def update_component(self) -> None:
        dsp_denom = np.zeros(self._n_reg + 1)
        dsp_denom[0] = 1.0
        super().__init__(
            self.name,
            self.sps,
            np.array([1.0]),
            dsp_denom,
            unit=Dimension(dimensionless=True),
        )
