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
from types import MappingProxyType
from typing import Mapping

import numpy as np


def set_opamp_parameters(
    gbp: float,
    aol: float,
    ccm: float,
    cdiff: float,
    *,
    db: bool = True,
) -> Mapping[str, float]:
    """Generate an immutable OpAmp parameter dictionary.

    Reference: https://www.tij.co.jp/jp/lit/an/sboa122/sboa122.pdf?ts=1662305678857

    Parameters
    ----------
    gbp : float
        Gain bandwidth product in Hz. Must be positive.
    aol : float
        Open-loop gain, in dB if db=True else linear. Must be positive.
    ccm : float
        Common-mode capacitance in F. Must be non-negative.
    cdiff : float
        Differential capacitance in F. Must be non-negative.
    db : bool, optional
        If True, aol is interpreted in dB; otherwise linear. Default is True.

    Returns
    -------
    Mapping[str, float]
        Immutable mapping with keys: GBP, AOL, omegaA, Ccm, Cdiff.

    Raises
    ------
    TypeError
        If any parameter is not numeric.
    ValueError
        If gbp <= 0, aol <= 0, ccm < 0, or cdiff < 0.
    """
    if not isinstance(gbp, numbers.Real):
        raise TypeError(f"gbp must be numeric, got {type(gbp).__name__}")
    if not isinstance(aol, numbers.Real):
        raise TypeError(f"aol must be numeric, got {type(aol).__name__}")
    if not isinstance(ccm, numbers.Real):
        raise TypeError(f"ccm must be numeric, got {type(ccm).__name__}")
    if not isinstance(cdiff, numbers.Real):
        raise TypeError(f"cdiff must be numeric, got {type(cdiff).__name__}")

    gbp_f = float(gbp)
    aol_f = float(aol)
    ccm_f = float(ccm)
    cdiff_f = float(cdiff)

    if gbp_f <= 0:
        raise ValueError(f"gbp must be positive, got {gbp_f}")
    if aol_f <= 0:
        raise ValueError(f"aol must be positive, got {aol_f}")
    if ccm_f < 0:
        raise ValueError(f"ccm must be non-negative, got {ccm_f}")
    if cdiff_f < 0:
        raise ValueError(f"cdiff must be non-negative, got {cdiff_f}")

    if db:
        aol_lin = 10 ** (aol_f / 20)
    else:
        aol_lin = aol_f

    if aol_lin <= 1:
        raise ValueError(
            f"aol (linear {aol_lin}) must be > 1 for valid omegaA; "
            "use db=True with dB value or db=False with linear value > 1"
        )

    omega_a = 2 * np.pi * gbp_f / (aol_lin - 1)

    return MappingProxyType({
        "GBP": gbp_f,
        "AOL": aol_lin,
        "omegaA": omega_a,
        "Ccm": ccm_f,
        "Cdiff": cdiff_f,
    })


# Immutable catalog of pre-defined OpAmp parameters (read-only).
OpAmp_dict: Mapping[str, Mapping[str, float]] = MappingProxyType({
    "LMH6624": set_opamp_parameters(
        gbp=1.5e9, aol=81, ccm=0.9e-12, cdiff=2.0e-12, db=True
    ),
    "OP27": set_opamp_parameters(
        gbp=8e6, aol=1.8e6, ccm=8e-12, cdiff=8e-12, db=False
    ),
})
