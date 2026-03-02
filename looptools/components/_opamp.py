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


def set_opamp_parameters(GBP, AOL, Ccm, Cdiff, dB=True):
    """Generate an OpAmp parameter dictionary.

    Reference: https://www.tij.co.jp/jp/lit/an/sboa122/sboa122.pdf?ts=1662305678857

    Args:
        GBP: Gain bandwidth (Hz)
        AOL: open-loop gain (dB)
        Ccm: common-mode capacitance (F)
        Cdiff: differential capacitance (F)
    """
    if dB:
        AOL_lin = 10 ** (AOL / 20)
    else:
        AOL_lin = AOL
    omegaA = 2 * np.pi * GBP / (AOL_lin - 1)

    return {"GBP": GBP, "AOL": AOL_lin, "omegaA": omegaA, "Ccm": Ccm, "Cdiff": Cdiff}


OpAmp_dict = {
    "LMH6624": set_opamp_parameters(GBP=1.5e9, AOL=81, Ccm=0.9e-12, Cdiff=2.0e-12, dB=True),
    "OP27": set_opamp_parameters(GBP=8e6, AOL=1.8e6, Ccm=8e-12, Cdiff=8e-12, dB=False),
}
