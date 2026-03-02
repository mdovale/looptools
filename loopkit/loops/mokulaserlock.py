# BSD 3-Clause License

# Copyright (c) 2025, Miguel Dovale

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.

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

# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting this software to
# foreign countries or providing access to foreign persons.
#
from typing import Optional, Sequence

import numpy as np

from loopkit.loops.laserlock import LaserLock


class MokuLaserLock(LaserLock):
    """
    Feedback loop model for a Moku-based laser locking system.

    This class simulates the signal flow and control elements used in a heterodyne phase-locking setup 
    implemented with a Liquid Instruments Moku:Pro or Moku:Lab device. It leverages components from 
    the loopkit library to approximate the Moku's internal signal processing pipeline using bit-shift 
    based log₂ gain representation.

    Inherits from LaserLock with Moku-specific sampling rates: 78 MHz for the main loop
    (Servo, Delay) and 78.125 MHz for the Mixer, LPF, and Gain components.
    """

    # Moku hardware sampling rates (Hz)
    SPS = 78e6
    SPS_MIXER = 78.125e6

    def __init__(
        self,
        Plant,
        Amp_reference,
        Amp_input,
        LPF_cutoff,
        LPF_n,
        Cshift,
        Kp_db,
        f_I=None,
        f_II=None,
        n_reg=None,
        off: Optional[Sequence[str]] = None,
        f_trans=None,
    ):
        super().__init__(
            sps=MokuLaserLock.SPS,
            Plant=Plant,
            Amp_reference=Amp_reference,
            Amp_input=Amp_input,
            LPF_cutoff=LPF_cutoff,
            LPF_n=LPF_n,
            Cshift=Cshift,
            Kp_db=Kp_db,
            f_I=f_I,
            f_II=f_II,
            n_reg=n_reg,
            sps_mixer=MokuLaserLock.SPS_MIXER,
            nume_mixer=[1.25*2*np.pi*Amp_input*Amp_reference/MokuLaserLock.SPS_MIXER], # Mixer transfer function numerator (rad/s)
            off=off,
            f_trans=f_trans
        )

    def __deepcopy__(self, memo):
        new_obj = MokuLaserLock.__new__(MokuLaserLock)
        new_obj.__init__(
                        self.Plant,
                        self.Amp_reference,
                        self.Amp_input,
                        self.LPF_cutoff,
                        self.LPF_n,
                        self.Cshift,
                        self.Kp_db,
                        self.f_I,
                        self.f_II,
                        self.n_reg,
                        self.off)
        new_obj.callbacks = self.callbacks
        return new_obj
