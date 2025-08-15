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
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

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
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
#
from looptools.loop import LOOP
from looptools.component import Component
import looptools.loopmath as lm
import looptools.components as lc
from looptools.dimension import Dimension
import numpy as np
import logging
logger = logging.getLogger(__name__)

class MokuLaserLock(LOOP):
    """
    Feedback loop model for a Moku-based laser locking system.

    This class simulates the signal flow and control elements used in a heterodyne phase-locking setup 
    implemented with a Liquid Instruments Moku:Pro or Moku:Lab device. It leverages components from 
    the looptools library to approximate the Moku's internal signal processing pipeline using bit-shift 
    based log₂ gain representation.

    Parameters
    ----------

    """

    def __init__(self,
                Plant, # looptools Component specifying the plant
                Amp_reference, # Mixer local oscillator amplitude (Vpp)
                Amp_input, # Beatnote amplitude (Vpp)
                LPF_cutoff, # Butterworth LPF: cutoff frequency 
                LPF_n, # Butterworth LPF: number of stages
                Cshift, # Gain reduction stage, number of bits for LeftBitShift
                Kp_db, # P-gain (dB)
                f_I=None, # First integrator crossover frequency (Hz)
                f_II=None, # Second integrator crossover frequency (Hz)
                n_reg=None, # DSP delay component (number of registers)
                off=[None],
                f_trans=None
                ):
        super().__init__(78e6)

        # Validate inputs
        assert isinstance(Plant, Component)
        assert Amp_reference > 0
        assert Amp_input > 0
        assert LPF_cutoff > 0
        assert LPF_n >= 0 and isinstance(LPF_n, int)
        assert Cshift >= 0 and isinstance(Cshift, int)
        if f_I is not None: assert f_I > 0
        if f_II is not None: assert f_II > 0

        self.Plant = Plant
        self.Amp_reference = Amp_reference
        self.Amp_input = Amp_input
        self.LPF_cutoff = LPF_cutoff
        self.LPF_n = LPF_n
        self.Cshift = Cshift
        self.Kp_db = Kp_db
        self.f_I = f_I
        self.f_II = f_II
        self.n_reg = n_reg
        self.off = off

        self.Kp_log2 = lm.db_to_log2_gain(Kp_db)

        if f_I is not None and f_II is None:
            self.Ki_log2 = lm.gain_for_crossover_frequency(self.Kp_log2, 78e6, f_I, kind='I')
            self.Kii_log2 = None
        elif (f_I, f_II) != (None, None):
            self.Ki_log2, self.Kii_log2 = lm.gain_for_crossover_frequency(self.Kp_log2, 78e6, (f_I, f_II), kind='II')
        else:
            self.Ki_log2 = None
            self.Kii_log2 = None

        if "Plant" not in off:
            self.add_component(Plant)

        if "Mixer" not in off:
            self.add_component(Component("Mixer", 78.125e6, nume=[1.25*2*np.pi*Amp_input*Amp_reference/78.125e6], deno=[1,-1]))

        if "LPF" not in off:
            self.add_component(lc.ButterworthLPFComponent("LPF", 78.125e6, LPF_cutoff, LPF_n))

        if "Gain" not in off:
            self.add_component(lc.LeftBitShiftComponent("Gain", 78.125e6, Cshift))

        if "Servo" not in off:
            self.add_component(lc.MokuPIDController("Servo", 78e6, Kp_db, f_I, f_II, None, f_trans=f_trans))

        if "Delay" not in off:
            self.add_component(lc.DSPDelayComponent("Delay", 78e6, n_reg=n_reg))

        self.update()
        self.register_component_properties()

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