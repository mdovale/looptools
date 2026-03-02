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
from looptools.loop import LOOP
from looptools.component import Component
import looptools.loopmath as lm
import looptools.components as lc
from looptools.dimension import Dimension
import numpy as np
import logging
logger = logging.getLogger(__name__)

class LaserLock(LOOP):
    """
    Generic feedback loop model for a heterodyne phase-locking laser lock system.

    This base class defines the signal flow and control elements (Plant, Mixer, LPF,
    Gain, Servo, Delay) without baked-in sampling frequencies. Subclasses provide
    hardware-specific sps values.

    Parameters
    ----------
    sps : float
        Main loop sample rate (Hz), used for Servo and Delay components.
    Plant : Component
        looptools Component specifying the plant.
    Amp_reference : float
        Mixer local oscillator amplitude (Vpp).
    Amp_input : float
        Beatnote amplitude (Vpp).
    LPF_cutoff : float
        Butterworth LPF cutoff frequency (Hz).
    LPF_n : int
        Butterworth LPF number of stages.
    Cshift : int
        Gain reduction stage, number of bits for LeftBitShift.
    Kp_db : float
        P-gain (dB).
    f_I : float, optional
        First integrator crossover frequency (Hz).
    f_II : float, optional
        Second integrator crossover frequency (Hz).
    n_reg : int, optional
        DSP delay component (number of registers).
    sps_mixer : float, optional
        Sample rate for Mixer, LPF, and Gain components (Hz). Defaults to sps.
    off : list, optional
        Component names to exclude from the loop.
    f_trans : float, optional
        Transfer function extrapolation frequency for the servo.
    """

    def __init__(self,
                sps,
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
                sps_mixer=None,
                nume_mixer=None,
                off=[None],
                f_trans=None
                ):
        super().__init__(sps)
        sps_mixer = sps if sps_mixer is None else sps_mixer
        nume_mixer = [2*np.pi*Amp_input*Amp_reference/sps_mixer] if nume_mixer is None else nume_mixer # Mixer transfer function numerator (rad/s)

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
        self.sps_mixer = sps_mixer
        self.nume_mixer = nume_mixer
        self.Kp_log2 = lm.db_to_log2_gain(Kp_db)

        if f_I is not None and f_II is None:
            self.Ki_log2 = lm.gain_for_crossover_frequency(self.Kp_log2, sps, f_I, kind='I')
            self.Kii_log2 = None
        elif (f_I, f_II) != (None, None):
            self.Ki_log2, self.Kii_log2 = lm.gain_for_crossover_frequency(self.Kp_log2, sps, (f_I, f_II), kind='II')
        else:
            self.Ki_log2 = None
            self.Kii_log2 = None

        if "Plant" not in off:
            self.add_component(Plant)

        if "Mixer" not in off:
            self.add_component(Component("Mixer", sps_mixer, nume=nume_mixer, deno=[1,-1]))

        if "LPF" not in off:
            self.add_component(lc.ButterworthLPFComponent("LPF", sps_mixer, LPF_cutoff, LPF_n))

        if "Gain" not in off:
            self.add_component(lc.LeftBitShiftComponent("Gain", sps_mixer, Cshift))

        if "Servo" not in off:
            self.add_component(lc.MokuPIDController("Servo", sps, Kp_db, f_I, f_II, None, f_trans=f_trans))

        if "Delay" not in off:
            self.add_component(lc.DSPDelayComponent("Delay", sps, n_reg=n_reg))

        self.update()
        self.register_component_properties()


class MokuLaserLock(LaserLock):
    """
    Feedback loop model for a Moku-based laser locking system.

    This class simulates the signal flow and control elements used in a heterodyne phase-locking setup 
    implemented with a Liquid Instruments Moku:Pro or Moku:Lab device. It leverages components from 
    the looptools library to approximate the Moku's internal signal processing pipeline using bit-shift 
    based log₂ gain representation.

    Inherits from LaserLock with Moku-specific sampling rates: 78 MHz for the main loop
    (Servo, Delay) and 78.125 MHz for the Mixer, LPF, and Gain components.
    """

    # Moku hardware sampling rates (Hz)
    SPS = 78e6
    SPS_MIXER = 78.125e6

    def __init__(self,
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
                off=[None],
                f_trans=None
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
