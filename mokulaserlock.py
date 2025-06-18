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
                C_l, # Gain reduction stage, number of bits for LeftBitShift
                Kp_db, # P-gain (dB)
                f_I=None, # First integrator crossover frequency (Hz)
                f_II=None, # Second integrator crossover frequency (Hz)
                n_reg=None, # DSP delay component (number of registers)
                off=[None],
                f_trans=None
                ):
        super().__init__(78e6)

        # Validate inputs
        assert Amp_reference > 0
        assert Amp_input > 0
        assert LPF_cutoff > 0
        assert LPF_n >= 0 and isinstance(LPF_n, int)
        assert C_l >= 0 and isinstance(C_l, int)
        if f_I is not None: assert f_I > 0
        if f_II is not None: assert f_II > 0

        self.Amp_reference = Amp_reference
        self.Amp_input = Amp_input
        self.LPF_cutoff = LPF_cutoff
        self.LPF_n = LPF_n
        self.C_l = C_l
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
            self.add_component(Component("Mixer", 78.125e6, nume=[2*np.pi*Amp_input*Amp_reference/78.125e6], deno=[1,-1]))
        if "LPF" not in off:
            self.add_component(lc.ButterworthLPFComponent("LPF", 78.125e6, LPF_cutoff, LPF_n))
        if "Gain Reduction" not in off:
            self.add_component(lc.LeftBitShiftComponent("Gain Reduction", 78.125e6, C_l))
        if "Servo" not in off:
            self.add_component(lc.MokuPIDController("Servo", 78e6, Kp_db, f_I, f_II, None, f_trans=f_trans))
        if "Delay" not in off:
            self.add_component(lc.DSPDelayComponent("Delay", 78e6, n_reg=n_reg))

        if off != [None]:
            logger.warning(f"The following components are not included in the loop {off}")

        self.Gc, self.Hc, self.Ec = self.system_transfer_components()
        self.Gf, self.Hf, self.Ef = self.system_transfer_functions()

        self.register_component_properties()

    def __deepcopy__(self, memo):
        new_obj = MokuLaserLock.__new__(MokuLaserLock)
        new_obj.__init__(self.Amp_reference,
                        self.Amp_input,
                        self.LPF_cutoff,
                        self.LPF_n,
                        self.C_l,
                        self.Kp_db,
                        self.f_I,
                        self.f_II,
                        self.n_reg,
                        self.off)
        new_obj.callbacks = self.callbacks
        return new_obj