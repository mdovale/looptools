from looptools.dimension import Dimension
from looptools.components import *
from looptools.loop import LOOP
from looptools.loopmath import db_to_log2_gain, gain_for_crossover_frequency
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
    sps : float
        Sampling rate [Hz].
    Amp : float
        Normalized input signal amplitude, defined as Vpk / Vpp_adc.
    Klf : float
        Gain per pass of the low-pass filter stage.
    nlf : int
        Number of IIR LPF passes.
    Kp_db : float
        Proportional gain in decibels (dB). Translated to log₂(Kp) internally.
    on_I : bool
        Enable Integrator #1.
    on_II : bool
        Enable Integrator #2.
    f_I : float
        Crossover frequency [Hz] for integrator #1.
    f_II : float
        Crossover frequency [Hz] for integrator #2.
    Kdac : float
        DAC output gain [V/arb].
    Kplant : float
        Actuator gain [Hz/V].
    Fplant : float
        Actuator cutoff frequency [Hz].
    off : list of str, optional
        List of component names to skip when building the loop (e.g., ["II", "LPF"]).
    f_trans : float or None, optional
        if float, perform extrapolation with that transition frequency.

    Attributes
    ----------
    Kp_db : float
        Externally visible proportional gain in dB.
    f_I : float
        Integrator crossover frequency [Hz].
    f_II : float
        Double integrator crossover frequency [Hz].
    Ki_db, Kii_db : float
        Computed integrator gains in dB.
    """

    def __init__(self, sps, 
                Amp, 
                Klf, nlf,
                Kp_db, 
                f_I, f_II,
                Kdac,
                Kplant, Fplant,
                off=[None],
                f_trans=None
                ):
        super().__init__(sps)

        # Validate inputs
        assert sps > 0
        assert Amp > 0
        assert Klf > 0
        assert nlf >= 0 and isinstance(nlf, int)
        if f_I is not None: assert f_I > 0
        if f_II is not None: assert f_II > 0

        self.Amp = Amp
        self.Klf = Klf
        self.nlf = nlf
        self.Kdac = Kdac
        self.Kplant = Kplant
        self.Fplant = Fplant
        self.off = off
        self.f_trans = f_trans

        # Store external API inputs
        self._Kp_db = Kp_db
        self._f_I = f_I
        self._f_II = f_II

        self.Kp_log2 = db_to_log2_gain(Kp_db)
        self.Ki_log2 = gain_for_crossover_frequency(self.Kp_log2, sps, f_I, kind='I', structure='mul')
        if f_II is not None:
            self.Kii_log2 = gain_for_crossover_frequency(self.Kp_log2, sps, f_II, kind='II', structure='mul')
        else:
            self.Kii_log2 = None

        # Add components
        if "Mixer" not in off:
            self.add_component(PDComponent("Mixer", sps, Amp))
        if "LPF" not in off:
            for i in range(nlf):
                self.add_component(LPFComponent(f"LPF{i+1}", sps, Klf))
        if "PI" not in off:
            self.add_component(MokuPIDController("PI", sps, Kp_db, f_I, f_II, None, f_trans=f_trans))
        if "DAC" not in off:
            self.add_component(MultiplierComponent("DAC", sps, Kdac, Dimension(dimensionless=True)))
        if "Plant" not in off:
            self.add_component(ActuatorComponent("Plant", sps, Kplant, Fplant, Dimension(["Hz"], ["V"])))
        if "PA" not in off:
            self.add_component(ImplicitAccumulatorComponent("PA", sps))

        if off != [None]:
            logger.warning(f"The following components are not included in the loop {off}")

        self.Gc, self.Hc, self.Ec = self.system_transfer_components()
        self.Gf, self.Hf, self.Ef = self.system_transfer_functions()

        self.register_component_properties()

    def __deepcopy__(self, memo):
        new_obj = MokuLaserLock.__new__(MokuLaserLock)
        new_obj.__init__(self.sps, self.Amp, self.Klf, self.nlf, self._Kp_db, self.on_I, self.on_II, self._f_I, self._f_II, self.Kdac, self.Kplant, self.Fplant, self.off, self.extrapolate)
        new_obj.callbacks = self.callbacks
        return new_obj

    # ======= Dynamic controller parameters (user-facing) =======

    @property
    def Kp_db(self):
        return self._Kp_db

    @Kp_db.setter
    def Kp_db(self, value):
        self._Kp_db = value
        self.Kp_log2 = db_to_log2_gain(value)
        if "PI" in self.components_dict:
            self.components_dict["PI"].P_dB = value

    @property
    def f_I(self):
        return self._f_I

    @f_I.setter
    def f_I(self, value):
        self._f_I = value
        if "PI" in self.components_dict:
            self.components_dict["PI"].w1 = value

    @property
    def f_II(self):
        return self._f_II

    @f_II.setter
    def f_II(self, value):
        self._f_II = value
        if "PI" in self.components_dict:
            self.components_dict["PI"].w2 = value

    @property
    def Ki_db(self):
        return 20 * np.log10(2 ** gain_for_crossover_frequency(self.Kp_log2, self.sps, self.f_I, kind='I', structure='mul'))

    @property
    def Kii_db(self):
        return 20 * np.log10(2 ** gain_for_crossover_frequency(self.Kp_log2, self.sps, self.f_II, kind='II', structure='mul'))