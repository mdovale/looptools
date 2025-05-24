from looptools.dimension import Dimension
from looptools.components import *
from looptools.loop import LOOP
import logging
logger = logging.getLogger(__name__)

class MokuLaserLock(LOOP):
    """
    Feedback loop model for a Moku-based laser locking system.

    This class simulates the signal flow and control elements used in a heterodyne phase-locking setup implemented 
    with a Liquid Instruments Moku:Pro or Moku:Lab device. It leverages components from the looptools library to 
    approximate the Moku's internal signal processing pipeline.

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
    Kp : float
        Proportional gain.
    Ki : float
        Integrator gain (1st integrator).
    Kii : float
        Double-integrator gain (2nd integrator).
    Kdac : float
        DAC output gain [V/arb].
    Kplant : float
        Actuator gain [Hz/V].
    Fplant : float
        Actuator cutoff frequency [Hz].
    off : list, optional
        List of component names to skip when building the loop (e.g., ["II", "LPF"]).
    extrapolate : list, optional
        Parameters to extrapolate the controller's TF: [bool, f_trans].

    Notes
    -----
    - The current implementation uses simplified modeling:
        * Low-pass filter is an n-pass IIR stage, not a Butterworth filter.
        * Control block includes only P, I, and II terms; D gain is not implemented.
        * Saturation for I and D blocks is not yet modeled.
    - The LeftBitShiftComponent simulates the P-controller and requires a gain parameter, but Moku 
      configures the P-controller using a gain in dB. A conversion tool (TODO) is needed to translate 
      between gain in dB and a  for accurate modeling.
    - The DoubleIntegratorComponent requires gain parameters, but Moku configures integration stages using 
      cross-over frequencies. A conversion tool (TODO) is needed to translate between cross-over frequency 
      and gain for accurate modeling.
    """
    def __init__(self, sps, 
                Amp, 
                Klf, nlf,
                Kp,
                Ki, Kii,
                Kdac,
                Kplant, Fplant,
                off=[None],
                extrapolate=[False,1e2]
                ):
        super().__init__(sps)
        self.Amp = Amp
        self.Kp = Kp
        self.Klf = Klf
        self.nlf = nlf
        self.Ki = Ki
        self.Kii = Kii
        self.Kdac = Kdac
        self.Kplant = Kplant
        self.Fplant = Fplant
        self.off = off
        self.extrapolate = extrapolate

        # : === PLL components ==========
        # : Phase detector (mixer) [V/rad]
        if "Mixer" not in off:
            self.add_component(PDComponent("Mixer", self.sps, Amp))

		# : n-order IIR LPF
        # TODO: Replace with ButterworthLPF component with configurable order and cutoff
        if "LPF" not in off:
            for i in range(nlf):
                self.add_component(LPFComponent(f"LPF{i+1}", self.sps, Klf))

        # : P+II-controller
        # TODO: Replace gain-based DoubleIntegratorComponent with a frequency-based one.
        if "PII" not in off:
            self.add_component(PIIControllerComponent("PII", self.sps, Kp, Ki, Kii, extrapolate))

        # : DAC
        if "Kdac" not in off:
            self.add_component(MultiplierComponent("DAC", self.sps, Kdac, Dimension(dimensionless=True)))

        # : Laser actuator [Hz/V]
        if "Plant" not in off:
            self.add_component(ActuatorComponent("Plant", self.sps, Kplant, Fplant, Dimension(["Hz"], ["V"])))

        # : Implicit accumulator [rad/Hz]
        if "PA" not in off:
            self.add_component(ImplicitAccumulatorComponent("PA", self.sps))

        if off != [None]:
            logger.warning(f"The following components are not included in the loop {off}")
            
        self.Gc, self.Hc, self.Ec = self.system_transfer_components()
        self.Gf, self.Hf, self.Ef = self.system_transfer_functions()

        self.register_component_properties()

    def __deepcopy__(self, memo):
        new_obj = MokuLaserLock.__new__(MokuLaserLock)
        new_obj.__init__(self.sps, self.Amp, self.Klf, self.nlf, self.Kp, self.Ki, self.Kii, self.Kdac, self.Kplant, self.Fplant, self.off, self.extrapolate)
        new_obj.callbacks = self.callbacks
        return new_obj
