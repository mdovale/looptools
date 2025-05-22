from looptools.components import *
from looptools.loop import LOOP
import logging
logger = logging.getLogger(__name__)


class PLL(LOOP):
    def __init__(self, sps, Amp, Cshift, Klf, Kp, Ki, twostages=True, n_reg=10, but=[None]):
        """ class to simulate PLL
        Args:
            sps: system clock frequency (Hz)
            Amp: normalized input signal amplitude, i.e. Vpk/Vpp,adc
            Cshift: number of bit shifts for Gain C
            Klf: gain of Low-Pass filter
            Kp: P gain (WARNING: number of bit shifts)
            Ki: I gain (WARNING: number of bit shifts)
            twostages: use two stages for IIR LF or not
            n_reg: number of registers in a loop
            but: list of components' names to be omitted from the loop
        """
        super().__init__(sps)
        self.but = but
        self.twostages = twostages
        self.Amp = Amp
        self.Cshift = Cshift 
        self.Klf = Klf
        self.Kp = Kp
        self.Ki = Ki
        self.n_reg = n_reg

        # : === PLL components ==========
        # : Phase detector
        if "PD" not in but:
            self.add_component(PDComponent("PD", self.sps, Amp))

		# : IIR LF
        if "LF" not in but:
            if twostages:
                self.add_component(TwoStageLFComponent("LF", self.sps, Klf))
            else:
                self.add_component(LFComponent("LF", self.sps, Klf))

        # : Gain
        if "Gain" not in but:
            self.add_component(LeftBitShiftComponent("Gain", self.sps, Cshift))

        # : PI controller
        if "PI" not in but:
            self.add_component(PIControllerComponent("PI", self.sps, Kp, Ki))

        # : PA
        if "PA" not in but:
            self.add_component(PAComponent("PA", self.sps))

        # : LUT
        if "LUT" not in but:
            self.add_component(LUTComponent("LUT", self.sps))

        # : DSP delay (= register delay)
        if "DSP" not in but:
            self.add_component(DSPDelayComponent("DSP", self.sps, n_reg))
            
        if but != [None]:
            logging.warning(f"The following components are not included in the loop {but}")

        #self.components = [self.PD, self.LF, self.Gain, self.PI, self.PA, self.LUT, self.DSP]
        self.Gc, self.Hc, self.Ec = self.system_transfer_components()
        self.Gf, self.Hf, self.Ef = self.system_transfer_functions()

        # : === miscellany ============
        # self.loop_simulator = plln.PLLnonlin(self)

        self.register_component_properties()
        
    def __deepcopy__(self, memo):
        new_obj = PLL.__new__(PLL)
        new_obj.__init__(self.sps, self.Amp, self.Cshift, self.Klf, self.Kp, self.Ki, self.twostages, self.n_reg, self.but)
        new_obj.callbacks = self.callbacks
        return new_obj

    def show_all_te(self):
        """ Display all transfer elements
        Args:
        """
        for comp in self.components_dict:
            print(f"=== transfer function of "+self.components_dict[comp].name+f" === {self.components_dict[comp].TE}")
            print(f"=== transfer function of G === {self.Gc.TE}")
            print(f"=== transfer function of H === {self.Hc.TE}")
            print(f"=== transfer function of E === {self.Ec.TE}")

    def point_to_point_component(self, _from, _to=None, suppression=False, view=False):
        """ compute a point-to-point loop component
        Args:
            _from: a staring PLL component [str]
            _to: a stopping PLL component (this component is NOT included) [str]
            suppression: suppression by 1/1+G or not 
            view: print information or not
        """

        component = super().point_to_point_component(_from, _to, suppression, view)
        return component

    def point_to_point_tf(self, f, _from, _to=None, suppression=False, view=False):
        """ compute a point-to-point loop transfer function
        Args:
            f: fourier frequencies (Hz)
            _from: a staring PLL component [str]
            _to: a stopping PLL component (this component is NOT included) [str]
            suppression: suppression by 1/1+G or not 
            view: print information or not
        """

        tf = super().point_to_point_tf(f, _from, _to, suppression, view)
        return tf