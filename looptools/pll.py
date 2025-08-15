from looptools.components import *
from looptools.loop import LOOP
import logging
logger = logging.getLogger(__name__)


class PLL(LOOP):
    def __init__(self, sps, Amp, Cshift, Klf, Kp, Ki, twostages=True, n_reg=10, but=[None], name=None):
        """
        Phase-Locked Loop (PLL) simulation loop subclass of LOOP.

        This class models a digital PLL system using a chain of configurable components such as a phase detector, low-pass filter,
        gain, PI controller, and output driver stages. This model is designed to match common FPGA/DSP-based implementations.

        Parameters
        ----------
        sps : float
            System sampling frequency in Hz.
        Amp : float
            Normalized input signal amplitude, defined as Vpk / Vpp_adc.
        Cshift : int
            Number of bit shifts applied in the gain stage (left shift).
        Klf : float
            Loop filter gain (typically normalized).
        Kp : int
            Proportional gain of PI controller, interpreted as bit shifts.
        Ki : int
            Integral gain of PI controller, interpreted as bit shifts.
        twostages : bool, optional
            If True, use a two-stage IIR loop filter (default is True).
        n_reg : int, optional
            Number of DSP delay registers to insert (default is 10).
        but : list of str or [None], optional
            List of component names to exclude from the loop. Default is [None].
        """
        super().__init__(sps)
        self.name = name or 'PLL'
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
        if "LPF" not in but:
            if twostages:
                self.add_component(TwoStageLPFComponent("LPF", self.sps, Klf))
            else:
                self.add_component(LPFComponent("LPF", self.sps, Klf))

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

        self.update()
        self.register_component_properties()
        
    def __deepcopy__(self, memo):
        new_obj = PLL.__new__(PLL)
        new_obj.__init__(self.sps, self.Amp, self.Cshift, self.Klf, self.Kp, self.Ki, self.twostages, self.n_reg, self.but)
        new_obj.callbacks = self.callbacks
        return new_obj

    def show_all_te(self):
        """
        Display the transfer elements of all components in the loop.

        This method prints the internal transfer element (TE) representation of
        each component, as well as the overall open-loop forward path (Gc),
        feedback path (Hc), and error function (Ec).
        """
        for comp in self.components_dict:
            print(f"=== transfer function of "+self.components_dict[comp].name+f" === {self.components_dict[comp].TE}")
            print(f"=== transfer function of G === {self.Gc.TE}")
            print(f"=== transfer function of H === {self.Hc.TE}")
            print(f"=== transfer function of E === {self.Ec.TE}")

    def point_to_point_component(self, _from, _to=None, suppression=False, view=False):
        """
        Compute the transfer element between two components in the loop.

        Parameters
        ----------
        _from : str
            Name of the starting component.
        _to : str, optional
            Name of the stopping component. This component is *not* included.
        suppression : bool, optional
            If True, apply loop suppression factor 1 / (1 + G). Default is False.
        view : bool, optional
            If True, print details about the path and resulting transfer element.

        Returns
        -------
        TransferElement
            Transfer element between the specified components.
        """
        component = super().point_to_point_component(_from, _to, suppression, view)
        return component

    def point_to_point_tf(self, f, _from, _to=None, suppression=False, view=False):
        """
        Compute the frequency response (transfer function) between two loop components.

        Parameters
        ----------
        f : ndarray
            Array of Fourier frequencies in Hz.
        _from : str
            Name of the starting component.
        _to : str, optional
            Name of the stopping component. This component is *not* included.
        suppression : bool, optional
            If True, apply loop suppression factor 1 / (1 + G). Default is False.
        view : bool, optional
            If True, print details about the path and resulting transfer function.

        Returns
        -------
        ndarray
            Complex frequency response of the transfer function between components.
        """

        tf = super().point_to_point_tf(f, _from, _to, suppression, view)
        return tf