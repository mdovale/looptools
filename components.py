import numpy as np
from functools import partial
from looptools.component import Component
from looptools.dimension import Dimension
from looptools import auxiliary as aux
import logging
logger = logging.getLogger(__name__)


def set_opamp_parameters(GBP, AOL, Ccm, Cdiff, dB=True):
	""" Generate an OpAmp parameter dictionary

    Reference: https://www.tij.co.jp/jp/lit/an/sboa122/sboa122.pdf?ts=1662305678857

	Args:
		GBP: Gain bandwidth (Hz)
		AOL: open-loop gain (dB)
		Ccm: common-mode capacitance (F)
		Cdiff: differntial capacitance (F)
	"""

	if dB:
		AOL_lin = 10**(AOL/20)
	else:
		AOL_lin = AOL
	omegaA = 2 * np.pi * GBP / (AOL_lin - 1)

	return {"GBP": GBP, "AOL": AOL_lin, "omegaA": omegaA, "Ccm": Ccm, "Cdiff": Cdiff}

# : Dictionary of OpAmps
OpAmp_dict = {
	"LMH6624": set_opamp_parameters(GBP=1.5e9, AOL=81, Ccm=0.9e-12, Cdiff=2.0e-12, dB=True), # https://www.ti.com/lit/ds/symlink/lmh6624.pdf
	"OP27": set_opamp_parameters(GBP=8e6, AOL=1.8e6, Ccm=8e-12, Cdiff=8e-12, dB=False), # https://www.analog.com/media/en/technical-documentation/data-sheets/op27.pdf
}

class PDComponent(Component):
    """
    Phase Detector (PD) component for phase-locked loop simulations.

    This component models the behavior of a digital phase detector, which mixes
    an input signal with a numerically controlled oscillator (NCO) in a PLL system.
    The output of the phase detector is proportional to the phase difference 
    between its input and reference signals.

    The internal transfer function is a static gain element with a value derived 
    from the amplitude (`Amp`), initialized as `Amp / 4.0`. This scaling factor 
    reflects typical digital mixing gain behavior.

    Parameters
    ----------
    name : str
        Name of the component.
    sps : float
        Sample rate in Hz.
    Amp : float
        Peak amplitude of the input signal; determines phase detector gain.

    Attributes
    ----------
    Amp : float
        Input amplitude used to compute the transfer gain.
    ival : float
        Internal gain value, computed as Amp / 4.0.
    """
    def __init__(self, name, sps, Amp):
        self._Amp = Amp
        self._ival = Amp/4.0
        super().__init__(name, sps, np.array([self._ival]), np.array([1.0]), unit=Dimension(dimensionless=True))
        self.properties = {'Amp': (lambda self=self: self.Amp, lambda value, self=self: setattr(self, 'Amp', value)),
                           'ival': (lambda self=self: self.ival, lambda value, self=self: setattr(self, 'ival', value))}
        
    def __deepcopy__(self, memo):
        new_obj = PDComponent.__new__(PDComponent)
        new_obj.__init__(self.name, self.sps, self._Amp)
        if getattr(self, '_loop', None) != None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Amp(self):
        return self._Amp

    @Amp.setter
    def Amp(self, value):
        self._Amp = float(value)
        self._ival = self._Amp/4.0
        self.update_component()

    @property
    def ival(self):
        return self._ival

    @ival.setter
    def ival(self, value):
        self._ival = float(value)
        self._Amp = 4.0*self._ival
        self.update_component()

    def update_component(self):
        super().__init__(self.name, self.sps, np.array([self._ival]), np.array([1.0]), unit=Dimension(dimensionless=True))


class MultiplierComponent(Component):
    """
    Static gain multiplier component.

    Parameters
    ----------
    name : str
        Name of the component.
    sps : float
        Sample rate in Hz.
    gain : float
        Gain value.
    unit : Dimension
        Dimensional unit of the signal after multiplication.

    Attributes
    ----------
    gain : float
        Gain applied to the input signal.
    """
    def __init__(self, name, sps, gain, unit):
        self._gain = gain
        super().__init__(name, sps, np.array([self._gain]), np.array([1.0]), unit=unit)
        self.properties = {'gain': (lambda: self.gain, lambda value: setattr(self, 'gain', value))}

    def __deepcopy__(self, memo):
        new_obj = MultiplierComponent.__new__(MultiplierComponent)
        new_obj.__init__(self.name, self.sps, self._gain, self.unit)
        if getattr(self, '_loop', None) != None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        self._gain = value
        self.update_component()

    def update_component(self):
        super().__init__(self.name, self.sps, np.array([self._gain]), np.array([1.0]), unit=self.unit)


class LeftBitShiftComponent(Component):
    """
    Simulates a left bit-shift operation (*2^Cshift).

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    Cshift : int or float
        Shift value; actual gain is 2^(-Cshift).

    Attributes
    ----------
    Cshift : float
        Exponent of the power-of-two shift.
    """
    def __init__(self, name, sps, Cshift):
        self._Cshift = 2.0**float(-Cshift)
        super().__init__(name, sps, np.array([self._Cshift]), np.array([1.0]), unit=Dimension(dimensionless=True))
        self.properties = {'Cshift': (lambda: self.Cshift, lambda value: setattr(self, 'Cshift', value))}

    def __deepcopy__(self, memo):
        new_obj = LeftBitShiftComponent.__new__(LeftBitShiftComponent)
        new_obj.__init__(self.name, self.sps, np.log2(self._Cshift))
        if getattr(self, '_loop', None) != None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Cshift(self):
        return self._Cshift

    @Cshift.setter
    def Cshift(self, value):
        self._Cshift = 2.0**float(-value)
        self.update_component()

    def update_component(self):
        super().__init__(self.name, self.sps, np.array([self._Cshift]), np.array([1.0]), unit=Dimension(dimensionless=True))
        

class LPFComponent(Component):
    """
    Low pass filter component (first-order IIR).

    Models a low-pass loop filter with tunable gain.

    Parameters
    ----------
    name : str
        Name of the component.
    sps : float
        Sample rate in Hz.
    Klf : float
        Log2 representation of loop gain (gain = 2^-Klf).

    Attributes
    ----------
    Klf : float
        Filter gain as 2^-Klf.
    """
    def __init__(self, name, sps, Klf):
        self._Klf = 2.0**float(-Klf)
        super().__init__(name, sps, np.array([self._Klf]), np.array([1.0, -(1.0 - self._Klf)]), unit=Dimension(dimensionless=True))
        self.properties = {'Klf': (lambda: self.Klf, lambda value: setattr(self, 'Klf', value))}

    def __deepcopy__(self, memo):
        new_obj = LPFComponent.__new__(LPFComponent)
        new_obj.__init__(self.name, self.sps, -np.log2(self._Klf))
        if getattr(self, '_loop', None) != None:
            new_obj._loop = self._loop
        return new_obj
        
    @property
    def Klf(self):
        return self._Klf

    @Klf.setter
    def Klf(self, value):
        self._Klf = 2**float(-value)
        self.update_component()

    def update_component(self):
        super().__init__(self.name, self.sps, np.array([self._Klf]), np.array([1.0, -(1.0-self._Klf)]), unit=Dimension(dimensionless=True))


class TwoStageLPFComponent(Component):
    """
    Cascaded low pass filter with two identical first-order stages.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    Klf : float
        Log2 representation of gain.

    Attributes
    ----------
    Klf : float
        Effective loop filter gain (applied twice in series).
    """
    def __init__(self, name, sps, Klf):
        self._Klf = 2**float(-Klf)
        LF = Component("LPF", sps, np.array([self._Klf]), np.array([1.0, -(1.0 - self._Klf)]), unit=Dimension(dimensionless=True))
        LF = LF*LF
        super().__init__(name, sps, LF.nume, LF.deno, unit=LF.unit)
        self.TE = LF.TE
        self.TE.name = name
        self.TF = LF.TF
        self.properties = {'Klf': (lambda: self.Klf, lambda value: setattr(self, 'Klf', value))}

    def __deepcopy__(self, memo):
        new_obj = TwoStageLPFComponent.__new__(TwoStageLPFComponent)
        new_obj.__init__(self.name, self.sps, -np.log2(self._Klf))
        if getattr(self, '_loop', None) != None:
            new_obj._loop = self._loop
        return new_obj
        
    @property
    def Klf(self):
        return self._Klf

    @Klf.setter
    def Klf(self, value):
        self._Klf = 2**float(-value)
        self.update_component()

    def update_component(self):
        LF = Component("LPF", self.sps, np.array([self._Klf]), np.array([1.0, -(1.0 - self._Klf)]), unit=Dimension(dimensionless=True))
        LF = LF*LF
        super().__init__(self.name, self.sps, LF.nume, LF.deno, unit=LF.unit)
        self.TE = LF.TE
        self.TE.name = self.name
        self.TF = LF.TF


class PIControllerComponent(Component):
    """
    Proportional-Integral controller component.

    Combines proportional and integral actions into a PI controller with
    bit-shift-based tunable gain.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    Kp : float
        Proportional gain as log2(Kp).
    Ki : float
        Integral gain as log2(Ki).

    Attributes
    ----------
    Kp : float
        Proportional gain.
    Ki : float
        Integral gain.
    """
    def __init__(self, name, sps, Kp, Ki):
        self._Kp = 2**float(Kp)  # convert a bit shift to gain
        self._Ki = 2**float(Ki)  # convert a bit shift to gain
        P = Component("P", sps, np.array([self._Kp]), np.array([1.0]), unit=Dimension(["cycle"], ["s","rad"]))
        I = Component("I", sps, np.array([self._Ki]), np.array([1.0, -1.0]), unit=Dimension(["cycle"], ["s","rad"]))
        PI = P + I
        super().__init__(name, sps, PI.nume, PI.deno, unit=PI.unit)
        self.properties = {'Kp': (lambda: self.Kp, lambda value: setattr(self, 'Kp', value)),
                           'Ki': (lambda: self.Ki, lambda value: setattr(self, 'Ki', value))}
        
    def __deepcopy__(self, memo):
        new_obj = PIControllerComponent.__new__(PIControllerComponent)
        new_obj.__init__(self.name, self.sps, np.log2(self._Kp), np.log2(self._Ki))
        if getattr(self, '_loop', None) != None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Kp(self):
        return self._Kp

    @Kp.setter
    def Kp(self, value):
        self._Kp = 2**float(value)
        self.update_component()

    @property
    def Ki(self):
        return self._Ki

    @Ki.setter
    def Ki(self, value):
        self._Ki = 2**float(value)
        self.update_component()

    def update_component(self):
        P = Component("P", self.sps, np.array([self._Kp]), np.array([1.0]), unit=Dimension(["cycle"], ["s","rad"]))
        I = Component("I", self.sps, np.array([self._Ki]), np.array([1.0, -1.0]), unit=Dimension(["cycle"], ["s","rad"]))
        PI = P + I
        super().__init__(self.name, self.sps, PI.nume, PI.deno, unit=PI.unit)


class DoubleIntegratorComponent(Component):
    """
    Second-order integrator.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    Ki : float
        Gain of first integrator (log2 scale).
    Kii : float
        Gain of second integrator (log2 scale).
    extrapolate : tuple(bool, float)
        (Enable extrapolation, transition frequency)

    Attributes
    ----------
    Ki : float
        Gain of the first integrator.
    Kii : float
        Gain of the second integrator.
    """
    def __init__(self, name, sps, Ki, Kii, extrapolate):
        self.extrapolate = extrapolate
        self._Ki = 2**float(Ki)
        self._Kii = 2**float(Kii)
        I = Component("I", sps, np.array([self._Ki]), np.array([1.0, -1.0]), unit=Dimension(dimensionless=True))
        II = Component("II", sps, np.array([self._Kii]), np.array([1.0, -2.0, 1.0]), unit=Dimension(dimensionless=True))
        II.TF = partial(II.TF, extrapolate=self.extrapolate[0], f_trans=self.extrapolate[1], power=-2) # avoid numerical errors
        DoubleI = I + II
        super().__init__(name, sps, DoubleI.nume, DoubleI.deno, unit=DoubleI.unit)
        self.TE = DoubleI.TE
        self.TE.name = name
        self.TF = partial(aux.add_transfer_function, tf1=I.TF, tf2=II.TF)
        self.properties = {'Ki': (lambda: self.Ki, lambda value: setattr(self, 'Ki', value)),
                           'Kii': (lambda: self.Kii, lambda value: setattr(self, 'Kii', value))}
        
    def __deepcopy__(self, memo):
        new_obj = DoubleIntegratorComponent.__new__(DoubleIntegratorComponent)
        new_obj.__init__(self.name, self.sps, np.log2(self._Ki), np.log2(self._Kii), self.extrapolate)
        if getattr(self, '_loop', None) != None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Ki(self):
        return self._Ki

    @Ki.setter
    def Ki(self, value):
        self._Ki = 2**(value)
        self.update_component()

    @property
    def Kii(self):
        return self._Kii

    @Kii.setter
    def Kii(self, value):
        self._Kii = 2**(value)
        self.update_component()

    def update_component(self):
        I = Component("I", self.sps, np.array([self._Ki]), np.array([1.0, -1.0]), unit=Dimension(dimensionless=True))
        II = Component("II", self.sps, np.array([self._Kii]), np.array([1.0, -2.0, 1.0]), unit=Dimension(dimensionless=True))
        II.TF = partial(II.TF, extrapolate=self.extrapolate[0], f_trans=self.extrapolate[1], power=-2) # avoid numerical errors
        DoubleI = I + II
        super().__init__(self.name, self.sps, DoubleI.nume, DoubleI.deno, unit=DoubleI.unit)
        self.TE = DoubleI.TE
        self.TE.name = self.name
        self.TF = partial(aux.add_transfer_function, tf1=I.TF, tf2=II.TF)


class PIIControllerComponent(Component):
    """
    Proportional + Integrator + Double Integrator controller component.

    This component models a control law consisting of:
        - A proportional term (P)
        - A first-order integrator (I)
        - A second-order integrator (II)

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    Kp : float
        Proportional gain (log₂ scale).
    Ki : float
        First integrator gain (log₂ scale).
    Kii : float
        Second integrator gain (log₂ scale).
    extrapolate : tuple(bool, float)
        Tuple (enable_extrapolation, transition_frequency) for the double integrator.

    Attributes
    ----------
    Kp : float
        Proportional gain.
    Ki : float
        First integrator gain.
    Kii : float
        Second integrator gain.
    """
    def __init__(self, name, sps, Kp, Ki, Kii, extrapolate=(False, 1e2)):
        self.sps = sps
        self.extrapolate = extrapolate
        self._Kp = 2**float(Kp)
        self._Ki = 2**float(Ki)
        self._Kii = 2**float(Kii)

        # Create the individual components
        P = Component("P", sps, np.array([self._Kp]), np.array([1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        I = Component("I", sps, np.array([self._Ki]), np.array([1.0, -1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        II = Component("II", sps, np.array([self._Kii]), np.array([1.0, -2.0, 1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        II.TF = partial(II.TF, extrapolate=self.extrapolate[0], f_trans=self.extrapolate[1], power=-2)

        PII = P + I + II
        super().__init__(name, sps, PII.nume, PII.deno, unit=PII.unit)

        self.TE = PII.TE
        self.TE.name = name
        self.TF = partial(aux.add_transfer_function, tf1=P.TF, tf2=partial(aux.add_transfer_function, tf1=I.TF, tf2=II.TF))

        self.properties = {
            'Kp': (lambda: self.Kp, lambda value: setattr(self, 'Kp', value)),
            'Ki': (lambda: self.Ki, lambda value: setattr(self, 'Ki', value)),
            'Kii': (lambda: self.Kii, lambda value: setattr(self, 'Kii', value)),
        }

    def __deepcopy__(self, memo):
        new_obj = PIIControllerComponent.__new__(PIIControllerComponent)
        new_obj.__init__(self.name, self.sps, np.log2(self._Kp), np.log2(self._Ki), np.log2(self._Kii), self.extrapolate)
        if getattr(self, '_loop', None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Kp(self):
        return self._Kp

    @Kp.setter
    def Kp(self, value):
        self._Kp = 2**float(value)
        self.update_component()

    @property
    def Ki(self):
        return self._Ki

    @Ki.setter
    def Ki(self, value):
        self._Ki = 2**float(value)
        self.update_component()

    @property
    def Kii(self):
        return self._Kii

    @Kii.setter
    def Kii(self, value):
        self._Kii = 2**float(value)
        self.update_component()

    def update_component(self):
        P = Component("P", self.sps, np.array([self._Kp]), np.array([1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        I = Component("I", self.sps, np.array([self._Ki]), np.array([1.0, -1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        II = Component("II", self.sps, np.array([self._Kii]), np.array([1.0, -2.0, 1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        II.TF = partial(II.TF, extrapolate=self.extrapolate[0], f_trans=self.extrapolate[1], power=-2)
        PII = P + I + II
        super().__init__(self.name, self.sps, PII.nume, PII.deno, unit=PII.unit)
        self.TE = PII.TE
        self.TE.name = self.name
        self.TF = partial(aux.add_transfer_function, tf1=P.TF, tf2=partial(aux.add_transfer_function, tf1=I.TF, tf2=II.TF))


class PAComponent(Component):
    """
    Phase accumulator.

    Implements a pure integrator (I(z) = 1 / (1 - z⁻¹)).

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    """
    def __init__(self, name, sps):
        super().__init__(name, sps, np.array([1.0]), np.array([1.0, -1.0]), unit=Dimension(["s"], []))

    def __deepcopy__(self, memo):
        new_obj = PAComponent.__new__(PAComponent)
        new_obj.__init__(self.name, self.sps)
        if getattr(self, '_loop', None) != None:
            new_obj._loop = self._loop
        return new_obj


class LUTComponent(Component):
    """
    Lookup table phase converter.

    Converts digital phase to analog signal (rad ↔ cycle).

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    """
    def __init__(self, name, sps):
        super().__init__(name, sps, np.array([2.0*np.pi]), np.array([1.0]), unit=Dimension(["rad"], ["cycle"]))

    def __deepcopy__(self, memo):
        new_obj = LUTComponent.__new__(LUTComponent)
        new_obj.__init__(self.name, self.sps)
        if getattr(self, '_loop', None) != None:
            new_obj._loop = self._loop
        return new_obj


class DSPDelayComponent(Component):
    """
    Discrete pipeline delay component.

    Implements delay through register depth `n_reg`.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    n_reg : int
        Number of DSP registers (delay in samples).

    Attributes
    ----------
    n_reg : int
        Length of the pipeline delay.
    """
    def __init__(self, name, sps, n_reg):
        self._n_reg = int(n_reg)
        DSP_denom = np.zeros(self._n_reg+1)
        DSP_denom[0] = 1.0
        super().__init__(name, sps, np.array([1.0]), DSP_denom, unit=Dimension(dimensionless=True))
        self.properties = {'n_reg': (lambda: self.n_reg, lambda value: setattr(self, 'n_reg', value))}

    def __deepcopy__(self, memo):
        new_obj = DSPDelayComponent.__new__(DSPDelayComponent)
        new_obj.__init__(self.name, self.sps, self._n_reg)
        if getattr(self, '_loop', None) != None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def n_reg(self):
        return self._n_reg

    @n_reg.setter
    def n_reg(self, value):
        self._n_reg = int(value)
        self.update_component()

    def update_component(self):
        DSP_denom = np.zeros(self._n_reg+1)
        DSP_denom[0] = 1
        super().__init__(self.name, self.sps, np.array([1.0]), DSP_denom, unit=Dimension(dimensionless=True))


class ActuatorComponent(Component):
    """
    PZT actuator model with gain and cutoff frequency.

    Converts s-domain coefficients into z-domain using aux.polynomial_conversion_s_to_z.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    Ka_pzt : float
        Actuator gain.
    Fa_pzt : float
        Actuator cutoff frequency (Hz).
    unit : Dimension
        Dimensional unit of the actuator.

    Attributes
    ----------
    Ka_pzt : float
        Gain.
    Fa_pzt : float
        Cutoff frequency.
    """
    def __init__(self, name, sps, Ka_pzt, Fa_pzt, unit):
        self._Fa_pzt = Fa_pzt
        self._Ka_pzt = Ka_pzt
        nume = aux.polynomial_conversion_s_to_z(np.array([self._Ka_pzt]), sps)
        deno = aux.polynomial_conversion_s_to_z(np.array([1.0/(2.0*np.pi*self._Fa_pzt), 1.0]), sps)
        super().__init__(name, sps, nume, deno, unit=unit)
        self.properties = {'Ka_pzt': (lambda: self.Ka_pzt, lambda value: setattr(self, 'Ka_pzt', value)),
                            'Fa_pzt': (lambda: self.Fa_pzt, lambda value: setattr(self, 'Fa_pzt', value))}
        
    def __deepcopy__(self, memo):
        new_obj = ActuatorComponent.__new__(ActuatorComponent)
        new_obj.__init__(self.name, self.sps, self._Ka_pzt, self._Fa_pzt, self.unit)
        if getattr(self, '_loop', None) != None:
            new_obj._loop = self._loop
        return new_obj
        
    @property
    def Fa_pzt(self):
        return self._Fa_pzt

    @Fa_pzt.setter
    def Fa_pzt(self, value):
        self._Fa_pzt = float(value)
        self.update_component()

    @property
    def Ka_pzt(self):
        return self._Ka_pzt

    @Ka_pzt.setter
    def Ka_pzt(self, value):
        self._Ka_pzt = float(value)
        self.update_component()

    def update_component(self):
        nume = aux.polynomial_conversion_s_to_z(np.array([self._Ka_pzt]), self.sps)
        deno = aux.polynomial_conversion_s_to_z(np.array([1.0/(2.0*np.pi*self._Fa_pzt), 1.0]), self.sps)
        super().__init__(self.name, self.sps, nume, deno, unit=self.unit)


class ImplicitAccumulatorComponent(Component):
    """
    Continuous-time accumulator modeled in discrete-time domain.

    Approximates a pure integrator with scaling factor (2pi).

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    """
    def __init__(self, name, sps):
        nume = aux.polynomial_conversion_s_to_z(np.array([2.0*np.pi]), sps)
        deno = aux.polynomial_conversion_s_to_z(np.array([1.0, 0.0]), sps)
        super().__init__(name, sps, nume, deno, unit=Dimension(["rad"], ["Hz"]))

    def __deepcopy__(self, memo):
        new_obj = ImplicitAccumulatorComponent.__new__(ImplicitAccumulatorComponent)
        new_obj.__init__(self.name, self.sps)
        if getattr(self, '_loop', None) != None:
            new_obj._loop = self._loop
        return new_obj
    
    
class LeadLagComponent(Component):
    """
    Lead-Lag controller component.

    Implements a compensator of the form:

        G(s) = K * (s + wz) / (s + wp)

    where `wz = 2π*fz` is the zero frequency and `wp = 2π*fp` is the pole frequency.

    This component is useful for phase compensation: when fz < fp, it behaves like
    a phase lead; when fz > fp, it's a lag; and when fz ≈ fp, it is a gain shaper.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    K : float
        Gain factor.
    fz : float
        Zero frequency in Hz.
    fp : float
        Pole frequency in Hz.
    unit : Dimension, optional
        Dimensional unit of the signal. Defaults to dimensionless.

    Attributes
    ----------
    K : float
        Gain factor.
    fz : float
        Zero frequency.
    fp : float
        Pole frequency.
    """
    def __init__(self, name, sps, K, fz, fp, unit=Dimension(dimensionless=True)):
        self._K = float(K)
        self._fz = float(fz)
        self._fp = float(fp)
        w_z = 2 * np.pi * self._fz
        w_p = 2 * np.pi * self._fp
        nume = np.array([self._K, self._K * w_z])
        deno = np.array([1.0, w_p])
        super().__init__(name, sps, nume, deno, unit=unit)
        self.properties = {
            'K': (lambda: self.K, lambda value: setattr(self, 'K', value)),
            'fz': (lambda: self.fz, lambda value: setattr(self, 'fz', value)),
            'fp': (lambda: self.fp, lambda value: setattr(self, 'fp', value)),
        }

    def __deepcopy__(self, memo):
        new_obj = LeadLagComponent.__new__(LeadLagComponent)
        new_obj.__init__(self.name, self.sps, self._K, self._fz, self._fp, self.unit)
        if getattr(self, '_loop', None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = float(value)
        self.update_component()

    @property
    def fz(self):
        return self._fz

    @fz.setter
    def fz(self, value):
        self._fz = float(value)
        self.update_component()

    @property
    def fp(self):
        return self._fp

    @fp.setter
    def fp(self, value):
        self._fp = float(value)
        self.update_component()

    def update_component(self):
        w_z = 2 * np.pi * self._fz
        w_p = 2 * np.pi * self._fp
        nume = np.array([self._K, self._K * w_z])
        deno = np.array([1.0, w_p])
        super().__init__(self.name, self.sps, nume, deno, unit=self.unit)