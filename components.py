import numpy as np
from functools import partial
from scipy.signal import butter
from looptools.component import Component
from looptools.dimension import Dimension
import looptools.loopmath as lm
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
    Low pass filter component (first-order or cascaded first-order IIR).

    Models a low-pass IIR filter with tunable gain. Higher-order filters can be 
    simulated by cascading multiple identical first-order sections.

    Parameters
    ----------
    name : str
        Name of the component.
    sps : float
        Sample rate in Hz.
    Klf : float
        Log2 representation of loop gain (gain = 2^-Klf).
    n : int, optional
        Number of cascaded first-order sections (default is 1).

    Attributes
    ----------
    Klf : float
        Filter gain as 2^-Klf.
    n : int
        Filter order (number of cascaded stages).
    """
    def __init__(self, name, sps, Klf, n=1):
        self.n = int(n)
        self._Klf = 2.0 ** float(-Klf)
        num = np.array([self._Klf]) ** self.n
        den = np.poly1d([1.0, -(1.0 - self._Klf)]) ** self.n
        super().__init__(name, sps, num, den.coeffs, unit=Dimension(dimensionless=True))
        self.properties = {
            'Klf': (lambda: self.Klf, lambda value: setattr(self, 'Klf', value)),
            'n': (lambda: self.n, lambda value: setattr(self, 'n', int(value))),
        }

    def __deepcopy__(self, memo):
        new_obj = LPFComponent.__new__(LPFComponent)
        new_obj.__init__(self.name, self.sps, -np.log2(self._Klf), self.n)
        if getattr(self, '_loop', None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Klf(self):
        return self._Klf

    @Klf.setter
    def Klf(self, value):
        self._Klf = 2 ** float(-value)
        self.update_component()

    def update_component(self):
        num = np.array([self._Klf]) ** self.n
        den = np.poly1d([1.0, -(1.0 - self._Klf)]) ** self.n
        super().__init__(self.name, self.sps, num, den.coeffs, unit=Dimension(dimensionless=True))

class ButterworthLPFComponent(Component):
    """
    Digital Butterworth low-pass filter component (n-th order IIR).

    Uses scipy's butter() design to produce a maximally flat low-pass filter.

    Parameters
    ----------
    name : str
        Name of the component.
    sps : float
        Sample rate in Hz.
    f_c : float
        -3 dB cutoff frequency in Hz.
    order : int, optional
        Filter order (default: 1). Must be >= 1.

    Attributes
    ----------
    f_c : float
        Cutoff frequency in Hz.
    order : int
        Filter order.
    """

    def __init__(self, name, sps, f_c, order=1):
        if order < 1:
            raise ValueError("Butterworth filter order must be >= 1.")
        self.f_c = f_c
        self.order = int(order)

        # Design Butterworth filter in digital (normalized) domain
        norm_cutoff = f_c / (0.5 * sps)  # Normalize to Nyquist
        b, a = butter(N=order, Wn=norm_cutoff, btype='low', analog=False)

        super().__init__(name=name, sps=sps,
                         nume=b, deno=a,
                         unit=Dimension(dimensionless=True))

        # Allow dynamic property access
        self.properties = {
            'f_c': (lambda: self.f_c, self._set_fc),
            'order': (lambda: self.order, self._set_order)
        }

    def _set_fc(self, f_c):
        self.f_c = float(f_c)
        self._update_tf()

    def _set_order(self, order):
        self.order = int(order)
        self._update_tf()

    def _update_tf(self):
        norm_cutoff = self.f_c / (0.5 * self.sps)
        b, a = butter(N=self.order, Wn=norm_cutoff, btype='low', analog=False)
        self.nume = np.atleast_1d(b)
        self.deno = np.atleast_1d(a)


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
    Proportional-Integral controller component, P+I.

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
    Second-order integrator, I+II.

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
        self.TF = partial(lm.add_transfer_function, tf1=I.TF, tf2=II.TF)
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
        self.TF = partial(lm.add_transfer_function, tf1=I.TF, tf2=II.TF)


class PIIControllerComponent(Component):
    """
    Proportional + Integrator + Double Integrator controller component, P+I+II.

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
        self.TF = partial(lm.add_transfer_function, tf1=P.TF, tf2=partial(lm.add_transfer_function, tf1=I.TF, tf2=II.TF))

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
        self.TF = partial(lm.add_transfer_function, tf1=P.TF, tf2=partial(lm.add_transfer_function, tf1=I.TF, tf2=II.TF))


class MokuPIDSymbolicController(Component):
    """
    Moku-style symbolic PID controller using P, I, II, and D terms.
    
    WARNING: II-term causes numerical instabilities at low frequencies, use MokuPIDController instead.

    This component constructs the transfer function symbolically using known corner frequencies
    and proportional gain in dB, as implemented by Liquid Instruments Moku devices.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate [Hz].
    Kp_dB : float
        Proportional gain in dB.
    Fc_i : float or None
        First integrator (I) crossover frequency [Hz].
    Fc_ii : float or None
        Second integrator (II) crossover frequency [Hz].
    Fc_d : float or None
        Derivative (D) crossover frequency [Hz].
    f_trans : float or None
        Transition frequency for regularization [Hz]. Used to improve numerical behavior of double integrator.

    Properties
    ----------
    Kp_dB, Ki_dB, Kii_dB, Kd_dB : float
        Gains in decibels.
    Fc_i, Fc_ii, Fc_d : float
        Crossover frequencies in Hz.
    """
    def __init__(self, name, sps, Kp_dB, Fc_i=None, Fc_ii=None, Fc_d=None, f_trans=None):
        self.name = name
        self.sps = sps
        self.f_trans = f_trans

        self._Kp_dB = Kp_dB
        self._Fc_i = Fc_i
        self._Fc_ii = Fc_ii
        self._Fc_d = Fc_d

        self.update_component()

        self.properties = {
            'Kp_dB': (lambda: self.Kp_dB, lambda value: setattr(self, 'Kp_dB', value)),
            'Ki_dB': (lambda: self.Ki_dB, lambda value: setattr(self, 'Ki_dB', value)),
            'Kii_dB': (lambda: self.Kii_dB, lambda value: setattr(self, 'Kii_dB', value)),
            'Kd_dB': (lambda: self.Kd_dB, lambda value: setattr(self, 'Kd_dB', value)),
            'Fc_i': (lambda: self.Fc_i, lambda value: setattr(self, 'Fc_i', value)),
            'Fc_ii': (lambda: self.Fc_ii, lambda value: setattr(self, 'Fc_ii', value)),
            'Fc_d': (lambda: self.Fc_d, lambda value: setattr(self, 'Fc_d', value)),
        }

    def update_component(self):
        tf_str = self.moku_pid_tf_string(
            self.sps,
            Kp_dB=self._Kp_dB,
            Ki_dB=None if self._Fc_i is not None else lm.log2_gain_to_db(np.log2(self._Ki)) if hasattr(self, '_Ki') else None,
            Kii_dB=None if self._Fc_ii is not None else lm.log2_gain_to_db(np.log2(self._Kii)) if hasattr(self, '_Kii') else None,
            Kd_dB=None if self._Fc_d is not None else lm.log2_gain_to_db(np.log2(self._Kd)) if hasattr(self, '_Kd') else None,
            Fc_i=self._Fc_i,
            Fc_ii=self._Fc_ii,
            Fc_d=self._Fc_d,
            f_trans=self.f_trans
        )

        super().__init__(self.name, tf=tf_str, sps=self.sps, unit=Dimension(["cycle"], ["s", "rad"]))

        Kp_log2 = lm.db_to_log2_gain(self._Kp_dB)
        self._Kp = 2 ** Kp_log2
        self._Ki = 0.0 if self._Fc_i is None else 2 ** lm.gain_for_crossover_frequency(0.0, self.sps, self._Fc_i, kind='I')
        self._Kii = 0.0 if self._Fc_ii is None else 2 ** lm.gain_for_crossover_frequency(0.0, self.sps, self._Fc_ii, kind='II')

        if self._Fc_d is not None:
            omega_d = 2 * np.pi * self._Fc_d / self.sps
            mag = abs((1 - np.exp(-1j * omega_d)) / (1 + np.exp(-1j * omega_d)))
            self._Kd = self._Kp / mag
        elif hasattr(self, '_Kd'):
            pass
        else:
            self._Kd = 0.0

    @staticmethod
    def moku_pid_tf_string(sps, Kp_dB=0.0, Ki_dB=None, Kii_dB=None, Kd_dB=None, Fc_i=None, Fc_ii=None, Fc_d=None, f_trans=None):
        Kp_log2 = lm.db_to_log2_gain(Kp_dB)
        Kp = 2 ** Kp_log2

        if Fc_i is not None:
            Ki_log2 = lm.gain_for_crossover_frequency(0.0, sps, Fc_i, kind='I')
        elif Ki_dB is not None:
            Ki_log2 = lm.db_to_log2_gain(Ki_dB)
        else:
            Ki_log2 = float('-inf')

        if Fc_ii is not None:
            Kii_log2 = lm.gain_for_crossover_frequency(0.0, sps, Fc_ii, kind='II')
        elif Kii_dB is not None:
            Kii_log2 = lm.db_to_log2_gain(Kii_dB)
        else:
            Kii_log2 = float('-inf')

        if Fc_d is not None:
            Kd_log2 = lm.gain_for_crossover_frequency(0.0, sps, Fc_d, kind='D')
        elif Kd_dB is not None:
            Kd_log2 = lm.db_to_log2_gain(Kd_dB)
        else:
            Kd_log2 = float('-inf')

        Ki = 0.0 if not np.isfinite(Ki_log2) else 2 ** Ki_log2
        Kii = 0.0 if not np.isfinite(Kii_log2) else 2 ** Kii_log2
        Kd = 0.0 if not np.isfinite(Kd_log2) else 2 ** Kd_log2

        if f_trans is not None:
            delta = (2 * np.pi * f_trans / sps) ** 2
            Kii_term = f"({Kii:.6g})/((1 - z**-1)**2 + {delta:.3e})"
        else:
            Kii_term = f"({Kii:.6g})/(1 - z**-1)**2"

        return (
            f"{Kp:.6g} * (1"
            f" + ({Ki:.6g})/(1 - z**-1)"
            f" + {Kii_term}"
            f" + ({Kd:.6g})*(1 - z**-1)/(1 + z**-1))"
        )

    @property
    def Kp_dB(self): return self._Kp_dB
    @Kp_dB.setter
    def Kp_dB(self, value): self._Kp_dB = float(value); self.update_component()

    @property
    def Ki_dB(self): return None if self._Ki == 0.0 else lm.log2_gain_to_db(np.log2(self._Ki))
    @Ki_dB.setter
    def Ki_dB(self, value): self._Fc_i = None; self._Ki = 2 ** lm.db_to_log2_gain(value); self.update_component()

    @property
    def Kii_dB(self): return None if self._Kii == 0.0 else lm.log2_gain_to_db(np.log2(self._Kii))
    @Kii_dB.setter
    def Kii_dB(self, value): self._Fc_ii = None; self._Kii = 2 ** lm.db_to_log2_gain(value); self.update_component()

    @property
    def Kd_dB(self): return None if self._Kd == 0.0 else lm.log2_gain_to_db(np.log2(self._Kd))
    @Kd_dB.setter
    def Kd_dB(self, value): self._Fc_d = None; self._Kd = 2 ** lm.db_to_log2_gain(value); self.update_component()

    @property
    def Fc_i(self): return self._Fc_i
    @Fc_i.setter
    def Fc_i(self, value): self._Fc_i = float(value); self.update_component()

    @property
    def Fc_ii(self): return self._Fc_ii
    @Fc_ii.setter
    def Fc_ii(self, value): self._Fc_ii = float(value); self.update_component()

    @property
    def Fc_d(self): return self._Fc_d
    @Fc_d.setter
    def Fc_d(self, value): self._Fc_d = float(value); self.update_component()

    def __deepcopy__(self, memo):
        return MokuPIDSymbolicController(
            name=self.name,
            sps=self.sps,
            Kp_dB=self._Kp_dB,
            Fc_i=self._Fc_i,
            Fc_ii=self._Fc_ii,
            Fc_d=self._Fc_d,
            f_trans=self.f_trans,
        )
    

class MokuPIDController(Component):
    """
    Moku-style PID controller with P, optional I, II, and D terms using symbolic structure
    and extrapolated low-frequency behavior for numerical stability.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate [Hz].
    Kp_dB : float
        Proportional gain in dB.
    Fc_i : float or None
        First integrator crossover frequency [Hz]. If None, I is omitted.
    Fc_ii : float or None
        Second integrator crossover frequency [Hz]. If None, II is omitted.
    Fc_d : float or None
        Derivative crossover frequency [Hz]. If None, D is omitted.
    f_trans : float
        Transition frequency below which extrapolation is applied [Hz].
    """

    def __init__(self, name, sps, Kp_dB, Fc_i=None, Fc_ii=None, Fc_d=None, f_trans=None):
        self.name = name
        self.sps = sps
        self.f_trans = f_trans
        self._Kp_dB = Kp_dB
        self._Fc_i = Fc_i
        self._Fc_ii = Fc_ii
        self._Fc_d = Fc_d

        self.update_component()

        self.properties = {
            'Kp_dB': (lambda: self.Kp_dB, lambda value: setattr(self, 'Kp_dB', value)),
            'Ki_dB': (lambda: self.Ki_dB, lambda value: setattr(self, 'Ki_dB', value)),
            'Kii_dB': (lambda: self.Kii_dB, lambda value: setattr(self, 'Kii_dB', value)),
            'Kd_dB': (lambda: self.Kd_dB, lambda value: setattr(self, 'Kd_dB', value)),
            'Fc_i': (lambda: self.Fc_i, lambda value: setattr(self, 'Fc_i', value)),
            'Fc_ii': (lambda: self.Fc_ii, lambda value: setattr(self, 'Fc_ii', value)),
            'Fc_d': (lambda: self.Fc_d, lambda value: setattr(self, 'Fc_d', value)),
        }

    def update_component(self):
        Kp_log2 = lm.db_to_log2_gain(self._Kp_dB)
        self._Kp = 2 ** Kp_log2

        P = Component("P", self.sps, np.array([self._Kp]), np.array([1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        components = [P]

        if self._Fc_i is not None and self._Fc_ii is None:
            self._Ki = 2 ** lm.gain_for_crossover_frequency(Kp_log2, self.sps, self._Fc_i, kind='I')
            I = Component("I", self.sps, np.array([self._Ki]), np.array([1.0, -1.0]), unit=P.unit)
            components.append(I)
        else:
            self._Ki = None

        if self._Fc_ii is not None and self._Fc_i is not None: # We cannot have double integrator with the first-stage integrator
            i_log2, ii_log2 = lm.gain_for_crossover_frequency(Kp_log2, self.sps, [self._Fc_i, self._Fc_ii], kind='II')
            self._Ki, self._Kii = 2 ** i_log2, 2 ** ii_log2
            I = Component("I", self.sps, np.array([self._Ki]), np.array([1.0, -1.0]), unit=P.unit)
            components.append(I)

            II = Component("II", self.sps, np.array([self._Kii]), np.array([1.0, -2.0, 1.0]), unit=P.unit)
            if self.f_trans is not None:
                II.TF = partial(II.TF, extrapolate=True, f_trans=self.f_trans, power=-2)
            components.append(II)
        else:
            self._Ki = None
            self._Kii = None

        if self._Fc_d is not None:
            self._Kd = 2 ** lm.gain_for_crossover_frequency(Kp_log2, self.sps, self._Fc_d, kind='D')
            D = Component("D", self.sps, np.array([self._Kd, -self._Kd]), np.array([1.0, 0.0, 1.0]), unit=P.unit)
            components.append(D)
        else:
            self._Kd = None

        PID = components[0]
        for comp in components[1:]:
            PID = PID + comp

        super().__init__(self.name, self.sps, PID.nume, PID.deno, unit=PID.unit)
        self.TF = PID.TF
        self.TE = PID.TE
        self.TE.name = self.name

    def __deepcopy__(self, memo):
        return MokuPIDController(
            name=self.name,
            sps=self.sps,
            Kp_dB=self._Kp_dB,
            Fc_i=self._Fc_i,
            Fc_ii=self._Fc_ii,
            Fc_d=self._Fc_d,
            f_trans=self.f_trans
        )

    # --- Gain dB Accessors ---
    @property
    def Kp_dB(self): return self._Kp_dB
    @Kp_dB.setter
    def Kp_dB(self, value): self._Kp_dB = float(value); self.update_component()

    @property
    def Ki_dB(self): return None if self._Ki is None else lm.log2_gain_to_db(np.log2(self._Ki))
    @Ki_dB.setter
    def Ki_dB(self, value): self._Fc_i = None; self._Ki = 2 ** lm.db_to_log2_gain(value); self.update_component()

    @property
    def Kii_dB(self): return None if self._Kii is None else lm.log2_gain_to_db(np.log2(self._Kii))
    @Kii_dB.setter
    def Kii_dB(self, value): self._Fc_ii = None; self._Kii = 2 ** lm.db_to_log2_gain(value); self.update_component()

    @property
    def Kd_dB(self): return None if self._Kd is None else lm.log2_gain_to_db(np.log2(self._Kd))
    @Kd_dB.setter
    def Kd_dB(self, value): self._Fc_d = None; self._Kd = 2 ** lm.db_to_log2_gain(value); self.update_component()

    # --- Crossover Frequency Accessors ---
    @property
    def Fc_i(self): return self._Fc_i
    @Fc_i.setter
    def Fc_i(self, value):
        self._Fc_i = float(value) if value is not None else None
        self.update_component()

    @property
    def Fc_ii(self): return self._Fc_ii
    @Fc_ii.setter
    def Fc_ii(self, value):
        self._Fc_ii = float(value) if value is not None else None
        self.update_component()

    @property
    def Fc_d(self): return self._Fc_d
    @Fc_d.setter
    def Fc_d(self, value):
        self._Fc_d = float(value) if value is not None else None
        self.update_component()


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

    Converts s-domain coefficients into z-domain using polynomial_conversion_s_to_z.

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
        nume = lm.polynomial_conversion_s_to_z(np.array([self._Ka_pzt]), sps)
        deno = lm.polynomial_conversion_s_to_z(np.array([1.0/(2.0*np.pi*self._Fa_pzt), 1.0]), sps)
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
        nume = lm.polynomial_conversion_s_to_z(np.array([self._Ka_pzt]), self.sps)
        deno = lm.polynomial_conversion_s_to_z(np.array([1.0/(2.0*np.pi*self._Fa_pzt), 1.0]), self.sps)
        super().__init__(self.name, self.sps, nume, deno, unit=self.unit)


class ImplicitAccumulatorComponent(Component):
    """
    Continuous-time accumulator modeled in discrete-time domain.

    Approximates a pure integrator with scaling factor of 2pi.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    """
    def __init__(self, name, sps):
        nume = lm.polynomial_conversion_s_to_z(np.array([2.0*np.pi]), sps)
        deno = lm.polynomial_conversion_s_to_z(np.array([1.0, 0.0]), sps)
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