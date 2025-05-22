import numpy as np
from functools import partial
from looptools.component import Component
from looptools.dimension import Dimension
from looptools import auxiliary as aux
import logging
logger = logging.getLogger(__name__)


class PDComponent(Component):
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
        

class LFComponent(Component):
    def __init__(self, name, sps, Klf):
        self._Klf = 2.0**float(-Klf)
        super().__init__(name, sps, np.array([self._Klf]), np.array([1.0, -(1.0 - self._Klf)]), unit=Dimension(dimensionless=True))
        self.properties = {'Klf': (lambda: self.Klf, lambda value: setattr(self, 'Klf', value))}

    def __deepcopy__(self, memo):
        new_obj = LFComponent.__new__(LFComponent)
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


class TwoStageLFComponent(Component):
    def __init__(self, name, sps, Klf):
        self._Klf = 2**float(-Klf)
        LF = Component("LF", sps, np.array([self._Klf]), np.array([1.0, -(1.0 - self._Klf)]), unit=Dimension(dimensionless=True))
        LF = LF*LF
        super().__init__(name, sps, LF.nume, LF.deno, unit=LF.unit)
        self.TE = LF.TE
        self.TE.name = name
        self.TF = LF.TF
        self.properties = {'Klf': (lambda: self.Klf, lambda value: setattr(self, 'Klf', value))}

    def __deepcopy__(self, memo):
        new_obj = TwoStageLFComponent.__new__(TwoStageLFComponent)
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
        LF = Component("LF", self.sps, np.array([self._Klf]), np.array([1.0, -(1.0 - self._Klf)]), unit=Dimension(dimensionless=True))
        LF = LF*LF
        super().__init__(self.name, self.sps, LF.nume, LF.deno, unit=LF.unit)
        self.TE = LF.TE
        self.TE.name = self.name
        self.TF = LF.TF


class PIControllerComponent(Component):
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


class PAComponent(Component):
    def __init__(self, name, sps):
        super().__init__(name, sps, np.array([1.0]), np.array([1.0, -1.0]), unit=Dimension(["s"], []))

    def __deepcopy__(self, memo):
        new_obj = PAComponent.__new__(PAComponent)
        new_obj.__init__(self.name, self.sps)
        if getattr(self, '_loop', None) != None:
            new_obj._loop = self._loop
        return new_obj


class LUTComponent(Component):
    def __init__(self, name, sps):
        super().__init__(name, sps, np.array([2.0*np.pi]), np.array([1.0]), unit=Dimension(["rad"], ["cycle"]))

    def __deepcopy__(self, memo):
        new_obj = LUTComponent.__new__(LUTComponent)
        new_obj.__init__(self.name, self.sps)
        if getattr(self, '_loop', None) != None:
            new_obj._loop = self._loop
        return new_obj


class DSPDelayComponent(Component):
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