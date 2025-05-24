import copy
from functools import partial
from looptools.dimension import Dimension
from looptools.components import *
from looptools.loopmath import *
from looptools.loop import LOOP
import logging
logger = logging.getLogger(__name__)


class NPROLaserLock():
    def __init__(self, sps, pll, 
                C1, C2,
                Ki1, Kii1,
                Kp2, Ki2,
                Kdac,
                Kc_pzt, Fc_pzt,
                Ka_pzt, Fa_pzt,
                Kc_temp, Fc_temp,
                Ka_temp, Fa_temp,
                Nreg1,
                OPpzt= None, OPtemp=None,
                mode='frequency',
                # extrapolate=False, f_trans=1e2, power=1,
                extrapolate={
                    'Fpll':[False, 1e2,1],
                    'p_II1':[False, 1e2],
                    't_II2':[False, 1e2],
                    },
                off=[None]
                ):
        """
        Composite model of a dual-loop NPRO laser frequency (or phase) lock system.

        This class combines a fast PZT control loop and a slower temperature loop
        to simulate the closed-loop frequency or phase locking of a Non-Planar Ring Oscillator (NPRO) laser.
        It wraps together two `LOOP`-based subsystems (`LaserLockPZT` and `LaserLockTemp`) and manages their
        shared structure and synchronization.

        Parameters
        ----------
        sps : float
            System clock frequency in Hz.
        pll : Component
            Phase-locked loop model providing the input sensing transfer function.
        C1 : int
            Bit shift gain stage 1 (digital).
        C2 : int
            Bit shift gain stage 2 (digital, used only in temperature loop).
        Ki1 : float
            Integral gain of the first digital controller (common to both loops).
        Kii1 : float
            Double-integrator gain of the first digital controller (common to both loops).
        Kp2 : float
            Proportional gain of the second digital controller (temperature loop only).
        Ki2 : float
            Integral gain of the second digital controller (temperature loop only).
        Kdac : float
            DAC gain [V/count].
        Kc_pzt : float
            Analog servo gain in the PZT path.
        Fc_pzt : float
            Corner frequency of the analog PZT servo.
        Ka_pzt : float
            Actuator gain of the PZT tuning mechanism [Hz/V].
        Fa_pzt : float
            Bandwidth of the PZT actuator [Hz].
        Kc_temp : float
            Analog servo gain in the temperature path.
        Fc_temp : float
            Corner frequency of the analog temperature servo.
        Ka_temp : float
            Actuator gain of the temperature tuning mechanism [Hz/V].
        Fa_temp : float
            Bandwidth of the temperature actuator [Hz].
        Nreg1 : int
            Number of registers used to model shared DSP delay in the loop.
        OPpzt : str, optional
            Label of operational amplifier used in the PZT analog path (lookup from `OpAmp_dict`).
        OPtemp : str, optional
            Label of operational amplifier used in the temperature analog path.
        mode : {'frequency', 'phase'}, default='frequency'
            Determines whether the loop stabilizes frequency or phase.
        extrapolate : dict, optional
            Dictionary configuring transfer function extrapolation for PLL and controllers.
            Keys: {'Fpll', 'p_II1', 't_II1'} → Values: [bool, f_trans, power] or [bool, f_trans].
        off : list of str or [None], optional
            List of component names to exclude from loop construction.

        Attributes
        ----------
        pzt : LaserLockPZT
            The fast PZT-based control loop.
        temp : LaserLockTemp
            The slow temperature-based control loop.
        Gf : Callable
            Closed-loop forward transfer function: `Gf(f) = G_pzt(f) + G_temp(f)`.

        Notes
        -----
        This structure enables modeling realistic lock behavior with multiple control paths
        acting on the same frequency tuning mechanism, each with different dynamics.
        """
        self.sps = sps

        if (mode != 'phase')&(mode != 'frequency'):
            raise ValueError(f'invalid mode for the laser lock loop {mode}!')

        # : === Instantiate common blocks =========================================================
        # : PLL
        from_pll = 'PD'
        to_pll = 'PD' if mode=='phase' else 'PA'
        pll_inst = pll.point_to_point_component(_from=from_pll, _to=to_pll, suppression=True)
        if mode=='frequency':
            # pll_inst.TF = partial(pll_inst.TF, extrapolate=extrapolate, f_trans=f_trans, power=power)
            pll_inst.TF = partial(pll_inst.TF, extrapolate=extrapolate['Fpll'][0], f_trans=extrapolate['Fpll'][1], power=extrapolate['Fpll'][2])
        pll_inst.name = "Fpll"

        # : generate PZT and temperature loops
        self.pzt = LaserLockPZT(self.sps, pll_inst, C1, Ki1, Kii1, Kdac,
            Kc_pzt, Fc_pzt, Ka_pzt, Fa_pzt, Nreg1, OPpzt, off, extrapolate['p_II1'])
        self.temp = LaserLockTemp(self.sps, pll_inst, C1, C2, Ki1, Kii1, Kp2, Ki2, Kdac,
            Kc_temp, Fc_temp, Ka_temp, Fa_temp, Nreg1, OPtemp, off, extrapolate['t_II1'])
        
        for name, comp in self.pzt.components_dict.items():
            if name in self.temp.components_dict:
                self.pzt.register_callback(self.temp.replace_component, comp.name, comp, loop_update=True)

        self.pzt.register_callback(self.update_NPROLaserLock)

        for name, comp in self.temp.components_dict.items():
            if name in self.pzt.components_dict:
                self.temp.register_callback(self.pzt.replace_component, comp.name, comp, loop_update=True)

        self.temp.register_callback(self.update_NPROLaserLock)
        
        self.update_NPROLaserLock()

    def __deepcopy__(self, memo):
        new_obj = NPROLaserLock.__new__(NPROLaserLock)
        new_obj.temp = copy.deepcopy(self.temp)
        new_obj.pzt = copy.deepcopy(self.pzt)
        new_obj.sps = self.sps
        new_obj.update_NPROLaserLock()
        return new_obj

    def update_NPROLaserLock(self):
        self.Gf = partial(add_transfer_function, tf1=self.pzt.Gf, tf2=self.temp.Gf)
        # self.Gc = self.temp.point_to_point_component(_from='Fpll', _to='Fpll', suppression=True) + self.pzt.point_to_point_component(_from='Fpll', _to=None, suppression=True)
        # self.Gc.name = "G"
        # self.Gf = partial(LOOP.tf_series, components=[self.Gc], mode=None, self=None)
        # self.Hf = partial(LOOP.tf_series, components=[self.Gc], mode="H", self=None)
        # self.Ef = partial(LOOP.tf_series, components=[self.Gc], mode="E", self=None)
        # H_TE = control.feedback(self.Gc.TE, 1)
        # self.Hc = Component("H", self.sps, tf=H_TE, unit=self.Gc.unit)
        # E_TE = control.feedback(1, self.Gc.TE)
        # self.Ec = Component("E", self.sps, tf=E_TE, unit=self.Gc.unit)

    def noise_propagation_asd(self, f, asd, unit=Dimension(dimensionless=True), _from='PD', _to=None, view=False, isTF=True):
        TF = self.Gf(f=f)
        TF = 1/(1+TF)
        mag = abs(TF)
        phase = np.angle(TF, deg=False)
        bode={'f':f, 'mag':mag, 'phase':phase}
        asd_prop = bode['mag'] * asd
        unit_prop = unit
        rms = integral_rms(f, asd_prop, [0, np.inf])
        return asd_prop, unit_prop, bode, rms
        


class LaserLockPZT(LOOP):
    def __init__(self, sps, pll,
                C1,
                Ki1, Kii1,
                Kdac,
                Kc_pzt, Fc_pzt,
                Ka_pzt, Fa_pzt,
                Nreg1,
                OPpzt, off=[None],
                extrapolate=[False,1e2]
                ):
        """
        Simulates a fast PZT-based feedback loop for laser frequency control.

        This subclass of `LOOP` builds a control chain that models the fast response of a 
        piezoelectric actuator used to finely tune the laser frequency. It includes gain stages,
        digital controllers, DAC, analog filters, actuator dynamics, and DSP delays.

        Parameters
        ----------
        sps : float
            System clock frequency in Hz.
        pll : Component
            PLL component representing sensing transfer function.
        C1 : int
            Shift bit for gain stage 1.
        Ki1 : float
            Integral gain of the digital controller.
        Kii1 : float
            Double-integrator gain of the digital controller.
        Kdac : float
            DAC gain [V/count].
        Kc_pzt : float
            Analog servo gain in the PZT path.
        Fc_pzt : float
            Corner frequency of the analog PZT servo.
        Ka_pzt : float
            Actuator gain of the PZT tuning mechanism [Hz/V].
        Fa_pzt : float
            Bandwidth of the PZT actuator [Hz].
        Nreg1 : int
            Number of DSP registers simulating control delay.
        OPpzt : str or None
            Optional operational amplifier name for modeling analog gain-bandwidth product effects.
        off : list of str or [None], optional
            List of component names to exclude.
        extrapolate : list, optional
            Parameters to extrapolate the controller's TF: [bool, f_trans].

        Attributes
        ----------
        Gc : Component
            Open-loop controller TF component.
        Gf : Callable
            Closed-loop transfer function G(f) from loop output to input.
        Hf : Callable
            Sensitivity function H(f).
        Ef : Callable
            Error transfer function E(f).
        """
        super().__init__(sps)
        self.pll = pll
        self.C1 = C1
        self.Ki1 = Ki1
        self.Kii1 = Kii1
        self.Kdac = Kdac
        self.Kc_pzt = Kc_pzt
        self.Fc_pzt = Fc_pzt
        self.Ka_pzt = Ka_pzt
        self.Fa_pzt = Fa_pzt
        self.Nreg1 = Nreg1
        self.OPpzt = OPpzt
        self.off = off
        self.extrapolate = extrapolate

        # : === Laser-lock-loop components ==========
        # : PLL
        self.add_component(pll)

        # : Overall gain stage 1
        if "Fgain1" not in off:
            self.add_component(LeftBitShiftComponent("Fgain1", self.sps, C1))

        # : PZT digital controller
        if "Fctrl1" not in off:
            self.add_component(DoubleIntegratorComponent("Fctrl1", self.sps, Ki1, Kii1, extrapolate))

        # : DAC
        if "Kdac" not in off:
            self.add_component(MultiplierComponent("Kdac", self.sps, Kdac, Dimension(dimensionless=True)))

        # : PZT analog low-pass filter
        if "p_Fcond" not in off:
            self.add_component(ActuatorComponent("p_Fcond", self.sps, Kc_pzt, Fc_pzt, Dimension(dimensionless=True)))
            if OPpzt is not None:
                self.add_component(ActuatorComponent("p_Fop", self.sps, 1.0, OpAmp_dict[OPpzt]["GBP"]/Kc_pzt), Dimension(dimensionless=True))

        # : Laser PZT actuator efficiency [Hz/V]
        if "p_Fplant" not in off:
            self.add_component(ActuatorComponent("p_Fplant", self.sps, Ka_pzt, Fa_pzt, Dimension(["Hz"], ["V"])))

        # : implicit accumulator [rad/Hz]
        if "Fnu2phi" not in off:
            self.add_component(ImplicitAccumulatorComponent("Fnu2phi", self.sps))

        # : DSP delay (= register delay)
        if "DSP" not in off:
            self.add_component(DSPDelayComponent("DSP", self.sps, Nreg1))

        if off != [None]:
            logger.warning(f"The following components are not included in the loop {off}")
            
        self.Gc, self.Hc, self.Ec = self.system_transfer_components()
        self.Gf, self.Hf, self.Ef = self.system_transfer_functions()

        self.register_component_properties()

    def __deepcopy__(self, memo):
        new_obj = LaserLockPZT.__new__(LaserLockPZT)
        new_obj.__init__(self.sps, self.pll, self.C1, self.Ki1, self.Kii1, self.Kdac, self.Kc_pzt, self.Fc_pzt, self.Ka_pzt, self.Fa_pzt, self.Nreg1, self.OPpzt, self.off, self.extrapolate)
        new_obj.callbacks = self.callbacks
        return new_obj


class LaserLockTemp(LOOP):
    def __init__(self, sps, pll,
                C1, C2,
                Ki1, Kii1,
                Kp2, Ki2,
                Kdac,
                Kc_temp, Fc_temp,
                Ka_temp, Fa_temp, 
                Nreg1,
                OPtemp, off,
                extrapolate=[False,1e2]
                ):
        """
        Simulates a slow, temperature-based feedback loop for laser frequency control.

        This subclass of `LOOP` models a thermal control path that compensates low-frequency drifts 
        in the laser frequency. It includes dual digital controllers (cascaded), gain stages, DAC,
        analog filters, actuator dynamics, and DSP delays.

        Parameters
        ----------
        sps : float
            System clock frequency in Hz.
        pll : Component
            PLL component representing sensing transfer function.
        C1 : int
            Shift bit for gain stage 1.
        C2 : int
            Shift bit for gain stage 2.
        Ki1 : float
            Integral gain of the shared digital controller.
        Kii1 : float
            Double-integrator gain of the shared digital controller.
        Kp2 : float
            Proportional gain of the second digital controller.
        Ki2 : float
            Integral gain of the second digital controller.
        Kdac : float
            DAC gain [V/count].
        Kc_temp : float
            Analog servo gain in the temperature path.
        Fc_temp : float
            Corner frequency of the analog temperature servo.
        Ka_temp : float
            Actuator gain of the temperature tuning mechanism [Hz/V].
        Fa_temp : float
            Bandwidth of the temperature actuator [Hz].
        Nreg1 : int
            Number of DSP registers simulating control delay.
        OPtemp : str or None
            Optional operational amplifier name for modeling analog gain-bandwidth product effects.
        off : list of str or [None]
            List of component names to exclude.
        extrapolate : list, optional
            Parameters to extrapolate the controller's TF: [bool, f_trans].

        Attributes
        ----------
        Gc : Component
            Open-loop controller TF component.
        Gf : Callable
            Closed-loop transfer function G(f) from loop output to input.
        Hf : Callable
            Sensitivity function H(f).
        Ef : Callable
            Error transfer function E(f).

        Notes
        -----
        Used in conjunction with `LaserLockPZT` to provide full-range stabilization
        of laser frequency over both fast and slow timescales.
        """
        super().__init__(sps)
        self.pll = pll
        self.C1 = C1
        self.C2 = C2
        self.Ki1 = Ki1
        self.Kii1 = Kii1
        self.Kp2 = Kp2
        self.Ki2 = Ki2
        self.Kdac = Kdac
        self.Kc_temp = Kc_temp
        self.Fc_temp = Fc_temp
        self.Ka_temp = Ka_temp
        self.Fa_temp = Fa_temp
        self.Nreg1 = Nreg1
        self.OPtemp = OPtemp
        self.off = off
        self.extrapolate = extrapolate

        # : === Laser-lock-loop components ==========
        # : PLL
        self.add_component(pll)

        # : Overall gain stage 1
        if "Fgain1" not in off:
            self.add_component(LeftBitShiftComponent("Fgain1", self.sps, C1))

        # : PZT digital controller
        self.add_component(DoubleIntegratorComponent("Fctrl1", self.sps, Ki1, Kii1, extrapolate))

        # : Overall gain stage 2
        if "Fgain2" not in off:
            self.add_component(LeftBitShiftComponent("Fgain2", self.sps, C2))

        # : Temperature digital controller
        if "Fctrl2" not in off:
            self.add_component(PIControllerComponent("Fctrl2", self.sps, Kp2, Ki2))

        # : DAC
        if "Kdac" not in off:
            self.add_component(MultiplierComponent("Kdac", self.sps, Kdac, Dimension(dimensionless=True)))

        # : Temp analog low-pass filter
        if "t_Fcond" not in off:
            self.add_component(ActuatorComponent("t_Fcond", self.sps, Kc_temp, Fc_temp, Dimension(dimensionless=True)))
            if OPtemp is not None:
                self.add_component(ActuatorComponent("t_Fop", self.sps, 1.0, OpAmp_dict[OPtemp]["GBP"]/Kc_temp), Dimension(dimensionless=True))                    

        # : Laser temperature actuator efficiency [Hz/V]
        if "t_Fplant" not in off:
            self.add_component(ActuatorComponent("t_Fplant", self.sps, Ka_temp, Fa_temp, unit=Dimension(["Hz"], ["V"])))

        # : implicit accumulator [rad/Hz]
        if "Fnu2phi" not in off:
            self.add_component(ImplicitAccumulatorComponent("Fnu2phi", self.sps))

        # : DSP delay (= register delay)
        if "DSP" not in off:
            self.add_component(DSPDelayComponent("DSP", self.sps, Nreg1))

        if off != [None]:
            logger.warning(f"The following components are not included in the loop {off}")

        self.Gc, self.Hc, self.Ec = self.system_transfer_components()
        self.Gf, self.Hf, self.Ef = self.system_transfer_functions()
    
        self.register_component_properties()

    def __deepcopy__(self, memo):
        new_obj = LaserLockTemp.__new__(LaserLockTemp)
        new_obj.__init__(self.sps, self.pll, self.C1, self.C2, self.Ki1, self.Kii1, self.Kp2, self.Ki2, self.Kdac, self.Kc_temp, self.Fc_temp, self.Ka_temp, self.Fa_temp, self.Nreg1, self.OPtemp, self.off, self.extrapolate)
        new_obj.callbacks = self.callbacks
        return new_obj
