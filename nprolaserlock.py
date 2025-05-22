import copy
from functools import partial
from looptools.dimension import Dimension
from looptools.components import *
from looptools.loop import LOOP
from looptools import auxiliary as aux
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
        """ class to simulate NPRO laser lock
        Args:
            sps: system clock frequency (Hz)
            pll: PLL instance
            C1: shifted bit at the gain stage 1
            C2: shifted bit at the gain stage 2
            Ki1: I gain of a 1st digital controller
            Kii1: II gain of a 1st digital controller
            Kp2: P gain of a 2nd digital controller unique to temperature
            Ki2: I gain of a 2nd digital controller unique to temperature
            Kdac: DAC gain
            Kc_pzt: Gain of an analog filter in a PZT loop
            Fc_pzt: Corner frequency of an analog filter in a PZT loop
            Ka_pzt: PZT actuator efficiency of a laser source
            Fa_pzt: PZT response bandwidth
            Kc_temp: Gain of an analog filter in a Temperature loop
            Fc_temp: Corner frequency of an analog filter in a Temperature loop
            Ka_temp: Temperature actuator efficiency of a laser source
            Fa_temp: temperature response bandwidth
            Nreg1: the number of registers which represents delays common for the entire laser-lock loop (e.g., IPU-PCU)
            OPpzt: OpAmp for PZT analog servo
            OPtemp: OpAmp for temperature analog servo
            mode: 'frequency' or 'phase' lock
            extrapolate: extrapolate PLL TF or not
            f_trans: transition frequency (for PLL TF extrapolate)
            power: power in power law (for PLL TF extrapolate)
            off: name list of components NOT to be included
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

        self.pzt.register_callback(self.update_LaserLock)

        for name, comp in self.temp.components_dict.items():
            if name in self.pzt.components_dict:
                self.temp.register_callback(self.pzt.replace_component, comp.name, comp, loop_update=True)

        self.temp.register_callback(self.update_LaserLock)
        
        self.update_LaserLock()

    def __deepcopy__(self, memo):
        new_obj = LaserLock.__new__(LaserLock)
        new_obj.temp = copy.deepcopy(self.temp)
        new_obj.pzt = copy.deepcopy(self.pzt)
        new_obj.sps = self.sps
        new_obj.update_LaserLock()
        return new_obj

    def update_LaserLock(self):
        self.Gf = partial(aux.add_transfer_function, tf1=self.pzt.Gf, tf2=self.temp.Gf)
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
        rms = aux.integral_rms(f, asd_prop, [0, np.inf])
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
        """ class to simulate PM-based laser lock
        Args:
            sps: system clock frequency (Hz)
            pll: PLL instance
            C1: shifted bit at the gain stage 1
            Ki1: I gain of a 1st digital controller
            Kii1: II gain of a 1st digital controller
            Kdac: DAC gain
            Kc_pzt: Gain of an analog filter in a PZT loop
            Fc_pzt: Corner frequency of an analog filter in a PZT loop
            Ka_pzt: PZT actuator efficiency of a laser source
            Fa_pzt: PZT response bandwidth
            Nreg1: the number of registers which represents delays common for the entire laser-lock loop (e.g., IPU-PCU)
            OPpzt: OpAmp for PZT analog servo
            off: name list of components NOT to be included
            extrapolate: extrapolation parameters for II1, [bool, f_trans] where f_trans: transition frequency)
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
        """ class to simulate PM-based laser lock
        Args:
            sps: system clock frequency (Hz)
            pll: PLL instance
            C1: shifted bit at the gain stage 1
            C2: shifted bit at the gain stage 2
            Ki1: I gain of a 1st digital controller
            Kii1: II gain of a 1st digital controller
            Kp2: P gain of a 2nd digital controller unique to temperature
            Ki2: I gain of a 2nd digital controller unique to temperature
            Kdac: DAC gain
            Kc_temp: Gain of an analog filter in a Temperature loop
            Fc_temp: Corner frequency of an analog filter in a Temperature loop
            Ka_temp: Temperature actuator efficiency of a laser source
            Fa_temp: temperature response bandwidth
            Nreg1: the number of registers which represents delays common for the entire laser-lock loop (e.g., IPU-PCU)
            OPtemp: OpAmp for temperature analog servo
            off: name list of components NOT to be included
            extrapolate: extrapolation parameters for II1, [bool, f_trans] where f_trans: transition frequency)
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
