import numpy as np
import control
import copy
from functools import partial
from looptools.component import Component
from looptools.dimension import Dimension
import looptools.auxiliary as aux
import pyhexagon.auxiliary as hexaux
import logging
logger = logging.getLogger(__name__)


class LOOP:
    def __init__(self, sps):
        """ parent class for a control loop
        Args:
            sps: loop rate (Hz)
        """

        self.sps = sps
        self.components_dict = {}
        self.property_list = []
        self.callbacks = []
        self.Gc = None
        self.Hc = None
        self.Ec = None
        self.Gf = None
        self.Hf = None
        self.Ef = None

    def update(self):
        self.system_transfer_components()
        self.system_transfer_functions()

    def notify_callbacks(self):
        for callback, args, kwargs in self.callbacks:
            callback(*args, **kwargs)
        
    def add_component(self, newcomp, loop_update=False):
        assert (newcomp.name != None)&(newcomp.name != ''), logger.error("Attempting to add unnamed component")
        if newcomp.name in self.components_dict:
            logger.error("Named component already exists in the System, use `replace_component` instead")
            return
        else:
            newcomp._loop = self
            self.components_dict[newcomp.name] = newcomp
            if loop_update:
                self.update()

    def remove_component(self, name, loop_update=False):
        assert name in self.components_dict, logger.error("Attempting to remove inexistent component")
        del self.components_dict[name]
        if loop_update:
            self.update()

    def replace_component(self, name, newcomp, loop_update=False):
        assert name in self.components_dict, logger.error("Attempting to replace inexistent component")
        self.components_dict[name] = newcomp
        if loop_update:
            self.update()

    def update_component(self, component, property, newvalue, loop_update=False):
        assert component in self.components_dict, logger.error("Attempting to update inexistent component")
        assert property in self.components_dict[component].properties,  logger.error("Attempting to modify inexistent component attribute")
        self.components_dict[component].properties[property][1](newvalue)
        if loop_update:
            self.update()

    def register_callback(self, callback, *args, **kwargs):
        self.callbacks.append((callback, args, kwargs))
        
    def register_component_properties(self):
        for name, comp in self.components_dict.items():
            if getattr(comp, 'properties', None) is not None:
                for prop in comp.properties:
                    self.create_property_delegator(name, prop)
        
    def create_property_delegator(self, component_name, prop_name):
        # Accessor method that retrieves the property from the specific component
        def get_prop(self):
            return getattr(self.components_dict[component_name], prop_name)

        # Mutator method that sets the property for the specific component
        def set_prop(self, value):
            setattr(self.components_dict[component_name], prop_name, value)

        # Create a new property on the fly and attach it to the class
        sys_property_name = component_name+'_'+prop_name

        setattr(self.__class__, sys_property_name, property(get_prop, set_prop))

        self.property_list.append(sys_property_name)

    def system_transfer_components(self):
        """ compute system transfer elements
        Args:
            components: list composed of loop components
        """

        # : open-loop transfer element G(z)
        self.Gc = np.prod(list(self.components_dict.values()))
        self.Gc.name = "G"
        # : system transfer element H(z)
        H_TE = control.feedback(self.Gc.TE, 1)
        self.Hc = Component("H", self.sps, tf=H_TE, unit=self.Gc.unit)
        # : error transfer element E(z)
        E_TE = control.feedback(1, self.Gc.TE)
        self.Ec = Component("E", self.sps, tf=E_TE, unit=self.Gc.unit)

        return self.Gc, self.Hc, self.Ec

    def system_transfer_functions(self):
        """ compute system transfer functions
        Args:
            components: list composed of loop components [child]
            f: fourier frequencies
        """

        self.Gf = partial(self.tf_series, mode=None)
        self.Hf = partial(self.tf_series, mode="H")
        self.Ef = partial(self.tf_series, mode="E")
        return self.Gf, self.Hf, self.Ef

    def noise_propagation_t(self, tau, noise, unit=Dimension(dimensionless=True), _from='PD', _to=None, view=False):
        """ noise propagain in time
        Args:
            tau: time array (sec)
            noise: noise in time
            unit: unit of the noise
            _from: a staring PLL component [str]
            _to: a stopping loop component (this component itself will NOT be included) [str]
            view: print information or not
        """

        component = self.point_to_point_component(_from, _to, suppression=True, view=view)
        pll_response = control.forced_response(component.TE, T=tau, U=noise)
        noise_prop = pll_response.outputs
        unit_prop = component.unit * unit

        return noise_prop, unit_prop

    def noise_propagation_asd(self, f, asd, unit=Dimension(dimensionless=True), _from='PD', _to=None, view=False, isTF=True):
        """ noise propagain in frequency
        Args:
            f: fourier frequencies (Hz)
            asd: noise ASD
            unit: unit of the noise (1/sqrt(Hz) is not included)
            _from: a staring PLL component [str]
            _to: a stopping PLL component (this component is NOT included) [str]
            view: print information or not
            isTF: use TF attribute to compute a transfer function
        """

        # : compute the transfer function through the propagation path
        component = self.point_to_point_component(_from, _to, suppression=True, view=view)

        # : compute TF
        if isTF:
            TF = self.point_to_point_tf(f, _from, _to, suppression=True, view=False)
            mag = abs(TF)
            phase = np.angle(TF, deg=False)
            bode={'f':f, 'mag':mag, 'phase':phase}
        else:
            bode, _, _ = component.bode(2*np.pi*f)

        # : compute the noise ASD
        asd_prop = bode['mag'] * asd
        unit_prop = component.unit * unit

        # : compute RMS of the new ASD
        rms = hexaux.integral_rms(f, asd_prop, [0, np.inf])

        return asd_prop, unit_prop, bode, rms

    def collect_components(self, _from=None, _to=None):
        """ collet loop components on the path (requires Python>=3.7)
        Args:
            components: list composed of loop components [child]
            _from: a staring PLL component [str]
            _to: a stopping PLL component (this component is NOT included) [str]
        """
        if _to is None:
            return [], ""

        keys = self.components_dict.keys()
        if _from is not None: assert _from in keys, logger.error("Starting component does not exist")
        if _to is not None: assert _to in keys, logger.error("End component does not exist")
        keys = list(self.components_dict.keys())
        propagation_path = "->"
        
        compo_list = []
        start_index = None
        end_index = None

        if _from is not None:
            start_index = keys.index(_from)
            if (_from == _to)or(_to == None):
                sequence = keys[start_index:] + keys[:start_index]
            else:
                end_index = keys.index(_to)
                if start_index < end_index:
                    sequence = keys[start_index:end_index]
                else:
                    sequence = keys[start_index:] + keys[:end_index]
        else:
            start_index = keys.index(_to)
            sequence = keys[start_index:] + keys[:start_index]
            sequence.reverse()

        for key in sequence:
            propagation_path += key + "->"
            compo_list.append(self.components_dict[key])

        return compo_list, propagation_path

    def point_to_point_component(self, _from=None, _to=None, suppression=False, view=False):
        """ compute a point-to-point loop component
        Args:
            components: list composed of loop components [child]
            _from: a staring PLL component [str]
            _to: a stopping PLL component (this component is NOT included) [str]
            suppression: suppression by 1/1+G or not 
            view: print information or not
        """

        # : collect components along with the path
        compo_list, propagation_path = self.collect_components(_from, _to)

        # : compute an output component
        if suppression:
            compo_list.append(self.Ec)
        output = np.prod(compo_list)
        if (_from == _to) and suppression: # WARNING: just a temporal solution to this case
            output = copy.deepcopy(self.Hc)

        if view:
            print(f"propagation path: {propagation_path}")

        return output

    def point_to_point_tf(self, f, _from, _to=None, suppression=False, view=False):
        """ compute a point-to-point loop transfer function
        Args:
            f: fourier frequencies (Hz)
            components: list composed of ALL loop components [child]
            _from: a staring PLL component [str]
            _to: a stopping PLL component (this component is NOT included) [str]
            suppression: suppression by 1/1+G or not 
            view: print information or not  
        """

        # : collect components along with the path
        compo_list, propagation_path = self.collect_components(_from, _to)

        # : compute an output transfer function
        output = self.tf_series(f=f, components=compo_list)
        if suppression:
            output *= self.Ef(f=f)
        if _from == _to: # WARNING: just a temporal solution to this case
            output = self.Hf(f=f)

        if view:
            print(f"propagation path: {propagation_path}")

        return output

    def tf_series(self, f, components=None, mode=None, extrapolate=False, f_trans=1e-1, power=-2, size=2, solver=True):
        """ extract a series of polar transfer funcitons from components
        Args:
            f: fourier frequencies (Hz)
            mode: None for a simple series, 'H' for a system function, or 'E' for an error function
            extrapolate: extrapolate the tf in a power law
            f_trans: transition frequency
            power: power in power law
            size: point size to be used for fit (not needed for solver)
            solver: use solver or not (not = fit)
        """

        tf = 1
        if components is None:
            for component in self.components_dict:
                tf *= self.components_dict[component].TF(f=f)
        else:
            for component in components:
                tf *= component.TF(f=f)

        if mode is None:
            output = tf
        elif mode == "H":
            output = tf/(1+tf)
        elif mode == "E":
            output = 1/(1+tf)
        else:
            raise ValueError(f"invalid mode {mode}")

        if extrapolate:
            output = aux.tf_power_extrapolate(f, output, f_trans=f_trans, power=power, size=size, solver=solver)

        return output