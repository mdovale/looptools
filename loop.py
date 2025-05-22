import numpy as np
import control
import copy
from functools import partial
from looptools.component import Component
from looptools.dimension import Dimension
import looptools.auxiliary as aux
import logging
logger = logging.getLogger(__name__)


class LOOP:
    def __init__(self, sps, component_list=None):
        """
        Base class for defining a control loop system.

        This class manages loop components, their transfer functions, 
        property delegation, and callback mechanisms. It is designed 
        to facilitate dynamic loop configuration and simulation.

        Parameters
        ----------
        sps : float
            Loop sample rate in Hz.
        component_list : list of Component
            If provided, add all components in this list to the loop. 

        Attributes
        ----------
        sps : float
            Sample rate of the control loop.
        components_dict : dict
            Dictionary mapping component names to component objects.
        property_list : list
            List of registered property delegators.
        callbacks : list
            List of registered callback functions with arguments.
        Gc, Hc, Ec, Gf, Hf, Ef : object or None
            Transfer function components (control and feedback path elements).
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

        if component_list is not None:
            for comp in component_list:
                self.add_component(comp)
            self.update()

    def update(self):
        """
        Update internal system transfer functions and component states.
        """
        self.system_transfer_components()
        self.system_transfer_functions()

    def notify_callbacks(self):
        """
        Execute all registered callback functions.
        """
        for callback, args, kwargs in self.callbacks:
            callback(*args, **kwargs)
        
    def add_component(self, newcomp, loop_update=False):
        """
        Add a new component to the control loop.

        Parameters
        ----------
        newcomp : object
            Component object to be added. Must have a non-empty `name` attribute.
        loop_update : bool, optional
            If True, updates the loop after adding the component.
        """
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
        """
        Remove a component from the loop by name.

        Parameters
        ----------
        name : str
            Name of the component to remove.
        loop_update : bool, optional
            If True, updates the loop after removing the component.
        """
        assert name in self.components_dict, logger.error("Attempting to remove inexistent component")
        del self.components_dict[name]
        if loop_update:
            self.update()

    def replace_component(self, name, newcomp, loop_update=False):
        """
        Replace an existing component with a new one.

        Parameters
        ----------
        name : str
            Name of the component to replace.
        newcomp : object
            New component object.
        loop_update : bool, optional
            If True, updates the loop after replacement.
        """
        assert name in self.components_dict, logger.error("Attempting to replace inexistent component")
        self.components_dict[name] = newcomp
        if loop_update:
            self.update()

    def update_component(self, component, property, newvalue, loop_update=False):
        """
        Update a specific property of a component in the loop.

        Parameters
        ----------
        component : str
            Name of the component to modify.
        property : str
            Property name to update.
        newvalue : any
            New value to set.
        loop_update : bool, optional
            If True, updates the loop after modifying the property.
        """
        assert component in self.components_dict, logger.error("Attempting to update inexistent component")
        assert property in self.components_dict[component].properties,  logger.error("Attempting to modify inexistent component attribute")
        self.components_dict[component].properties[property][1](newvalue)
        if loop_update:
            self.update()

    def register_callback(self, callback, *args, **kwargs):
        """
        Register a callback function to be executed on trigger.

        Parameters
        ----------
        callback : callable
            Function to be called.
        *args : tuple
            Positional arguments for the callback.
        **kwargs : dict
            Keyword arguments for the callback.
        """
        self.callbacks.append((callback, args, kwargs))
        
    def register_component_properties(self):
        """
        Register delegators for all properties of components in the loop.
        """
        for name, comp in self.components_dict.items():
            if getattr(comp, 'properties', None) is not None:
                for prop in comp.properties:
                    self.create_property_delegator(name, prop)
        
    def create_property_delegator(self, component_name, prop_name):
        """
        Create dynamic getter and setter for a component property.

        Parameters
        ----------
        component_name : str
            Name of the component.
        prop_name : str
            Property name to delegate.
        """
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
        """
        Compute system-level transfer elements of the control loop.

        Calculates:
        - Gc: The open-loop transfer element G(z)
        - Hc: The closed-loop (complementary sensitivity) transfer element H(z)
        - Ec: The error transfer element E(z)

        Returns
        -------
        tuple
            (Gc, Hc, Ec) components as `Component` objects.
        """
        self.Gc = np.prod(list(self.components_dict.values()))
        self.Gc.name = "G"

        H_TE = control.feedback(self.Gc.TE, 1)
        self.Hc = Component("H", self.sps, tf=H_TE, unit=self.Gc.unit)

        E_TE = control.feedback(1, self.Gc.TE)
        self.Ec = Component("E", self.sps, tf=E_TE, unit=self.Gc.unit)

        return self.Gc, self.Hc, self.Ec

    def system_transfer_functions(self):
        """
        Prepare callable frequency-domain transfer functions.

        Defines partial function handles for evaluating the system's transfer
        functions in the frequency domain:
        - Gf: Open-loop
        - Hf: Closed-loop
        - Ef: Error

        Returns
        -------
        tuple
            (Gf, Hf, Ef) as callable partial functions.
        """

        self.Gf = partial(self.tf_series, mode=None)
        self.Hf = partial(self.tf_series, mode="H")
        self.Ef = partial(self.tf_series, mode="E")
        return self.Gf, self.Hf, self.Ef

    def noise_propagation_t(self, tau, noise, unit=Dimension(dimensionless=True), _from='PD', _to=None, view=False):
        """
        Propagate a time-domain noise signal through a defined segment of the loop.

        Parameters
        ----------
        tau : array_like
            Time vector in seconds.
        noise : array_like
            Noise signal (time-domain).
        unit : Dimension, optional
            Unit of the noise signal.
        _from : str
            Starting component name.
        _to : str, optional
            Stopping component name (excluded).
        view : bool, optional
            Print propagation path information if True.

        Returns
        -------
        tuple
            (noise_prop, unit_prop), the propagated noise signal and its resulting unit.
        """
        component = self.point_to_point_component(_from, _to, suppression=True, view=view)
        pll_response = control.forced_response(component.TE, T=tau, U=noise)
        noise_prop = pll_response.outputs
        unit_prop = component.unit * unit

        return noise_prop, unit_prop

    def noise_propagation_asd(self, f, asd, unit=Dimension(dimensionless=True), _from='PD', _to=None, view=False, isTF=True):
        """
        Propagate a noise ASD through a defined segment of the loop.

        Parameters
        ----------
        f : array_like
            Frequency vector in Hz.
        asd : array_like
            Amplitude spectral density of the noise.
        unit : Dimension, optional
            Unit of the ASD (excluding 1/sqrt(Hz)).
        _from : str
            Starting component name.
        _to : str, optional
            Stopping component name (excluded).
        view : bool, optional
            Print propagation path if True.
        isTF : bool, optional
            If True, compute propagation using point-to-point transfer function.

        Returns
        -------
        tuple
            (asd_prop, unit_prop, bode, rms)
            - Propagated ASD
            - Resulting unit
            - Bode dictionary with 'f', 'mag', 'phase'
            - RMS value of the propagated ASD
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
        rms = aux.integral_rms(f, asd_prop, [0, np.inf])

        return asd_prop, unit_prop, bode, rms

    def collect_components(self, _from=None, _to=None):
        """
        Collect a list of components between two loop nodes.

        This defines the directional propagation path (can wrap around the loop).

        Parameters
        ----------
        _from : str, optional
            Starting component name.
        _to : str, optional
            Stopping component name (excluded).

        Returns
        -------
        tuple
            (compo_list, propagation_path) where `compo_list` is a list of components,
            and `propagation_path` is a string representation.
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
        """
        Compute a compound component representing a segment of the loop.

        This component reflects a product of all components between `_from` and `_to`,
        optionally applying error suppression (i.e., multiplying by E(z)).

        Parameters
        ----------
        _from : str, optional
            Starting component name.
        _to : str, optional
            Stopping component name (excluded).
        suppression : bool, optional
            If True, apply suppression using E(z).
        view : bool, optional
            Print propagation path if True.

        Returns
        -------
        Component
            Combined component object over the defined path.
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
        """
        Compute a compound transfer function for a defined loop segment.

        Parameters
        ----------
        f : array_like
            Frequency vector in Hz.
        _from : str
            Starting component name.
        _to : str, optional
            Stopping component name (excluded).
        suppression : bool, optional
            If True, apply suppression using E(z).
        view : bool, optional
            Print propagation path if True.

        Returns
        -------
        array_like
            Frequency-domain transfer function of the segment.
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
        """
        Compute the frequency-domain transfer function product for a series of components.

        Parameters
        ----------
        f : array_like
            Frequency vector in Hz.
        components : list, optional
            List of components; if None, all components in the loop are used.
        mode : str or None, optional
            Output mode:
            - None: basic product
            - 'H': closed-loop (G / (1+G))
            - 'E': error (1 / (1+G))
        extrapolate : bool, optional
            If True, extend transfer function using power-law extrapolation.
        f_trans : float, optional
            Transition frequency for extrapolation (Hz).
        power : float, optional
            Power-law exponent.
        size : int, optional
            Number of points to use for fit.
        solver : bool, optional
            Use solver-based extrapolation instead of point-fit.

        Returns
        -------
        array_like
            Resulting frequency response.
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