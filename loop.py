import numpy as np
import control
import copy
import itertools
import warnings
from functools import partial
from looptools.component import Component
from looptools.dimension import Dimension
from looptools.plots import default_rc
from looptools.loopmath import *
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)


class LOOP:
    def __init__(self, sps, component_list=None, name='Loop'):
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
        self.name = name
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

    def block_diagram(self, filename='loop_diagram.tex', transfer_functions=True):
        """
        Generate a rectangular-loop TikZ block diagram of the LOOP structure.

        Components are laid out in a rectangular loop:
        top row flows left to right, then turns downward;
        bottom row flows right to left, then upward to close the loop.

        Parameters
        ----------
        filename : str
            Output LaTeX filename.
        transfer_functions : bool
            If True, show transfer function inside component boxes.

        Returns
        -------
        tikz.Picture
            The rendered TikZ picture object.
        """
        import tikz
        import html

        names = list(self.components_dict.keys())
        if not names:
            print("[block_diagram] No components found.")
            return None

        n = len(names)
        top_names = names[: (n + 1) // 2]
        bottom_names = names[(n + 1) // 2:]

        pic = tikz.Picture()
        pic.add_preamble(r"""
\usetikzlibrary{arrows.meta,positioning}
\tikzset{
block/.style={draw, fill=white, rectangle, minimum height=3em, minimum width=6em},
sum/.style={draw, fill=white, circle},
arrow/.style={->, >=latex}
}
        """)
        code = []
        top_nodes = []

        # --- Top row ---
        for idx, name in enumerate(top_names):
            label = html.escape(name)
            pos = "" if idx == 0 else f"right=of {top_nodes[-1]}"
            code.append(f"\\node [sum] (sum_{name}) [{pos}] {{+}};")
            code.append(f"\\node [block, right=of sum_{name}] ({name}) {{{label}}};")
            top_nodes.append(name)

        # --- Bottom row (reversed visually) ---
        for idx, name in enumerate(bottom_names):
            label = html.escape(name)
            anchor = top_names[-(idx + 1)] if idx < len(top_names) else top_nodes[-1]
            code.append(f"\\node [sum, below=2cm of {anchor}] (sum_{name}) {{+}};")
            code.append(f"\\node [block, left=of sum_{name}] ({name}) {{{label}}};")

        # --- Draw arrows through loop ---
        flow = top_names + bottom_names
        print(flow)
        for i, name in enumerate(flow):
            code.append(f"\\draw [arrow] (sum_{name}) -- ({name});")
            next_idx = (i + 1) % len(flow)
            next_name = flow[next_idx]
            from_node = name
            to_node = f"sum_{next_name}"

            if i == len(top_names) - 1:  # Top-to-bottom corner
                code.append(f"\\draw [arrow] ({from_node}) -- ({to_node});")
            elif next_idx == 0:  # Bottom-to-top loop closure
                code.append(f"\\draw [arrow] ({from_node}) -- ({to_node}) node[midway, left]{{{self.name}}};")
            elif i < len(top_names):
                code.append(f"\\draw [arrow] ({from_node}) -- ({to_node});")
            else:
                code.append(f"\\draw [arrow] ({from_node}) -- ({to_node});")

        raw = tikz.Raw('\n'.join(code))
        pic._append(raw)

        with open(filename, 'w') as f:
            f.write(pic.document_code())

        self.pic = pic
        return pic

    def bode_plot(self, frfr, figsize=(5,5), title=None, which='all', axes=None, label="", *args, **kwargs):
        """Plot the Bode diagram of the loop's Gf, Hf, and Ef.

        Parameters
        ----------
        frfr : array_like
            Frequency array in Hz at which to evaluate the transfer functions.
        axes : tuple of matplotlib.axes.Axes, optional
            Existing axes to plot into. If None, creates new figure and axes.
        label : str, optional
            Base label prefix for this loop's lines (e.g., 'Loop 1'). If None, defaults to empty.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plots.
        axes : tuple of matplotlib.axes.Axes
            The magnitude and phase axes used.
        """
        with plt.rc_context(default_rc):
            if axes is None:
                fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=figsize, sharex=True)
            else:
                ax_mag, ax_phase = axes
                fig = ax_mag.figure

            prefix = f"{label}: " if label else ""

            if which == 'all':
                which = ['G', 'H', 'E']
            else:
                which = [w.upper() for w in which]
                for w in which:
                    if w not in ('G', 'H', 'E'):
                        raise ValueError(f"Invalid transfer function key: '{w}'. Use 'G', 'H', or 'E'.")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # Plot open-loop transfer function
                if 'G' in which:
                    G_val = self.Gf(f=frfr)
                    ax_mag.loglog(frfr, np.abs(G_val), label=prefix + "G(f)", *args, **kwargs)
                    ax_phase.semilogx(frfr, np.angle(G_val, deg=True), *args, **kwargs)

                # Plot system function
                if 'H' in which:
                    H_val = self.Hf(f=frfr)
                    ax_mag.loglog(frfr, np.abs(H_val), label=prefix + "H(f)", *args, **kwargs)
                    ax_phase.semilogx(frfr, np.angle(H_val, deg=True), *args, **kwargs)

                # Plot error function
                if 'E' in which:
                    E_val = self.Ef(f=frfr)
                    ax_mag.loglog(frfr, np.abs(E_val), label=prefix + "E(f)", *args, **kwargs)
                    ax_phase.semilogx(frfr, np.angle(E_val, deg=True), *args, **kwargs)

            ax_phase.set_xlabel("Frequency (Hz)")
            ax_mag.set_ylabel("Magnitude")
            ax_phase.set_ylabel("Phase (deg)")

            ax_mag.set_xlim(frfr[0], frfr[-1])
            ax_phase.set_xlim(frfr[0], frfr[-1])

            if label is not False:
                ax_mag.legend(loc='upper left', 
                            bbox_to_anchor=(1, 1), 
                            edgecolor='black', 
                            fancybox=True, 
                            shadow=True, 
                            framealpha=1,
                            fontsize=8)
            
            ax_mag.minorticks_on()
            ax_phase.minorticks_on()
            ax_mag.grid(True, which='minor', linestyle='--', linewidth=0.5)
            ax_phase.grid(True, which='minor', linestyle='--', linewidth=0.5)

            if title is not None:
                ax_mag.set_title(title)

            fig.tight_layout()
            fig.align_ylabels()

            return fig, (ax_mag, ax_phase)
        
    def nyquist_plot(self, frfr, which='all', critical_point=False,
                    arrow_scale=1.0, arrow_frequency=None,
                    figsize=(4, 4), title=None, ax=None, label="",
                    logx=False, logy=False, *args, **kwargs):
        """
        Plot the Nyquist diagram of the loop's Gf, Hf, and Ef transfer functions.

        Parameters
        ----------
        frfr : array_like
            Frequency array in Hz at which to evaluate the transfer functions.
        which : {'all', 'G', 'H', 'E'}, optional
            Which transfer functions to plot.
        critical_point : bool, optional
            Mark the critical point (-1, 0).
        arrow_scale : float, optional
            Size scale of the directional arrowhead.
        arrow_frequency : float, optional
            Frequency (Hz) at which to draw the direction arrow.
        figsize : tuple of float, optional
            Size of the matplotlib figure.
        title : str, optional
            Title for the plot.
        ax : matplotlib.axes.Axes, optional
            Axis to plot on.
        label : str, optional
            Prefix label for the legend.
        logx : bool, optional
            Use symmetric log scaling on x-axis.
        logy : bool, optional
            Use symmetric log scaling on y-axis.
        *args, **kwargs :
            Passed to `plot`.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        with plt.rc_context(default_rc), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure

            prefix = f"{label}: " if label else ""

            if which == 'all':
                which = ['G', 'H', 'E']
            else:
                which = [w.upper() for w in which]
                for w in which:
                    if w not in ('G', 'H', 'E'):
                        raise ValueError(f"Invalid transfer function key: '{w}'.")

            tf_map = {'G': self.Gf, 'H': self.Hf, 'E': self.Ef}
            color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

            for key in which:
                tf_func = tf_map[key]
                vals = tf_func(f=frfr)
                color = next(color_cycle)

                ax.plot(vals.real, vals.imag, label=prefix + f"{key}(f)", color=color, *args, **kwargs)

                dx = np.diff(vals.real)
                dy = np.diff(vals.imag)
                ds = np.hypot(dx, dy)
                arc_len = np.concatenate([[0], np.cumsum(ds)])
                total_len = arc_len[-1]

                if total_len == 0:
                    continue  # flat trace

                if arrow_frequency is not None:
                    idx = np.argmin(np.abs(frfr - arrow_frequency))
                    if idx >= len(vals) - 1:
                        idx = len(vals) - 2
                else:
                    midpoint_target = total_len / 2
                    idx = np.searchsorted(arc_len, midpoint_target)
                    if idx >= len(vals) - 1:
                        idx = len(vals) - 2

                x0, y0 = vals.real[idx], vals.imag[idx]
                dx = vals.real[idx + 1] - x0
                dy = vals.imag[idx + 1] - y0

                ax.annotate('', xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                            arrowprops=dict(
                                arrowstyle=f'->,head_length={arrow_scale},head_width={arrow_scale/2}',
                                color=color, lw=1.0))

            ax.set_xlabel("Re")
            ax.set_ylabel("Im")
            if logx:
                ax.set_xscale('symlog', linthresh=1e-2)
            if logy:
                ax.set_yscale('symlog', linthresh=1e-2)
            if not (logx or logy):
                ax.set_aspect('equal', adjustable='datalim')

            ax.axhline(0, color='#FFD700', linewidth=1.0)
            ax.axvline(0, color='#FFD700', linewidth=1.0)
            if critical_point:
                ax.plot([-1], [0], 'x', color='k', markersize=6, linewidth=3)

            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle='--', linewidth=0.5)

            if label is not False:
                ax.legend(loc='best', edgecolor='black', fancybox=True,
                        shadow=True, framealpha=1, fontsize=8)

            if title is not None:
                ax.set_title(title)

            try:
                fig.tight_layout()
            except Exception as e:
                warnings.warn(f"tight_layout() failed: {e}")

            return fig, ax
        
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
        rms = dsp.integral_rms(f, asd_prop, [0, np.inf])

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
            output = tf_power_extrapolate(f, output, f_trans=f_trans, power=power, size=size, solver=solver)

        return output