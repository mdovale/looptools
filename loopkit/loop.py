# BSD 3-Clause License

# Copyright (c) 2025, Miguel Dovale

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
#
from __future__ import annotations

import base64
import copy
import html
import itertools
import logging
import warnings
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import control
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from loopkit import dsp
from loopkit.component import Component
from loopkit.dimension import Dimension
from loopkit.loopmath import tf_power_extrapolate
from loopkit.plots import default_rc

logger = logging.getLogger(__name__)


class LOOP:
    def __init__(
        self,
        sps: float,
        component_list: Optional[Sequence[Component]] = None,
        name: str = "Loop",
    ) -> None:
        """
        Base class for defining a control loop system.

        This class manages loop components, their transfer functions,
        property delegation, and callback mechanisms. It is designed
        to facilitate dynamic loop configuration and simulation.

        Parameters
        ----------
        sps : float
            Loop sample rate in Hz.
        component_list : sequence of Component, optional
            If provided, add all components in this list to the loop.
        name : str, optional
            Name of the loop.

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
        if not isinstance(sps, (int, float)) or sps <= 0:
            raise ValueError(f"sps must be a positive number, got {sps!r}")
        if name is None or not isinstance(name, str) or not str(name).strip():
            raise ValueError(f"name must be a non-empty string, got {name!r}")
        self.sps = float(sps)
        self.name = str(name).strip()
        self.components_dict = {}
        self.property_list = []
        self.callbacks = []
        self.Gc = None
        self.Hc = None
        self.Ec = None
        self.Gf = None
        self.Hf = None
        self.Ef = None
        self.phase = None
        self.phase_deg = None
        self.mag = None
        self.mag_dB = None
        self.phase_unwrapped = None
        self.phase_deg_unwrapped = None

        if component_list is not None:
            for comp in component_list:
                self.add_component(comp)
            self.update()

    def update(self) -> None:
        """
        Compute transfer elements and prepare callable transfer functions of the loop.

        Calculates:
        - Gc: The open-loop transfer element G(z)
        - Hc: The closed-loop (complementary sensitivity) transfer element H(z)
        - Ec: The error transfer element E(z)
        - Gf: Open-loop transfer function (callable)
        - Hf: Closed-loop transfer function (callable)
        - Ef: Error function (callable)
        - mag, phase, mag_dB, phase_deg, and unwrapped variants for Gf
        """
        # Transfer elements:
        self.Gc = np.prod(list(self.components_dict.values()))

        H_TE = control.feedback(self.Gc.TE, 1)
        self.Hc = Component("H", self.sps, tf=H_TE, unit=self.Gc.unit)

        E_TE = control.feedback(1, self.Gc.TE)
        self.Ec = Component("E", self.sps, tf=E_TE, unit=self.Gc.unit)
        
        # Transfer functions (expect Hz input):
        self.Gf = partial(self.tf_series, mode=None)
        self.Hf = partial(self.tf_series, mode="H")
        self.Ef = partial(self.tf_series, mode="E")

        def _get_phase(tf_func, frfr, deg):
            return np.angle(tf_func(frfr), deg=deg)

        def _get_magnitude(tf_func, frfr, dB):
            mag = np.abs(tf_func(frfr))
            return control.mag2db(mag) if dB else mag

        self.phase = lambda frfr: _get_phase(self.Gf, frfr, deg=False)
        self.phase_deg = lambda frfr: _get_phase(self.Gf, frfr, deg=True)
        self.mag = lambda frfr: _get_magnitude(self.Gf, frfr, dB=False)
        self.mag_dB = lambda frfr: _get_magnitude(self.Gf, frfr, dB=True)
        self.phase_unwrapped = lambda frfr: np.unwrap(self.phase(frfr))
        self.phase_deg_unwrapped = lambda frfr: np.unwrap(self.phase_deg(frfr), period=360)


    def notify_callbacks(self) -> None:
        """
        Execute all registered callback functions.
        """
        for callback, args, kwargs in self.callbacks:
            callback(*args, **kwargs)
        
    def add_component(
        self,
        newcomp: Component,
        loop_update: bool = False,
    ) -> None:
        """
        Add a new component to the control loop.

        Parameters
        ----------
        newcomp : Component
            Component object to be added. Must have a non-empty `name` attribute.
        loop_update : bool, optional
            If True, updates the loop after adding the component.

        Raises
        ------
        ValueError
            If component has empty name or name already exists.
        """
        if not newcomp.name or not str(newcomp.name).strip():
            raise ValueError("Attempting to add unnamed component")
        if newcomp.name in self.components_dict:
            logger.error("Named component already exists in the System, use `replace_component` instead")
            return
        else:
            newcomp._loop = self
            self.components_dict[newcomp.name] = newcomp
            if loop_update:
                self.update()

    def remove_component(
        self,
        name: str,
        loop_update: bool = False,
    ) -> None:
        """
        Remove a component from the loop by name.

        Parameters
        ----------
        name : str
            Name of the component to remove.
        loop_update : bool, optional
            If True, updates the loop after removing the component.

        Raises
        ------
        ValueError
            If component name does not exist.
        """
        if name not in self.components_dict:
            raise ValueError(f"Attempting to remove inexistent component: {name!r}")
        del self.components_dict[name]
        if loop_update:
            self.update()

    def replace_component(
        self,
        name: str,
        newcomp: Component,
        loop_update: bool = False,
    ) -> None:
        """
        Replace an existing component with a new one.

        Parameters
        ----------
        name : str
            Name of the component to replace.
        newcomp : Component
            New component object.
        loop_update : bool, optional
            If True, updates the loop after replacement.

        Raises
        ------
        ValueError
            If component name does not exist.
        """
        if name not in self.components_dict:
            raise ValueError(f"Attempting to replace inexistent component: {name!r}")
        self.components_dict[name] = newcomp
        if loop_update:
            self.update()

    def update_component(
        self,
        component: str,
        prop_name: str,
        newvalue: Any,
        loop_update: bool = False,
    ) -> None:
        """
        Update a specific property of a component in the loop.

        Parameters
        ----------
        component : str
            Name of the component to modify.
        prop_name : str
            Property name to update.
        newvalue : any
            New value to set.
        loop_update : bool, optional
            If True, updates the loop after modifying the property.

        Raises
        ------
        ValueError
            If component or property does not exist.
        """
        if component not in self.components_dict:
            raise ValueError(f"Attempting to update inexistent component: {component!r}")
        comp = self.components_dict[component]
        if getattr(comp, "properties", None) is None or prop_name not in comp.properties:
            raise ValueError(
                f"Attempting to modify inexistent component attribute: {prop_name!r}"
            )
        comp.properties[prop_name][1](newvalue)
        if loop_update:
            self.update()

    def register_callback(
        self,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
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
        
    def register_component_properties(self) -> None:
        """
        Register delegators for all properties of components in the loop.
        """
        for name, comp in self.components_dict.items():
            if getattr(comp, 'properties', None) is not None:
                for prop in comp.properties:
                    self.create_property_delegator(name, prop)
        
    def create_property_delegator(
        self,
        component_name: str,
        prop_name: str,
    ) -> None:
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
        sys_property_name = component_name + "_" + prop_name

        setattr(self.__class__, sys_property_name, property(get_prop, set_prop))

        self.property_list.append(sys_property_name)

    def block_diagram(
        self,
        dpi: int = 150,
        filename: Optional[str] = None,
        transfer_functions: bool = True,
    ) -> Optional[Any]:
        """
        Generate a TikZ block diagram of the LOOP structure.

        Components are laid out in a rectangular loop:
        top row flows left to right, then turns downward;
        bottom row flows right to left, then upward to close the loop.

        Special case: if the loop has only one component, a minimal vertical layout is used.

        Parameters
        ----------
        dpi : int
            Dots per inch for the rendered PNG image.
        filename : str
            If given, saves the LaTeX code to this file.
        transfer_functions : bool
            If True, show transfer function inside component boxes.

        Returns
        -------
        tikz.Picture
            The rendered TikZ picture object.

        Raises
        ------
        ImportError
            If the optional dependencies (PyMuPDF, pytikz, IPython) are not installed.
            Install with: pip install loopkit[diagram]
            pytikz must be installed from git: pip install git+https://github.com/allefeld/pytikz.git
        """
        try:
            import fitz
            import IPython.display
            import tikz
        except ImportError as e:
            raise ImportError(
                "block_diagram() requires optional dependencies: PyMuPDF, pytikz, and IPython. "
                "Install with: pip install loopkit[diagram] and "
                "pip install git+https://github.com/allefeld/pytikz.git"
            ) from e

        import html
        import base64

        def tikz_safe(name):
            return name.replace(" ", "").replace("-", "")

        def tex_fraction(tf):
            """
            Generate a LaTeX math expression for a transfer function in s or z domain.

            Parameters
            ----------
            tf : control.TransferFunction
                The transfer function object to render.

            Returns
            -------
            str
                A LaTeX string in the form "$\\frac{numerator}{denominator}$", or None if tf is invalid.
            """
            def fmt(c):
                if abs(c) < 1e-3 or abs(c) > 1e3:
                    base, exp = f"{c:.1e}".split("e")
                    return f"{base}\\times 10^{{{int(exp)}}}"
                else:
                    return f"{c:.3g}"

            def poly_to_tex(p, var='s', use_zinv=False):
                terms = []
                N = len(p)
                for i, c in enumerate(p):
                    if abs(c) < 1e-12:
                        continue

                    power = N - i - 1
                    sign = "-" if c < 0 else "+"
                    abs_c = abs(c)

                    # Skip coefficient display for ±1 unless constant term
                    if abs_c == 1.0 and power != 0:
                        coeff_str = ""
                    else:
                        coeff_str = fmt(abs_c)

                    # Format variable
                    if power == 0:
                        term = f"{coeff_str}"
                    elif use_zinv:
                        term = f"{coeff_str}{var}^{{-{power}}}"
                    elif power == 1:
                        term = f"{coeff_str}{var}"
                    else:
                        term = f"{coeff_str}{var}^{{{power}}}"

                    if not terms:
                        terms.append(f"-{term}" if c < 0 else term)  # First term
                    else:
                        terms.append(f" {sign} {term}")

                return "".join(terms) if terms else "0"

            try:
                num = tf.num[0][0]
                den = tf.den[0][0]
            except Exception:
                return None

            # Detect domain
            if tf.dt is not None:
                var = 'z'
                use_zinv = True  # Discrete TFs rendered in z⁻¹ form
            else:
                var = 's'
                use_zinv = False

            num_tex = poly_to_tex(num, var, use_zinv)
            den_tex = poly_to_tex(den, var, use_zinv)
            return f"$\\frac{{{num_tex}}}{{{den_tex}}}$"
        
        def render_and_display(pic):
            pic._update()
            self.pic = pic
            self.block_diagram_code = pic.document_code()

            if filename is not None:
                with open(filename, 'w') as f:
                    f.write(self.block_diagram_code)

            zoom = dpi / 72
            doc = fitz.open(pic.temp_pdf)
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            pngdata = pix.tobytes("png")
            png_base64 = base64.b64encode(pngdata).decode('ascii')
            demo_template = '\n'.join([
                '<div style="background-color:#e0e0e0;padding:10px;">',
                '  <img src="data:image/png;base64,{0}">',
                '</div>'
            ])
            IPython.display.display(IPython.display.HTML(demo_template.format(png_base64)))

        def make_block_label(name, tf_obj):
            label = html.escape(name)
            tf_latex = tex_fraction(tf_obj) if transfer_functions and tf_obj else None
            return f"{label}\\\\{tf_latex}" if tf_latex else label

        names = list(self.components_dict.keys())
        if not names:
            print("[block_diagram] No components found.")
            return None

        pic = tikz.Picture()
        preamble = r"""
\usetikzlibrary{arrows.meta,positioning,calc}
\tikzset{
block/.style={draw, fill=white, rectangle, minimum height=3em, minimum width=6em},
sum/.style={draw, fill=white, circle},
arrow/.style={->, >=latex}
    }
        """
        pic.add_preamble(preamble)
        code = []

        # --- Special case: single-component loop ---
        if len(names) == 1:
            raw_name = names[0]
            safe_name = tikz_safe(raw_name)
            label = make_block_label(raw_name, getattr(self.components_dict[raw_name], 'TE', None))

            code.append(f"\\node [block, align=center] ({safe_name}) {{{label}}};")
            code.append(f"\\node [sum, below=1cm of {safe_name}] (sum{safe_name}) {{+}};")
            code += [
                f"\\coordinate (fwd1) at ($({safe_name}.east) + (5mm, 0)$);",
                f"\\coordinate (fwd2) at ($(fwd1) + (0, -1mm)$);",
                f"\\draw [arrow] ({safe_name}.east) -- (fwd1) -- (fwd2) |- (sum{safe_name}.east);",
                f"\\coordinate (back1) at ($({safe_name}.west) + (-5mm, 0)$);",
                f"\\coordinate (back2) at ($(back1) + (0, -1mm)$);",
                f"\\draw [<-, >=latex] ({safe_name}.west) -| (back1) -- (back2) |- (sum{safe_name}.west) node[midway, left]{{{self.name}}};"
            ]
            pic._append(tikz.Raw('\n'.join(code)))
            render_and_display(pic)
            return

        # --- General case: multiple components ---
        n = len(names)
        top_names = names[: (n + 1) // 2]
        bottom_names = names[(n + 1) // 2:]

        top_nodes, bottom_nodes = [], []

        for idx, raw_name in enumerate(top_names):
            safe_name = tikz_safe(raw_name)
            label = make_block_label(raw_name, getattr(self.components_dict[raw_name], 'TE', None))
            pos = "" if idx == 0 else f"right=of {top_nodes[-1]}"
            code.append(f"\\node [sum] (sum{safe_name}) [{pos}] {{+}};")
            code.append(f"\\node [block, right=of sum{safe_name}, align=center] ({safe_name}) {{{label}}};")
            top_nodes.append(safe_name)

        for idx, raw_name in enumerate(bottom_names):
            safe_name = tikz_safe(raw_name)
            label = make_block_label(raw_name, getattr(self.components_dict[raw_name], 'TE', None))
            if idx == 0:
                anchor = top_nodes[-1]
                code.append(f"\\node [sum, below=2cm of {anchor}] (sum{safe_name}) {{+}};")
                code.append(f"\\node [block, left=of sum{safe_name}, align=center] ({safe_name}) {{{label}}};")
            else:
                prev_safe = tikz_safe(bottom_names[idx - 1])
                code.append(f"\\node [sum, left=of {prev_safe}] (sum{safe_name}) {{+}};")
                code.append(f"\\node [block, left=of sum{safe_name}, align=center] ({safe_name}) {{{label}}};")
            bottom_nodes.append(safe_name)

        flow = top_nodes + bottom_nodes
        for i, name in enumerate(flow):
            code.append(f"\\draw [arrow] (sum{name}) -- ({name});")
            next_idx = (i + 1) % len(flow)
            next_name = flow[next_idx]
            from_node = name
            to_node = f"sum{next_name}"

            if i == len(top_nodes) - 1:
                code.append(f"\\draw [arrow] ({from_node}) -- ({to_node});")
            elif next_idx == 0:
                code.append(f"""
% Always extend loop to the left of both nodes
\\path let
\\p1 = ({to_node}.west),
\\p2 = ({from_node}.west),
\\n1 = {{min(\\x1,\\x2)-5mm}},
\\n2 = {{\\y2}}
in
coordinate (loop_corner) at (\\n1,\\n2);

\\draw [-] ({from_node}.west) -- (loop_corner);
\\draw [arrow] (loop_corner) |- ({to_node}.west) node[midway, left]{{{self.name}}};""")
            else:
                code.append(f"\\draw [arrow] ({from_node}) -- ({to_node});")

        pic._append(tikz.Raw('\n'.join(code)))
        render_and_display(pic)

    def magnitude_plot(
        self,
        frfr: ArrayLike,
        figsize: Tuple[float, float] = (5, 4),
        title: Optional[str] = None,
        which: Union[str, Sequence[str]] = "G",
        ax: Optional[Any] = None,
        label: Optional[str] = None,
        label_prefix: Optional[str] = None,
        legend: bool = True,
        dB: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Plot the magnitude of selected loop transfer functions at specified frequencies.

        Parameters
        ----------
        frfr : array_like
            Frequency array in Hz at which to evaluate the transfer functions.
        figsize : tuple, optional
            Figure size for the plot.
        title : str, optional
            Title for the magnitude plot.
        which : str or list, optional
            Which transfer functions to plot: 'G', 'H', 'E', or a list thereof.
        axes : matplotlib.axes.Axes, optional
            Existing matplotlib Axes to plot into. If None, creates a new figure and axes.
        label : str, optional
            Label prefix for the plotted line (e.g., 'Loop 1: '). Defaults to empty string.
        dB : bool, optional
            If True, plot magnitude in dB (20*log10). Otherwise, plot absolute magnitude.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axis object used for the plot.
        """
        with plt.rc_context(default_rc):
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=figsize)
            else:
                ax = ax
                fig = ax.figure

            label_prefix = f"{label_prefix}: " if label_prefix else ""

            if which == 'all':
                which = ['G', 'H', 'E']
            else:
                which = [w.upper() for w in which] if isinstance(which, (list, tuple)) else [which.upper()]
                for w in which:
                    if w not in ('G', 'H', 'E'):
                        raise ValueError(f"Invalid transfer function key: '{w}'. Use 'G', 'H', or 'E'.")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                def plot_tf(tf_func, name):
                    val = tf_func(f=frfr)
                    mag = 20 * np.log10(np.abs(val)) if dB else np.abs(val)
                    if label is not None:
                        ax.semilogx(frfr, mag, label=label_prefix+label, *args, **kwargs)
                    else:
                        ax.semilogx(frfr, mag, label=label_prefix+name, *args, **kwargs)

                if 'G' in which:
                    plot_tf(self.Gf, 'G')
                if 'H' in which:
                    plot_tf(self.Hf, 'H')
                if 'E' in which:
                    plot_tf(self.Ef, 'E')

            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude (dB)" if dB else "Magnitude")
            ax.set_xlim(frfr[0], frfr[-1])

            if legend:
                ax.legend(loc='upper left', 
                        bbox_to_anchor=(1, 1), 
                        edgecolor='black', 
                        fancybox=True, 
                        shadow=True, 
                        framealpha=1,
                        fontsize=8)

            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle='--', linewidth=0.5)

            if title is not None:
                ax.set_title(title)

            fig.tight_layout()
            return ax

    def bode_plot(
        self,
        frfr: ArrayLike,
        figsize: Tuple[float, float] = (5, 5),
        title: Optional[str] = None,
        which: Union[str, Sequence[str]] = "all",
        axes: Optional[Tuple[Any, Any]] = None,
        label: Optional[str] = None,
        label_prefix: Optional[str] = None,
        legend: bool = True,
        dB: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Plot the Bode diagram of the loop's Gf, Hf, and Ef.

        Parameters
        ----------
        frfr : array_like
            Frequency array in Hz at which to evaluate the transfer functions.
        figsize : tuple, optional
            Figure size for the plot.
        title : str, optional
            Title for the magnitude plot.
        which : str or list, optional
            Which transfer functions to plot: 'G', 'H', 'E', or 'all'.
        axes : tuple of matplotlib.axes.Axes, optional
            Existing axes to plot into. If None, creates new figure and axes.
        label : str, optional
            Base label prefix for this loop's lines (e.g., 'Loop 1'). If None, defaults to empty.
        dB : bool, optional
            If True, plot magnitude in dB (20*log10). Otherwise, plot absolute magnitude.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plots.
        axes : tuple of matplotlib.axes.Axes
            The magnitude and phase axes used.
        """
        frfr = np.asarray(frfr)
        
        with plt.rc_context(default_rc):
            if axes is None:
                fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=figsize, sharex=True)
            else:
                ax_mag, ax_phase = axes
                fig = ax_mag.figure

            label_prefix = f"{label_prefix}: " if label_prefix else ""

            if which == 'all':
                which = ['G', 'H', 'E']
            else:
                which = [w.upper() for w in which]
                for w in which:
                    if w not in ('G', 'H', 'E'):
                        raise ValueError(f"Invalid transfer function key: '{w}'. Use 'G', 'H', or 'E'.")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                def plot_tf(tf_func, name):
                    val = tf_func(f=frfr)
                    mag = 20 * np.log10(np.abs(val)) if dB else np.abs(val)
                    ax_mag_func = ax_mag.semilogx if dB else ax_mag.loglog
                    if label is not None:
                        ax_mag_func(frfr, mag, label=label_prefix+label, *args, **kwargs)
                    else:
                        ax_mag_func(frfr, mag, label=label_prefix+name, *args, **kwargs)
                    ax_phase.semilogx(frfr, np.angle(val, deg=True), *args, **kwargs)

                if 'G' in which:
                    plot_tf(self.Gf, "G")
                if 'H' in which:
                    plot_tf(self.Hf, "H")
                if 'E' in which:
                    plot_tf(self.Ef, "E")

            ax_phase.set_xlabel("Frequency (Hz)")
            ax_mag.set_ylabel("Magnitude (dB)" if dB else "Magnitude")
            ax_phase.set_ylabel("Phase (deg)")

            ax_mag.set_xlim(frfr[0], frfr[-1])
            ax_phase.set_xlim(frfr[0], frfr[-1])

            if legend:
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

            return (ax_mag, ax_phase)
        
    def nyquist_plot(
        self,
        frfr: ArrayLike,
        which: Union[str, Sequence[str]] = "all",
        critical_point: bool = False,
        arrow_scale: float = 1.0,
        arrow_frequency: Optional[float] = None,
        figsize: Tuple[float, float] = (4, 4),
        title: Optional[str] = None,
        ax: Optional[Any] = None,
        label: Union[str, bool] = "",
        logx: bool = False,
        logy: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
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

            return ax
        
    def noise_propagation_t(
        self,
        tau: ArrayLike,
        noise: ArrayLike,
        unit: Optional[Dimension] = None,
        _from: str = "PD",
        _to: Optional[str] = None,
        view: bool = False,
    ) -> Tuple[NDArray[np.floating[Any]], Dimension]:
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
        if unit is None:
            unit = Dimension(dimensionless=True)
        component = self.point_to_point_component(_from, _to, closed=True, view=view)
        pll_response = control.forced_response(component.TE, T=tau, U=noise)
        noise_prop = pll_response.outputs
        unit_prop = component.unit * unit

        return noise_prop, unit_prop

    def noise_propagation_asd(
        self,
        f: ArrayLike,
        asd: ArrayLike,
        unit: Optional[Dimension] = None,
        _from: str = "PD",
        _to: Optional[str] = None,
        view: bool = False,
        isTF: bool = True,
    ) -> Tuple[NDArray[np.floating[Any]], Dimension, dict, float]:
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
        if unit is None:
            unit = Dimension(dimensionless=True)

        # Compute the transfer function through the propagation path
        component = self.point_to_point_component(_from, _to, closed=True, view=view)

        # Compute TF
        if isTF:
            TF = self.point_to_point_tf(f, _from, _to, closed=True, view=False)
            mag = abs(TF)
            phase = np.angle(TF, deg=False)
            bode = {"f": f, "mag": mag, "phase": phase}
        else:
            bode, _, _ = component.bode(2*np.pi*f)

        # Compute the noise ASD
        asd_prop = bode['mag'] * asd
        unit_prop = component.unit * unit

        # Compute RMS of the new ASD
        rms = dsp.integral_rms(f, asd_prop, [0, np.inf])

        return asd_prop, unit_prop, bode, rms

    def collect_components(
        self,
        _from: Optional[str] = None,
        _to: Optional[str] = None,
    ) -> Tuple[List[Component], str]:
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

        keys = list(self.components_dict.keys())
        if _from is not None and _from not in keys:
            raise ValueError(f"Starting component does not exist: {_from!r}")
        if _to is not None and _to not in keys:
            raise ValueError(f"End component does not exist: {_to!r}")
        propagation_path = "->"
        
        compo_list = []
        start_index = None
        end_index = None

        if _from is not None:
            start_index = keys.index(_from)
            if _from == _to or _to is None:
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

    def point_to_point_component(
        self,
        _from: Optional[str] = None,
        _to: Optional[str] = None,
        closed: bool = False,
        view: bool = False,
    ) -> Component:
        """
        Compute a compound component representing a segment of the loop.

        This component reflects a product of all components between `_from` and `_to`,
        optionally applying closed-loop transfer (i.e., multiplying by E(z)).

        Parameters
        ----------
        _from : str, optional
            Starting component name.
        _to : str, optional
            Stopping component name (excluded).
        closed : bool, optional
            If True, apply closed-loop transfer using E(z).
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
        if closed:
            compo_list.append(self.Ec)
        output = np.prod(compo_list)
        if (_from == _to) and closed: # WARNING: just a temporal solution to this case
            output = copy.deepcopy(self.Hc)

        if view:
            print(f"propagation path: {propagation_path}")

        return output

    def point_to_point_tf(
        self,
        f: ArrayLike,
        _from: str,
        _to: Optional[str] = None,
        closed: bool = False,
        view: bool = False,
    ) -> NDArray[np.complexfloating[Any, Any]]:
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
        closed : bool, optional
            If True, apply closed-loop transfer using E(z).
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
        if closed:
            output *= self.Ef(f=f)
        if _from == _to: # WARNING: just a temporal solution to this case
            output = self.Hf(f=f)

        if view:
            print(f"propagation path: {propagation_path}")

        return output

    def tf_series(
        self,
        f: ArrayLike,
        components: Optional[Sequence[Component]] = None,
        mode: Optional[str] = None,
        extrapolate: bool = False,
        f_trans: float = 1e-1,
        power: float = -2,
        size: int = 2,
        solver: bool = True,
    ) -> NDArray[np.complexfloating[Any, Any]]:
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