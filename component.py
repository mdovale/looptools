from looptools.loopmath import *
from looptools.utils import normalize_tf_string
from looptools import dimension as dim
from looptools.plots import default_rc

import warnings
import copy
import numbers
import control
import numpy as np
from functools import partial
import scipy.signal as sig
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)



class Component:
    def __init__(self, name, sps, nume=np.array([1.0]), deno=np.array([1.0]), tf=None, domain='z', unit=dim.Dimension(dimensionless=True)):
        """
        Represents a component in a control loop with its transfer function.

        This class encapsulates both symbolic and frequency-domain descriptions
        of a transfer function element, and supports arithmetic composition,
        unit handling, Bode analysis, and integration with a control loop.

        Parameters
        ----------
        name : str
            Name of the component.
        sps : float
            Sample rate of the control loop (Hz). Required if `domain='s'`.
        nume : array_like, optional
            Numerator coefficients of the transfer function (if `tf` is None).
        deno : array_like, optional
            Denominator coefficients of the transfer function (if `tf` is None).
        tf : str | float | control.TransferFunction | tuple
            Transfer function specification. Can be:
            - A string expression in 'z' or 's' (depending on `domain`)
            - A scalar (interpreted as gain)
            - A control.TransferFunction object
            - A (nume, deno) tuple or list
        domain : {'z', 's'}, optional
            Specifies how to interpret string-based transfer functions:
            - 'z' (default): interpret the expression as a Z-domain TF
            - 's': interpret as an S-domain TF, and discretize using the bilinear transform
        unit : Dimension, optional
            Unit of the component.

        Attributes
        ----------
        TE : control.TransferFunction
            The symbolic transfer function representation.
        TF : callable
            Callable function for evaluating the transfer function in frequency domain.
        unit : Dimension
            Unit associated with the component.
        sps : float
            Sample rate in Hz.
        """
        self.name = name
        self.sps = sps
        self.unit = unit

        if tf is None:
            self.nume = nume
            self.deno = deno
            self.TE = control.tf(self.nume, self.deno, 1 / self.sps, name=name)

        elif isinstance(tf, str):
            import sympy as sp
            from sympy import symbols, Poly, sympify, Symbol, together, fraction, simplify
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
            tf_clean = normalize_tf_string(tf, debug=False)

            try:
                # Determine domain variable and sympy context
                if domain == 's':
                    if self.sps is None:
                        raise ValueError("Sample rate `sps` must be specified for domain='s'")
                    var = Symbol('s')
                    local_dict = {'s': var}
                elif domain == 'z':
                    var = Symbol('z')
                    local_dict = {'z': var}
                else:
                    raise ValueError(f"Unrecognized domain '{domain}'. Use 's' or 'z'.")

                expr = parse_expr(
                    tf_clean,
                    local_dict=local_dict,
                    transformations=standard_transformations + (implicit_multiplication_application,)
                )

                # Validate free symbols
                symbols = expr.free_symbols
                if not symbols:
                    # Constant TF
                    value = float(expr)
                    self.nume = np.array([value])
                    self.deno = np.array([1.0])
                    self.TE = control.tf(self.nume, self.deno, 1 / self.sps, name=name)

                elif len(symbols) == 1:
                    sym = symbols.pop()
                    if sym.name != var.name:
                        raise ValueError(
                            f"TF expression used symbol '{sym}', but the domain was set to '{domain}' "
                            f"(expected variable '{var.name}').\n"
                            f"Did you mean to use `domain='{sym.name}'` instead?"
                        )

                    # Get numerator and denominator expressions
                    nume_expr, deno_expr = expr.as_numer_denom()

                    try:
                        # Attempt to extract polynomial coefficients
                        nume_poly = Poly(nume_expr, var)
                        deno_poly = Poly(deno_expr, var)
                        nume_raw = np.array(nume_poly.all_coeffs(), dtype=float)
                        deno_raw = np.array(deno_poly.all_coeffs(), dtype=float)
                    except sympy.polys.polyerrors.PolynomialError as e:
                        raise ValueError(
                            f"TF must be a rational polynomial in '{var}'. Got:\n"
                            f"  Numerator: {nume_expr}\n"
                            f"  Denominator: {deno_expr}\n"
                            f"Sympy error: {e}"
                        )

                    # Discretize if necessary
                    if domain == 'z':
                        self.nume = nume_raw
                        self.deno = deno_raw
                        self.TE = control.tf(self.nume, self.deno, 1 / self.sps, name=name)

                    elif domain == 's':
                        from scipy.signal import cont2discrete
                        sysd = cont2discrete((nume_raw, deno_raw), dt=1 / self.sps, method='bilinear')
                        self.nume = np.asarray(sysd[0]).flatten()
                        self.deno = np.asarray(sysd[1]).flatten()
                        self.TE = control.tf(self.nume, self.deno, 1 / self.sps, name=name)

                else:
                    raise ValueError(f"Expected a single symbolic variable in TF expression, got: {symbols}")

            except Exception as e:
                raise ValueError(f"Failed to parse TF expression '{tf}': {e}")

        elif isinstance(tf, numbers.Number):
            self.nume = np.array([float(tf)])
            self.deno = np.array([1.0])
            self.TE = control.tf(self.nume, self.deno, 1 / self.sps, name=name)

        elif isinstance(tf, control.TransferFunction):
            (nume, deno) = control.tfdata(tf)
            deno = np.array(deno)[0, 0, :]
            nume = np.array(nume)[0, 0, :]
            self.nume = np.around(nume, 16)
            self.deno = np.around(deno, 16)
            self.TE = copy.deepcopy(tf)

        elif isinstance(tf, (tuple, list)) and len(tf) == 2:
            self.nume = np.array(tf[0], dtype=float)
            self.deno = np.array(tf[1], dtype=float)
            self.TE = control.tf(self.nume, self.deno, 1 / self.sps, name=name)

        else:
            raise ValueError(f"Unsupported tf format: {type(tf)}")

        self.TE.name = name
        self.update()

    def __add__(self, other):
        """
        Define parallel addition (+) between two components.

        Parameters
        ----------
        other : Component
            The other component to add.

        Returns
        -------
        Component
            New component representing the parallel connection.
        """
        new_TF = control.parallel(self.TE, other.TE)
        new = Component(self.name+'+'+other.name, sps=self.sps, tf=new_TF, unit=self.unit)
        new.TF = partial(add_transfer_function, tf1=self.TF, tf2=other.TF)

        return new

    def __mul__(self, other):
        """
        Define series multiplication (*) between two components.

        Parameters
        ----------
        other : Component
            The other component to multiply.

        Returns
        -------
        Component
            New component representing the series connection.
        """
        new_unit = self.unit * other.unit
        new_name = self.name + '*' + other.name

        new_TF = control.series(self.TE, other.TE)
        new_TF.name = new_name

        new = Component(new_name, sps=self.sps, tf=new_TF, unit=new_unit)
        new.TF = partial(mul_transfer_function, tf1=self.TF, tf2=other.TF)

        return new

    def modify(self, new_nume, new_deno=None):
        """
        Modify the transfer function coefficients of the component.

        Parameters
        ----------
        new_nume : array_like
        New numerator coefficients.
        new_deno : array_like, optional
        New denominator coefficients; if None, original denominator is kept.
        """
        self.nume = np.array([new_nume])
        if new_deno != None:
            self.deno = np.array(self.deno)
        self.TE = control.tf(self.nume, self.deno, 1/self.sps)
        self.TE.name = self.name
        self.TF = partial(transfer_function, com=self)

    def update(self):
        """
        Refresh internal symbolic and callable representations of the transfer function.

        If part of a control loop, triggers any registered callbacks.
        """
        self.TE = control.tf(self.nume, self.deno, 1/self.sps)
        self.TE.name = self.name
        self.TF = partial(transfer_function, com=self)
        if getattr(self, '_loop', None) != None:
            self._loop.update()
            self._loop.notify_callbacks()

    def extrapolate_tf(self, f_trans, power=-2, size=2, solver=True):
        """
        Replace the component's transfer function with an extrapolated version.

        This method sets `self.TF` to a partial function of `transfer_function`
        that enables extrapolation beyond the provided frequency data. It is useful 
        when simulating or analyzing loop behavior outside the measured or modeled 
        frequency range. It is specially useful to avoid numerical errors when
        simulating a double integrator.

        Parameters
        ----------
        f_trans : float
            The transition frequency (in Hz) beyond which the extrapolation 
            is applied (at f>f_trans there will be no extrapolation).
        power : float, optional
            Power-law exponent used in extrapolation (e.g., -2 for 1/f² roll-off).
        size : int, optional
            Number of points used in extrapolation fitting (not used if `solver=True`).
        solver : bool, optional
            If True, use solver-based extrapolation. Otherwise use simple fit.

        Notes
        -----
        This replaces the existing `self.TF` with a new callable that includes 
        extrapolation logic. Use with caution when accuracy at extrapolated 
        frequencies is critical.
        """
        self.TF = partial(
            transfer_function,
            com=self,
            extrapolate=True,
            f_trans=f_trans,
            power=power,
            size=size,
            solver=solver
        )

    def group_delay(self, omega):
        """
        Compute the group delay of the component.

        Parameters
        ----------
        omega : array_like
            Angular frequency vector (rad/s).

        Returns
        -------
        array_like
            Group delay in seconds.
        """
        # todo: remove this after the consistency check with tf_group_delay() in auxiliary.py 
        _, delay = sig.group_delay((self.nume, self.deno), omega, fs=2*np.pi*self.sps)
        return delay/self.sps
    
    def bode(self, frfr, dB=False, deg=True, wrap=True):
        """
        Compute the Bode magnitude and phase of the component at given frequencies.

        Parameters
        ----------
        frfr : array_like
            Frequency array in Hz.
        dB : bool, optional
            If True, return magnitude in decibels. Default is False (linear).
        deg : bool, optional
            If True, return phase in degrees. Default is True (radians if False).
        wrap : bool, optional
            If True, wrap phase to [-180, 180] deg or [-π, π] rad. Default is True.

        Returns
        -------
        mag : ndarray
            Magnitude response. Units depend on `dB` flag.
        phase : ndarray
            Phase response. Units depend on `deg` flag.
        """
        f = np.asarray(frfr)
        val = self.TF(f=f)

        # Compute magnitude
        mag = 20 * np.log10(np.abs(val)) if dB else np.abs(val)

        # Compute phase
        phase = np.angle(val, deg=deg)

        # Optionally wrap phase
        if wrap:
            if deg:
                phase = (phase + 180) % 360 - 180
            else:
                phase = (phase + np.pi) % (2 * np.pi) - np.pi

        return mag, phase
    
    def bode_plot(self, frfr, figsize=(4, 4), title=None, dB=False, deg=True, wrap=True, axes=None, label=None, *args, **kwargs):
        """
        Plot the Bode diagram (magnitude and phase) of this component.

        Parameters
        ----------
        frfr : array_like
            Frequency array in Hz at which to evaluate the transfer function.
        figsize : tuple, optional
            Size of the figure if creating new one.
        dB : bool, optional
            Display magnitude in dB. Default is False (linear).
        deg : bool, optional
            Display phase in degrees. Default is True.
        wrap : bool, optional
            Wrap phase between -π and π. Default is True.
        axes : tuple of matplotlib.axes.Axes, optional
            Existing axes to plot into. If None, creates new figure and axes.
        label : str or None
            Label for this component in the plot. If None, uses self.name.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plots.
        axes : tuple of matplotlib.axes.Axes
            The magnitude and phase axes used.
        """
        f = np.asarray(frfr)
        
        with plt.rc_context(default_rc), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            val = self.TF(f=f)

            mag = 20 * np.log10(np.abs(val)) if dB else np.abs(val)
            phase = np.angle(val, deg=deg)
            if wrap and deg:
                phase = (phase + 180) % 360 - 180
            elif wrap and not deg:
                phase = (phase + np.pi) % (2 * np.pi) - np.pi

            if axes is None:
                fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=figsize, sharex=True)
            else:
                ax_mag, ax_phase = axes
                fig = ax_mag.figure

            lbl = label if label is not None else self.name
            ax_mag_func = ax_mag.semilogx if dB else ax_mag.loglog
            ax_mag_func(f, mag, label=lbl, *args, **kwargs)
            ax_phase.semilogx(f, phase, label=lbl, *args, **kwargs)

            ax_mag.set_ylabel("Magnitude (dB)" if dB else "Magnitude")
            ax_phase.set_ylabel("Phase (deg)" if deg else "Phase (rad)")
            ax_phase.set_xlabel("Frequency (Hz)")

            ax_mag.set_xlim(f[0], f[-1])
            ax_phase.set_xlim(f[0], f[-1])

            ax_mag.minorticks_on()
            ax_phase.minorticks_on()
            ax_mag.grid(True, which='minor', linestyle='--', linewidth=0.5)
            ax_phase.grid(True, which='minor', linestyle='--', linewidth=0.5)

            ax_mag.legend(loc='best',
                        edgecolor='black',
                        fancybox=True,
                        shadow=True,
                        framealpha=1,
                        fontsize=8)

            if title is not None:
                ax_mag.set_title(title)

            fig.tight_layout()
            fig.align_ylabels()

            return (ax_mag, ax_phase)

def transfer_function(f, com, extrapolate=False, f_trans=1e-1, power=-2, size=2, solver=True):
    """
    Evaluate the discrete-time transfer function of a Component in the frequency domain.

    Uses the Z-transform evaluated on the unit circle to compute the transfer function
    response at given Fourier frequencies. Optionally applies power-law extrapolation
    to extend the response outside the design band.

    Parameters
    ----------
    f : array_like
        Fourier frequencies in Hz.
    com : Component
        Component instance whose transfer function is evaluated.
    extrapolate : bool, optional
        If True, apply power-law extrapolation to extend the response.
    f_trans : float, optional
        Transition frequency (Hz) for the power-law extrapolation.
    power : float, optional
        Power-law exponent used in extrapolation (e.g., -2 for 1/f² roll-off).
    size : int, optional
        Number of points used in extrapolation fitting (not used if `solver=True`).
    solver : bool, optional
        If True, use solver-based extrapolation. Otherwise use simple fit.

    Returns
    -------
    array_like
        Complex-valued frequency response evaluated at the specified frequencies.
    """

    omega = 2*np.pi*f
    z = np.exp(1j*omega/com.sps)

    try:
        n_nume = com.nume.size
        n_deno = com.deno.size
    except AttributeError:
        n_nume = len(com.nume)
        n_deno = len(com.deno)
    numerator = 0
    denominator = 0
    for i, n in enumerate(com.nume):
        numerator += n*z**(n_nume-(i+1))
    for i, d in enumerate(com.deno):
        denominator += d*z**(n_deno-(i+1))
    tf = numerator/denominator

    if extrapolate:
        tf = tf_power_extrapolate(f, tf, f_trans=f_trans, power=power, size=size, solver=solver)

    return tf