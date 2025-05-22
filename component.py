import numpy as np
import control
import copy
from functools import partial
import scipy.signal as sig
import matplotlib.pyplot as plt
from looptools import auxiliary as aux
from looptools import dimension as dim
import logging
logger = logging.getLogger(__name__)


class Component:
    def __init__(self, name, sps, nume=np.array([1.0]), deno=np.array([1.0]), tf=None, unit=dim.Dimension(dimensionless=True)):
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
            Sample rate of the control loop (Hz).
        tf : str | float | control.TransferFunction | tuple
            Transfer function specification. Can be:
            - A string using 's' (e.g. '1 / (s^2 + 10*s + 20)')
            - A scalar (interpreted as gain)
            - A control.TransferFunction object
            - A (nume, deno) tuple/list
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

        from sympy import symbols, Poly, sympify
        import numbers

        s = symbols('s')

        # Handle various tf formats
        if tf is None:
            self.nume = nume
            self.deno = deno
            self.TE = control.tf(self.nume, self.deno, 1/self.sps, name=name)

        elif isinstance(tf, str):
            expr = sympify(tf.replace('^', '**'))
            nume_expr, deno_expr = expr.as_numer_denom()
            nume_poly = Poly(nume_expr, s)
            deno_poly = Poly(deno_expr, s)
            self.nume = np.array(nume_poly.all_coeffs(), dtype=float)
            self.deno = np.array(deno_poly.all_coeffs(), dtype=float)
            self.TE = control.tf(self.nume, self.deno, 1/self.sps, name=name)

        elif isinstance(tf, numbers.Number):
            self.nume = np.array([float(tf)])
            self.deno = np.array([1.0])
            self.TE = control.tf(self.nume, self.deno, 1/self.sps, name=name)

        elif isinstance(tf, control.TransferFunction):
            self.nume, self.deno = aux.mytfdata(tf)
            self.TE = copy.deepcopy(tf)
            self.TE.name = name

        elif isinstance(tf, (tuple, list)) and len(tf) == 2:
            self.nume = np.array(tf[0], dtype=float)
            self.deno = np.array(tf[1], dtype=float)
            self.TE = control.tf(self.nume, self.deno, 1/self.sps, name=name)

        else:
            raise ValueError(f"Unsupported tf format: {type(tf)}")

        # : transfer function
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
        new = Component(self.name+other.name, sps=self.sps, tf=new_TF, unit=self.unit)
        new.TF = partial(aux.add_transfer_function, tf1=self.TF, tf2=other.TF)

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
        new_TF = control.series(self.TE, other.TE)
        new = Component(self.name+other.name, sps=self.sps, tf=new_TF, unit=new_unit)
        new.TF = partial(aux.mul_transfer_function, tf1=self.TF, tf2=other.TF)

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
            self._loop.notify_callbacks()

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

    def bode(self, omega, dB=False, deg=True, wrap=True):
        """
        Compute the Bode plot data (magnitude and phase) of the component.

        Parameters
        ----------
        omega : array_like
            Angular frequency vector (rad/s).
        dB : bool, optional
            If True, returns magnitude in dB.
        deg : bool, optional
            If True, returns phase in degrees.
        wrap : bool, optional
            If True, wraps phase between -π and π.

        Returns
        -------
        tuple
            (bode_dict, ugf, margin) where:
            - bode_dict: dict with 'f', 'mag', and 'phase'
            - ugf: unity-gain frequency
            - margin: phase margin at UGF
        """
        bode, ugf, margin = compute_bode(self.TE, omega, sps=self.sps, dB=dB, deg=deg, wrap=wrap)
        #bode, ugf, margin = compute_bode(self.TF, omega, sps=self.sps, dB=dB, deg=deg, wrap=wrap)
        return bode, ugf, margin

    def bode_plot(self, omega, returns=False, dB=False, deg=True, wrap=True):
        """
        Plot the Bode diagram of the component.

        Parameters
        ----------
        omega : array_like
            Angular frequency vector (rad/s).
        returns : bool, optional
            If True, returns Bode data and metrics.
        dB : bool, optional
            If True, displays magnitude in dB.
        deg : bool, optional
            If True, displays phase in degrees.
        wrap : bool, optional
            If True, wraps phase between -π and π.

        Returns
        -------
        tuple, optional
            (bode_dict, ugf, margin) if `returns` is True.
        """
        bode, ugf, margin = self.bode(omega, dB=dB, deg=deg, wrap=wrap)
        plt.figure(figsize=(15,10))
        plt.subplots_adjust(wspace=0.4, hspace=0.15)
        ax1 = plt.subplot2grid((2,1), (0,0), colspan=1)
        ax2 = plt.subplot2grid((2,1), (1,0), colspan=1)
        # : magnitude
        if dB:
            ax1.semilogx(bode["f"], bode["mag"])
            ax1.set_ylabel('magnitude (dB)')
        else:
            ax1.loglog(bode["f"], bode["mag"])
            ax1.set_ylabel('magnitude (linear)')
        ax1.axvline(ugf, ls = "-.", lw=2, color="m")
        ax1.grid(which='major')
        ax1.text(0.1, 0.1, f'UGF = {ugf:.2e} Hz', fontsize=15, transform=ax1.transAxes)
        # : phase
        ax2.semilogx(bode["f"], bode["phase"])
        ax2.axvline(ugf, ls = "-.", lw=2, color="m")
        ax2.text(0.1, 0.1, f'phase margin = {margin:.2e} deg', fontsize=15, transform=ax2.transAxes)
        ax2.grid(which='both')
        ax2.set_xlabel(f'frequency (Hz)')
        ax2.set_ylabel('phase (deg)' if deg else 'phase (rad)')
        plt.show()
        plt.close()

        if returns:
            return bode, ugf, margin

def compute_bode(tf, omega, sps=80e6, dB=False, deg=True, wrap=True):
    """
    Compute Bode diagram data and loop performance metrics.

    Parameters
    ----------
    tf : control.TransferFunction
        Transfer function to analyze.
    omega : array_like
        Angular frequency vector (rad/s).
    sps : float, optional
        Sampling rate (Hz).
    dB : bool, optional
        Return magnitude in dB if True.
    deg : bool, optional
        Return phase in degrees if True.
    wrap : bool, optional
        Wrap phase between -π and π if True.

    Returns
    -------
    tuple
        (bode_dict, ugf, margin)
        - bode_dict: dict with 'f', 'mag', and 'phase'
        - ugf: unity-gain frequency
        - margin: phase margin at UGF
    """
    # : compute bode
    mag, phase, _ = control.bode(tf, omega, dB=False, plot=False)
    fourier_freq = omega/(2*np.pi) # Hz of control.bode does not work for some reason

    # : some treatments with options
    if dB:
        mag = 20*np.log10(mag)
    if wrap:
        phase = aux.wrap_phase(phase)
    if deg: # deg of control.bode does not work for some reason
        phase *= (180/np.pi)
    bode = {"f":fourier_freq, "mag":mag, "phase":phase}

    # : compute UGF
    ugf, margin = aux.get_margin([mag,phase], fourier_freq, dB=dB, deg=deg)

    return bode, ugf, margin

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

    n_nume = com.nume.size
    n_deno = com.deno.size
    numerator = 0
    denominator = 0
    for i, n in enumerate(com.nume):
        numerator += n*z**(n_nume-(i+1))
    for i, d in enumerate(com.deno):
        denominator += d*z**(n_deno-(i+1))
    tf = numerator/denominator

    if extrapolate:
        tf = aux.tf_power_extrapolate(f, tf, f_trans=f_trans, power=power, size=size, solver=solver)

    return tf