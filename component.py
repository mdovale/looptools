import numpy as np
import control
import copy
from functools import partial
import scipy.signal as scisig
import matplotlib.pyplot as plt
from looptools import auxiliary as aux
from looptools import dimension as dim
import logging
logger = logging.getLogger(__name__)


def transfer_function(f, com, extrapolate=False, f_trans=1e-1, power=-2, size=2, solver=True):
	""" compute a transfer function
	Args:
		f: fourier frequencies (Hz)
		com: Component instance
		extrapolate: extrapolate the tf in a power law
		f_trans: transition frequency
		power: power in power law
		size: point size to be used for fit (not needed for solver)
		solver: use solver or not (not = fit)
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


class Component:
    def __init__(self, name, sps, nume=np.array([1]), deno=np.array([1]), tf=None, unit=dim.Dimension(dimensionless=True)):
        """ class for a component of a control loop
        Args:
            name: name of this component
            nume: numerator coefficients
            deno: denominator coefficients
            sps: loop rate
            tf: TF instance. If not None, this component is defined by it
            unit: unit of this component [Dimension class]
        """

        self.name = name
        self.sps = sps
        self.unit = unit

        # : transfer element
        if tf is None:
            self.nume = nume
            self.deno = deno
            self.TE = control.tf(self.nume, self.deno, 1/self.sps, name=name)
        else:
            self.nume, self.deno = aux.mytfdata(tf)
            self.TE = copy.deepcopy(tf)
            self.TE.name = name

        # : transfer function
        self.update()

    def __add__(self, other):
        """ define + operator
        Args:
            other: another Component instance
        """

        new_TF = control.parallel(self.TE, other.TE)
        new = Component(self.name+other.name, sps=self.sps, tf=new_TF, unit=self.unit)
        new.TF = partial(aux.add_transfer_function, tf1=self.TF, tf2=other.TF)

        return new

    def __mul__(self, other):
        """ define * operator
        Args:
            other: another Component instance
        """

        new_unit = self.unit * other.unit
        new_TF = control.series(self.TE, other.TE)
        new = Component(self.name+other.name, sps=self.sps, tf=new_TF, unit=new_unit)
        new.TF = partial(aux.mul_transfer_function, tf1=self.TF, tf2=other.TF)

        return new

    def modify(self, new_nume, new_deno=None):
        self.nume = np.array([new_nume])
        if new_deno != None:
            self.deno = np.array(self.deno)
        self.TE = control.tf(self.nume, self.deno, 1/self.sps)
        self.TE.name = self.name
        self.TF = partial(transfer_function, com=self)

    def update(self):
        self.TE = control.tf(self.nume, self.deno, 1/self.sps)
        self.TE.name = self.name
        self.TF = partial(transfer_function, com=self)
        if getattr(self, '_loop', None) != None:
            self._loop.notify_callbacks()

    def group_delay(self, omega):
        """ compute a group delay
        Args:
            omega: angular fourier frequency (rad*Hz)
        """

        # todo: remove this after the consistency check with tf_group_delay() in auxiliary.py 

        _, delay = scisig.group_delay((self.nume, self.deno), omega, fs=2*np.pi*self.sps)
        return delay/self.sps

    def bode(self, omega, dB=False, deg=True, wrap=True):
        """ compute a bode diagram
        Args:
            omega: angular fourier frequency (rad*Hz)
            dB: magnitude in dB or not
            deg: phase in degree or not
            wrap: wrap phase or not
        """

        bode, ugf, margin = compute_bode(self.TE, omega, sps=self.sps, dB=dB, deg=deg, wrap=wrap)
        #bode, ugf, margin = compute_bode(self.TF, omega, sps=self.sps, dB=dB, deg=deg, wrap=wrap)
        return bode, ugf, margin

    def bode_plot(self, omega, returns=True, dB=False, deg=True, wrap=True):
        """ generator/display a bode plot
        Args:
            omega: angular fourier frequency (rad*Hz)
            returns: return bode and ugf or not
            dB: magnitude in dB or not
            deg: phase in degree or not
            wrap: wrap phase or not
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
    """ compute a bode diagram
    Args:
        tf: tf instance
        omega: angular fourier frequency (rad*Hz)
        sps: sampling rate
        dB: magnitude in dB or not
        deg: phase in degree or not
        wrap: wrap phase or not
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