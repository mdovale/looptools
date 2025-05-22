import numpy as np
import control
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from functools import singledispatch
from pyhexagon.processing import spectrums as spc
import scipy.constants as scc
import logging
logger = logging.getLogger(__name__)


def loop_crossover(loop1, loop2, frfr):
    Gf1 = np.abs(loop1.Gf(f=frfr)) # Loop 1 open-loop transfer function magnitude
    Gf2 = np.abs(loop2.Gf(f=frfr)) # Loop 2 open-loop transfer function magnitude
    diff = np.array(Gf1-Gf2)
    signs = np.sign(diff)
    signs[signs == 0] = 1  # Replace zeros with 1 to avoid sign change detection issues
    # sign_changes = np.where(np.diff(signs) != 0)[0] + 1
    # return frfr[sign_changes[0]] if sign_changes.size > 0 else np.nan
    sign_changes = np.diff(signs) != 0
    # Find the first crossover frequency
    crossover_indices = np.where(sign_changes)[0] + 1
    if crossover_indices.size > 0:
        return frfr[crossover_indices[0]]
    else:
        return np.nan

def wrap_phase(phase, deg=False):
	""" wrap a phase
	Args:
		phase: phase to wrap
		deg: input phase in degree or not
	"""

	if deg:
		phase_new = (phase + 180.0) % (2 * 180.0) - 180.0
	else:
		phase_new = (phase + np.pi) % (2 * np.pi) - np.pi

	return phase_new

def mytfdata(tf, rounding=16):
	""" extract numerators and denominators from transfer element
	Args:
		tf: Transfer function instance
	"""

	(nume, deno) = control.tfdata(tf)
	deno = np.array(deno)[0,0,:]
	nume = np.array(nume)[0,0,:]
	deno = np.around(deno, rounding)
	nume = np.around(nume, rounding)

	return nume, deno

def crop_data(x,y,xmin,xmax):
	""" crop data
	Args:
		x: data in x
		y: data in y
		xmin: lower bound of x
		xmax: upper bound of x
	"""

	x_tmp = []
	y_tmp = []
	for i in range(x.size):
		if(x[i] >= xmin and x[i] <= xmax):
			x_tmp.append(x[i])
			y_tmp.append(y[i])

	return np.array(x_tmp), np.array(y_tmp)

def bode2tf(bode, dB=False, deg=False):
	""" convert a bode dictionary to a transfer function
	Args:
		bode: bode dict. ['f', 'mag', 'phase']
		dB: if a bode magnitude is in dB
		deg: if a bode phase is in degree
	"""

	mag = bode['mag']
	phase = bode['phase']
	if dB:
		mag = 10**(mag/20)
	if deg:
		phase = (np.pi/180)*phase
	tf = mag*np.exp(phase*1j)
	return tf

def rad_asd_to_CN0(asd, dBHz=False):
	""" conversion from phase asd rad/sqrt(Hz) to carrier-to-noise-density ratio dB-Hz
	Args:
		asd: phase asd (rad/sqrt(Hz))
		dBHz: return a result in dBHz or not
	"""

	CN0 = 1/asd**2
	if dBHz:
		CN0 = 10*np.log10(CN0)
	return CN0

def CN0_to_rad_asd(CN0, dBHz=False):
	""" conversion from carrier-to-noise-density ratio dB-Hz to phase asd rad/sqrt(Hz)
	Args:
		CN0: carrier-to-noise-density ratio
		dBHz: given in dBHz or not
	"""

	if dBHz:
		CN0_tmp = 10**(CN0/10)
		asd = 1/np.sqrt(CN0_tmp)
	else:
		asd = 1/np.sqrt(CN0)

	return asd

def tf_group_delay(f, tf):
	""" compute groupd delay of TF (sec)
	Args: 
		f: fourier frequency (Hz)
		tf: complex transfer function
	"""

	# : To avoid making all Nan due to np.unwrap, the case with Nan is carefully treated
	tfnew, nanarray= nan_checker(tf)
	isnan = True in nanarray

	phase = np.angle(tfnew, deg=False)
	phase = np.unwrap(phase)
	gd = - np.gradient(phase, 2*np.pi*f[~nanarray])

	if not isnan:
		output = gd
	else:
		output = np.zeros(tf.size)
		idx = 0
		for i, t in enumerate(tf):
			if not nanarray[i]:
				output[i] = gd[idx]
				idx += 1
			else:
				output[i] = t

	return output

def nan_checker(x):
	""" Check if Nan exists in x. If yes, Nan is removed 
	Args:
		x: data to be checked
	"""

	if True in np.isnan(x):
		logger.warning('Nan was detected in the input array')
		nanarray = np.isnan(x)
		xnew = x[~nanarray]
	else:
		xnew, nanarray = x, np.full(x.size, False)

	return xnew, nanarray

def polynomial_conversion_s_to_z(s_coeffs, sps):
	""" convert polynomial coefficients in s to z
	Args:
		s_coeffs: polynomial coefficients in the Laplace domain
		sps: sampling frequency of the new z domain
	"""

	size = np.shape(s_coeffs)[0]
	z_coeffs = np.zeros(size)
	for i,c in enumerate(s_coeffs):
		order = size-(i+1)
		base0 = np.array([sps, -sps])
		base = np.ones(1)
		for j in range(order):
			base = np.convolve(base, base0)
		coord = c*base
		coord = np.concatenate((np.zeros(size-order-1), coord), axis=0)
		z_coeffs += coord

	return z_coeffs

def plot_transfer_functions(f, tfs, isG=None, labels=None, dB=False, deg=True, wrap=True, gd=True, filen=None, output=False, phase_comp='default'):
	""" plot transfer functions
	Args:
		f: fourier frequencies (Hz)
		tfs: complex transfer functions
		isG: if the TF is G or not (if yes, ugf is computed) (list)
		labels: label of a data (list)
		dB: dB for magnitude (boolean)
		deg: degree for phase (boolean)
		wrap: display wrapped phases (boolean)
		gd: plot group delay or not (boolean)
		filen: name of a file to be saved
		output: if return the transfer functions or not
	"""

	if isG is None:
		isG = [False]*len(tfs)
	if labels is None:
		labels = [f'data {i}' for i in range(len(tfs))]
	if phase_comp=='default':
		phase_comp = np.zeros(len(tfs))

	# : prepare arrays in case the output is true
	mag_array = np.empty([f.size, len(tfs)])
	phase_array = np.empty([f.size, len(tfs)])
	gd_array = np.empty([f.size, len(tfs)])

	plt.figure(figsize=(15,10))
	plt.subplots_adjust(wspace=0.4, hspace=0.15)
	ax1 = plt.subplot2grid((2,1), (0,0), colspan=1)
	ax2 = plt.subplot2grid((2,1), (1,0), colspan=1)
	if gd:
		ax2b = ax2.twinx()
	ugf_txt = f'UGF (Hz) = '
	margin_txt = f'phase margin (deg) = '
	if labels is None:
		legends=[f'TF{i}' for i in range(len(tfs))]
	else:
		legends=labels
	for i, tf in enumerate(tfs):
		mag = abs(tf)
		# : magnitude
		if dB:
			mag = 20*np.log10(mag)
			ax1.semilogx(f, mag, label=legends[i], color=f'C{i}')
		else:
			ax1.loglog(f, mag, label=legends[i], color=f'C{i}')
		# : phase
		phase = np.angle(tf, deg=deg)
		phase = wrap_phase(phase, deg=deg) if wrap else np.unwrap(phase, period=360 if deg else 2*np.pi)
		group_delay = tf_group_delay(f, tf)
		group_delay *= scc.c # convert sec to meter
		ax2.semilogx(f, phase+phase_comp[i], color=f'C{i}')
		# : group delay
		if gd:
			ax2b.semilogx(f, group_delay, ls='--', color=f'C{i}')
		# : UGF for open-loop transfer function
		if isG[i]:
			ugf, margin = get_margin(tf, f, deg=deg)
			margin += phase_comp[i]
			ax1.axvline(ugf, ls = "-.", lw=2, color=f'C{i}')
			ax2.axvline(ugf, ls = "-.", lw=2, color=f'C{i}')
			ugf_txt += f'{ugf:.2e}, '
			margin_txt += f'{margin:.1f}, '
			print(legends[i]+' spec: '+f'UGF = {ugf:.2e} (Hz)'+f', phase margin = {margin:.2e} (deg)')
		# : substitute results to output arrays
		mag_array[:, i] = mag
		phase_array[:, i] = phase
		gd_array[:, i] = group_delay
	ax1.set_ylabel(r'dB', fontsize=15) if dB else ax1.set_ylabel(r'magnitude', fontsize=15)
	ax1.tick_params(axis='both', labelsize=13)
	ax1.legend()
	ax1.grid(which='both')
	ax2.set_xlabel(r'frequency (Hz)', fontsize=15)
	ax2.set_ylabel('phase (deg)' if deg else 'phase (rad)', fontsize=15)
	ax2.tick_params(axis='both', labelsize=13)
	ax2.grid(which='both')
	if gd:
		ax2b.set_ylabel('group delay (m)', fontsize=15)
		ax2b.tick_params(axis='y', labelsize=13)
	if True in isG:
		ax1.text(0.1, 0.1, ugf_txt, fontsize=15, transform=ax1.transAxes)
		ax2.text(0.1, 0.1, margin_txt, fontsize=15, transform=ax2.transAxes)
	if filen is not None:
		plt.subplots_adjust(left=0.08, right=0.94, bottom=0.08, top=0.96)
		plt.savefig(filen)
	plt.show()
	plt.close()

	if output:
		return mag_array, phase_array, gd_array

def index_of_the_nearest(data, value):
	""" extract an index at which data gets closest to a value
	Args:
		data: data array
		value: target value
	"""

	idx = np.argmin(np.abs(np.array(data) - value))
	return idx

@singledispatch
def get_margin(tf, f, dB=False, deg=True):
	""" compute a phase margin
	Args:
		tf: list of [mag, phase]
		f: fourier frequencies (Hz)
		dB: magnitude in dB or not
		deg: phase in degree or not
	"""

	# : To avoid making all Nan due to numpy processing, the case with Nan is carefully treated
	mag, nanarray = nan_checker(tf[0])
	phase = tf[1][~nanarray]
	#phase = np.unwrap(phase, period=360 if deg else 2*np.pi)
	fnew = f[~nanarray]

	if dB:
		index = index_of_the_nearest(mag, 0)
	else:
		index = index_of_the_nearest(mag, 1)
	ugf = fnew[index]
	if deg:
		margin = 180 + phase[index]
	else:
		margin = np.pi + phase[index]
	return ugf, margin

@get_margin.register(np.ndarray)
def _(tf, f, deg=True):
	""" compute a phase margin
	Args:
		tf: complex transfer function array
		f: fourier frequencies (Hz)
		deg: phase in degree or not
	"""

	# : To avoid making all Nan due to numpy processing, the case with Nan is carefully treated
	tfnew, nanarray = nan_checker(tf)
	fnew = f[~nanarray]

	mag = abs(tfnew)
	phase = np.angle(tfnew, deg=deg)
	#phase = np.unwrap(phase, period=360 if deg else 2*np.pi)
	index = index_of_the_nearest(mag, 1)
	
	ugf = fnew[index]
	if deg:
		margin = 180 + phase[index]
	else:
		margin = np.pi + phase[index]
	return ugf, margin

def get_fourier_freq(tau, p_lpsd):
	""" get a fourier frequency array generated by lpsd with a time array
	Args:
		tau: time array (sec)
	"""

	fs = 1/(tau[1]-tau[0])
	dammy = np.random.normal(0, 1, tau.size)
	fourier_freq, _, _ = spc.lasd(dammy, fs, p_lpsd)
	return fourier_freq

def tf_power_fitting(f, tf, fnew, power):
    """ fit a transfer function based on the power law
    Args:
        f: fourier frequencies for fit
        tf: complex transfer function  for fit
        fnew: output fourier frequencies
        power: power in power law
    """

    def power_law(freq, a, b, mag=False):
        omega = 2*np.pi*freq
        s = 1j*omega
        out = a*s**b
        return np.abs(out) if mag else out

    fabs = lambda f, a: power_law(f, a, b=power, mag=True)
    popt, pcov = curve_fit(fabs, f, np.abs(tf))
    tfnew = power_law(fnew, popt[0], b=power, mag=False)

    return tfnew

def tf_power_solver(f, tf, fnew, power):
    """ analytically solve a transfer function with only one sample based on the power law
    Args:
        f: fourier frequency (float)
        tf: complex transfer function (complex)
        fnew: output fourier frequencies
        power: power in power law
    """

    if not isinstance(f, float):
        raise ValueError(f'invalid type of f {type(f)}')

    s = 1j*2*np.pi*f
    a = np.abs(tf/s**power)
    snew = 1j*2*np.pi*fnew
    tfnew = a*snew**power

    return tfnew

def tf_power_extrapolate(f, tf, f_trans, power, size=2, solver=True):
    """ extrapolate tf with a power law below a given transition frequency
    Args:
        f: fourier frequencies
        tf: complex transfer function (Hz)
        f_trans: transition frequency
        power: power in power law
        size: point size to be used for fit (not needed for solver)
        solver: use solver or not (not = fit)
    """

    idx = index_of_the_nearest(f, f_trans)

    if solver:
        tfnew = tf_power_solver(f[idx], tf[idx], f, power=power)
    else:
        ftmp, tftmp = crop_data(f, tf, xmin=f_trans, xmax=np.max(f))
        ftmp = ftmp[:size]
        tftmp = tftmp[:size]
        tfnew = tf_power_fitting(ftmp, tftmp, f, power=power)

    tfnew[f>f_trans] = tf[f>f_trans]

    return tfnew

def add_transfer_function(f, tf1, tf2, extrapolate=False, f_trans=1e-1, power=-2, size=2, solver=True):
	""" return the sum of transfer functions
	Args:
		f: fourier frequencies (Hz)
		tf1: TF instance 1
		tf2: TF instance 2
		extrapolate: extrapolate the tf in a power law
		f_trans: transition frequency
		power: power in power law
		size: point size to be used for fit (not needed for solver)
		solver: use solver or not (not = fit)
	"""

	tf1f = tf1(f)
	tf2f = tf2(f)

	if extrapolate:
		tf = tf_power_extrapolate(f, tf1f + tf2f, f_trans=f_trans, power=power, size=size, solver=solver)
	else:
		tf = tf1f + tf2f

	return tf

def mul_transfer_function(f, tf1, tf2, extrapolate=False, f_trans=1e-1, power=-2, size=2, solver=True):
	""" return the product of transfer functions
	Args:
		f: fourier frequencies (Hz)
		tf1: TF instance 1
		tf2: TF instance 2
		extrapolate: extrapolate the tf in a power law
		f_trans: transition frequency
		power: power in power law
		size: point size to be used for fit (not needed for solver)
		solver: use solver or not (not = fit)
	"""

	tf1f = tf1(f)
	tf2f = tf2(f)

	if extrapolate:
		tf = tf_power_extrapolate(f, tf1f * tf2f, f_trans=f_trans, power=power, size=size, solver=solver)
	else:
		tf = tf1f * tf2f

	return tf