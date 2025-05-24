import numpy as np
from functools import singledispatch
from scipy.optimize import curve_fit
from looptools import dsp

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

def tf_group_delay(f, tf):
	""" compute groupd delay of TF (sec)
	Args: 
		f: fourier frequency (Hz)
		tf: complex transfer function
	"""

	# : To avoid making all Nan due to np.unwrap, the case with Nan is carefully treated
	tfnew, nanarray = dsp.nan_checker(tf)
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
	mag, nanarray = dsp.nan_checker(tf[0])
	phase = tf[1][~nanarray]
	#phase = np.unwrap(phase, period=360 if deg else 2*np.pi)
	fnew = f[~nanarray]

	if dB:
		index = dsp.index_of_the_nearest(mag, 0)
	else:
		index = dsp.index_of_the_nearest(mag, 1)
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
	tfnew, nanarray = dsp.nan_checker(tf)
	fnew = f[~nanarray]

	mag = abs(tfnew)
	phase = np.angle(tfnew, deg=deg)
	#phase = np.unwrap(phase, period=360 if deg else 2*np.pi)
	index = dsp.index_of_the_nearest(mag, 1)
	
	ugf = fnew[index]
	if deg:
		margin = 180 + phase[index]
	else:
		margin = np.pi + phase[index]
	return ugf, margin

def tf_power_fitting(f, tf, fnew, power):
    """
    Fit and extrapolate a transfer function magnitude using a power-law model.

    Parameters
    ----------
    f : array_like
        Frequencies (Hz) at which the transfer function is known.
    tf : array_like
        Complex-valued transfer function at `f`.
    fnew : array_like
        Frequencies (Hz) at which to evaluate the extrapolated transfer function.
    power : float
        Power-law exponent (e.g., -2 implies 1/f² behavior).

    Returns
    -------
    tfnew : ndarray
        Complex-valued extrapolated transfer function evaluated at `fnew`.

    Notes
    -----
    The fitting is done on the magnitude |TF(f)|. The resulting extrapolation assumes
    a phase consistent with the complex model `TF ∝ (j2πf)^power`.
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
    """
    Extrapolate a transfer function using a power-law model from a single reference point.

    Parameters
    ----------
    f : float
        Single frequency (Hz) at which the transfer function is known.
    tf : complex
        Transfer function value at frequency `f`.
    fnew : array_like
        Frequencies (Hz) at which to evaluate the extrapolated transfer function.
    power : float
        Power-law exponent (e.g., -2 implies 1/f² behavior).

    Returns
    -------
    tfnew : ndarray
        Complex-valued extrapolated transfer function evaluated at `fnew`.

    Raises
    ------
    ValueError
        If `f` is not a float.
    """
    if not isinstance(f, float):
        raise ValueError(f'invalid type of f {type(f)}')

    s = 1j*2*np.pi*f
    a = np.abs(tf/s**power)
    snew = 1j*2*np.pi*fnew
    tfnew = a*snew**power

    return tfnew

def tf_power_extrapolate(f, tf, f_trans, power, size=2, solver=True):
    """
    Extrapolate a transfer function below a transition frequency using a power-law model.

    Parameters
    ----------
    f : array_like
        Frequencies (Hz) at which the transfer function is defined.
    tf : array_like
        Complex-valued transfer function at each frequency in `f`.
    f_trans : float
        Transition frequency (Hz) below which extrapolation is applied.
    power : float
        Exponent of the power law (e.g., -2 implies 1/f² behavior).
    size : int, optional
        Number of points above `f_trans` to use for fitting if `solver=False`. Default is 2.
    solver : bool, optional
        If True, use solver-based extrapolation from one sample. If False, use magnitude fitting. Default is True.

    Returns
    -------
    tfnew : ndarray
        Complex-valued transfer function over `f`, extrapolated below `f_trans`.
    """
    idx = dsp.index_of_the_nearest(f, f_trans)

    if solver:
        tfnew = tf_power_solver(f[idx], tf[idx], f, power=power)
    else:
        ftmp, tftmp = dsp.crop_data(f, tf, xmin=f_trans, xmax=np.max(f))
        ftmp = ftmp[:size]
        tftmp = tftmp[:size]
        tfnew = tf_power_fitting(ftmp, tftmp, f, power=power)

    tfnew[f>f_trans] = tf[f>f_trans]

    return tfnew

def add_transfer_function(f, tf1, tf2, extrapolate=False, f_trans=1e-1, power=-2, size=2, solver=True):
    """
    Return the sum of two transfer functions, with optional extrapolation below a transition frequency.

    Parameters
    ----------
    f : array_like
        Fourier frequencies (Hz) at which to evaluate the transfer functions.
    tf1 : callable
        First transfer function. Should accept `f` and return complex values.
    tf2 : callable
        Second transfer function. Should accept `f` and return complex values.
    extrapolate : bool, optional
        Whether to extrapolate the sum below `f_trans` using a power law. Default is False.
    f_trans : float, optional
        Transition frequency (Hz) below which extrapolation is applied. Default is 1e-1.
    power : float, optional
        Power-law exponent (e.g., -2 implies 1/f² behavior). Default is -2.
    size : int, optional
        Number of points above `f_trans` to use for fitting if `solver=False`. Default is 2.
    solver : bool, optional
        If True, use solver-based extrapolation. If False, use least-squares fitting. Default is True.

    Returns
    -------
    tf : ndarray
        Complex-valued array of the summed transfer functions (extrapolated if enabled).
    """
    tf1f = tf1(f)
    tf2f = tf2(f)

    if extrapolate:
        tf = tf_power_extrapolate(f, tf1f + tf2f, f_trans=f_trans, power=power, size=size, solver=solver)
    else:
        tf = tf1f + tf2f

    return tf

def mul_transfer_function(f, tf1, tf2, extrapolate=False, f_trans=1e-1, power=-2, size=2, solver=True):
    """
    Return the product of two transfer functions, with optional extrapolation below a transition frequency.

    Parameters
    ----------
    f : array_like
        Fourier frequencies (Hz) at which to evaluate the transfer functions.
    tf1 : callable
        First transfer function. Should accept `f` and return complex values.
    tf2 : callable
        Second transfer function. Should accept `f` and return complex values.
    extrapolate : bool, optional
        Whether to extrapolate the product below `f_trans` using a power law. Default is False.
    f_trans : float, optional
        Transition frequency (Hz) below which extrapolation is applied. Default is 1e-1.
    power : float, optional
        Power-law exponent (e.g., -2 implies 1/f² behavior). Default is -2.
    size : int, optional
        Number of points above `f_trans` to use for fitting if `solver=False`. Default is 2.
    solver : bool, optional
        If True, use solver-based extrapolation. If False, use least-squares fitting. Default is True.

    Returns
    -------
    tf : ndarray
        Complex-valued array of the multiplied transfer functions (extrapolated if enabled).
    """
    tf1f = tf1(f)
    tf2f = tf2(f)

    if extrapolate:
        tf = tf_power_extrapolate(f, tf1f * tf2f, f_trans=f_trans, power=power, size=size, solver=solver)
    else:
        tf = tf1f * tf2f

    return tf

def gain_for_crossover_frequency(Kp_log2, f_cross, kind='I'):
    """
    Compute log2 gain for an I or II block so that its magnitude matches the P gain at f_cross.

    This function helps translate a Moku-style crossover frequency between P and I (or P and II)
    into the gain value (in log₂ units) needed for I or II blocks in bit-shift-based controllers.

    Parameters
    ----------
    Kp_log2 : float
        Proportional gain in log₂ scale (i.e., log₂(Kp)).
    f_cross : float
        Desired crossover frequency [Hz] between P and I (or P and II).
    kind : {'I', 'II'}
        Which gain to compute: 'I' for integrator (1/s), 'II' for double integrator (1/s²).

    Returns
    -------
    float
        Log₂ gain for the I or II block that matches |P| at `f_cross`.

    Examples
    --------
    >>> gain_for_crossover_frequency(3, 1e4, 80e6, kind='I')
    9.3219  # equivalent to Ki = 2^9.32
    """
    assert kind in ['I', 'II'], "kind must be 'I' or 'II'"
    
    # Linear gain of P
    Kp = 2 ** Kp_log2

    # Compute angular frequency
    w = 2 * np.pi * f_cross

    if kind == 'I':
        # |Ki / (jw)| = |Kp| --> Ki = Kp * w
        Ki = Kp * w
    elif kind == 'II':
        # |Kii / (jw)^2| = |Kp| --> Kii = Kp * w^2
        Ki = Kp * w**2

    return np.log2(Ki)