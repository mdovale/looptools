import numpy as np
from functools import singledispatch
from scipy.optimize import curve_fit
from looptools import dsp

def loop_crossover(loop1, loop2, frfr):
    """
    Find the frequency at which the open-loop gain of two control loops crosses over.

    This function computes the magnitude of the open-loop gain for two control
    loops (`loop1` and `loop2`) over a frequency grid and identifies the first frequency at
    which their magnitudes intersect (i.e., cross each other). This is useful in control
    systems to detect dominance crossover points between nested or competing loops.

    Parameters
    ----------
    loop1 : object
        Control loop object that implements a method `Gf(f)` returning the complex-valued
        open-loop transfer function evaluated at frequencies `f`.
    loop2 : object
        Another control loop object with a compatible `Gf(f)` interface.
    frfr : array_like
        Frequency grid (Hz) over which to evaluate the transfer functions.

    Returns
    -------
    crossover_freq : float
        The first frequency (Hz) where the magnitude of `loop1.Gf` crosses that of `loop2.Gf`.
        Returns `np.nan` if no crossover is detected.

    Notes
    -----
    - The crossover is detected by evaluating the sign of the difference between |G1(f)| and |G2(f)|
      and looking for the first sign change.
    - Zeros in the difference are treated as positive to ensure consistent sign behavior.
    - This function is useful in tuning multi-loop control systems to identify the frequency
      at which control authority shifts between loops.
    """
    Gf1 = np.abs(loop1.Gf(f=frfr)) # Loop 1 open-loop gain
    Gf2 = np.abs(loop2.Gf(f=frfr)) # Loop 2 open-loop gain
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
    """
    Wrap phase values to the principal interval [-π, π) or [-180°, 180°).

    This function ensures that the input phase values are wrapped into the standard
    interval, either in radians or degrees, depending on the `deg` flag. This is
    useful for visualizing or processing phase responses without discontinuities
    exceeding a full cycle.

    Parameters
    ----------
    phase : array_like or float
        Input phase value(s) to wrap. Can be a scalar or array.
    deg : bool, optional
        If True, assume input is in degrees and wrap to [-180, 180).
        If False, assume radians and wrap to [-π, π). Default is False.

    Returns
    -------
    phase_new : ndarray or float
        Wrapped phase values within the specified interval.

    Notes
    -----
    - Wrapping is done using modular arithmetic:
        - In degrees:    (phase + 180) % 360 - 180
        - In radians:    (phase + π)   % (2π) - π
    - This does not perform unwrapping or differentiation.
    - Useful as a post-processing step after phase extraction or manipulation.
    """

    if deg:
        phase_new = (phase + 180.0) % (2 * 180.0) - 180.0
    else:
        phase_new = (phase + np.pi) % (2 * np.pi) - np.pi

    return phase_new

def tf_group_delay(f, tf):
    """
    Compute the group delay of a complex transfer function.

    The group delay is calculated as the negative derivative of the unwrapped phase
    with respect to angular frequency ω = 2πf. This represents the time delay
    experienced by the envelope of a modulated signal through the system.

    Parameters
    ----------
    f : array_like
        Fourier frequencies (Hz) at which the transfer function is evaluated.
    tf : array_like
        Complex-valued transfer function evaluated at frequencies `f`.

    Returns
    -------
    gd : ndarray
        Group delay in seconds. If `tf` contains NaNs, the output retains their positions
        and fills corresponding delays accordingly.

    Notes
    -----
    - Internally unwraps the phase using `np.unwrap` before differentiating.
    - Uses `np.gradient` to estimate the derivative of the unwrapped phase over `2πf`.
    - Handles `NaN` values in `tf` robustly using `dsp.nan_checker`, preserving array size.
    - Group delay is defined as:
        gd(f) = -d(phase) / d(ω) = -d(∠TF) / d(2πf)
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
    """
    Convert polynomial coefficients from the Laplace (s) domain to the discrete-time (z) domain.

    This function performs a polynomial transformation based on the backward difference
    approximation:
        s ≈ (1 - z⁻¹) * sps
    where `sps` is the sampling frequency. Each power of `s` in the polynomial is mapped
    to its corresponding polynomial in `z⁻¹`.

    Parameters
    ----------
    s_coeffs : array_like
        Coefficients of the polynomial in the Laplace domain (highest power first).
        For example, [a_n, a_{n-1}, ..., a_0] corresponds to: a_n sⁿ + a_{n-1} sⁿ⁻¹ + ... + a_0.
    sps : float
        Sampling frequency (Hz) used for the s-to-z transformation.

    Returns
    -------
    z_coeffs : ndarray
        Coefficients of the transformed polynomial in the z domain (highest power first),
        using the backward difference approximation.

    Notes
    -----
    - This conversion maps an analog transfer function (in `s`) to a digital one (in `z`)
        by applying the backward difference formula to each term of the polynomial.
    - The resulting polynomial is not necessarily normalized.
    - This transformation preserves the shape of the analog response for low frequencies,
        but is only accurate for systems sampled at sufficiently high rates (relative to bandwidth).
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
    """
    Compute the phase margin from a transfer function given as [magnitude, phase] arrays.

    The phase margin is calculated at the frequency where the magnitude is closest to
    unity gain (|TF| ≈ 1) or 0 dB, depending on `dB`. This is useful for assessing stability
    margins in control systems.

    Parameters
    ----------
    tf : list of array_like
        A list containing two arrays: [magnitude, phase].
        - magnitude should be in linear scale or dB, depending on `dB`.
        - phase should be in degrees or radians, depending on `deg`.
    f : array_like
        Frequencies (Hz) corresponding to the values in `tf`.
    dB : bool, optional
        If True, treat magnitude as decibels and find where magnitude is closest to 0 dB.
        If False, find where magnitude is closest to 1 (linear). Default is False.
    deg : bool, optional
        If True, return phase margin in degrees. If False, return in radians. Default is True.

    Returns
    -------
    ugf : float
        Unity gain frequency (Hz), where magnitude crosses unity or 0 dB.
    margin : float
        Phase margin at the unity gain frequency (in degrees or radians).

    Notes
    -----
    - NaNs in magnitude or phase are removed before processing.
    - Phase is not unwrapped before evaluating the margin. This may introduce ±360° jumps near boundaries.
    - A positive phase margin generally implies stable feedback behavior.

    See Also
    --------
    get_gain_margin : Compute the gain margin at the phase crossover frequency.
    get_margin.register(np.ndarray) : Overload for complex-valued transfer functions.
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
def _(tf, f, deg=True, unwrap_phase=False, interpolate=False):
    """
    Compute the phase margin from a complex-valued transfer function array.

    The phase margin is evaluated at the frequency where the transfer function's
    magnitude is closest to unity (|TF| ≈ 1). This overload handles full complex
    transfer function arrays and supports optional interpolation and phase unwrapping
    to improve numerical stability and accuracy.

    Parameters
    ----------
    tf : ndarray
        Complex-valued transfer function evaluated at frequencies `f`.
    f : array_like
        Frequencies (Hz) corresponding to the values in `tf`.
    deg : bool, optional
        If True, return phase margin in degrees. If False, return in radians. Default is True.
    unwrap_phase : bool, optional
        If True, unwrap the phase before evaluating margin. This avoids phase discontinuities
        near ±180° and is recommended for systems with steep phase roll-off. Default is False.
    interpolate : bool, optional
        If True, interpolate the magnitude and phase onto a dense logarithmic grid to improve
        resolution around the unity gain frequency. Default is False.

    Returns
    -------
    ugf : float
        Unity gain frequency (Hz), where |TF| ≈ 1.
    margin : float
        Phase margin at the unity gain frequency (in degrees or radians).

    Notes
    -----
    - NaNs in the transfer function are removed prior to processing.
    - By default, phase is wrapped in the interval [−π, π] or [−180°, 180°].
      This can lead to apparent discontinuities near ±180°, especially problematic
      near the unity gain frequency. Setting `unwrap_phase=True` applies `np.unwrap`
      to ensure phase continuity.
    - Interpolation is disabled by default to preserve the original behavior.
      Enabling it may yield more accurate margins for sparse frequency grids.

    See Also
    --------
    get_gain_margin : Compute the gain margin at the phase crossover frequency.
    get_margin : Generic overload accepting magnitude/phase pairs.
    """
    import numpy as np
    from scipy.interpolate import interp1d

    tf = np.asarray(tf)
    f = np.asarray(f)

    # Remove NaNs in transfer function
    valid = ~np.isnan(tf)
    tfnew = tf[valid]
    fnew = f[valid]

    mag = np.abs(tfnew)

    # Compute phase in radians, unwrap if requested
    phase_rad = np.angle(tfnew, deg=False)
    if unwrap_phase:
        phase_rad = np.unwrap(phase_rad)
    if deg:
        phase = np.rad2deg(phase_rad)
    else:
        phase = phase_rad

    if interpolate:
        # Interpolate magnitude and phase onto a fine frequency grid
        fine_f = np.logspace(np.log10(fnew[0]), np.log10(fnew[-1]), 10000)
        mag_interp = interp1d(fnew, mag, kind="linear", bounds_error=False, fill_value="extrapolate")
        phase_interp = interp1d(fnew, phase, kind="linear", bounds_error=False, fill_value="extrapolate")

        fine_mag = mag_interp(fine_f)
        ugf_index = np.argmin(np.abs(fine_mag - 1))
        ugf = fine_f[ugf_index]
        phase_at_ugf = phase_interp(ugf)
    else:
        # Use the nearest point to unity gain
        index = np.argmin(np.abs(mag - 1))
        ugf = fnew[index]
        phase_at_ugf = phase[index]

    # Compute phase margin
    if deg:
        margin = 180 + phase_at_ugf
    else:
        margin = np.pi + phase_at_ugf

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

def tf_power_extrapolate(f, tf, f_trans, power, side='left', size=2, solver=True):
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

    if side == 'left':
        tfnew[f>f_trans] = tf[f>f_trans]
    elif side == 'right':
        tfnew[f<f_trans] = tf[f<f_trans]

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

def gain_for_crossover_frequency(Kp_log2, sps, f_cross, kind='I'):
    """
    Compute log₂ gain for I, II, or D block so that its magnitude matches the target at f_cross.

    For 'I' and 'D', the target is |P|. For 'II', both I and II gains are returned so that:
        - |I| = |P| at f_I
        - |II| = |P + I| at f_II

    Parameters
    ----------
    Kp_log2 : float
        Proportional gain in log₂ scale (i.e., log₂(Kp)).
    sps : float
        Sampling rate [Hz].
    f_cross : float or tuple(float, float)
        Desired crossover frequency [Hz]. For 'II', must be a tuple: (f_I, f_II).
    kind : {'I', 'II', 'D'}
        Type of block to match.

    Returns
    -------
    float or tuple(float, float)
        - For 'I' or 'D': log₂ gain for the block.
        - For 'II': tuple (Ki_log2, Kii_log2)

    Raises
    ------
    AssertionError
        If parameters are missing or invalid.
    """
    assert kind in ['I', 'II', 'D'], f"Invalid kind: {kind}"
    Kp = 2 ** Kp_log2

    if kind == 'II':
        assert isinstance(f_cross, (tuple, list)) and len(f_cross) == 2, \
            "For kind='II', f_cross must be a tuple: (f_I, f_II)."
        f_I, f_II = f_cross

        omega_I = 2 * np.pi * f_I / sps
        omega_II = 2 * np.pi * f_II / sps
        sin_I = 2 * np.sin(omega_I / 2)
        sin_II = 2 * np.sin(omega_II / 2)

        # Step 1: compute Ki from f_I (|I| = |P| → Ki = Kp * sin(ω_I / 2))
        Ki = Kp * sin_I

        # Step 2: evaluate I(f_II) as 1 / (1 - exp(-jω))
        I_val = Ki / (1 - np.exp(-1j * omega_II))  # corrected
        PI_val = Kp + I_val
        mag_PI = abs(PI_val)

        # Step 3: match II gain
        Kii = mag_PI * (sin_II ** 2)

        return np.log2(Ki), np.log2(Kii)

    elif kind == 'I':
        omega = 2 * np.pi * f_cross / sps
        sin_term = 2 * np.sin(omega / 2)
        return np.log2(Kp * sin_term)

    elif kind == 'D':
        omega = 2 * np.pi * f_cross / sps
        exp_negj = np.exp(-1j * omega)
        mag_discrete = abs((1 - exp_negj) / (1 + exp_negj))
        mag_discrete = max(mag_discrete, 1e-12)
        return np.log2(Kp / mag_discrete)
    
def Klf_from_cutoff(f_c, sps, n=1):
    """
    Compute log2 loop gain `Klf` from cutoff frequency for an n-stage IIR LPF.

    Uses numerical solver to ensure cascaded filter reaches -3 dB at `f_c`.

    Parameters
    ----------
    f_c : float
        Desired cutoff frequency (-3 dB point) in Hz.
    sps : float
        Sampling rate in Hz.
    n : int, optional
        Number of cascaded stages. Default is 1.

    Returns
    -------
    Klf : float
        Loop filter gain in log2 scale (i.e., α = 2^-Klf).
    """
    from scipy.optimize import root_scalar
    T = 1.0 / sps
    omega = 2 * np.pi * f_c * T  # Normalized digital frequency (rad/sample)

    def H_mag(alpha):
        # Magnitude of a single IIR stage at ω
        num = alpha
        den = np.sqrt(1 - 2*(1 - alpha)*np.cos(omega) + (1 - alpha)**2)
        mag = num / den
        return mag ** n

    def error_fn(alpha):
        return 20 * np.log10(H_mag(alpha)) + 3  # match -3 dB

    sol = root_scalar(error_fn, bracket=[1e-6, 1.0 - 1e-6], method='bisect')
    if not sol.converged:
        raise RuntimeError("Failed to converge while solving for α.")

    alpha = sol.root
    return -np.log2(alpha)

def log2_gain_to_db(log2_gain):
    """
    Converts base-2 logarithmic gain to dB.

    Parameters
    ----------
    log2_gain : float
        Gain as log₂(gain), as used in PIControllerComponent.

    Returns
    -------
    float
        Equivalent gain in dB.

    Examples
    --------
    >>> log2_gain_to_db(3.3219)
    20.0
    """
    linear_gain = 2**log2_gain
    return 20 * np.log10(linear_gain)

def db_to_log2_gain(db_gain):
    """
    Converts gain in dB to base-2 logarithmic gain.

    Parameters
    ----------
    db_gain : float
        Gain in decibels.

    Returns
    -------
    float
        Gain as log₂(gain), as used in PIControllerComponent.

    Examples
    --------
    >>> db_to_log2_gain(20.0)
    3.3219...
    """
    linear_gain = 10**(db_gain / 20)
    return np.log2(linear_gain)

def linear_to_log2_gain(linear_gain):
    """
    Converts linear gain to base-2 logarithmic gain.

    Parameters
    ----------
    linear_gain : float
        Linear gain (unitless ratio).

    Returns
    -------
    float
        Gain as log₂(gain), as used in PIControllerComponent.

    Examples
    --------
    >>> linear_to_log2_gain(10.0)
    3.3219...
    """
    return np.log2(linear_gain)