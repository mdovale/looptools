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

from functools import singledispatch
from typing import Callable, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d
from scipy.optimize import brentq, curve_fit, root_scalar

from looptools import dsp


class _LoopWithGf(Protocol):
    """Protocol for objects with a Gf(f) method returning complex transfer function."""

    def Gf(self, f: ArrayLike | None = None) -> NDArray[np.complexfloating]: ...


def loop_crossover(
    loop1: _LoopWithGf,
    loop2: _LoopWithGf,
    frfr: ArrayLike,
) -> float:
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
    frfr_arr = np.asarray(frfr)
    if frfr_arr.size == 0:
        return float("nan")

    Gf1 = np.abs(loop1.Gf(f=frfr_arr))
    Gf2 = np.abs(loop2.Gf(f=frfr_arr))
    diff = np.asarray(Gf1 - Gf2, dtype=float)
    signs = np.sign(diff)
    signs[signs == 0] = 1  # Replace zeros with 1 to avoid sign change detection issues
    # sign_changes = np.where(np.diff(signs) != 0)[0] + 1
    # return frfr[sign_changes[0]] if sign_changes.size > 0 else np.nan
    sign_changes = np.diff(signs) != 0
    crossover_indices = np.where(sign_changes)[0] + 1
    if crossover_indices.size > 0:
        return float(frfr_arr[crossover_indices[0]])
    return float("nan")


def wrap_phase(
    phase: ArrayLike | float,
    deg: bool = False,
) -> NDArray[np.floating] | float:
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
    phase_arr = np.asarray(phase, dtype=float)
    if deg:
        phase_new = (phase_arr + 180.0) % 360.0 - 180.0
    else:
        phase_new = (phase_arr + np.pi) % (2 * np.pi) - np.pi

    return float(phase_new) if phase_arr.ndim == 0 else phase_new


def tf_group_delay(f: ArrayLike, tf: ArrayLike) -> NDArray[np.floating]:
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
    f_arr = np.asarray(f)
    tf_arr = np.asarray(tf, dtype=complex)
    if f_arr.size != tf_arr.size:
        raise ValueError(
            f"f and tf must have the same length, got {f_arr.size} and {tf_arr.size}"
        )

    tfnew, nanarray = dsp.nan_checker(tf_arr)
    has_nan = np.any(nanarray)

    phase = np.angle(tfnew, deg=False)
    phase = np.unwrap(phase)
    gd = -np.gradient(phase, 2 * np.pi * f_arr[~nanarray])

    if not has_nan:
        return np.asarray(gd, dtype=float)
    output = np.full(tf_arr.size, np.nan, dtype=float)
    idx = 0
    for i in range(tf_arr.size):
        if not nanarray[i]:
            output[i] = gd[idx]
            idx += 1
    return output


def polynomial_conversion_s_to_z(
    s_coeffs: ArrayLike,
    sps: float,
) -> NDArray[np.floating]:
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
    s_arr = np.asarray(s_coeffs, dtype=float)
    if s_arr.ndim != 1:
        raise ValueError(f"s_coeffs must be 1D, got ndim={s_arr.ndim}")
    if sps <= 0:
        raise ValueError(f"sps must be positive, got {sps}")

    size = s_arr.size
    z_coeffs = np.zeros(size, dtype=float)
    for i, c in enumerate(s_arr):
        order = size - (i + 1)
        base0 = np.array([sps, -sps])
        base = np.ones(1)
        for _ in range(order):
            base = np.convolve(base, base0)
        coord = c * base
        coord = np.concatenate((np.zeros(size - order - 1), coord), axis=0)
        z_coeffs += coord

    return z_coeffs


@singledispatch
def get_margin(
    tf: list[ArrayLike],
    f: ArrayLike,
    *,
    dB: bool = False,
    deg: bool = True,
) -> tuple[float, float]:
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
    if len(tf) != 2:
        raise ValueError(
            f"tf must be [magnitude, phase] with 2 elements, got {len(tf)}"
        )
    f_arr = np.asarray(f)
    mag, nanarray = dsp.nan_checker(tf[0])
    phase = np.asarray(tf[1])[~nanarray]
    fnew = f_arr[~nanarray]

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
def _get_margin_ndarray(
    tf: NDArray[np.complexfloating],
    f: ArrayLike,
    *,
    deg: bool = True,
    unwrap_phase: bool = True,
    interpolate: bool = True,
) -> tuple[float, float]:
    """
    Compute the phase margin from a complex-valued transfer function array.

    The phase margin is evaluated at the unity gain frequency (UGF), where the
    transfer function's magnitude is 1.

    Parameters
    ----------
    tf : ndarray
        Complex-valued transfer function evaluated at frequencies `f`.
    f : array_like
        Frequencies (Hz) corresponding to the values in `tf`.
    deg : bool, optional
        If True, return phase margin in degrees. If False, return in radians.
        Default is True.
    unwrap_phase : bool, optional
        If True (default), unwrap the phase before evaluating the margin.
        This is crucial for a correct margin calculation in most systems, as it
        ensures phase continuity across the -180° boundary. Default is True.
    interpolate : bool, optional
        If True (default), interpolate the magnitude and phase to find a more
        precise unity gain frequency. If False, the nearest point in the data is
        used. Default is True.

    Returns
    -------
    ugf : float
        Unity gain frequency (Hz), where |TF| = 1. Returns NaN if the gain
        never crosses unity.
    margin : float
        Phase margin at the unity gain frequency (in degrees or radians).
        Returns NaN if the gain never crosses unity.

    Notes
    -----
    - The phase margin is defined as: PM = Phase(at UGF) + 180°. A positive
      margin indicates stability, while a negative margin indicates instability.
    - NaNs in the input transfer function are removed before processing.
    """
    tf_arr = np.asarray(tf, dtype=complex)
    f_arr = np.asarray(f, dtype=float)
    if tf_arr.size != f_arr.size:
        raise ValueError(
            f"tf and f must have the same length, got {tf_arr.size} and {f_arr.size}"
        )

    # Remove NaNs in transfer function
    valid = ~np.isnan(tf_arr)
    if not np.any(valid):
        return float("nan"), float("nan")

    fnew = f_arr[valid]
    tfnew = tf_arr[valid]

    if len(fnew) < 2:
        interpolate = False

    mag = np.abs(tfnew)

    # Find the unity-gain frequency (UGF)
    # Check if the magnitude crosses 1 at all.
    # It crosses if there is at least one point > 1 and one point < 1.
    if not (np.any(mag > 1) and np.any(mag < 1)):
        # Gain never crosses 1, phase margin is not well-defined.
        # We can find the point closest to 1, but it's often better to return NaN.
        ugf_idx = np.argmin(np.abs(mag - 1))
        # Optional: you could return the margin at this point, but it's ambiguous.
        # For this implementation, we state it's undefined.
        # A perfectly-gained system (mag all 1) would be an exception.
        if not np.all(np.isclose(mag, 1)):
            return float("nan"), float("nan")

    # Compute phase in radians
    phase_rad = np.angle(tfnew)
    if unwrap_phase:
        phase_rad = np.unwrap(phase_rad)

    if deg:
        phase = np.rad2deg(phase_rad)
    else:
        phase = phase_rad

    if interpolate:
        log_f = np.log10(fnew)
        phase_interp = interp1d(
            log_f, phase, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        log_mag_interp = interp1d(log_f, np.log10(mag), kind="linear")

        # Find all crossings by looking at sign changes
        signs = np.sign(np.log10(mag)[:-1] * np.log10(mag)[1:])
        crossing_indices = np.where(signs < 0)[0]

        if len(crossing_indices) == 0:
            ugf_idx = np.argmin(np.abs(mag - 1))
            ugf = float(fnew[ugf_idx])
            phase_at_ugf = phase[ugf_idx]
        else:
            try:
                log_ugf = brentq(
                    log_mag_interp,
                    log_f[crossing_indices[0]],
                    log_f[crossing_indices[0] + 1],
                )
                ugf = 10**log_ugf
                phase_at_ugf = float(phase_interp(log_ugf))
            except (ValueError, RuntimeError):
                ugf_idx = np.argmin(np.abs(mag - 1))
                ugf = float(fnew[ugf_idx])
                phase_at_ugf = phase[ugf_idx]

    else:
        ugf_idx = np.argmin(np.abs(mag - 1))
        ugf = float(fnew[ugf_idx])
        phase_at_ugf = phase[ugf_idx]

    # Phase margin = Phase(at UGF) + 180°, wrapped to [-180, 180)
    # The formula is simply Phase + 180 (or pi).
    # This is correct ONLY if the phase is unwrapped.
    if deg:
        margin = phase_at_ugf + 180.0
    else:
        margin = phase_at_ugf + np.pi

    return ugf, (margin + 180) % 360 - 180


def tf_power_fitting(
    f: ArrayLike,
    tf: ArrayLike,
    fnew: ArrayLike,
    power: float,
) -> NDArray[np.complexfloating]:
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
    f_arr = np.asarray(f)
    tf_arr = np.asarray(tf, dtype=complex)
    fnew_arr = np.asarray(fnew)

    def power_law(freq: ArrayLike, a: float, b: float, *, mag: bool = False) -> NDArray:
        omega = 2 * np.pi * freq
        s = 1j * omega
        out = a * s**b
        return np.abs(out) if mag else out

    def fabs(freq: ArrayLike, a: float) -> NDArray:
        return power_law(freq, a, power, mag=True)

    popt, _ = curve_fit(fabs, f_arr, np.abs(tf_arr))
    return power_law(fnew_arr, popt[0], power, mag=False)


def tf_power_solver(
    f: float | np.floating,
    tf: complex | np.complexfloating,
    fnew: ArrayLike,
    power: float,
) -> NDArray[np.complexfloating]:
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
        If `f` is not a scalar (float or numpy scalar).
    """
    if not (np.isscalar(f) and isinstance(f, (int, float, np.integer, np.floating))):
        raise ValueError(f"f must be a scalar frequency in Hz, got {type(f).__name__}")

    f_val = float(f)
    s = 1j * 2 * np.pi * f_val
    a = np.abs(tf / s**power)
    fnew_arr = np.asarray(fnew)
    snew = 1j * 2 * np.pi * fnew_arr
    return np.asarray(a * snew**power, dtype=complex)


def tf_power_extrapolate(
    f: ArrayLike,
    tf: ArrayLike,
    f_trans: float,
    power: float,
    *,
    side: Literal["left", "right"] = "left",
    size: int = 2,
    solver: bool = True,
) -> NDArray[np.complexfloating]:
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
    f_arr = np.asarray(f)
    tf_arr = np.asarray(tf, dtype=complex)
    if f_arr.size != tf_arr.size:
        raise ValueError(
            f"f and tf must have the same length, got {f_arr.size} and {tf_arr.size}"
        )
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")

    idx = dsp.index_of_the_nearest(f_arr, f_trans)

    if solver:
        tfnew = tf_power_solver(float(f_arr[idx]), tf_arr[idx], f_arr, power=power)
    else:
        ftmp, tftmp = dsp.crop_data(
            f_arr, tf_arr, xmin=f_trans, xmax=float(np.max(f_arr))
        )
        ftmp = ftmp[:size]
        tftmp = tftmp[:size]
        tfnew = tf_power_fitting(ftmp, tftmp, f_arr, power=power)

    tfnew = np.asarray(tfnew, dtype=complex).copy()
    if side == "left":
        mask = f_arr > f_trans
        tfnew[mask] = tf_arr[mask]
    else:
        mask = f_arr < f_trans
        tfnew[mask] = tf_arr[mask]

    return tfnew


def add_transfer_function(
    f: ArrayLike,
    tf1: Callable[[ArrayLike], NDArray[np.complexfloating]],
    tf2: Callable[[ArrayLike], NDArray[np.complexfloating]],
    *,
    extrapolate: bool = False,
    f_trans: float = 1e-1,
    power: float = -2,
    size: int = 2,
    solver: bool = True,
) -> NDArray[np.complexfloating]:
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
    f_arr = np.asarray(f)
    tf1f = np.asarray(tf1(f_arr), dtype=complex)
    tf2f = np.asarray(tf2(f_arr), dtype=complex)

    if extrapolate:
        return tf_power_extrapolate(
            f_arr, tf1f + tf2f, f_trans, power, size=size, solver=solver
        )
    return tf1f + tf2f


def mul_transfer_function(
    f: ArrayLike,
    tf1: Callable[[ArrayLike], NDArray[np.complexfloating]],
    tf2: Callable[[ArrayLike], NDArray[np.complexfloating]],
    *,
    extrapolate: bool = False,
    f_trans: float = 1e-1,
    power: float = -2,
    size: int = 2,
    solver: bool = True,
) -> NDArray[np.complexfloating]:
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
    f_arr = np.asarray(f)
    tf1f = np.asarray(tf1(f_arr), dtype=complex)
    tf2f = np.asarray(tf2(f_arr), dtype=complex)

    if extrapolate:
        return tf_power_extrapolate(
            f_arr, tf1f * tf2f, f_trans, power, size=size, solver=solver
        )
    return tf1f * tf2f


def gain_for_crossover_frequency(
    Kp_log2: float,
    sps: float,
    f_cross: float | tuple[float, float],
    *,
    kind: Literal["I", "II", "D"] = "I",
) -> float | tuple[float, float]:
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
    ValueError
        If parameters are missing or invalid.
    """
    if kind not in ("I", "II", "D"):
        raise ValueError(f"kind must be 'I', 'II', or 'D', got {kind!r}")
    if sps <= 0:
        raise ValueError(f"sps must be positive, got {sps}")

    Kp = 2**Kp_log2

    if kind == "II":
        if not isinstance(f_cross, (tuple, list)) or len(f_cross) != 2:
            raise ValueError(
                "For kind='II', f_cross must be a tuple of two frequencies: (f_I, f_II)"
            )
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
        Kii = mag_PI * (sin_II**2)

        return np.log2(Ki), np.log2(Kii)

    if kind == "I":
        omega = 2 * np.pi * f_cross / sps
        sin_term = 2 * np.sin(omega / 2)
        return np.log2(Kp * sin_term)

    # kind == 'D'
    omega = 2 * np.pi * f_cross / sps
    exp_negj = np.exp(-1j * omega)
    mag_discrete = abs((1 - exp_negj) / (1 + exp_negj))
    mag_discrete = max(mag_discrete, 1e-12)
    return np.log2(Kp / mag_discrete)


def Klf_from_cutoff(
    f_c: float,
    sps: float,
    *,
    n: int = 1,
) -> float:
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
    if f_c <= 0 or sps <= 0:
        raise ValueError(f"f_c and sps must be positive, got f_c={f_c}, sps={sps}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    T = 1.0 / sps
    omega = 2 * np.pi * f_c * T  # Normalized digital frequency (rad/sample)

    def H_mag(alpha: float) -> float:
        num = alpha
        den = np.sqrt(1 - 2 * (1 - alpha) * np.cos(omega) + (1 - alpha) ** 2)
        mag = num / den
        return mag**n

    def error_fn(alpha):
        return 20 * np.log10(H_mag(alpha)) + 3  # match -3 dB

    sol = root_scalar(error_fn, bracket=[1e-6, 1.0 - 1e-6], method="bisect")
    if not sol.converged:
        raise RuntimeError("Failed to converge while solving for α.")

    alpha = sol.root
    return float(-np.log2(alpha))


def log2_gain_to_db(log2_gain: float) -> float:
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
    return float(20 * np.log10(linear_gain))


def db_to_log2_gain(db_gain: float) -> float:
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
    linear_gain = 10 ** (db_gain / 20)
    return float(np.log2(linear_gain))


def linear_to_log2_gain(linear_gain: float) -> float:
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
    if linear_gain <= 0:
        raise ValueError(f"linear_gain must be positive, got {linear_gain}")
    return float(np.log2(linear_gain))
