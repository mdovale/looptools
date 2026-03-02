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

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import cumulative_trapezoid

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


def index_of_the_nearest(data: ArrayLike, value: float | int) -> int:
    """
    Return the index of the element in `data` closest to a given target value.

    This function computes the absolute difference between each element in `data`
    and the target `value`, and returns the index of the element with the smallest difference.

    Parameters
    ----------
    data : array_like
        Input array of numeric values to search through.
    value : float or int
        Target value to which the closest element is sought.

    Returns
    -------
    idx : int
        Index of the element in `data` that is closest to `value`.

    Raises
    ------
    ValueError
        If `data` is empty.
    """
    arr = np.asarray(data)
    if arr.size == 0:
        raise ValueError("Cannot find nearest index in empty array")
    idx = int(np.argmin(np.abs(arr - value)))
    return idx


def crop_data(
    x: ArrayLike,
    y: ArrayLike,
    xmin: float,
    xmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop paired data arrays to the range [xmin, xmax] based on the x-axis values.

    This function filters the input arrays `x` and `y` to include only those
    elements where `xmin <= x[i] <= xmax`. It returns the filtered `x` and `y`
    arrays with corresponding indices preserved.

    Parameters
    ----------
    x : array_like
        1D array representing the independent variable (e.g., frequency, time).
    y : array_like
        1D array representing the dependent variable. Must be the same length as `x`.
    xmin : float
        Lower bound of the `x` range to retain.
    xmax : float
        Upper bound of the `x` range to retain.

    Returns
    -------
    x_out : ndarray
        Cropped `x` array containing values within [xmin, xmax].
    y_out : ndarray
        Cropped `y` array containing values corresponding to `x_out`.

    Raises
    ------
    ValueError
        If `x` and `y` have different lengths, or if `xmin > xmax`.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    if x_arr.size != y_arr.size:
        raise ValueError(
            f"x and y must have the same length, got {x_arr.size} and {y_arr.size}"
        )
    if xmin > xmax:
        raise ValueError(f"xmin must be <= xmax, got xmin={xmin}, xmax={xmax}")

    mask = (x_arr >= xmin) & (x_arr <= xmax)
    return x_arr[mask].copy(), y_arr[mask].copy()


def integral_rms(
    fourier_freq: ArrayLike,
    asd: ArrayLike,
    pass_band: tuple[float, float] | list[float] | None = None,
) -> float:
    """
    Compute the root-mean-square (RMS) value from an amplitude spectral density (ASD) via integration.

    This function calculates the total RMS over a frequency band by integrating the square of the
    ASD using the trapezoidal rule. The integration can be limited to a specific passband.

    Parameters
    ----------
    fourier_freq : array_like
        Frequency array (Hz) corresponding to the ASD values.
    asd : array_like
        Amplitude spectral density values (same length as `fourier_freq`).
        Units are assumed to be consistent with desired RMS (e.g., m/√Hz → RMS in m).
    pass_band : list or tuple of float, optional
        Two-element list or tuple [f_min, f_max] specifying the frequency range (Hz) over which
        to compute the RMS. If None, the full range of `fourier_freq` is used. Default is None.

    Returns
    -------
    rms : float
        Root-mean-square value computed as the square root of the integrated ASD² over the passband.
        Returns 0.0 if the cropped frequency range is empty or has a single point.

    Raises
    ------
    ValueError
        If `fourier_freq` and `asd` have different lengths, or if `pass_band` is invalid.

    Notes
    -----
    - Internally uses `scipy.integrate.cumulative_trapezoid` with `initial=0` for numerical integration.
    - Automatically restricts integration bounds to the overlap between `fourier_freq` and `pass_band`.
    - This function assumes that `asd` represents single-sided amplitude spectral density.
    """
    f_arr = np.asarray(fourier_freq)
    asd_arr = np.asarray(asd)

    if f_arr.size != asd_arr.size:
        raise ValueError(
            f"fourier_freq and asd must have the same length, "
            f"got {f_arr.size} and {asd_arr.size}"
        )

    if pass_band is None:
        pass_band = (-np.inf, np.inf)
    else:
        if len(pass_band) != 2:
            raise ValueError(
                f"pass_band must have exactly 2 elements [f_min, f_max], got {len(pass_band)}"
            )
        f_min, f_max = float(pass_band[0]), float(pass_band[1])
        if f_min > f_max:
            raise ValueError(
                f"pass_band f_min must be <= f_max, got f_min={f_min}, f_max={f_max}"
            )
        pass_band = (f_min, f_max)

    integral_range_min = max(np.min(f_arr), pass_band[0])
    integral_range_max = min(np.max(f_arr), pass_band[1])
    if integral_range_min > integral_range_max:
        return 0.0
    f_tmp, asd_tmp = crop_data(f_arr, asd_arr, integral_range_min, integral_range_max)

    if f_tmp.size < 2:
        return 0.0

    integral_rms2 = cumulative_trapezoid(asd_tmp**2, f_tmp, initial=0)
    return float(np.sqrt(integral_rms2[-1]))


def nan_checker(x: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Check for NaNs in an array and return a cleaned version with a mask of NaN locations.

    If any NaN values are present in the input array `x`, they are removed, and a boolean
    mask is returned to indicate the original locations of the NaNs.

    Parameters
    ----------
    x : array_like
        Input array to check for NaN values. Must be numeric (float or complex).

    Returns
    -------
    xnew : ndarray
        Array with all NaN entries removed. If no NaNs are present, returns the input unchanged.
    nanarray : ndarray of bool
        Boolean array of the same shape as `x`, where `True` indicates a NaN in the original input.

    Raises
    ------
    TypeError
        If the input cannot be converted to a numeric array.

    Notes
    -----
    - Logs a warning message if any NaNs are detected using `logger.warning(...)`.
    - Useful for preprocessing before numerical operations (e.g., `unwrap`, `gradient`)
        that do not handle NaNs gracefully.
    - Maintains alignment with original indices using `nanarray`.

    """
    x_arr = np.asarray(x)
    # Integer arrays cannot contain NaN
    if np.issubdtype(x_arr.dtype, np.integer):
        return x_arr, np.zeros(x_arr.size, dtype=bool)
    has_nan = np.any(np.isnan(x_arr))

    if has_nan:
        logger.warning("NaN was detected in the input array")
        nanarray = np.isnan(x_arr)
        xnew = x_arr[~nanarray].copy()
    else:
        xnew = x_arr
        nanarray = np.zeros(x_arr.size, dtype=bool)

    return xnew, nanarray
