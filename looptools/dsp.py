import numpy as np
from scipy.integrate import cumulative_trapezoid
import logging
logger = logging.getLogger(__name__)

def index_of_the_nearest(data, value):
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
    """
    idx = np.argmin(np.abs(np.array(data) - value))
    return idx

def crop_data(x,y,xmin,xmax):
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
    """
    x_tmp = []
    y_tmp = []
    for i in range(x.size):
        if(x[i] >= xmin and x[i] <= xmax):
            x_tmp.append(x[i])
            y_tmp.append(y[i])

    return np.array(x_tmp), np.array(y_tmp)

def integral_rms(fourier_freq, asd, pass_band=None):
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

    Notes
    -----
    - Internally uses `scipy.integrate.cumulative_trapezoid` with `initial=0` for numerical integration.
    - Automatically restricts integration bounds to the overlap between `fourier_freq` and `pass_band`.
    - This function assumes that `asd` represents single-sided amplitude spectral density.
    """
    if pass_band is None:
        pass_band = [-np.inf,np.inf]

    integral_range_min = max(np.min(fourier_freq), pass_band[0])
    integral_range_max = min(np.max(fourier_freq), pass_band[1])
    f_tmp, asd_tmp = crop_data(fourier_freq, asd, integral_range_min, integral_range_max)
    integral_rms2 = cumulative_trapezoid(asd_tmp**2, f_tmp, initial=0)
    return np.sqrt(integral_rms2[-1])


def nan_checker(x):
    """
    Check for NaNs in an array and return a cleaned version with a mask of NaN locations.

    If any NaN values are present in the input array `x`, they are removed, and a boolean
    mask is returned to indicate the original locations of the NaNs.

    Parameters
    ----------
    x : array_like
        Input array to check for NaN values.

    Returns
    -------
    xnew : ndarray
        Array with all NaN entries removed. If no NaNs are present, returns the input unchanged.
    nanarray : ndarray of bool
        Boolean array of the same shape as `x`, where `True` indicates a NaN in the original input.

    Notes
    -----
    - Logs a warning message if any NaNs are detected using `logger.warning(...)`.
    - Useful for preprocessing before numerical operations (e.g., `unwrap`, `gradient`)
        that do not handle NaNs gracefully.
    - Maintains alignment with original indices using `nanarray`.

    """

    if True in np.isnan(x):
        logger.warning('Nan was detected in the input array')
        nanarray = np.isnan(x)
        xnew = x[~nanarray]
    else:
        xnew, nanarray = x, np.full(x.size, False)

    return xnew, nanarray