import numpy as np
from scipy.integrate import cumulative_trapezoid
import logging
logger = logging.getLogger(__name__)

def index_of_the_nearest(data, value):
	""" extract an index at which data gets closest to a value
	Args:
		data: data array
		value: target value
	"""

	idx = np.argmin(np.abs(np.array(data) - value))
	return idx

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

def integral_rms(fourier_freq, asd, pass_band=None):
    """ Compute the RMS as integral of an ASD.
    
    Parameters
    ----------
        fourier_freq: fourier frequency (Hz)
        asd: amplitude spectral density from which RMS is computed
        pass_band: [0] = min, [1] = max 
    """
    if pass_band is None:
        pass_band = [-np.inf,np.inf]

    integral_range_min = max(np.min(fourier_freq), pass_band[0])
    integral_range_max = min(np.max(fourier_freq), pass_band[1])
    f_tmp, asd_tmp = crop_data(fourier_freq, asd, integral_range_min, integral_range_max)
    integral_rms2 = cumulative_trapezoid(asd_tmp**2, f_tmp, initial=0)
    return np.sqrt(integral_rms2[-1])


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