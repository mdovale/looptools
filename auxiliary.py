import numpy as np

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