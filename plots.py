import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as scc

default_rc = {
    'figure.dpi': 150,
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.grid': True,
    'grid.color': '#FFD700',
    'grid.linewidth': 0.7,
    'grid.linestyle': '--',
    'axes.prop_cycle': plt.cycler('color', [
        '#000000', '#DC143C', '#00BFFF', '#FFD700', '#32CD32',
        '#FF69B4', '#FF4500', '#1E90FF', '#8A2BE2', '#FFA07A', '#8B0000'
    ]),
}

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