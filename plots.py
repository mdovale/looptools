import matplotlib.pyplot as plt

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