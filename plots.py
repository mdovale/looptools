import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from looptools.loopmath import *
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

def plot_sweep_result(result, metric, *,
                      logx=True, logy=False,
                      ax=None, cmap="RdYlGn", levels=20,
                      show_colorbar=True, title=None):
    """
    Plot a performance/stability metric from a parameter sweep result.

    Parameters
    ----------
    result : dict
        Output from `parameter_sweep_1d` or `parameter_sweep_nd`.
    metric : str
        Name of the metric to plot (e.g., 'phase_margin', 'ugf').
    logx : bool, optional
        If True, use log scale on X-axis (default: True).
    logy : bool, optional
        If True, use log scale on Y-axis (2D case only).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    cmap : str, optional
        Colormap for 2D plots. Default is 'RdYlGn'.
    levels : int or list, optional
        Number of contour levels (or list of contour values).
    show_colorbar : bool, optional
        Whether to display the colorbar in 2D plots.
    title : str, optional
        Title for the plot. If None, a descriptive title is auto-generated.
    """
    param_names = result["parameter_names"]
    metric_values = result["metrics"][metric]

    with plt.rc_context(default_rc):
        if ax is None:
            fig, ax = plt.subplots()

        if len(param_names) == 1:
            # --- 1D plot ---
            x = result["parameter_values"]
            y = metric_values
            ax.plot(x, y, marker="o")
            ax.set_xlabel(param_names[0])
            ax.set_ylabel(metric.replace('_', ' ').title())

            if logx:
                ax.set_xscale("log")
            ax.grid(True)

        elif len(param_names) == 2:
            # --- 2D contour plot ---
            x = result["parameter_grid"][param_names[0]]
            y = result["parameter_grid"][param_names[1]]
            z = metric_values

            contour = ax.contourf(x, y, z, levels=levels, cmap=cmap, extend='both')

            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")

            ax.set_xlabel(param_names[0])
            ax.set_ylabel(param_names[1])
            if show_colorbar:
                plt.colorbar(contour, ax=ax, label=metric.replace('_', ' ').title())

        else:
            raise ValueError("plot_sweep_result supports only 1D or 2D parameter sweeps.")

        if title is None:
            title = f"{metric.replace('_', ' ').title()} vs " + ", ".join(param_names)
        ax.set_title(title)
        return ax
