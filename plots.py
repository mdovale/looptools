import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
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
                      show_colorbar=True, title=None,
                      interpolation=False):
    """
    Plot a performance/stability metric from a parameter sweep result.

    Parameters
    ----------
    ... (other params) ...
    interpolation : bool, optional
        If True, plots a smooth, interpolated contour plot. If False (default),
        plots a discrete grid. For 'phase_margin', this toggles between a
        smooth gradient and a categorical plot.
    """
    param_names = result["parameter_names"]
    metric_values = result["metrics"][metric]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if len(param_names) == 1:
        # (1D plot logic remains the same)
        param_name = param_names[0]
        x = result["parameter_grid"][param_name]
        y = metric_values
        ax.plot(x, y, marker="o")
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric.replace('_', ' ').title())
        if logx:
            ax.set_xscale("log")
        ax.grid(True)

    elif len(param_names) == 2:
        x = result["parameter_grid"][param_names[0]]
        y = result["parameter_grid"][param_names[1]]
        z = metric_values

        # Special handling for phase margin
        if metric == "phase_margin":
            if interpolation:
                # --- SMOOTH, INTERPOLATED PLOT ---
                # Define the colors and the phase margin values they correspond to.
                # We'll create a gradient from 0 to 90 degrees.
                colors = ['#DC143C', '#FFD700', '#32CD32', '#3CB371', '#87CEEB']
                nodes = [0.0, 30.0, 60.0, 76.0, 90.0]

                # Normalize the nodes to the [0, 1] range for the colormap
                norm_nodes = [n / max(nodes) for n in nodes]
                
                # Create a continuous colormap
                cmap_gradient = LinearSegmentedColormap.from_list(
                    "phase_margin_gradient", list(zip(norm_nodes, colors))
                )
                
                # Set colors for values outside the 0-90 range
                cmap_gradient.set_under('#DC143C') # Unstable
                cmap_gradient.set_over('#6495ED')  # Very Sluggish

                # Normalize the data to the 0-90 range
                norm = plt.Normalize(vmin=0, vmax=90)
                
                # Use contourf for a smooth plot
                contour = ax.contourf(x, y, z, levels=512, cmap=cmap_gradient, norm=norm, extend='both')

                if show_colorbar:
                    cbar = fig.colorbar(contour, ax=ax, label=metric.replace('_', ' ').title())
                    cbar.set_ticks(nodes) # Add ticks at key locations

            else:
                # --- DISCRETE, CATEGORICAL PLOT (pcolormesh) ---
                bounds = [-np.inf, 0, 30, 60, 76, 90, np.inf]
                labels = ['Unstable (<0°)', 'Marginally Stable (0-30°)', 'Well Damped (30-60°)',
                          'Optimally Damped (60-76°)', 'Overdamped (76-90°)', 'Very Sluggish (>90°)']
                colors = ['#DC143C', '#FFD700', '#32CD32', '#3CB371', '#87CEEB', '#6495ED']
                
                cmap_custom = ListedColormap(colors)
                norm = BoundaryNorm(bounds, cmap_custom.N)

                ax.pcolormesh(x, y, z, cmap=cmap_custom, norm=norm, shading='auto')

                if show_colorbar: # This flag now controls the legend for this plot type
                    legend_handles = [Patch(facecolor=color, edgecolor='black', label=label)
                                      for color, label in zip(colors, labels)]
                    ax.legend(handles=legend_handles, title="Phase Margin Classification",
                              loc="upper right", fontsize=7, title_fontsize=8)
        else:
            # --- General case for other metrics (e.g., 'ugf') ---
            # Default behavior is already interpolated via contourf
            contour = ax.contourf(x, y, z, levels=levels, cmap=cmap, extend='both')
            if show_colorbar:
                fig.colorbar(contour, ax=ax, label=metric.replace('_', ' ').title())

        # Axis settings for all 2D plots
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")

        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])

    else:
        raise ValueError("plot_sweep_result supports only 1D or 2D parameter sweeps.")

    if title is None:
        title = f"{metric.replace('_', ' ').title()} vs " + ", ".join(param_names)
    ax.set_title(title)
    
    return ax