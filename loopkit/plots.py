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

from types import MappingProxyType
from typing import Any, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import (
    BoundaryNorm,
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
)
from matplotlib.patches import Patch

# Immutable default rc params for matplotlib (prevents accidental mutation)
_DEFAULT_RC = {
    "figure.dpi": 150,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.color": "#FFD700",
    "grid.linewidth": 0.7,
    "grid.linestyle": "--",
    "axes.prop_cycle": plt.cycler(
        "color",
        [
            "#000000",
            "#DC143C",
            "#00BFFF",
            "#FFD700",
            "#32CD32",
            "#FF69B4",
            "#FF4500",
            "#1E90FF",
            "#8A2BE2",
            "#FFA07A",
            "#8B0000",
        ],
    ),
}
default_rc: MappingProxyType[str, Any] = MappingProxyType(_DEFAULT_RC)


def _validate_sweep_result(result: Any) -> None:
    """Validate sweep result structure from parameter_sweep_nd."""
    if not isinstance(result, dict):
        raise TypeError(f"result must be a dict, got {type(result).__name__}")
    for key in ("parameter_names", "parameter_grid", "metrics"):
        if key not in result:
            raise ValueError(f"result must contain '{key}'")
    param_names = result["parameter_names"]
    if not isinstance(param_names, (list, tuple)) or len(param_names) not in (1, 2):
        raise ValueError(
            "result['parameter_names'] must be a sequence of length 1 or 2"
        )
    grid = result["parameter_grid"]
    if not isinstance(grid, dict):
        raise TypeError("result['parameter_grid'] must be a dict")
    for name in param_names:
        if name not in grid:
            raise ValueError(f"parameter_grid missing key '{name}'")


def plot_sweep_result(
    result: dict[str, Any],
    metric: str,
    *,
    logx: bool = True,
    logy: bool = False,
    ax: Optional[Axes] = None,
    cmap: Union[str, Colormap] = "RdYlGn",
    levels: int = 20,
    show_colorbar: bool = True,
    title: Optional[str] = None,
    interpolation: bool = False,
) -> Axes:
    """
    Plot a performance/stability metric from a parameter sweep result.

    Expects the result structure from ``parameter_sweep_nd`` (1D or 2D sweeps).
    Supported metrics include ``'ugf'`` and ``'phase_margin'``.

    Parameters
    ----------
    result : dict
        Sweep result with keys ``parameter_names``, ``parameter_grid``, and
        ``metrics``. Must be from a 1D or 2D parameter sweep.
    metric : str
        Metric key in ``result['metrics']`` to plot (e.g. ``'ugf'``,
        ``'phase_margin'``).
    logx : bool, optional
        Use log scale for x-axis. Default is True.
    logy : bool, optional
        Use log scale for y-axis (2D only). Default is False.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure and axes.
    cmap : str or Colormap, optional
        Colormap name or instance for non-phase-margin 2D plots. Default is
        ``"RdYlGn"``.
    levels : int, optional
        Number of contour levels for non-phase-margin 2D plots. Default is 20.
    show_colorbar : bool, optional
        Show colorbar or legend. Default is True.
    title : str, optional
        Plot title. If None, auto-generated from metric and parameter names.
    interpolation : bool, optional
        If True, plots a smooth, interpolated contour plot. If False (default),
        plots a discrete grid. For ``'phase_margin'``, this toggles between a
        smooth gradient and a categorical plot.

    Returns
    -------
    Axes
        The matplotlib axes containing the plot.

    Raises
    ------
    TypeError
        If result is not a dict or has invalid structure.
    ValueError
        If result structure is invalid, metric is missing, or sweep is not 1D/2D.
    """
    _validate_sweep_result(result)

    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("metric must be a non-empty string")
    if metric not in result["metrics"]:
        raise ValueError(
            f"metric '{metric}' not in result['metrics']. "
            f"Available: {list(result['metrics'].keys())}"
        )
    if not isinstance(levels, int) or levels < 1:
        raise ValueError("levels must be a positive integer")

    param_names = result["parameter_names"]
    metric_values = result["metrics"][metric]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if len(param_names) == 1:
        param_name = param_names[0]
        x = np.asarray(result["parameter_grid"][param_name])
        y = np.asarray(metric_values)
        if x.shape != y.shape:
            raise ValueError(
                f"1D grid and metric shape mismatch: {x.shape} vs {y.shape}"
            )
        ax.plot(x, y, marker="o")
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric.replace("_", " ").title())
        if logx:
            ax.set_xscale("log")
        ax.grid(True)

    elif len(param_names) == 2:
        x = np.asarray(result["parameter_grid"][param_names[0]])
        y = np.asarray(result["parameter_grid"][param_names[1]])
        z = np.asarray(metric_values)

        if z.ndim != 2:
            raise ValueError(f"2D metric must have shape (nx, ny), got {z.shape}")
        if x.ndim == 1 and y.ndim == 1:
            expected_shape = (len(y), len(x))
        else:
            expected_shape = z.shape
        if z.shape != expected_shape:
            raise ValueError(
                f"2D grid/metric shape mismatch: z {z.shape}, expected {expected_shape}"
            )

        if metric == "phase_margin":
            if interpolation:
                colors = [
                    "#DC143C",
                    "#FFD700",
                    "#32CD32",
                    "#3CB371",
                    "#87CEEB",
                ]
                nodes = [0.0, 30.0, 60.0, 76.0, 90.0]
                norm_nodes = [n / max(nodes) for n in nodes]
                cmap_gradient = LinearSegmentedColormap.from_list(
                    "phase_margin_gradient", list(zip(norm_nodes, colors))
                )
                cmap_gradient.set_under("#DC143C")
                cmap_gradient.set_over("#6495ED")
                norm = plt.Normalize(vmin=0, vmax=90)
                contour = ax.contourf(
                    x,
                    y,
                    z,
                    levels=512,
                    cmap=cmap_gradient,
                    norm=norm,
                    extend="both",
                )
                if show_colorbar:
                    cbar = fig.colorbar(
                        contour,
                        ax=ax,
                        label=metric.replace("_", " ").title(),
                    )
                    cbar.set_ticks(nodes)
            else:
                bounds = [-np.inf, 0, 30, 60, 76, 90, np.inf]
                labels = [
                    "Unstable (<0°)",
                    "Marginally Stable (0-30°)",
                    "Well Damped (30-60°)",
                    "Optimally Damped (60-76°)",
                    "Overdamped (76-90°)",
                    "Very Sluggish (>90°)",
                ]
                colors = [
                    "#DC143C",
                    "#FFD700",
                    "#32CD32",
                    "#3CB371",
                    "#87CEEB",
                    "#6495ED",
                ]
                cmap_custom = ListedColormap(colors)
                norm = BoundaryNorm(bounds, cmap_custom.N)
                ax.pcolormesh(
                    x,
                    y,
                    z,
                    cmap=cmap_custom,
                    norm=norm,
                    shading="auto",
                )
                if show_colorbar:
                    legend_handles = [
                        Patch(
                            facecolor=c,
                            edgecolor="black",
                            label=lbl,
                        )
                        for c, lbl in zip(colors, labels)
                    ]
                    ax.legend(
                        handles=legend_handles,
                        title="Phase Margin Classification",
                        loc="upper right",
                        fontsize=7,
                        title_fontsize=8,
                    )
        else:
            contour = ax.contourf(
                x,
                y,
                z,
                levels=levels,
                cmap=cmap,
                extend="both",
            )
            if show_colorbar:
                fig.colorbar(
                    contour,
                    ax=ax,
                    label=metric.replace("_", " ").title(),
                )

        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])

    if title is None:
        title = f"{metric.replace('_', ' ').title()} vs " + ", ".join(param_names)
    ax.set_title(title)

    return ax
