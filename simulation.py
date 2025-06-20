"""
This module provides tools for control loop analysis, including N-dimensional
parameter sweeps and optimization, with support for parallel processing.
"""
import copy
import logging
import warnings
from contextlib import contextmanager
from itertools import product
from typing import Any, Dict, List, Tuple, Callable, Sequence

import numpy as np
from joblib import Parallel, delayed
from joblib.parallel import BatchCompletionCallBack
from scipy.optimize import minimize, OptimizeResult
from tqdm.auto import tqdm

# Assuming these are from a library this file is part of.
# If these are not available, placeholders would be needed.
from looptools.loopmath import get_margin, loop_crossover

# Placeholder for the main loop object type for type hinting
# from my_loop_library import LOOP
LOOP = Any

logger = logging.getLogger(__name__)

# --- Parallel Progress Bar Utility ---

@contextmanager
def tqdm_joblib(tqdm_object: tqdm) -> tqdm:
    """
    Context manager to patch joblib to report progress into a tqdm bar.
    Note: Newer versions of joblib may have built-in support for tqdm,
    but this remains a reliable method.
    """
    class TqdmBatchCompletionCallback(BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = Parallel.__init__.__globals__['BatchCompletionCallBack']
    Parallel.__init__.__globals__['BatchCompletionCallBack'] = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        Parallel.__init__.__globals__['BatchCompletionCallBack'] = old_callback
        tqdm_object.close()

# --- Core Sweep Logic ---

def _calculate_point(
    loop_prototype: LOOP,
    params_to_set: Dict[str, Any],
    frequencies: np.ndarray,
    deg: bool,
    unwrap_phase: bool,
    interpolate: bool,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Worker function for parallel sweep. Calculates metrics for one parameter point.

    This function creates a local, independent copy of the loop object to
    ensure thread/process safety.
    """
    loop = copy.deepcopy(loop_prototype)

    for name, val in params_to_set.items():
        setattr(loop, name, val)

    tf = loop.Gf(frequencies)
    mag = np.abs(tf)
    phase_rad = np.angle(tf, deg=False)
    if unwrap_phase:
        phase_rad = np.unwrap(phase_rad)
    
    phase = np.rad2deg(phase_rad) if deg else phase_rad

    ugf, pm = get_margin(
        tf, frequencies, deg=deg, unwrap_phase=unwrap_phase, interpolate=interpolate
    )

    return ugf, pm, mag, phase

# --- Modern Parameter Sweep Functions ---

def parameter_sweep_nd(
    loop: LOOP,
    param_grid: Dict[str, Sequence],
    frequencies: Sequence,
    deg: bool = True,
    unwrap_phase: bool = False,
    interpolate: bool = False,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Performs an N-dimensional parameter sweep in parallel.

    This function can be used to analyze loop stability and performance over a 
    grid of parameters, using joblib for parallel computation to improve performance.

    Parameters
    ----------
    loop : LOOP
        The loop object to use as a template. It is NOT modified in-place.
    param_grid : dict
        Maps property names (str) to arrays of values to sweep.
    frequencies : array_like
        Frequencies (Hz) for evaluating the loop's transfer function.
    deg : bool, optional
        If True, compute phase and margin in degrees. Defaults to True.
    unwrap_phase : bool, optional
        If True, unwrap phase before computing margin. Defaults to False.
    interpolate : bool, optional
        If True, interpolate TF to refine margin calculations. Defaults to False.
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all cores. 1 disables parallelism.
        Defaults to -1.

    Returns
    -------
    dict
        A dictionary containing the sweep results, including parameter grids,
        frequencies, and calculated metrics (UGF, phase margin, magnitude, phase).
    
    Raises
    ------
    ValueError
        If a parameter name in `param_grid` is not a valid property of the loop.
    """
    # Validate parameter names before starting expensive computation
    for name in param_grid.keys():
        if not hasattr(loop, name):
            raise ValueError(f"Property '{name}' not found on the provided loop object.")

    param_names = list(param_grid.keys())
    sweep_axes = [np.asarray(param_grid[k]) for k in param_names]
    mesh = np.meshgrid(*sweep_axes, indexing='ij')
    shape = mesh[0].shape
    freqs_arr = np.asarray(frequencies)
    num_freqs = len(freqs_arr)
    total_points = np.prod(shape)

    # Create a flat list of tasks (dictionaries of parameter settings)
    param_values_flat = [m.flatten() for m in mesh]
    tasks = [dict(zip(param_names, p_set)) for p_set in zip(*param_values_flat)]
    
    cores_str = "all available" if n_jobs == -1 else n_jobs
    logger.info(
        f"Starting parallel sweep over {total_points} points using {cores_str} cores."
    )
    
    # Run tasks in parallel with a progress bar
    # The 'loop' object is deep-copied by the worker, not here, to ensure
    # safety regardless of the joblib backend (threading vs. multiprocessing).
    with tqdm_joblib(tqdm(total=total_points, desc="Sweeping Parameters")) as progress_bar:
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_calculate_point)(
                loop, params, freqs_arr, deg, unwrap_phase, interpolate
            ) for params in tasks
        )

    # --- Collect and reshape results ---
    ugf_flat, pm_flat, mag_flat, phase_flat = zip(*results)

    # Reshape scalar metrics to the N-D grid shape
    ugf_arr = np.array(ugf_flat).reshape(shape)
    pm_arr = np.array(pm_flat).reshape(shape)
    
    # Reshape vector metrics (per frequency) to N-D grid shape + frequency dim
    mag_arr = np.array(mag_flat).reshape(shape + (num_freqs,))
    phase_arr = np.array(phase_flat).reshape(shape + (num_freqs,))

    return {
        "parameter_names": param_names,
        "parameter_grid": {name: mesh[i] for i, name in enumerate(param_names)},
        "frequencies": freqs_arr,
        "metrics": {
            "ugf": ugf_arr,
            "phase_margin": pm_arr,
        },
        "open_loop": {
            "magnitude": mag_arr,
            "phase": phase_arr,
        }
    }


def parameter_sweep_1d(
    loop: LOOP,
    prop_name: str,
    values: Sequence,
    frequencies: Sequence,
    deg: bool = True,
    unwrap_phase: bool = False,
    interpolate: bool = False
) -> Dict[str, Any]:
    """
    Sweeps a single parameter and computes stability metrics.

    This is a convenience wrapper around `parameter_sweep_nd` for 1D sweeps.

    Parameters
    ----------
    loop : LOOP
        The loop object to analyze. It is NOT modified in-place.
    prop_name : str
        Name of the property to sweep.
    values : array_like
        Values to assign to the property.
    frequencies : array_like
        Frequencies (Hz) for evaluation.
    deg : bool, optional
        If True, use degrees for phase. Defaults to True.
    unwrap_phase : bool, optional
        If True, unwrap phase. Defaults to False.
    interpolate : bool, optional
        If True, interpolate for margin calculation. Defaults to False.

    Returns
    -------
    dict
        A dictionary with sweep results in a 1D-friendly format.
    """
    param_grid = {prop_name: values}
    
    # Run the N-D sweep with n_jobs=1 (sequential) for a single-threaded 1D sweep
    results_nd = parameter_sweep_nd(
        loop, param_grid, frequencies, deg, unwrap_phase, interpolate, n_jobs=1
    )

    # Unpack and format results for the 1D case
    return {
        "parameter_name": prop_name,
        "parameter_values": np.asarray(values),
        "frequencies": results_nd["frequencies"],
        "metrics": {
            "ugf": results_nd["metrics"]["ugf"],
            "phase_margin": results_nd["metrics"]["phase_margin"],
        },
        "open_loop": {
            "magnitude": results_nd["open_loop"]["magnitude"],
            "phase": results_nd["open_loop"]["phase"],
        },
    }

# --- Deprecated and Legacy Functions ---

def old_parameter_sweep_1d(
    loop: LOOP,
    frfr: Sequence,
    noise: Sequence,
    comp: str,
    sweep: str,
    space: Sequence,
    tf_from: str,
    tf_to: str,
    isTF: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    [DEPRECATED] Sweep a component parameter and evaluate loop error and performance.

    This function is preserved for legacy use cases but `parameter_sweep_1d` or
    `parameter_sweep_nd` are recommended for new development.

    This sweep is inefficient and uses an outdated, error-prone API.
    """
    warnings.warn(
        "`old_parameter_sweep_1d` is deprecated. Use `parameter_sweep_1d` or "
        "`parameter_sweep_nd` for better performance and a cleaner API.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    _loop = copy.deepcopy(loop)
    
    if comp not in _loop.components_dict:
        raise ValueError(f"Component '{comp}' not found in the loop.")
    component = _loop.components_dict[comp]
    if sweep not in component.properties:
        raise ValueError(f"Property '{sweep}' not found in component '{comp}'.")
    if tf_from not in _loop.components_dict:
        raise ValueError(f"Component '{tf_from}' not found in the loop.")
    if tf_to not in _loop.components_dict:
        raise ValueError(f"Component '{tf_to}' not found in the loop.")

    fsweep = component.properties[sweep][1]
    
    # Pre-allocate arrays for efficiency
    N = len(space)
    F = len(frfr)
    ugf = np.empty(N)
    margin = np.empty(N)
    rms = np.empty(N)
    asd = np.empty((F, N))  # Shape (freqs, params) for direct column assignment
    
    freq_arr = np.asarray(frfr)

    for i, p in enumerate(tqdm(space, desc=f"Sweeping {comp}.{sweep}")):
        fsweep(p)
        if isTF:
            tf = _loop.Gf(f=freq_arr)
            ugf_tmp, margin_tmp = get_margin(tf, freq_arr, deg=True)
        else:
            _, ugf_tmp, margin_tmp = _loop.Gc.bode(2 * np.pi * freq_arr)
        
        asd_tmp, _, _, rms_tmp = _loop.noise_propagation_asd(
            freq_arr, noise, _from=tf_from, _to=to_to, isTF=isTF, view=False
        )
        
        ugf[i] = ugf_tmp
        margin[i] = margin_tmp
        rms[i] = rms_tmp
        asd[:, i] = asd_tmp

    return ugf, margin, rms, asd


def loop_crossover_optimizer(
    loop1: LOOP,
    loop2: LOOP,
    frfr: Sequence,
    desired_f_cross: float,
    meta: Dict[str, Tuple],
    method: str = "Nelder-Mead",
    options: Dict = {"maxiter": 1000},
) -> Tuple[OptimizeResult, LOOP]:
    """
    Optimizes `loop1` parameters to match the crossover frequency of `loop2`.

    Parameters
    ----------
    loop1 : LOOP
        The loop to optimize. A deep copy is used, so the original is not modified.
    loop2 : LOOP
        The reference loop for crossover comparison.
    frfr : array_like
        Frequency grid (Hz) for evaluation.
    desired_f_cross : float
        Target crossover frequency (Hz).
    meta : dict
        Parameter specification: {name: (min, max, initial_guess, is_int), ...}
    method : str, optional
        Solver for `scipy.optimize.minimize`. Defaults to "Nelder-Mead".
    options : dict, optional
        Solver options. Defaults to {"maxiter": 1000}.

    Returns
    -------
    Tuple[scipy.optimize.OptimizeResult, LOOP]
        A tuple containing the optimization result object and the optimized loop copy.
    """
    _loop_1 = copy.deepcopy(loop1)
    freq_arr = np.asarray(frfr)

    def cost_function(x: np.ndarray, desired: float, meta_spec: dict,
                      loop_to_opt: LOOP, ref_loop: LOOP) -> float:
        for i, (attr, spec) in enumerate(meta_spec.items()):
            is_integer = spec[3]
            value = int(x[i]) if is_integer else x[i]
            setattr(loop_to_opt, attr, value)
        
        f_cross = loop_crossover(loop_to_opt, ref_loop, freq_arr)
        return (f_cross - desired)**2

    x0 = [m[2] for m in meta.values()]
    bounds = [(m[0], m[1]) for m in meta.values()]

    logger.info("# ===== Optimization start ==========")
    logger.info(f"    x0 = {x0}")
    logger.info(f"    bounds = {bounds}")
  
    popt = minimize(
        fun=cost_function, 
        x0=x0, 
        bounds=bounds, 
        args=(desired_f_cross, meta, _loop_1, loop2), 
        method=method, 
        options=options
    )
    
    # Apply final optimized parameters to the loop copy
    for i, (attr, spec) in enumerate(meta.items()):
        is_integer = spec[3]
        value = int(popt.x[i]) if is_integer else popt.x[i]
        setattr(_loop_1, attr, value)

    final_f_cross = loop_crossover(_loop_1, loop2, freq_arr)

    # Log results for user feedback
    logger.info("# ===== Optimization result ==========")
    for i, attr in enumerate(meta):
        is_integer = meta[attr][3]
        value = int(popt.x[i]) if is_integer else popt.x[i]
        logger.info(f"    Optimized {attr} = {value}")
    logger.info(f"    Achieved f_cross (Hz) = {final_f_cross:.4g}")
    logger.info(f"    Success = {popt.success}")
    logger.info(f"    Message = {popt.message}")
    
    return popt, _loop_1