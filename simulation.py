import copy
import numpy as np
from itertools import product
from scipy.optimize import minimize
from looptools.loopmath import *
import logging
logger = logging.getLogger(__name__)


def new_parameter_sweep_1d(_loop, prop_name, values, frequencies, deg=True,
                       unwrap_phase=False, interpolate=False):
    """
    Sweep a single tunable property of a LOOP object and compute stability metrics.

    Parameters
    ----------
    _loop : LOOP
        The control loop object to modify and analyze. It is modified in-place.
    prop_name : str
        Name of the property to sweep (must be in `loop.property_list`).
    values : array_like
        Values to assign to the specified property for each sweep iteration.
    frequencies : array_like
        Fourier frequency array (Hz) over which to evaluate the loop transfer functions.
    deg : bool, optional
        If True, report phase margin in degrees (default). If False, use radians.
    unwrap_phase : bool, optional
        If True, unwrap phase before computing phase margin.
    interpolate : bool, optional
        If True, interpolate TFs to refine unity gain crossing.

    Returns
    -------
    result : dict
        Dictionary with the following keys:
            - 'parameter_name' : str, the swept property name
            - 'parameter_values' : ndarray, sweep values
            - 'frequencies' : ndarray, Fourier frequencies
            - 'metrics' : dict, with:
                - 'ugf' : ndarray of unity gain frequencies
                - 'phase_margin' : ndarray of phase margins
            - 'open_loop' : dict, with:
                - 'magnitude' : ndarray of shape (N, F)
                - 'phase' : ndarray of shape (N, F)

    Raises
    ------
    ValueError
        If the property name is not found in loop.property_list
    """
    loop = copy.deepcopy(_loop)

    values = np.asarray(values)
    frfr = np.asarray(frequencies)
    N, F = len(values), len(frfr)

    if prop_name not in loop.property_list:
        raise ValueError(f"Property '{prop_name}' not found in loop.property_list")

    mag_array = np.empty((N, F))
    phase_array = np.empty((N, F))
    ugf_array = np.empty(N)
    pm_array = np.empty(N)

    for i, val in enumerate(values):
        setattr(loop, prop_name, val)
        tf = loop.Gf(frfr)
        mag = np.abs(tf)
        phase = np.angle(tf, deg=False)

        if unwrap_phase:
            phase = np.unwrap(phase)
        if deg:
            phase_deg = np.rad2deg(phase)
            phase_use = phase_deg
        else:
            phase_use = phase

        mag_array[i, :] = mag
        phase_array[i, :] = phase_use

        ugf, pm = get_margin(tf, frfr, deg=deg, unwrap_phase=unwrap_phase, interpolate=interpolate)
        ugf_array[i] = ugf
        pm_array[i] = pm

    return {
        "parameter_name": prop_name,
        "parameter_values": values,
        "frequencies": frfr,
        "metrics": {
            "ugf": ugf_array,
            "phase_margin": pm_array,
        },
        "open_loop": {
            "magnitude": mag_array,
            "phase": phase_array,
        },
    }

def parameter_sweep_nd(loop, param_grid, frequencies, deg=True,
                       unwrap_phase=False, interpolate=False):
    """
    Sweep multiple LOOP parameters over an N-dimensional grid and analyze stability.

    Parameters
    ----------
    loop : LOOP
        Loop object to modify and evaluate. Modified in-place.
    param_grid : dict of str -> array_like
        Dictionary mapping property names (must be in loop.property_list)
        to arrays of values to sweep.
    frequencies : array_like
        Frequencies (Hz) over which to evaluate the loop.
    deg : bool, optional
        Whether to compute phase margin in degrees.
    unwrap_phase : bool, optional
        Whether to unwrap phase before computing margin.
    interpolate : bool, optional
        Whether to interpolate TFs before computing margins.

    Returns
    -------
    result : dict
        Dictionary containing:
            - 'parameter_names' : list of parameter names
            - 'parameter_grid' : dict of broadcasted parameter arrays
            - 'frequencies' : ndarray of shape (F,)
            - 'metrics' : dict with:
                - 'ugf' : ndarray of shape (...,)
                - 'phase_margin' : ndarray of shape (...,)
            - 'open_loop' : dict with:
                - 'magnitude' : ndarray of shape (..., F)
                - 'phase' : ndarray of shape (..., F)
    """
    param_names = list(param_grid.keys())
    sweep_axes = [np.asarray(param_grid[k]) for k in param_names]
    mesh = np.meshgrid(*sweep_axes, indexing='ij')
    shape = mesh[0].shape
    frfr = np.asarray(frequencies)
    F = len(frfr)

    # Preallocate result arrays
    ugf_arr = np.empty(shape)
    pm_arr = np.empty(shape)
    mag_arr = np.empty(shape + (F,))
    phase_arr = np.empty(shape + (F,))

    it = np.ndindex(shape)
    for idx in it:
        for i, name in enumerate(param_names):
            val = mesh[i][idx]
            if name not in loop.property_list:
                raise ValueError(f"Property '{name}' not found in loop.property_list")
            setattr(loop, name, val)

        tf = loop.L(frfr)
        mag = np.abs(tf)
        phase_rad = np.angle(tf, deg=False)
        if unwrap_phase:
            phase_rad = np.unwrap(phase_rad)
        phase = np.rad2deg(phase_rad) if deg else phase_rad

        mag_arr[idx] = mag
        phase_arr[idx] = phase

        ugf, pm = get_margin(tf, frfr, deg=deg,
                             unwrap_phase=unwrap_phase,
                             interpolate=interpolate)
        ugf_arr[idx] = ugf
        pm_arr[idx] = pm

    return {
        "parameter_names": param_names,
        "parameter_grid": {name: mesh[i] for i, name in enumerate(param_names)},
        "frequencies": frfr,
        "metrics": {
            "ugf": ugf_arr,
            "phase_margin": pm_arr,
        },
        "open_loop": {
            "magnitude": mag_arr,
            "phase": phase_arr,
        }
    }


def parameter_sweep_1d(frfr, noise, loop, comp, sweep, space, _from, _to, isTF=True):
    """
    Sweep a component parameter and evaluate its impact on loop error and performance.

    This function performs a 1D parameter sweep over a specified component property
    in a control loop, analyzing how it affects the unity gain frequency (UGF),
    phase margin, and noise contribution to loop error. It returns key performance
    metrics as functions of the swept parameter.

    Parameters
    ----------
    frfr : array_like
        Array of Fourier frequencies (Hz) for transfer function and noise evaluation.
    noise : array_like
        Noise ASD (amplitude spectral density), assumed to be in rad/√Hz.
    loop : LOOP
        An instance of the LOOP class representing the full control loop configuration.
    comp : str
        Name of the component within the loop whose parameter will be swept.
    sweep : str
        Name of the parameter (property) within the component to sweep.
    space : array_like
        Array of values to sweep through for the chosen parameter.
    _from : str
        Name of the starting component for transfer function computation.
    _to : str
        Name of the ending component for transfer function computation.
    isTF : bool, optional
        If True, use the loop's `Gf(f)` method (frequency-domain transfer function).
        If False, use the loop's internal `Gc.bode()` computation. Default is True.

    Returns
    -------
    ugf : ndarray
        Array of unity gain frequencies (Hz) obtained for each parameter value.
    margin : ndarray
        Array of phase margins (in degrees) corresponding to each UGF.
    rms : ndarray
        Array of RMS noise contributions computed from the propagated ASD.
    asd : ndarray
        2D array of propagated noise ASDs (rad/√Hz), shape = (len(frfr), len(space)).

    Notes
    -----
    - The loop is deep-copied before modification to preserve the original.
    - The function assumes that the component's parameter is exposed via
      a `.properties` dictionary with (value, setter) pairs.
    - Use this function to evaluate loop robustness and sensitivity to component tuning,
      such as filter corner frequencies, actuator gains, or delay elements.

    Raises
    ------
    AssertionError
        If the component or property name does not exist in the loop.
        If `_from` or `_to` are not valid loop components.

    See Also
    --------
    looptools.noise_propagation_asd : Computes noise propagation in the loop.
    get_margin : Estimates phase margin from transfer function.
    LOOP.Gf : Transfer function evaluation between components.
    LOOP.Gc.bode : Frequency response evaluation if `isTF=False`.
    """
    ugf = np.array([])
    margin = np.array([])
    rms = np.array([])
    asd = []

    _loop = copy.deepcopy(loop)

    assert comp in _loop.components_dict, logger.error("Cannot find this component in the loop")
    assert sweep in _loop.components_dict[comp].properties, logger.error("Cannot find this component property")
    if _from is not None:
        assert _from in _loop.components_dict, logger.error(f"Cannot find the {_from} component in the loop")
    if _to is not None:
        assert _to in _loop.components_dict, logger.error(f"Cannot find the {_to} component in the loop")

    fsweep = _loop.components_dict[comp].properties[sweep][1]

    for p in space:
        fsweep(p)
        if isTF:
            ugf_tmp, margin_tmp = get_margin(_loop.Gf(f=frfr), frfr, deg=True)
        else:
            _,ugf_tmp, margin_tmp = _loop.Gc.bode(2*np.pi*frfr)
        asd_tmp,_,_,rms_tmp = _loop.noise_propagation_asd(frfr, noise, _from=_from, _to=_to, isTF=isTF, view=False)

        ugf = np.append(ugf, ugf_tmp)
        margin = np.append(margin, margin_tmp)
        rms = np.append(rms, rms_tmp)
        asd.append(asd_tmp)

    return ugf, margin, rms, np.array(asd).T


def loop_crossover_optimizer(loop1, loop2, frfr, desired_f_cross, meta, method="Nelder-Mead", options={"maxiter": 1000}):
    """
    Optimize parameters in `loop1` to match the crossover frequency of `loop2`.

    This function tunes selected attributes of `loop1` to minimize the squared error between
    its crossover frequency and a desired target, based on the difference with `loop2`.
    The optimization uses SciPy's `minimize` with configurable method and options.

    Parameters
    ----------
    loop1 : object
        Control loop object whose attributes will be optimized.
    loop2 : object
        Reference control loop to match crossover behavior against.
    frfr : array_like
        Frequency grid (Hz) over which crossover comparison is performed.
    desired_f_cross : float
        Desired crossover frequency (Hz) to match between the two loops.
    meta : dict
        Dictionary specifying the parameters to optimize. Each entry is:
            key : str
                Name of an attribute in `loop1`.
            value : tuple
                (min_val, max_val, initial_val, is_integer)
                - min_val : lower bound for parameter
                - max_val : upper bound for parameter
                - initial_val : initial guess
                - is_integer : bool, whether parameter should be cast to int
    method : str, optional
        Optimization method passed to `scipy.optimize.minimize`. Default is "Nelder-Mead".
    options : dict, optional
        Dictionary of solver options passed to `scipy.optimize.minimize`.

    Returns
    -------
    None
        The function prints the optimization results and updates a copy of `loop1` internally.
        No values are returned. Results are shown via `print()` for diagnostic purposes.

    Notes
    -----
    - Uses `copy.deepcopy(loop1)` to avoid modifying the original loop in-place.
    - Crossover frequency is computed using `loop_crossover()`, which finds the first
      frequency where the gain difference between loops changes sign.
    - Prints optimized parameter values, achieved crossover frequency, and solver status.
    - Attributes marked as `is_integer=True` in `meta` are cast to integers during optimization.

    See Also
    --------
    loop_crossover : Computes the crossover frequency between two loop gain profiles.
    scipy.optimize.minimize : Underlying optimizer used for tuning.
    """
    def fun(x, desired, meta, loop_1, loop_2):
        for i, attr in enumerate(meta):
            if meta[attr][3]:
                setattr(loop_1, attr, int(x[i]))
            else:
                setattr(loop_1, attr, x[i])
        f_cross = loop_crossover(loop_1, loop_2, frfr)
        return  np.abs(f_cross - desired)**2
    
    x0 = []
    bounds = []
    for atrr in meta:
        bounds.append((meta[atrr][0], meta[atrr][1]))
        x0.append(meta[atrr][2])

    print(f"# ===== Optimization start ==========")
    print(f"	x0 = {x0}")
    print(f"	bounds = {bounds}")

    _loop_1 = copy.deepcopy(loop1)
  
    popt = minimize(fun=fun, 
                    x0=x0, 
                    bounds=bounds, 
                    args=(desired_f_cross, meta, _loop_1, loop2), 
                    method=method, 
                    options=options)
    
    for i, attr in enumerate(meta):
        if meta[attr][3]:
            setattr(_loop_1, attr, int(popt.x[i]))
        else:
            setattr(_loop_1, attr, popt.x[i])

    f_cross = loop_crossover(_loop_1, loop2, frfr)

    print(f"# ===== Optimization result ==========")
    for i, attr in enumerate(meta):
        if meta[attr][3]:
            print(f"	Result[{i}] = {int(popt.x)}")
        else:
            print(f"	Result[{i}] = {popt.x}")
    print(f"	fcross (Hz) = {f_cross:.4}")
    print(f"	Success = {popt.success}")
    print(f"	Message = {popt.message}")