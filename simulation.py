import copy
import numpy as np
from scipy.optimize import minimize
import looptools.auxiliary as aux
import logging
logger = logging.getLogger(__name__)


def parameter_sweep_1d(frfr, noise, loop, comp, sweep, space, _from, _to, isTF=True):
    """ sweep the loop bandwidth with a particular parameter designated by the 'sweep' argument
        and compute a contribution of the noise to the loop error
    Args: 
        frfr:: array of fourier frequencies (Hz)
        noise: array of noise asd (rad/sqrt(Hz))
        loop: member of LOOP class
        comp: component that will be modified in the sweep (member of Component class)
        sweep: string indicating the component's parameter to be swept
        space: array of values of the parameter to be swept
        _from: string indicating a staring loop component for the TF calculation
        _to: string indicating a stopping loop component for the TF calculation
        isTF: use TF attribute to compute a transfer function
        isPLL: use PLL (if not DLL)
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
            ugf_tmp, margin_tmp = aux.get_margin(_loop.Gf(f=frfr), frfr, deg=True)
        else:
            _,ugf_tmp, margin_tmp = _loop.Gc.bode(2*np.pi*frfr)
        asd_tmp,_,_,rms_tmp = _loop.noise_propagation_asd(frfr, noise, _from=_from, _to=_to, isTF=isTF, view=False)

        ugf = np.append(ugf, ugf_tmp)
        margin = np.append(margin, margin_tmp)
        rms = np.append(rms, rms_tmp)
        asd.append(asd_tmp)

    return ugf, margin, rms, np.array(asd).T


def loop_crossover_optimizer(loop1, loop2, frfr, desired_f_cross, meta, method="Nelder-Mead", options={"maxiter": 1000}):
    
    def fun(x, desired, meta, loop_1, loop_2):
        for i, attr in enumerate(meta):
            if meta[attr][3]:
                setattr(loop_1, attr, int(x[i]))
            else:
                setattr(loop_1, attr, x[i])
        f_cross = aux.loop_crossover(loop_1, loop_2, frfr)
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

    f_cross = aux.loop_crossover(_loop_1, loop2, frfr)

    print(f"# ===== Optimization result ==========")
    for i, attr in enumerate(meta):
        if meta[attr][3]:
            print(f"	Result[{i}] = {int(popt.x)}")
        else:
            print(f"	Result[{i}] = {popt.x}")
    print(f"	fcross (Hz) = {f_cross:.4}")
    print(f"	Success = {popt.success}")
    print(f"	Message = {popt.message}")