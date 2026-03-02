#!/usr/bin/env python3
"""
Generate golden reference data for PLL regression tests.

Uses parameters from notebooks/0.1_pll-demo.ipynb.
Run this script when PLL behavior has been verified correct to update the golden file.

Usage:
    python tests/generate_pll_golden.py
"""

import json
from pathlib import Path

import numpy as np

from loopkit.loops import PLL
from loopkit.loopmath import get_margin

# Parameters from pll-demo notebook (real-world PLL config)
PARAMS = {
    "sps": 80e6,
    "Amp": 1e-5,
    "Cshift": 19,
    "Klf": 5,
    "Kp": 18,
    "Ki": 4,
    "twostages": True,
    "n_reg": 10,
}

# Frequency grid (smaller than notebook for compact golden file)
N_FREQ = 500
FRFR = np.logspace(np.log10(1e-6), np.log10(1e6), N_FREQ)


def main():
    golden_dir = Path(__file__).parent / "golden"
    golden_dir.mkdir(exist_ok=True)

    pll = PLL(
        sps=PARAMS["sps"],
        Amp=PARAMS["Amp"],
        Cshift=PARAMS["Cshift"],
        Klf=PARAMS["Klf"],
        Kp=PARAMS["Kp"],
        Ki=PARAMS["Ki"],
        twostages=PARAMS["twostages"],
        n_reg=PARAMS["n_reg"],
    )

    # Transfer functions
    Gf = pll.Gf(f=FRFR)
    Hf = pll.Hf(f=FRFR)
    Ef = pll.Ef(f=FRFR)

    # Stability margins (as in notebooks)
    ugf, phase_margin = get_margin(Gf, FRFR, deg=True)

    # Point-to-point transfer (PD -> LPF segment)
    tf_pd_to_lpf = pll.point_to_point_tf(FRFR, _from="PD", _to="LPF", suppression=False)

    # Point-to-point with suppression (PD to end, error suppression)
    tf_pd_suppressed = pll.point_to_point_tf(
        FRFR, _from="PD", _to=None, suppression=True
    )

    np.savez_compressed(
        golden_dir / "pll_golden.npz",
        frfr=FRFR,
        Gf=Gf,
        Hf=Hf,
        Ef=Ef,
        ugf=np.array(ugf),
        phase_margin=np.array(phase_margin),
        tf_pd_to_lpf=tf_pd_to_lpf,
        tf_pd_suppressed=tf_pd_suppressed,
    )

    with open(golden_dir / "pll_golden_params.json", "w") as f:
        json.dump(PARAMS, f, indent=2)

    print(f"Golden file written to {golden_dir}/")
    print(f"  pll_golden.npz: frfr, Gf, Hf, Ef, ugf, phase_margin, tf_pd_to_lpf, tf_pd_suppressed")
    print(f"  pll_golden_params.json: {PARAMS}")
    print(f"  ugf = {ugf:.6e} Hz, phase_margin = {phase_margin:.4f} deg")


if __name__ == "__main__":
    main()
