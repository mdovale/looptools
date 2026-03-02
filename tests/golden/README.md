# PLL Golden Reference Data

Golden files for PLL regression tests. Parameters match `notebooks/0.1_pll-demo.ipynb`.

## Regenerating

After intentional PLL changes:

```bash
python tests/generate_pll_golden.py
```

Then commit the updated `pll_golden.npz` and `pll_golden_params.json`.

## Contents

- **pll_golden.npz**: `frfr`, `Gf`, `Hf`, `Ef`, `ugf`, `phase_margin`, `tf_pd_to_lpf`, `tf_pd_closed`
- **pll_golden_params.json**: PLL constructor parameters
