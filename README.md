# Adaptive Hedge — SM4ML project

Implementation and empirical comparison of:

- **Hedge** with several fixed learning rates,
- **AdaHedge** with a data-dependent learning rate,

on three synthetic loss regimes (stochastic, adversarial-shifting, low-gap).

## Layout

```
src/
  algorithms.py     # Hedge, AdaHedge
  losses.py         # synthetic loss generators
  experiments.py    # runner, regret computation, plotting helpers
notebooks/
  experiments.ipynb # runs all experiments and saves figures
```

## Reproducing the results

```bash
python3 -m pip install -r requirements.txt
jupyter notebook notebooks/experiments.ipynb
```
