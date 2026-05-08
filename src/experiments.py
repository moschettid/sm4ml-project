"""Experiment helpers.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import matplotlib.pyplot as plt

from .algorithms import Hedge, AdaHedge, hedge_eta_theory


# Helper functions for the single run of an algorithm

def run(algo, losses: np.ndarray) -> np.ndarray:
    """Apply `algo` to a (T, K) loss matrix; return the per-round losses h_t."""
    T = losses.shape[0]
    h = np.empty(T)
    for t in range(T):
        h[t] = algo.step(losses[t])
    return h


def cumulative_regret(round_losses: np.ndarray, losses: np.ndarray) -> np.ndarray:
    """R_t = sum_{s<=t} h_s - min_k sum_{s<=t} l_{s,k}."""
    cum_learner = np.cumsum(round_losses)
    cum_experts = np.cumsum(losses, axis=0)
    best_in_hindsight = cum_experts.min(axis=1)
    return cum_learner - best_in_hindsight


# Aggregation of results across seeds

LossFn = Callable[[np.random.Generator], np.ndarray]
AlgoFactory = Callable[[], object]


def run_with_seeds(
    loss_fn: LossFn,
    algo_factories: Dict[str, AlgoFactory],
    n_seeds: int,
    base_seed: int = 0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run each algorithm on `n_seeds` independent loss sequences.

    Returns a dict name -> {"loss": (n_seeds, T), "regret": (n_seeds, T)}.
    All algorithms in a given seed see the *same* loss sequence (paired runs).
    """
    out = {name: {"loss": [], "regret": []} for name in algo_factories}
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        losses = loss_fn(rng)
        for name, factory in algo_factories.items():
            algo = factory()
            h = run(algo, losses)
            R = cumulative_regret(h, losses)
            out[name]["loss"].append(np.cumsum(h))
            out[name]["regret"].append(R)
    return {
        name: {k: np.array(v) for k, v in d.items()}
        for name, d in out.items()
    }


def best_expert_cumloss(loss_fn: LossFn, n_seeds: int, base_seed: int = 0) -> np.ndarray:
    """Average cumulative loss of the best expert in hindsight, per seed."""
    series = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        losses = loss_fn(rng)
        cum = np.cumsum(losses, axis=0)
        series.append(cum.min(axis=1))
    return np.array(series)


# Default pool of algorithms to run in the experiments

def default_algos(T: int, K: int) -> Dict[str, AlgoFactory]:
    """Hedge with several etas (theory-optimal and mistuned) plus AdaHedge.

    Names are kept short and stable so STYLES can key on them.
    """
    eta_star = hedge_eta_theory(T, K)
    return {
        "Hedge eta=0.001":  (lambda: Hedge(K, 0.001)),
        "Hedge eta=eta*/4": (lambda eta=eta_star / 4: Hedge(K, eta)),
        "Hedge eta=eta*":   (lambda eta=eta_star:     Hedge(K, eta)),
        "Hedge eta=4 eta*": (lambda eta=4 * eta_star: Hedge(K, eta)),
        "AdaHedge":         (lambda: AdaHedge(K)),
    }


# Plotting

STYLES: Dict[str, dict] = {
    "Hedge eta=0.001":  dict(color="#9ecae1", lw=1.3, linestyle="--", zorder=2),
    "Hedge eta=eta*/4": dict(color="#4292c6", lw=1.3, linestyle="-.", zorder=3),
    "Hedge eta=eta*":   dict(color="#08519c", lw=1.6, linestyle="-",  zorder=4),
    "Hedge eta=4 eta*": dict(color="#08306b", lw=1.3, linestyle=":",  zorder=3),
    "AdaHedge":         dict(color="#d62728", lw=2.2, linestyle="-",  zorder=6),
}


def _style(name: str) -> dict:
    return STYLES.get(name, dict(lw=1.4))


def _shade(ax, t, mean, std, name: str):
    style = _style(name)
    line, = ax.plot(t, mean, label=name, **style)
    ax.fill_between(t, mean - std, mean + std, alpha=0.18, color=line.get_color(), zorder=style.get("zorder", 1) - 1)


def plot_regret(results, title, ax=None, log_y=True):
    """Regret R_t over time. Uses symlog y so curves spanning ~1 to ~10^3 are all readable."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    T = next(iter(results.values()))["regret"].shape[1]
    t = np.arange(1, T + 1)
    for name, d in results.items():
        R = d["regret"]
        mean = R.mean(axis=0)
        std = R.std(axis=0) if R.shape[0] > 1 else np.zeros_like(mean)
        _shade(ax, t, mean, std, name)
    if log_y:
        ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("round $t$")
    ax.set_ylabel("regret $R_t$")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_avg_loss(results, best_expert, title, ax=None):
    """Average loss per round L_t / t. Bounded in [0,1]; the asymptote of each
    algorithm is its mean per-round loss, much easier to compare than raw L_t."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    T = next(iter(results.values()))["loss"].shape[1]
    t = np.arange(1, T + 1)
    for name, d in results.items():
        L = d["loss"] / t                        # broadcast per-round
        mean = L.mean(axis=0)
        std = L.std(axis=0) if L.shape[0] > 1 else np.zeros_like(mean)
        _shade(ax, t, mean, std, name)
    be_mean = (best_expert / t).mean(axis=0)
    ax.plot(t, be_mean, color="black", linestyle="--", lw=1.0,
            label="best expert (hindsight)", zorder=1)
    ax.set_xlabel("round $t$")
    ax.set_ylabel(r"average loss $L_t / t$")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_cumloss(results, best_expert, title, ax=None):
    """Cumulative learner loss against the cumulative loss of the best fixed
    expert in hindsight. Required for completeness, but visually less
    discriminative than `plot_avg_loss` and `plot_regret`: all curves are
    O(T), so well-tuned algorithms tend to overlap with the comparator."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    T = next(iter(results.values()))["loss"].shape[1]
    t = np.arange(1, T + 1)
    for name, d in results.items():
        L = d["loss"]
        mean = L.mean(axis=0)
        std = L.std(axis=0) if L.shape[0] > 1 else np.zeros_like(mean)
        _shade(ax, t, mean, std, name)
    be_mean = best_expert.mean(axis=0)
    ax.plot(t, be_mean, color="black", linestyle="--", lw=1.0,
            label="best expert (hindsight)", zorder=1)
    ax.set_xlabel("round $t$")
    ax.set_ylabel(r"cumulative loss $L_t$")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return ax
