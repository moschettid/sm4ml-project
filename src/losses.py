"""Synthetic loss-sequence generators for the three regimes studied:

* stochastic    — one good expert, the rest worse, all i.i.d. Bernoulli;
* adversarial   — best expert shifts in fixed-length segments (deterministic);
* low_gap       — one expert is only marginally better than the rest.

All generators return an array of shape (T, K) with values in {0, 1} or [0, 1].
"""

from __future__ import annotations

import numpy as np


def stochastic_losses(
    T: int,
    K: int,
    good_p: float = 0.3,
    bad_p: float = 0.5,
    good_idx: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Expert `good_idx` is Bernoulli(good_p); the others Bernoulli(bad_p)."""
    rng = rng if rng is not None else np.random.default_rng()
    probs = np.full(K, bad_p, dtype=float)
    probs[good_idx] = good_p
    return (rng.random((T, K)) < probs).astype(float)


def adversarial_shifting(
    T: int,
    K: int,
    n_segments: int = 5,
    low: float = 0.1,
    high: float = 0.9,
    rng: np.random.Generator | None = None,  # actually it is unused. but it is kept for a uniform API and potential future use
) -> np.ndarray:
    """Best expert changes every T/n_segments rounds.

    In each segment the current best expert suffers loss `low`, all others
    suffer loss `high`. Deterministic.
    """
    losses = np.full((T, K), high, dtype=float)
    seg_len = T // n_segments
    for i in range(n_segments):
        start = i * seg_len
        end = (i + 1) * seg_len if i < n_segments - 1 else T
        best = i % K
        losses[start:end, best] = low
    return losses


def low_gap_losses(
    T: int,
    K: int,
    best_p: float = 0.49,
    others_p: float = 0.5,
    best_idx: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Hard stochastic regime: best expert wins by a tiny margin."""
    rng = rng if rng is not None else np.random.default_rng()
    probs = np.full(K, others_p, dtype=float)
    probs[best_idx] = best_p
    return (rng.random((T, K)) < probs).astype(float)
