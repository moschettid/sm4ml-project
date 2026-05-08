"""
Hedge and AdaHedge algorithms and helper functions.
"""

from __future__ import annotations

import numpy as np


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


class Hedge:
    """Hedge with a fixed learning rate eta."""

    def __init__(self, K: int, eta: float):
        if eta <= 0:
            raise ValueError("eta must be positive")
        self.K = K
        self.eta = float(eta)
        self.L = np.zeros(K)            # cumulative loss per expert
        self.cumulative_loss = 0.0      # cumulative learner loss

    def predict(self) -> np.ndarray:
        """
        Current distribution over experts, computed in log-domain for stability."""
        x = -self.eta * self.L
        x -= x.max()
        w = np.exp(x)
        return w / w.sum()

    def step(self, loss_vec: np.ndarray) -> float:
        """Play one round, observe losses, return expected loss h_t."""
        p = self.predict()
        h_t = float(np.dot(p, loss_vec))
        self.L += loss_vec
        self.cumulative_loss += h_t
        return h_t


class AdaHedge:
    """AdaHedge: data-dependent learning rate via the cumulative mixability gap.
    """

    def __init__(self, K: int):
        self.K = K
        self.L = np.zeros(K)
        self.Delta = 0.0
        self.cumulative_loss = 0.0

    def predict(self) -> np.ndarray:
        if self.Delta <= 0.0:
            leaders = (self.L == self.L.min())
            return leaders.astype(float) / leaders.sum()
        eta = np.log(self.K) / self.Delta
        x = -eta * self.L
        x -= x.max()
        w = np.exp(x)
        return w / w.sum()

    def step(self, loss_vec: np.ndarray) -> float:
        if self.Delta <= 0.0:
            leaders = (self.L == self.L.min())
            p = leaders.astype(float) / leaders.sum()
            h_t = float(np.dot(p, loss_vec))
            # Mix loss is the limit as eta -> infinity: the leader's loss.
            m_t = float(loss_vec[leaders].min())
        else:
            eta = np.log(self.K) / self.Delta
            log_w = -eta * self.L
            log_Z_old = _logsumexp(log_w)
            p = np.exp(log_w - log_Z_old)
            h_t = float(np.dot(p, loss_vec))
            log_Z_new = _logsumexp(log_w - eta * loss_vec)
            m_t = -(log_Z_new - log_Z_old) / eta

        # Clip to [0, inf) to avoid possible negative values due to floating-point precision issues.
        delta = max(h_t - m_t, 0.0)
        self.Delta += delta
        self.L += loss_vec
        self.cumulative_loss += h_t
        return h_t


def hedge_eta_theory(T: int, K: int) -> float:
    """Horizon-aware optimal learning rate for Hedge."""
    return float(np.sqrt(8.0 * np.log(K) / T))
