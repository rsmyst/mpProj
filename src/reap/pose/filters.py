"""Temporal smoothing.

OneEuroFilter: working baseline now (low lag, cheap).
NEuroPredictor: interface stub for the N-euro paper work (Apr milestone) —
predicts near-future joints from short history instead of reactive filtering.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np


class OneEuroFilter:
    def __init__(self, freq: float = 30.0, mincutoff: float = 1.0,
                 beta: float = 0.007, dcutoff: float = 1.0) -> None:
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        te = 1.0 / freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x.copy()
        freq = 1.0 / max(1e-6, t - self._t_prev)
        self._t_prev = t
        a_d = self._alpha(self.dcutoff, freq)
        dx = (x - self._x_prev) * freq
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.mincutoff + self.beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) * freq)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev, self._dx_prev = x_hat.copy(), dx_hat.copy()
        return x_hat


class NEuroPredictor:
    """Placeholder — residual MLP predicting x(t+1) from history.

    Kept API-compatible with OneEuroFilter usage so the Apr
    algorithmic-iteration milestone can swap it in without
    touching the pipeline.
    """

    def __init__(self, hist: int = 8) -> None:
        self.buf: deque[np.ndarray] = deque(maxlen=hist)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        self.buf.append(x.copy())
        if len(self.buf) < 3:
            return x.copy()
        # Naive constant-velocity prediction as a stand-in
        v = self.buf[-1] - self.buf[-2]
        return np.clip(self.buf[-1] + 0.5 * v, 0, 1)
