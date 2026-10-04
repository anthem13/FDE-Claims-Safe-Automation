"""Confidence calibration, stdlib-only.

* IsotonicCalibrator  - pool-adjacent-violators on the raw score (monotone, non-parametric)
* LogisticCalibrator  - logistic regression on the decision features (Platt-style, multivariate)

Both are fitted on a held-out calibration split and judged by expected calibration error
(ECE) and Brier score on the test split. Never on the data they were fit on.
"""
from __future__ import annotations

import bisect
import math
from typing import List, Sequence, Tuple


class IsotonicCalibrator:
    name = "isotonic"

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "IsotonicCalibrator":
        pairs = sorted(zip(scores, labels))
        blocks: List[List[float]] = []  # [sum_y, count, min_x, max_x]
        for x, y in pairs:
            blocks.append([float(y), 1.0, x, x])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
                s, n, lo, hi = blocks.pop()
                blocks[-1][0] += s; blocks[-1][1] += n; blocks[-1][3] = hi
        self.xs = [b[3] for b in blocks]
        # Laplace-smoothed block means so a pure block never claims certainty.
        self.ys = [(b[0] + 0.5) / (b[1] + 1.0) for b in blocks]
        return self

    def predict(self, score: float, _features=None, _action=None) -> float:
        i = bisect.bisect_left(self.xs, score)
        return self.ys[min(i, len(self.ys) - 1)]


class LogisticCalibrator:
    name = "logistic"

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int], epochs: int = 600, lr: float = 0.5, l2: float = 1e-3):
        d = len(X[0])
        self.w = [0.0] * d
        self.b = 0.0
        n = len(X)
        for _ in range(epochs):
            gw = [0.0] * d
            gb = 0.0
            for xi, yi in zip(X, y):
                p = _sigmoid(self.b + sum(w * v for w, v in zip(self.w, xi)))
                err = p - yi
                gb += err
                for j in range(d):
                    gw[j] += err * xi[j]
            self.b -= lr * gb / n
            for j in range(d):
                self.w[j] -= lr * (gw[j] / n + l2 * self.w[j])
        return self

    def predict(self, _score: float, features: Sequence[float], _action=None) -> float:
        return _sigmoid(self.b + sum(w * v for w, v in zip(self.w, features)))


class Identity:
    name = "raw"

    def predict(self, score: float, _features=None, _action=None) -> float:
        return score


class PerAction:
    """One calibrator per proposed action: a wrong approval and a wrong evidence request
    fail for different reasons, so they get separate confidence curves."""

    def __init__(self, factory, rows):
        self.name = factory().name
        self.models = {}
        for action in sorted({r["action"] for r in rows}):
            sub = [r for r in rows if r["action"] == action]
            m = factory()
            if isinstance(m, IsotonicCalibrator):
                m.fit([r["raw"] for r in sub], [r["correct"] for r in sub])
            else:
                m.fit([r["features"] for r in sub], [r["correct"] for r in sub])
            self.models[action] = m

    def predict(self, score: float, features=None, action=None) -> float:
        m = self.models.get(action)
        return m.predict(score, features) if m else 0.0


def _sigmoid(z: float) -> float:
    if z < -30:
        return 0.0
    if z > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def ece(probs: Sequence[float], labels: Sequence[int], bins: int = 10) -> float:
    total = len(probs)
    err = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(probs[i] for i in idx) / len(idx)
            acc = sum(labels[i] for i in idx) / len(idx)
            err += len(idx) / total * abs(conf - acc)
    return err


def brier(probs: Sequence[float], labels: Sequence[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def reliability(probs: Sequence[float], labels: Sequence[int], bins: int = 10) -> List[Tuple[float, float, int]]:
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if idx:
            out.append((sum(probs[i] for i in idx) / len(idx), sum(labels[i] for i in idx) / len(idx), len(idx)))
    return out
