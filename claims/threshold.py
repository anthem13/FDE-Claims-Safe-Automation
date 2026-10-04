"""Sweep tau in [0, 1] and find the highest automation rate you would actually ship.

Semantics per claim at threshold tau:
  * model says route_reviewer          -> human handles it (not automated, no error)
  * calibrated confidence < tau        -> human handles it (not automated, no error)
  * otherwise the action executes automatically:
      approve when truth != approve         -> FALSE APPROVAL (money out the door)
      request_evidence when truth = approve -> FALSE DENIAL (good customer delayed)
      request_evidence when truth = route   -> MISSED REVIEW (a reviewer should have seen it)

Humans are assumed to decide correctly; that cost shows up as (1 - automation_rate).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class Row:
    action: str
    truth: str
    p_correct: float


@dataclass
class Point:
    tau: float
    automation_rate: float
    false_approval_rate: float
    false_denial_rate: float
    missed_review_rate: float
    fa_upper95: float
    n_auto: int


def wilson_upper(k: int, n: int, z: float = 1.645) -> float:
    """One-sided 95% upper bound on a rate, so the shipped tau survives sampling noise."""
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + rad) / denom


def sweep(rows: Sequence[Row], steps: int = 101) -> List[Point]:
    n = len(rows)
    out = []
    for s in range(steps):
        tau = s / (steps - 1)
        auto = [r for r in rows if r.action != "route_reviewer" and r.p_correct >= tau]
        fa = sum(1 for r in auto if r.action == "approve" and r.truth != "approve")
        fd = sum(1 for r in auto if r.action == "request_evidence" and r.truth == "approve")
        mr = sum(1 for r in auto if r.action == "request_evidence" and r.truth == "route_reviewer")
        out.append(Point(round(tau, 3), len(auto) / n, fa / n, fd / n, mr / n, wilson_upper(fa, n), len(auto)))
    return out


def best_threshold(points: Sequence[Point], max_false_approval: float = 0.01, conservative: bool = True) -> Optional[Point]:
    """Max automation subject to false-approval < limit (on the 95% upper bound when conservative)."""
    ok = [p for p in points if (p.fa_upper95 if conservative else p.false_approval_rate) < max_false_approval]
    if not ok:
        return None
    return max(ok, key=lambda p: (p.automation_rate, -p.tau))
