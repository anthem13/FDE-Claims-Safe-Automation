"""Turn an extraction into one of three actions plus a raw confidence score."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List

from .data import INCIDENTS, Policy
from .extract import Extraction

ACTIONS = ("approve", "request_evidence", "route_reviewer")


@dataclass
class Decision:
    action: str
    reason: str
    raw_conf: float
    features: List[float]


def decide(ex: Extraction, policies: Dict[str, Policy]) -> Decision:
    f, c = ex.fields, ex.conf
    pol = policies.get(str(f.get("policy_number")))
    needed = ["policy_number", "incident_date", "incident_type", "amount"]
    if any(f.get(k) in (None, "") for k in needed):
        action, reason = "route_reviewer", "unreadable_field"
    elif pol is None:
        action, reason = "route_reviewer", "unknown_policy"
    elif not (pol.start <= str(f["incident_date"]) <= pol.end):
        action, reason = "route_reviewer", "outside_coverage_window"
    elif f["incident_type"] not in pol.coverages:
        action, reason = "route_reviewer", "not_covered"
    elif float(f["amount"]) > min(pol.per_claim_limit, 25000):  # type: ignore[arg-type]
        action, reason = "route_reviewer", "over_limit"
    elif (dt.date.fromisoformat(str(f["incident_date"])) - dt.date.fromisoformat(pol.start)).days < 14:
        action, reason = "route_reviewer", "early_claim"
    elif any(r not in f["evidence"] for r in INCIDENTS[str(f["incident_type"])]):  # type: ignore[operator]
        action, reason = "request_evidence", "missing_evidence"
    else:
        action, reason = "approve", "all_checks_pass"

    # Raw confidence: weakest link among the fields that drove the decision.
    relevant = ["policy_number", "incident_date", "incident_type", "amount", "claimant"]
    if action != "route_reviewer":
        relevant.append("evidence")
    raw = min(c.get(k, 0.0) for k in relevant)
    # How close the amount sits to the limit matters: an OCR slip near the line flips the action.
    margin = 1.0
    if pol is not None and isinstance(f.get("amount"), (int, float)):
        lim = min(pol.per_claim_limit, 25000)
        margin = min(1.0, abs(lim - float(f["amount"])) / lim)  # type: ignore[arg-type]
    amt = f.get("amount")
    round_amt = float(isinstance(amt, (int, float)) and float(amt) % 50 == 0)
    email = ex.signals.get("email", 0.0)
    feats = [raw] + [c.get(k, 0.0) for k in ("claimant", "policy_number", "incident_date", "incident_type", "amount", "evidence")] + [
        ex.signals.get("garble", 0.0), ex.signals.get("negation", 0.0), ex.signals.get("scan", 0.0), email,
        margin, round_amt, email * round_amt, float(action == "approve")]
    return Decision(action, reason, round(raw, 4), feats)
