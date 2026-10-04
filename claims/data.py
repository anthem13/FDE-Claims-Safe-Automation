"""Synthetic insurance-claim corpus that mimics public claim-form layouts.

Three channels, matching what a claims inbox actually receives:
  * form  - structured claim form (clean text, like a fillable PDF export)
  * scan  - the same form after a bad scan + OCR (character confusions, dropped chars)
  * email - free-text email with attachments named in prose

Messiness is deliberate: negated evidence ("I'll send the police report later"),
deductibles mentioned before the claimed amount, name variants, and OCR noise in
policy numbers. Ground truth comes from the business rules in `truth_decision`.
"""
from __future__ import annotations

import datetime as dt
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

FIRST = ["James", "Maria", "Wei", "Aisha", "Carlos", "Priya", "John", "Elena", "Kwame", "Sofia",
         "Liam", "Yuki", "Omar", "Grace", "Mateo", "Hannah", "Ravi", "Chloe", "Ivan", "Fatima"]
LAST = ["Smith", "Garcia", "Chen", "Khan", "Lopez", "Patel", "Okafor", "Rossi", "Mensah", "Novak",
        "Brown", "Tanaka", "Haddad", "Kim", "Silva", "Muller", "Sharma", "Dubois", "Petrov", "Ali"]

INCIDENTS = {
    "auto_collision": ["police_report", "photos", "repair_estimate"],
    "water_damage": ["photos", "plumber_report"],
    "theft": ["police_report", "receipt"],
    "medical": ["medical_bill", "doctor_note"],
    "fire": ["fire_department_report", "photos", "inventory_list"],
}
INCIDENT_PROSE = {
    "auto_collision": ["rear-ended at a stop light", "hit a guard rail on the highway", "collided with another car in a parking lot"],
    "water_damage": ["a pipe burst under the kitchen sink", "the water heater leaked into the basement", "roof leak during the storm"],
    "theft": ["my laptop was stolen from my car", "the house was broken into", "my bike was taken from the garage"],
    "medical": ["I broke my wrist", "emergency room visit after a fall", "outpatient surgery on my knee"],
    "fire": ["a kitchen fire damaged the cabinets", "electrical fire in the garage", "smoke damage from a neighbour's fire"],
}
EVIDENCE_PROSE = {
    "police_report": "police report", "photos": "photos", "repair_estimate": "repair estimate",
    "plumber_report": "plumber's report", "receipt": "receipt", "medical_bill": "medical bill",
    "doctor_note": "doctor's note", "fire_department_report": "fire department report",
    "inventory_list": "inventory list",
}
OCR_CONFUSIONS = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5", "8": "B", "B": "8",
                  "rn": "m", "m": "rn", "e": "c", "a": "o", "_": " ", ",": "."}


@dataclass
class Policy:
    policy_number: str
    holder: str
    start: str
    end: str
    coverages: List[str]
    per_claim_limit: float


@dataclass
class Claim:
    claim_id: str
    channel: str
    text: str
    truth: Dict[str, object] = field(default_factory=dict)


def truth_decision(fields: Dict[str, object], policies: Dict[str, Policy]) -> str:
    """The business rules a human adjuster applies. This is the label, not the model."""
    pol = policies.get(str(fields.get("policy_number")))
    inc_date = str(fields.get("incident_date"))
    if pol is None or not (pol.start <= inc_date <= pol.end):
        return "route_reviewer"
    if fields["incident_type"] not in pol.coverages:
        return "route_reviewer"
    if float(fields["amount"]) > pol.per_claim_limit or float(fields["amount"]) > 25000:
        return "route_reviewer"
    start = dt.date.fromisoformat(pol.start)
    if (dt.date.fromisoformat(inc_date) - start).days < 14:
        return "route_reviewer"  # fraud signal: claim right after binding
    required = INCIDENTS[str(fields["incident_type"])]
    if any(r not in fields["evidence"] for r in required):  # type: ignore[operator]
        return "request_evidence"
    return "approve"


def _ocr_noise(text: str, rng: random.Random, rate: float) -> str:
    out = []
    i = 0
    while i < len(text):
        two = text[i:i + 2]
        if two in OCR_CONFUSIONS and rng.random() < rate:
            out.append(OCR_CONFUSIONS[two]); i += 2; continue
        ch = text[i]
        r = rng.random()
        if ch in OCR_CONFUSIONS and r < rate:
            out.append(OCR_CONFUSIONS[ch])
        elif ch.isalpha() and r < rate * 0.15:
            pass  # dropped character
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _fmt_date(d: dt.date, rng: random.Random) -> str:
    return rng.choice([d.isoformat(), d.strftime("%m/%d/%Y"), d.strftime("%B %d, %Y"), d.strftime("%d %b %Y")])


def generate(n: int = 2000, seed: int = 7):
    rng = random.Random(seed)
    policies: Dict[str, Policy] = {}
    claims: List[Claim] = []
    today = dt.date(2026, 9, 1)

    for i in range(n):
        holder = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        pnum = f"POL-{rng.randint(100000, 999999)}"
        start = today - dt.timedelta(days=rng.randint(30, 900))
        end = start + dt.timedelta(days=365 * rng.choice([1, 2, 3]))
        covs = rng.sample(list(INCIDENTS), k=rng.randint(2, 5))
        limit = rng.choice([5000, 10000, 20000, 50000])
        pol = Policy(pnum, holder, start.isoformat(), end.isoformat(), covs, float(limit))
        policies[pnum] = pol

        # Most claims are legit; the rest exercise each rule.
        roll = rng.random()
        inc_type = rng.choice(covs) if roll > 0.06 else rng.choice([k for k in INCIDENTS if k not in covs] or covs)
        if roll < 0.10:
            inc_date = start + dt.timedelta(days=rng.randint(0, 12))  # early-claim fraud signal
        elif roll < 0.14:
            inc_date = end + dt.timedelta(days=rng.randint(1, 60))  # outside coverage window
        else:
            inc_date = start + dt.timedelta(days=rng.randint(15, max(16, min((end - start).days, (today - start).days))))
        amount = round(rng.choice([rng.uniform(300, 4000), rng.uniform(4000, 15000), rng.uniform(15000, 60000)]), 2)
        required = INCIDENTS[inc_type]
        evidence = list(required) if rng.random() < 0.72 else rng.sample(required, k=len(required) - 1)
        if rng.random() < 0.3:
            evidence.append(rng.choice([e for e in EVIDENCE_PROSE if e not in evidence]))
        # Some claims reference a policy number that does not exist (typo by claimant)
        claimed_pnum = pnum if rng.random() > 0.03 else f"POL-{rng.randint(100000, 999999)}"

        fields = {
            "claimant": holder, "policy_number": claimed_pnum, "incident_date": inc_date.isoformat(),
            "incident_type": inc_type, "amount": amount, "evidence": sorted(evidence),
        }
        decision = truth_decision(fields, policies)
        channel = rng.choices(["form", "scan", "email"], weights=[0.35, 0.3, 0.35])[0]
        text = _render(channel, fields, rng)
        claims.append(Claim(f"CLM-{i:05d}", channel, text, {**fields, "decision": decision}))
    return policies, claims


def _render(channel: str, f: Dict[str, object], rng: random.Random) -> str:
    d = dt.date.fromisoformat(str(f["incident_date"]))
    name = str(f["claimant"])
    if rng.random() < 0.15:  # name variant: "Garcia, Maria" or initial
        first, last = name.split()
        name = rng.choice([f"{last}, {first}", f"{first[0]}. {last}"])
    missing = [e for e in INCIDENTS[str(f["incident_type"])] if e not in f["evidence"]]  # type: ignore[operator]

    if channel in ("form", "scan"):
        lines = [
            "PROPERTY & CASUALTY CLAIM FORM",
            f"Claimant Name: {name}",
            f"Policy Number: {f['policy_number']}",
            f"Date of Loss: {_fmt_date(d, rng)}",
            f"Type of Loss: {str(f['incident_type']).replace('_', ' ')}",
            f"Amount Claimed: ${float(f['amount']):,.2f}",
            f"Description: {rng.choice(INCIDENT_PROSE[str(f['incident_type'])])}.",
            "Attachments: " + ", ".join(sorted(f["evidence"])),  # type: ignore[arg-type]
        ]
        text = "\n".join(lines)
        if channel == "scan":
            text = _ocr_noise(text, rng, rate=rng.choice([0.01, 0.04, 0.08, 0.14]))
        return text

    ev_prose = [EVIDENCE_PROSE[e] for e in f["evidence"]]  # type: ignore[union-attr]
    parts = [rng.choice(["Hello,", "Hi there,", "To the claims team,"])]
    parts.append(f"My name is {name} and my policy number is {f['policy_number']}.")
    parts.append(f"On {_fmt_date(d, rng)} {rng.choice(INCIDENT_PROSE[str(f['incident_type'])])}.")
    if rng.random() < 0.35:
        parts.append(f"I know my deductible is ${rng.choice([250, 500, 1000]):,}.")
    amt = float(f["amount"])
    style = rng.random()
    if style < 0.6:
        parts.append(f"The total cost came to ${amt:,.2f}.")
    elif style < 0.75:
        parts.append(f"All in, it was about ${amt / 1000:.1f}k.")
    elif style < 0.9:
        parts.append(f"The bill was {amt:,.0f} dollars.")
    else:
        parts.append(f"I'm claiming ${amt:.0f} for the repairs.")
    parts.append("I've attached the " + ", ".join(ev_prose) + ".")
    if missing and rng.random() < 0.6:
        ev = EVIDENCE_PROSE[missing[0]]
        parts.append(rng.choice([  # negation traps; only some match the extractor's cue list
            f"I will send the {ev} later this week.",
            f"I'm still waiting on the {ev}.",
            f"The {ev} is pending from the other party.",
            f"I can get you the {ev} if you need it.",
            f"I don't have the {ev} yet.",
        ]))
    parts.append("Thanks,\n" + name)
    return "Subject: New claim\n\n" + " ".join(parts)


def save(policies, claims, path: str) -> None:
    with open(path, "w") as fh:
        json.dump({"policies": {k: asdict(v) for k, v in policies.items()},
                   "claims": [asdict(c) for c in claims]}, fh, indent=1)


def load(path: str):
    with open(path) as fh:
        raw = json.load(fh)
    policies = {k: Policy(**v) for k, v in raw["policies"].items()}
    claims = [Claim(**c) for c in raw["claims"]]
    return policies, claims


def load_folder(folder: str) -> List[Claim]:
    """Adapter for real documents: drop .txt/.png/.jpg/.pdf files into a folder."""
    import os

    out = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        ext = os.path.splitext(name)[1].lower()
        if ext == ".txt":
            with open(p) as fh:
                out.append(Claim(name, "email" if "subject:" in fh.read(200).lower() else "form", open(p).read()))
        elif ext in (".png", ".jpg", ".jpeg", ".pdf"):
            out.append(Claim(name, "scan", p))  # text field holds the file path; OCR stage reads it
    return out


def lookup(policies: Dict[str, Policy], pnum: Optional[str]) -> Optional[Policy]:
    return policies.get(pnum or "")
