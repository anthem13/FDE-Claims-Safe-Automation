"""OCR + field extraction.

Two extractors share one output shape (`Extraction`):
  * HeuristicExtractor - regex + fuzzy label matching + OCR repair. Deterministic, offline.
  * ClaudeExtractor    - multimodal structured-output call; handles scans/PDFs natively.

Each field carries a raw confidence. These raw numbers are NOT trusted downstream: the
calibration stage maps them to empirical correctness probabilities.
"""
from __future__ import annotations

import datetime as dt
import difflib
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .data import EVIDENCE_PROSE, INCIDENT_PROSE, INCIDENTS, Claim, Policy

FIELDS = ["claimant", "policy_number", "incident_date", "incident_type", "amount", "evidence"]
DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "s": "5", "B": "8"})
NEGATION_CUES = ("later", "will send", "don't have", "do not have", "missing", "haven't", "still waiting")
VOCAB = set(
    "property casualty claim form claimant name policy number date of loss type amount claimed description attachments "
    "subject new hello hi there to the claims team my and is on i know deductible total cost came ive attached "
    "will send later this week thanks a at in from".split()
)
for _v in list(EVIDENCE_PROSE.values()) + [k.replace("_", " ") for k in INCIDENTS] + [p for ps in INCIDENT_PROSE.values() for p in ps]:
    VOCAB.update(re.findall(r"[a-z]+", _v.lower()))


@dataclass
class Extraction:
    fields: Dict[str, object]
    conf: Dict[str, float]
    signals: Dict[str, float] = field(default_factory=dict)  # features for calibration
    source: str = "heuristic"


# ----------------------------------------------------------------------------- OCR

def ocr_text(claim: Claim) -> Tuple[str, str]:
    """Return (text, method). Real image/PDF scans go through Tesseract when installed."""
    if os.path.exists(claim.text) and os.path.splitext(claim.text)[1].lower() in (".png", ".jpg", ".jpeg"):
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore

            return pytesseract.image_to_string(Image.open(claim.text)), "tesseract"
        except ImportError:
            return "", "unavailable"  # Claude multimodal path handles these instead
    return claim.text, "passthrough"


def garble_score(text: str) -> float:
    """Share of alphabetic tokens that are not in the domain vocabulary: a cheap OCR-quality proxy."""
    toks = [t for t in re.findall(r"[A-Za-z]+", text.lower()) if len(t) > 2]
    if not toks:
        return 1.0
    known_names = 0
    unknown = 0
    for t in toks:
        if t in VOCAB:
            continue
        if t[0].isalpha() and len(t) <= 9 and t.isalpha():
            known_names += 1  # likely a proper noun; count half
        unknown += 1
    return round((unknown - 0.5 * known_names) / len(toks), 4)


# ----------------------------------------------------------------------------- heuristic

LABELS = {
    "claimant": ["claimant name"], "policy_number": ["policy number"], "incident_date": ["date of loss"],
    "incident_type": ["type of loss"], "amount": ["amount claimed"], "evidence": ["attachments"],
    "description": ["description"],
}


def _labelled_lines(text: str) -> Dict[str, Tuple[str, float]]:
    out: Dict[str, Tuple[str, float]] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        label = label.strip().lower()
        best, score = None, 0.0
        for key, names in LABELS.items():
            for n in names:
                s = difflib.SequenceMatcher(None, label, n).ratio()
                if s > score:
                    best, score = key, s
        if best and score >= 0.7 and best not in out:
            out[best] = (value.strip(), score)
    return out


def _parse_date(raw: str) -> Tuple[Optional[str], float]:
    raw = raw.strip().rstrip(".")
    candidates = [(raw, 1.0)]
    fixed = re.sub(r"[OolISsB](?=[\dOolISsB/\-, ])|(?<=[\d/\-])[OolISsB]", lambda m: m.group(0).translate(DIGIT_FIX), raw)
    if fixed != raw:
        candidates.append((fixed, 0.8))
    for cand, c in candidates:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%d %b %Y"):
            try:
                return dt.datetime.strptime(cand, fmt).date().isoformat(), c
            except ValueError:
                pass
    # last resort: fuzzy month name
    m = re.match(r"([A-Za-z]+) (\d{1,2}), (\d{4})", raw.translate(DIGIT_FIX) if not raw[:1].isalpha() else raw)
    if m:
        months = [dt.date(2000, i, 1).strftime("%B") for i in range(1, 13)]
        best = difflib.get_close_matches(m.group(1), months, n=1, cutoff=0.6)
        if best:
            try:
                d = dt.datetime.strptime(f"{best[0]} {m.group(2).translate(DIGIT_FIX)}, {m.group(3).translate(DIGIT_FIX)}", "%B %d, %Y")
                return d.date().isoformat(), 0.6
            except ValueError:
                pass
    return None, 0.0


def _parse_amount(raw: str) -> Tuple[Optional[float], float]:
    s = raw.strip().lstrip("$").rstrip(".")
    conf = 1.0
    t = s.translate(DIGIT_FIX)
    if t != s:
        conf = 0.85
    if re.fullmatch(r"\d{1,3}(,\d{3})*\.\d{2}", t):
        return float(t.replace(",", "")), conf
    if re.fullmatch(r"\d+\.\d{2}", t):
        return float(t), conf
    # OCR turned ',' into '.' -> "45.000.00"; the naive read keeps the first group.
    m = re.fullmatch(r"(\d{1,3})\.(\d{3})\.(\d{2})", t)
    if m:
        return float(m.group(1) + m.group(2) + "." + m.group(3)), 0.7
    m = re.search(r"\d[\d,]*(\.\d+)?", t)
    if m:
        try:
            return float(m.group(0).replace(",", "")), 0.5
        except ValueError:
            pass
    return None, 0.0


def _incident_type(text: str) -> Tuple[Optional[str], float]:
    t = text.lower().replace("_", " ")
    for k in INCIDENTS:
        if k.replace("_", " ") in t:
            return k, 1.0
    scores = []
    for k, prose in INCIDENT_PROSE.items():
        for p in prose + [k.replace("_", " ")]:
            scores.append((difflib.SequenceMatcher(None, t, p).ratio(), k))
            if p in t:
                return k, 0.9
    scores.sort(reverse=True)
    if scores and scores[0][0] > 0.5:
        return scores[0][1], round(scores[0][0], 2)
    return None, 0.0


def _evidence(text: str, structured: bool) -> Tuple[List[str], float, float]:
    """Returns (evidence list, confidence, negation_flag)."""
    found: List[str] = []
    negated = 0.0
    low = text.lower()
    sentences = re.split(r"(?<=[.!?])\s+", low)
    conf = 1.0
    for key, prose in EVIDENCE_PROSE.items():
        variants = {key, key.replace("_", " "), prose}
        hit_sentences = [s for s in sentences if any(v in s for v in variants)]
        if not hit_sentences and structured:
            # fuzzy match tokens of the Attachments line (OCR-damaged)
            parts = re.split(r"[,;]\s*", low)
            for p in parts:
                if max(difflib.SequenceMatcher(None, p.strip(), v).ratio() for v in variants) > 0.8:
                    hit_sentences = [p]; conf = min(conf, 0.75)
        if not hit_sentences:
            continue
        if all(any(c in s for c in NEGATION_CUES) for s in hit_sentences):
            negated = 1.0
            continue
        found.append(key)
    return sorted(found), conf, negated


class HeuristicExtractor:
    name = "heuristic"

    def __init__(self, policies: Dict[str, Policy]):
        self.policies = policies

    def extract(self, claim: Claim) -> Extraction:
        text, _ = ocr_text(claim)
        lab = _labelled_lines(text)
        f: Dict[str, object] = {}
        c: Dict[str, float] = {}
        structured = "claimant" in lab or "policy_number" in lab

        # policy number + OCR repair + DB cross-check
        m = re.search(r"P[O0o]L[-\s]?([0-9OolISsB]{6})", text, re.I)
        pnum, pc = None, 0.0
        if m:
            digits = m.group(1)
            pnum = "POL-" + digits.translate(DIGIT_FIX)
            pc = 1.0 if digits.isdigit() else 0.8
            if pnum not in self.policies:
                near = difflib.get_close_matches(pnum, self.policies.keys(), n=2, cutoff=0.83)
                if len(near) == 1:
                    pnum, pc = near[0], 0.55
                else:
                    pc = 0.6  # number reads cleanly but no such policy
        f["policy_number"], c["policy_number"] = pnum, pc

        # claimant
        if "claimant" in lab:
            name, c_name = lab["claimant"][0], lab["claimant"][1]
        else:
            mm = re.search(r"(?i:my name is) ([A-Z][\w.]*,? [A-Z]\w+)", text)
            name, c_name = (mm.group(1), 0.95) if mm else (None, 0.0)
        if name and "," in name:
            last, first = [p.strip() for p in name.split(",", 1)]
            name = f"{first} {last}"
        pol = self.policies.get(pnum or "")
        if name and pol:
            sim = difflib.SequenceMatcher(None, name.lower(), pol.holder.lower()).ratio()
            initial_match = re.fullmatch(r"([A-Z])\. (\w+)", name)
            if initial_match and pol.holder.split()[0][0] == initial_match.group(1) and pol.holder.split()[1] == initial_match.group(2):
                sim = 0.9
            c_name *= 1.0 if sim > 0.85 else (0.7 if sim > 0.6 else 0.3)
            if sim > 0.6:
                name = pol.holder
        f["claimant"], c["claimant"] = name, round(c_name, 3)

        # date
        raw_date = lab.get("incident_date", ("", 0))[0]
        if not raw_date:
            mm = re.search(r"\bOn ([A-Za-z0-9 ,/\-]+?)(?= [a-z])", text)
            raw_date = mm.group(1) if mm else ""
        f["incident_date"], c["incident_date"] = _parse_date(raw_date) if raw_date else (None, 0.0)

        # incident type
        it_src = lab.get("incident_type", (None, 0))[0] or text
        f["incident_type"], c["incident_type"] = _incident_type(it_src)
        if c["incident_type"] < 0.9 and "description" in lab:
            alt, ac = _incident_type(lab["description"][0])
            if ac > c["incident_type"]:
                f["incident_type"], c["incident_type"] = alt, ac

        # amount
        if "amount" in lab:
            f["amount"], c["amount"] = _parse_amount(lab["amount"][0])
        else:
            approx = re.search(r"\$(\d+(?:\.\d)?)k\b", text)
            amts = re.findall(r"\$([\d,]+(?:\.\d{2})?)(?![\d.]*k)", text)
            vals = [float(a.replace(",", "")) for a in amts if a.replace(",", "").replace(".", "").isdigit()]
            if approx:
                f["amount"], c["amount"] = float(approx.group(1)) * 1000, 0.7
            elif vals:
                tot = re.search(r"(total|cost|came to)[^$]*\$([\d,]+(?:\.\d{2})?)", text, re.I)
                f["amount"] = float(tot.group(2).replace(",", "")) if tot else max(vals)
                c["amount"] = 1.0 if (tot or len(vals) == 1) else 0.75
            else:
                f["amount"], c["amount"] = None, 0.0

        ev_src = lab["evidence"][0] if "evidence" in lab else text
        ev, evc, negated = _evidence(ev_src, structured)
        f["evidence"], c["evidence"] = ev, evc

        return Extraction(f, c, {"garble": garble_score(text), "negation": negated,
                                 "scan": float(claim.channel == "scan"), "email": float(claim.channel == "email")})


# ----------------------------------------------------------------------------- Claude

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "claimant": {"type": ["string", "null"]},
        "policy_number": {"type": ["string", "null"], "description": "Format POL-NNNNNN; repair OCR confusions"},
        "incident_date": {"type": ["string", "null"], "description": "ISO YYYY-MM-DD"},
        "incident_type": {"type": ["string", "null"], "enum": list(INCIDENTS) + [None]},
        "amount": {"type": ["number", "null"], "description": "Total claimed, not the deductible"},
        "evidence": {"type": "array", "items": {"type": "string", "enum": list(EVIDENCE_PROSE)}},
        "field_confidence": {
            "type": "object",
            "properties": {k: {"type": "number"} for k in FIELDS},
            "required": FIELDS, "additionalProperties": False,
        },
        "missing_or_promised_evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": FIELDS + ["field_confidence", "missing_or_promised_evidence"],
    "additionalProperties": False,
}

SYSTEM = (
    "You extract fields from insurance claim documents (forms, OCR'd scans, emails). "
    "Only list evidence as attached if the document says it is attached now; evidence the claimant "
    "promises to send later is missing. If a field is unreadable, return null rather than guessing. "
    "field_confidence is your probability (0-1) that each extracted value is exactly right."
)


class ClaudeExtractor:
    name = "claude"

    def __init__(self, policies: Dict[str, Policy]):
        self.policies = policies

    def extract(self, claim: Claim) -> Extraction:
        from .llm import call_json, pdf_or_image_block

        block = pdf_or_image_block(claim.text) if os.path.exists(claim.text) else None
        text, _ = ocr_text(claim)
        content = [block] if block else []
        content.append({"type": "text", "text": f"Channel: {claim.channel}\n\n{text}" if not block else "Extract the claim fields."})
        out = call_json(SYSTEM, content, EXTRACT_SCHEMA)
        conf = {k: float(out["field_confidence"].get(k, 0.0)) for k in FIELDS}
        fields = {k: out.get(k) for k in FIELDS}
        fields["evidence"] = sorted(set(fields["evidence"] or []))
        pnum = fields.get("policy_number")
        if pnum and pnum not in self.policies:
            conf["policy_number"] = min(conf["policy_number"], 0.6)
        return Extraction(fields, conf, {"garble": garble_score(text), "negation": float(bool(out["missing_or_promised_evidence"])),
                                         "scan": float(claim.channel == "scan"), "email": float(claim.channel == "email")},
                          source="claude")
