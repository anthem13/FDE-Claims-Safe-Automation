#!/usr/bin/env python3
"""Claims agent + safe automation threshold: end-to-end experiment.

    python3 run.py                    # offline heuristic extractor, 2,000 synthetic claims
    python3 run.py --n 5000 --limit 0.005
    USE_CLAUDE=1 python3 run.py --n 200   # Claude multimodal extractor (needs API credentials)

Outputs land in ./output: report.md, threshold_sweep.svg, reliability.svg, sweep.csv, decisions.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from claims import calibrate as cal  # noqa: E402
from claims.data import generate, save  # noqa: E402
from claims.decide import decide  # noqa: E402
from claims.extract import ClaudeExtractor, HeuristicExtractor  # noqa: E402
from claims.llm import claude_enabled  # noqa: E402
from claims.svgplot import line_chart  # noqa: E402
from claims.threshold import Row, best_threshold, sweep  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--limit", type=float, default=0.01, help="max false-approval rate")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_dir = os.path.join(HERE, "output")
    os.makedirs(out_dir, exist_ok=True)
    policies, claims = generate(args.n, args.seed)
    save(policies, claims, os.path.join(out_dir, "dataset.json"))

    extractor = ClaudeExtractor(policies) if claude_enabled() else HeuristicExtractor(policies)
    print(f"[1/5] extracting {len(claims)} claims with {extractor.name} extractor")
    recs = []
    field_hits = collections.Counter()
    for cl in claims:
        try:
            ex = extractor.extract(cl)
            d = decide(ex, policies)
        except RuntimeError as e:  # refusal / API failure -> human
            recs.append({"id": cl.claim_id, "channel": cl.channel, "action": "route_reviewer", "reason": f"error:{e}",
                         "truth": cl.truth["decision"], "raw": 0.0, "features": [0.0] * 15, "correct": 0}); continue
        for k in ("claimant", "policy_number", "incident_date", "incident_type", "amount", "evidence"):
            gold = cl.truth[k]
            got = ex.fields.get(k)
            ok = (abs(float(got) - float(gold)) < 0.01) if k == "amount" and got is not None else got == gold
            field_hits[k] += int(bool(ok))
        recs.append({"id": cl.claim_id, "channel": cl.channel, "action": d.action, "reason": d.reason,
                     "truth": cl.truth["decision"], "raw": d.raw_conf, "features": d.features,
                     "correct": int(d.action == cl.truth["decision"]), "extracted": ex.fields})

    # Split: calibration fit on one half, every reported number from the other half.
    rng = random.Random(args.seed)
    rng.shuffle(recs)
    half = len(recs) // 2
    calib, test = recs[:half], recs[half:]
    auto_calib = [r for r in calib if r["action"] != "route_reviewer"]

    print("[2/5] calibrating confidence on held-out split")
    calibrators = {"raw": cal.Identity(),
                   "isotonic": cal.PerAction(lambda: cal.IsotonicCalibrator(), auto_calib),
                   "logistic": cal.PerAction(lambda: cal.LogisticCalibrator(), auto_calib)}

    auto_test = [r for r in test if r["action"] != "route_reviewer"]
    labels = [r["correct"] for r in auto_test]
    cal_stats = {}
    for name, c in calibrators.items():
        probs = [c.predict(r["raw"], r["features"], r["action"]) for r in auto_test]
        cal_stats[name] = {"ece": cal.ece(probs, labels), "brier": cal.brier(probs, labels),
                           "reliability": cal.reliability(probs, labels)}
    # Model selection uses the calibration split only (2-fold inside it) to avoid peeking at test.
    chosen = min(("isotonic", "logistic"), key=lambda n: _cv_brier(n, auto_calib))
    print(f"      ECE raw={cal_stats['raw']['ece']:.3f} isotonic={cal_stats['isotonic']['ece']:.3f} "
          f"logistic={cal_stats['logistic']['ece']:.3f} -> using {chosen}")

    print("[3/5] sweeping threshold tau 0 -> 1")
    sweeps = {}
    for name, c in calibrators.items():
        rows = [Row(r["action"], r["truth"], c.predict(r["raw"], r["features"], r["action"])) for r in test]
        sweeps[name] = sweep(rows)
    pts = sweeps[chosen]
    best = best_threshold(pts, args.limit, conservative=True)
    best_point = best_threshold(pts, args.limit, conservative=False)
    raw_best = best_threshold(sweeps["raw"], args.limit, conservative=True)

    with open(os.path.join(out_dir, "sweep.csv"), "w") as fh:
        fh.write("calibrator,tau,automation_rate,false_approval_rate,false_approval_upper95,false_denial_rate,missed_review_rate\n")
        for name, ps in sweeps.items():
            for p in ps:
                fh.write(f"{name},{p.tau},{p.automation_rate:.4f},{p.false_approval_rate:.4f},{p.fa_upper95:.4f},"
                         f"{p.false_denial_rate:.4f},{p.missed_review_rate:.4f}\n")
    with open(os.path.join(out_dir, "decisions.jsonl"), "w") as fh:
        for r in test:
            fh.write(json.dumps({k: v for k, v in r.items() if k != "features"}) + "\n")

    print("[4/5] plotting")
    vl = [(best.tau, f"ship tau={best.tau:.2f}")] if best else []
    with open(os.path.join(out_dir, "threshold_sweep.svg"), "w") as fh:
        fh.write(line_chart({
            "automation rate": [(p.tau, p.automation_rate) for p in pts],
            "false approvals": [(p.tau, p.false_approval_rate) for p in pts],
            "false approvals (95% UB)": [(p.tau, p.fa_upper95) for p in pts],
            "false denials": [(p.tau, p.false_denial_rate) for p in pts],
            "missed reviews": [(p.tau, p.missed_review_rate) for p in pts],
        }, f"Automation vs errors ({chosen}-calibrated confidence)", "threshold tau", "share of all claims",
            vlines=vl, hlines=[(args.limit, f"{args.limit:.1%} FA limit")], ymax=1.0, xmin=0, xmax=1))
    with open(os.path.join(out_dir, "tradeoff.svg"), "w") as fh:
        fh.write(line_chart({n: [(p.automation_rate, p.false_approval_rate) for p in s] for n, s in sweeps.items()},
                            "Automation / false-approval frontier by calibrator", "automation rate", "false approval rate",
                            hlines=[(args.limit, "limit")], xmin=0, xmax=1))
    with open(os.path.join(out_dir, "reliability.svg"), "w") as fh:
        fh.write(line_chart({n: [(a, b) for a, b, _ in s["reliability"]] for n, s in cal_stats.items()},
                            "Reliability diagram (test split)", "predicted P(correct)", "observed accuracy",
                            ymax=1.0, xmin=0, xmax=1, diagonal=True))

    print("[5/5] writing report")
    by_channel = collections.defaultdict(lambda: [0, 0])
    for r in test:
        by_channel[r["channel"]][0] += r["correct"]; by_channel[r["channel"]][1] += 1
    reasons = collections.Counter(r["reason"] for r in test)
    report = _report(args, extractor.name, len(claims), field_hits, cal_stats, chosen, best, best_point, raw_best, pts, by_channel, reasons, test)
    with open(os.path.join(out_dir, "report.md"), "w") as fh:
        fh.write(report)
    print("\n" + report.split("## Threshold sweep")[0])
    print(f"Full report: {os.path.join(out_dir, 'report.md')}")


def _cv_brier(name: str, rows) -> float:
    a, b = rows[::2], rows[1::2]
    total = 0.0
    for fit, ev in ((a, b), (b, a)):
        factory = cal.IsotonicCalibrator if name == "isotonic" else cal.LogisticCalibrator
        c = cal.PerAction(factory, fit)
        total += cal.brier([c.predict(r["raw"], r["features"], r["action"]) for r in ev], [r["correct"] for r in ev])
    return total


def _report(args, ext_name, n, field_hits, cal_stats, chosen, best, best_point, raw_best, pts, by_channel, reasons, test) -> str:
    def pct(x):
        return f"{x:.1%}"
    lines = [
        "# Claims agent: safe automation threshold", "",
        f"Extractor: **{ext_name}** · claims: **{n}** (half calibration, half test) · false-approval limit: **{pct(args.limit)}**", "",
        "## Answer", "",
    ]
    if best:
        lines += [
            f"**Ship tau = {best.tau:.2f}**: automate **{pct(best.automation_rate)}** of claims with a measured false-approval rate of "
            f"{pct(best.false_approval_rate)} (95% upper bound {pct(best.fa_upper95)}), false denials {pct(best.false_denial_rate)}, "
            f"missed reviews {pct(best.missed_review_rate)}.", "",
            (f"Using the point estimate instead of the upper bound would allow tau={best_point.tau:.2f} / {pct(best_point.automation_rate)} automation, "
             "but that margin is inside sampling noise, so it isn't shipped.") if best_point and best_point.tau != best.tau else "", "",
        ]
    else:
        lines += ["No threshold meets the false-approval limit, so every claim goes to a human.", ""]
    if raw_best:
        lines += [f"Thresholding the *uncalibrated* score instead lands at tau={raw_best.tau:.2f}: {pct(raw_best.automation_rate)} automation.", ""]
    lines += ["## Extraction accuracy (field exact-match, all claims)", "", "| field | accuracy |", "|---|---|"]
    lines += [f"| {k} | {pct(v / n)} |" for k, v in field_hits.items()]
    lines += ["", "## Decision accuracy by channel (test)", "", "| channel | accuracy | n |", "|---|---|---|"]
    lines += [f"| {k} | {pct(a / t)} | {t} |" for k, (a, t) in sorted(by_channel.items())]
    lines += ["", "## Calibration (test split, auto-actionable claims)", "", "| calibrator | ECE | Brier |", "|---|---|---|"]
    lines += [f"| {k}{' ← chosen' if k == chosen else ''} | {v['ece']:.3f} | {v['brier']:.3f} |" for k, v in cal_stats.items()]
    lines += ["", "![reliability](reliability.svg)", "",
              "## Threshold sweep", "", "![sweep](threshold_sweep.svg)", "", "![frontier](tradeoff.svg)", "",
              "| tau | automation | false approval | FA 95% UB | false denial | missed review |", "|---|---|---|---|---|---|"]
    for p in pts[::5]:
        lines.append(f"| {p.tau:.2f} | {pct(p.automation_rate)} | {pct(p.false_approval_rate)} | {pct(p.fa_upper95)} | "
                     f"{pct(p.false_denial_rate)} | {pct(p.missed_review_rate)} |")
    fa_examples = [r for r in test if r["action"] == "approve" and r["truth"] != "approve"][:5]
    lines += ["", "## Decision reasons (test)", "", "| reason | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in reasons.most_common()]
    lines += ["", "## Sample false approvals (what the threshold must screen out)", ""]
    for r in fa_examples:
        lines.append(f"- `{r['id']}` ({r['channel']}): truth={r['truth']}, raw conf={r['raw']:.2f}, extracted amount={r.get('extracted', {}).get('amount')}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
