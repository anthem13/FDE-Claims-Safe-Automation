# 1 · Claims agent + safe automation threshold

> **Question:** How much of a claims workflow can you automate before the error rate becomes unacceptable?
>
> `risk = P(error | confidence >= tau)` · **find max automation subject to false approval < 1%**

## What it does

```
claim docs ──► OCR ──► extraction ──► business rules ──► action + raw confidence
(form/scan/email)        (fields + per-field conf)        approve | request_evidence | route_reviewer
                                                                   │
                       held-out calibration split ──► calibrator (isotonic / logistic, per action)
                                                                   │
                                     sweep tau 0→1 ──► automation vs false approvals / false denials
                                                                   │
                                     ship tau = max automation s.t. FA 95% upper bound < limit
```

1. **Ingest.** `claims/data.py` generates claims through three channels: clean forms, OCR'd scans with character confusions and dropped letters, and free-text emails. The documents carry the traps real inboxes do: negated evidence ("I'll send the police report later"), deductibles mentioned beside the claimed amount, "$4.3k" / "4,250 dollars" amounts, name variants, and policy-number typos. To use real documents, drop `.txt/.png/.jpg/.pdf` files into a folder and load them with `data.load_folder()`.
2. **OCR and extraction** (`claims/extract.py`):
   - `HeuristicExtractor`: fuzzy label matching, OCR repair (`O→0`, `l→1`), policy-DB cross-checks, and negation cues. Each field gets a raw confidence.
   - `ClaudeExtractor`: one multimodal structured-output call (images/PDFs go in natively) that returns fields plus per-field confidence. If Tesseract is installed, it's used for image OCR.
3. **Decide** (`claims/decide.py`). Coverage window, covered peril, per-claim limit, early-claim fraud signal, and required evidence produce one of the three actions.
4. **Calibrate** (`claims/calibrate.py`). The raw confidence isn't trusted. Calibrators (PAV isotonic, and logistic over per-field confidences plus OCR-garble, channel, negation, and round-amount signals) are fitted **per action** on the calibration half. Model choice uses 2-fold CV inside that half, and ECE, Brier, and reliability are reported on the test half.
5. **Sweep and pick a threshold** (`claims/threshold.py`). Below tau, or when the model says `route_reviewer`, a human decides. Above tau the action runs automatically, and each automated action is scored:
   - **False approval:** auto-approved when the truth isn't approve.
   - **False denial:** evidence auto-requested from a claim that should have been approved.
   - **Missed review:** evidence auto-requested from a claim that a reviewer should have handled.

   The shipped tau maximises automation subject to the **one-sided 95% Wilson upper bound** of the false-approval rate staying under the limit, so the result holds up against sampling noise.

## Results (offline run, 2,000 claims, seed 7)

| | tau | automation | false approvals |
|---|---|---|---|
| Uncalibrated score | 0.96 | 21.1% | ≤1% (UB) |
| **Calibrated (logistic, per action)** | **0.92** | **33.0%** | **0.2% (UB 0.6%)** |
| Point estimate instead of the upper bound | 0.85 | 41.1% | not shipped: the margin is within noise |

On the same claims and the same error limit, calibration raises automation from 21% to 33%. Raw-score ECE is 0.073; calibrated ECE is 0.013. See `output/report.md`, `output/threshold_sweep.svg`, `output/tradeoff.svg`, and `output/reliability.svg`.

## Run

```bash
python3 run.py                       # offline, ~3 s, no dependencies beyond Python 3.9 stdlib
python3 run.py --n 5000 --limit 0.005
python3 -m unittest discover -s tests
```

With Claude (uses `claude-opus-5-5` structured outputs, with server-side refusal fallback; a claim that is refused or errors goes to a human):

```bash
export ANTHROPIC_API_KEY=...
USE_CLAUDE=1 python3 run.py --n 200
```

## Files

| path | purpose |
|---|---|
| `claims/data.py` | synthetic corpus, policy DB, ground-truth rules, real-folder adapter |
| `claims/extract.py` | OCR, heuristic and Claude extractors |
| `claims/decide.py` | rules → action and features |
| `claims/calibrate.py` | isotonic, logistic, per-action, ECE/Brier/reliability |
| `claims/threshold.py` | tau sweep, Wilson bound, threshold selection |
| `output/` | report, charts, `sweep.csv`, per-claim `decisions.jsonl` |
