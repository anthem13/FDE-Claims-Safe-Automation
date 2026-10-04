import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from claims.calibrate import IsotonicCalibrator, ece  # noqa: E402
from claims.data import Claim, Policy, generate, truth_decision  # noqa: E402
from claims.decide import decide  # noqa: E402
from claims.extract import HeuristicExtractor, _parse_amount, _parse_date  # noqa: E402
from claims.threshold import Row, best_threshold, sweep, wilson_upper  # noqa: E402

POL = {"POL-123456": Policy("POL-123456", "Maria Garcia", "2025-01-01", "2026-12-31", ["auto_collision", "theft"], 10000.0)}


class ExtractionTests(unittest.TestCase):
    def test_clean_form(self):
        text = ("PROPERTY & CASUALTY CLAIM FORM\nClaimant Name: Maria Garcia\nPolicy Number: POL-123456\n"
                "Date of Loss: 2026-03-14\nType of Loss: auto collision\nAmount Claimed: $4,250.00\n"
                "Description: rear-ended.\nAttachments: photos, police_report, repair_estimate")
        ex = HeuristicExtractor(POL).extract(Claim("c1", "form", text))
        self.assertEqual(ex.fields["policy_number"], "POL-123456")
        self.assertEqual(ex.fields["amount"], 4250.0)
        self.assertEqual(ex.fields["incident_date"], "2026-03-14")
        self.assertEqual(decide(ex, POL).action, "approve")

    def test_ocr_repair_policy_number(self):
        text = "Claimant Name: Maria Garcia\nPolicy Number: POL-l23456\nDate of Loss: 2026-03-14"
        ex = HeuristicExtractor(POL).extract(Claim("c2", "scan", text))
        self.assertEqual(ex.fields["policy_number"], "POL-123456")
        self.assertLess(ex.conf["policy_number"], 1.0)  # repaired values are not fully trusted

    def test_negated_evidence_is_missing(self):
        text = ("Subject: New claim\n\nMy name is Maria Garcia and my policy number is POL-123456. On 2026-03-14 my laptop "
                "was stolen from my car. The total cost came to $900.00. I've attached the receipt. "
                "I will send the police report later this week.")
        ex = HeuristicExtractor(POL).extract(Claim("c3", "email", text))
        self.assertNotIn("police_report", ex.fields["evidence"])
        self.assertEqual(decide(ex, POL).action, "request_evidence")

    def test_parsers(self):
        self.assertEqual(_parse_date("March 14, 2026")[0], "2026-03-14")
        self.assertEqual(_parse_date("O3/14/2O26")[0], "2026-03-14")
        self.assertEqual(_parse_amount("$1,234.50")[0], 1234.5)


class RuleTests(unittest.TestCase):
    def test_truth_rules(self):
        base = {"policy_number": "POL-123456", "incident_date": "2026-03-14", "incident_type": "theft",
                "amount": 500.0, "evidence": ["police_report", "receipt"]}
        self.assertEqual(truth_decision(base, POL), "approve")
        self.assertEqual(truth_decision({**base, "amount": 20000.0}, POL), "route_reviewer")
        self.assertEqual(truth_decision({**base, "evidence": ["receipt"]}, POL), "request_evidence")
        self.assertEqual(truth_decision({**base, "incident_date": "2025-01-05"}, POL), "route_reviewer")

    def test_generator_is_deterministic(self):
        a = generate(50, 1)[1]
        b = generate(50, 1)[1]
        self.assertEqual([c.text for c in a], [c.text for c in b])


class CalibrationAndThresholdTests(unittest.TestCase):
    def test_isotonic_is_monotone_and_calibrates(self):
        scores = [i / 100 for i in range(100)] * 5
        labels = [1 if (i % 100) > 60 else int((i % 7) == 0) for i in range(500)]
        iso = IsotonicCalibrator().fit(scores, labels)
        preds = [iso.predict(s) for s in sorted(set(scores))]
        self.assertEqual(preds, sorted(preds))
        self.assertLess(ece([iso.predict(s) for s in scores], labels), ece(scores, labels))

    def test_sweep_semantics(self):
        rows = [Row("approve", "approve", 0.99), Row("approve", "route_reviewer", 0.6),
                Row("request_evidence", "approve", 0.7), Row("route_reviewer", "approve", 0.99)]
        pts = {p.tau: p for p in sweep(rows, steps=11)}
        self.assertAlmostEqual(pts[0.0].automation_rate, 0.75)  # route_reviewer never counts as automated
        self.assertAlmostEqual(pts[0.0].false_approval_rate, 0.25)
        self.assertAlmostEqual(pts[0.0].false_denial_rate, 0.25)
        self.assertAlmostEqual(pts[0.8].automation_rate, 0.25)
        self.assertEqual(pts[0.8].false_approval_rate, 0.0)

    def test_best_threshold_respects_limit(self):
        rows = [Row("approve", "approve", 0.95)] * 900 + [Row("approve", "route_reviewer", 0.5)] * 100
        best = best_threshold(sweep(rows), 0.01)
        self.assertGreater(best.tau, 0.5)
        self.assertAlmostEqual(best.automation_rate, 0.9)

    def test_wilson(self):
        self.assertGreater(wilson_upper(0, 100), 0.0)
        self.assertLess(wilson_upper(10, 1000), 0.02)


if __name__ == "__main__":
    unittest.main()
