import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.extend([str(ROOT / "03_code" / "analysis"), str(ROOT / "03_code" / "preprocessing"), str(ROOT / "03_code" / "models"), str(ROOT / "03_code" / "audit")])
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import vtac_safety_thresholds as vtac


SCRIPT = ROOT / "03_code" / "analysis" / "vtac_safety_thresholds.py"


class SafetyThresholdTests(unittest.TestCase):
    def test_challenge_score_uses_percentage_scale(self):
        result = vtac.metrics([1, 1, 0, 0], [0.9, 0.1, 0.8, 0.2], 0.5)
        self.assertAlmostEqual(result["challenge_score"], 25.0)

    def test_two_stage_cli_and_manifest_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            validation = tmp / "validation.csv"
            evaluation = tmp / "evaluation.csv"
            model = tmp / "model.bin"
            lock_dir = tmp / "lock"
            result_dir = tmp / "results"
            model.write_bytes(b"frozen model")
            pd.DataFrame({
                "split": ["validation"] * 6,
                "record": [f"v{i}" for i in range(6)],
                "event": [f"v{i}" for i in range(6)],
                "y_true": [1, 1, 1, 0, 0, 0],
                "score": [0.9, 0.8, 0.3, 0.7, 0.2, 0.1],
            }).to_csv(validation, index=False)
            pd.DataFrame({
                "split": ["test"] * 4 + ["external"] * 4,
                "record": ["t1", "t2", "t3", "t4", "x1", "x2", "x3", "x4"],
                "event": [f"e{i}" for i in range(8)],
                "y_true": [1, 1, 0, 0, 1, 1, 0, 0],
                "score": [0.9, 0.4, 0.8, 0.1, 0.8, 0.2, 0.7, 0.1],
            }).to_csv(evaluation, index=False)

            lock_command = [
                sys.executable, str(SCRIPT), "lock-threshold", str(validation),
                "--model-artifact", str(model), "--outdir", str(lock_dir),
                "--bootstrap", "20", "--skip-vtac-count-check",
            ]
            subprocess.run(lock_command, check=True, capture_output=True, text=True)
            lock_manifest = json.loads(
                (lock_dir / "threshold_lock_manifest.json").read_text(encoding="ascii")
            )
            self.assertEqual(lock_manifest["schema"], "vtac-threshold-lock-v1")

            apply_command = [
                sys.executable, str(SCRIPT), "apply-frozen", str(evaluation),
                "--thresholds", str(lock_dir / "locked_thresholds.csv"),
                "--outdir", str(result_dir), "--bootstrap", "20",
                "--external-cluster-mode", "event", "--skip-vtac-count-check",
            ]
            subprocess.run(apply_command, check=True, capture_output=True, text=True)
            results = pd.read_csv(result_dir / "safety_utility_metrics.csv")
            self.assertEqual(set(results["split"]), {"test", "external"})
            self.assertTrue((results["challenge_score"] <= 100).all())
            application_manifest = json.loads(
                (result_dir / "frozen_application_manifest.json").read_text(encoding="ascii")
            )
            self.assertEqual(application_manifest["schema"], "vtac-frozen-application-v1")

            thresholds = lock_dir / "locked_thresholds.csv"
            thresholds.write_text(thresholds.read_text() + "\n", encoding="utf-8")
            failed = subprocess.run(apply_command, capture_output=True, text=True, check=False)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("SHA-256 does not match", failed.stderr)

    def test_lock_rejects_test_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            scores = tmp / "mixed.csv"
            model = tmp / "model.bin"
            model.write_bytes(b"model")
            pd.DataFrame({
                "split": ["validation", "test"],
                "record": ["v", "t"],
                "event": ["v", "t"],
                "y_true": [1, 0],
                "score": [0.9, 0.1],
            }).to_csv(scores, index=False)
            command = [
                sys.executable, str(SCRIPT), "lock-threshold", str(scores),
                "--model-artifact", str(model), "--skip-vtac-count-check",
            ]
            failed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Allowed splits", failed.stderr)


if __name__ == "__main__":
    unittest.main()
