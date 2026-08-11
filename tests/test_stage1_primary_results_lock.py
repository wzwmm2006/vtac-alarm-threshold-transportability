from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]

def fmt(x): return format(float(x), ".4f")

def test_v2_rendered_values_follow_canonical_rounding():
    c = json.loads((ROOT / "00_protocol/frozen/STAGE1_PRIMARY_RESULTS_CANONICAL_v2.json").read_text(encoding="utf-8"))
    y = (ROOT / "00_protocol/frozen/STAGE1_PRIMARY_RESULTS_LOCK_v2.yaml").read_text(encoding="utf-8")
    m = (ROOT / "00_protocol/frozen/STAGE1_PRIMARY_RESULTS_LOCK_v2.md").read_text(encoding="utf-8")
    for cohort in c["cohorts"]:
        assert fmt(cohort["sensitivity_ci"][0]) in y
        assert fmt(cohort["sensitivity_ci"][0]) in m
        assert fmt(cohort["sensitivity_ci"][1]) in y
        assert fmt(cohort["sensitivity_ci"][1]) in m
    assert fmt(0.90625) == "0.9062"
