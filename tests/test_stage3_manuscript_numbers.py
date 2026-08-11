from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[3]
TEXT=(ROOT/"07_manuscript/current/Manuscript_v0.1.md").read_text(encoding="utf-8")
C=json.loads((ROOT/"00_protocol/frozen/STAGE1_PRIMARY_RESULTS_CANONICAL_v2.json").read_text(encoding="utf-8"))
def f(x): return format(float(x), ".4f")
def test_manuscript_primary_numbers_match_v2():
    for c in C["cohorts"]:
        for key in ("auroc", "sensitivity", "fasr"):
            assert f(c[key]) in TEXT
        for key in ("auroc_ci", "sensitivity_ci", "fasr_ci"):
            assert f"{f(c[key][0])}-{f(c[key][1])}" in TEXT
        assert f"{c['tp']}/{c['fn']}/{c['tn']}/{c['fp']}" in TEXT
    assert "0.9062-0.9897" in TEXT
    assert "0.9063-0.9897" not in TEXT