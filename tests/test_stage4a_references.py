from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[3]
TEXT = (ROOT / "07_manuscript/current/Manuscript_v0.2_REFERENCES_VERIFIED.md").read_text(encoding="utf-8")
CANONICAL = json.loads((ROOT / "00_protocol/frozen/STAGE1_PRIMARY_RESULTS_CANONICAL_v2.json").read_text(encoding="utf-8"))

def f(x):
    return format(float(x), ".4f")

def test_v02_citations_and_numbers():
    assert "[CITATION:" not in TEXT
    references = re.findall(r"^([1-8])\. ", TEXT, re.M)
    assert references == [str(i) for i in range(1, 9)]
    cited = {int(x) for group in re.findall(r"\[([0-9,]+)\]", TEXT) for x in group.split(",")}
    assert cited == set(range(1, 9))
    for cohort in CANONICAL["cohorts"]:
        assert f"{cohort['tp']}/{cohort['fn']}/{cohort['tn']}/{cohort['fp']}" in TEXT
        assert f"{f(cohort['sensitivity_ci'][0])}-{f(cohort['sensitivity_ci'][1])}" in TEXT
    assert "0.9062-0.9897" in TEXT
    assert "0.9063-0.9897" not in TEXT
