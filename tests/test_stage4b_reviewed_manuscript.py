from pathlib import Path
import json, re
ROOT=Path(__file__).resolve().parents[3]
TEXT=(ROOT/'07_manuscript/current/Manuscript_v0.3_SCIENTIFIC_REVIEWED.md').read_text(encoding='utf-8')
C=json.loads((ROOT/'00_protocol/frozen/STAGE1_PRIMARY_RESULTS_CANONICAL_v2.json').read_text(encoding='utf-8'))
def f(x): return format(float(x),'.4f')
def test_v03_reviewed_manuscript_integrity():
    assert 'Alarm Safety' not in TEXT
    assert '[CITATION:' not in TEXT
    assert '0.9062-0.9897' in TEXT and '0.9063-0.9897' not in TEXT
    assert '226 waveform records' in TEXT
    assert 'not the Challenge test set' in TEXT
    assert 'not claim exact numerical reproduction' in TEXT
    assert not re.search(r'\b(was|were|is|are) (equivalent|noninferior)\b', TEXT, re.I)
    assert 'deployment-ready' not in TEXT
    for c in C['cohorts']:
        assert f"{c['tp']}/{c['fn']}/{c['tn']}/{c['fp']}" in TEXT
        assert f"{f(c['sensitivity_ci'][0])}-{f(c['sensitivity_ci'][1])}" in TEXT
