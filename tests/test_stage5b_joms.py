from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[3]
T=(ROOT/'07_manuscript/targets/JOMS/Manuscript_JOMS_v1.0.md').read_text(encoding='utf-8')
C=json.loads((ROOT/'00_protocol/frozen/STAGE1_PRIMARY_RESULTS_CANONICAL_v2.json').read_text(encoding='utf-8'))
def f(x): return format(float(x),'.4f')
def test_joms_manuscript_integrity():
 assert '0.9062-0.9897' in T and '0.9063-0.9897' not in T
 assert 'No score transformation, recalibration, threshold modification, or model selection' in T
 assert '[CITATION:' not in T
 for c in C['cohorts']: assert f"{c['tp']}/{c['fn']}/{c['tn']}/{c['fp']}" in T
