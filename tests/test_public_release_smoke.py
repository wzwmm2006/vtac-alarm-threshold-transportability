from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1]
def test_public_release_smoke():
    files=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts]
    assert not any(p.suffix.lower() in {'.dat','.hea','.atr','.mat','.zip','.pt','.pth'} for p in files)
    assert not any(re.search(r'D:/|D:\\|老婆资料|BEGIN (RSA|OPENSSH)|api[_-]?key|password',p.read_text(errors='ignore'),re.I) for p in files if p.suffix.lower() in {'.py','.csv','.json','.yaml','.yml','.cff','.txt'} and p.name not in {'PUBLIC_RELEASE_SECURITY_AUDIT.md','test_public_release_smoke.py'})
    cfg=json.loads((ROOT/'configs/SELECTED_TRAINING_EPOCH_v1.json').read_text(encoding='utf-8'))
    assert cfg['selected_epoch']==31
    lock=(ROOT/'manifests/STAGE1_PRIMARY_RESULTS_LOCK_v2.yaml').read_text(encoding='utf-8')
    assert 'primary_threshold: 0.102635692' in lock
    assert 'sensitivity_ci: [0.9062, 0.9897]' in lock
    assert (ROOT/'src/evaluation/vtac_safety_thresholds.py').exists()
