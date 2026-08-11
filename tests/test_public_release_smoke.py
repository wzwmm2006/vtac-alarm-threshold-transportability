from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_public_release_smoke():
    forbidden = {".dat", ".hea", ".atr", ".mat", ".zip", ".pt", ".pth", ".env"}
    files = [p for p in ROOT.rglob("*") if p.is_file()]
    assert not [p for p in files if p.suffix.lower() in forbidden]
    text_files = [p for p in files if p.suffix.lower() in {".md", ".yml", ".yaml", ".json", ".csv", ".py", ".cff", ".txt"}]
    secret_pattern = re.compile(r"(?:BEGIN (?:RSA|OPENSSH)|api[_-]?key|ghp_[A-Za-z0-9_]+|sk-[A-Za-z0-9_-]+)", re.I)
    for path in text_files:
        if path.name == "PUBLIC_RELEASE_SECURITY_AUDIT.md":
            continue
        assert not secret_pattern.search(path.read_text(encoding="utf-8", errors="ignore")), path
    epoch = __import__("json").loads((ROOT / "configs/SELECTED_TRAINING_EPOCH_v1.json").read_text(encoding="utf-8"))
    assert epoch["selected_epoch"] == 31
    config = yaml.safe_load((ROOT / "configs/PRIMARY_FCN_CONFIG_LOCK_v3.yaml").read_text(encoding="utf-8"))
    assert config["input_window"] == "[alarm-2500, alarm)"
    thresholds = (ROOT / "configs/locked_thresholds.csv").read_text(encoding="utf-8")
    assert "tau_95,0.102635692" in thresholds
    lock = yaml.safe_load((ROOT / "manifests/STAGE1_PRIMARY_RESULTS_LOCK_v2.yaml").read_text(encoding="utf-8"))
    assert lock["cohorts"]["challenge2015_external"]["tau_95"]["sensitivity_ci"] == [0.9062, 0.9897]
    assert (ROOT / "src/evaluation/vtac_safety_thresholds.py").exists()
