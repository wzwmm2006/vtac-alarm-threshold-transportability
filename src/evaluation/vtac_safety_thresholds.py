#!/usr/bin/env python3
"""Two-stage locked VTaC safety-threshold analysis.

Run ``lock-threshold`` with validation scores only. It writes frozen numerical
thresholds and a hash manifest. Run ``apply-frozen`` later with test/external
scores and that threshold file. The application stage never reads validation
scores or selects a threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

TARGETS = (0.95, 0.975, 0.99)
VTAC_EXPECTED = {
    "validation": {"n": 495, "records": 226, "true_vt": 141},
    "test": {"n": 482, "records": 226, "true_vt": 137},
}


def _safe_div(a, b):
    return float(a / b) if b else float("nan")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y, score, threshold):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = score >= threshold
    tp = int(np.sum((y == 1) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    tn = int(np.sum((y == 0) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    n = len(y)
    out = {
        "n": n, "true_vt": int(np.sum(y == 1)), "false_vt": int(np.sum(y == 0)),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": _safe_div(tp, tp + fn),
        "fasr": _safe_div(tn, tn + fp),
        "specificity": _safe_div(tn, tn + fp),
        "ppv": _safe_div(tp, tp + fp),
        "npv": _safe_div(tn, tn + fn),
        "all_alarm_suppression": _safe_div(tn, n),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "challenge_score": 100.0 * _safe_div(tp + tn, tp + tn + fp + 5 * fn),
    }
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, score))
        out["auprc"] = float(average_precision_score(y, score))
    else:
        out["auroc"] = out["auprc"] = float("nan")
    return out


def candidate_thresholds(scores):
    unique = np.unique(np.asarray(scores, dtype=float))
    unique = unique[np.isfinite(unique)]
    if len(unique) == 0:
        raise ValueError("No finite validation scores")
    if len(unique) == 1:
        return np.array([-np.inf, np.inf], dtype=float)
    mids = unique[:-1] + (unique[1:] - unique[:-1]) / 2.0
    return np.concatenate(([-np.inf], mids, [np.inf]))


def select_threshold(validation, target):
    candidates = []
    for threshold in candidate_thresholds(validation["score"].to_numpy()):
        result = metrics(validation["y_true"], validation["score"], threshold)
        if result["sensitivity"] >= target:
            candidates.append((result["fasr"], result["sensitivity"],
                               -float(threshold), float(threshold), result))
    if not candidates:
        raise RuntimeError(f"No threshold satisfies sensitivity >= {target}")
    best = max(candidates, key=lambda row: (row[0], row[1], row[2]))
    return best[3], best[4]


def cluster_bootstrap(df, threshold, cluster_col="record", B=10000, seed=20260810):
    rng = np.random.default_rng(seed)
    groups = {key: group for key, group in df.groupby(cluster_col, sort=False)}
    keys = np.array(list(groups), dtype=object)
    metrics_to_interval = ["sensitivity", "fasr", "all_alarm_suppression", "ppv", "npv"]
    draws = {key: [] for key in metrics_to_interval}
    for _ in range(B):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        sample = pd.concat([groups[key] for key in sampled], ignore_index=True)
        result = metrics(sample["y_true"], sample["score"], threshold)
        for key in metrics_to_interval:
            draws[key].append(result[key])
    output = {}
    for key, values in draws.items():
        values = np.asarray(values, dtype=float)
        output[f"{key}_ci_low"] = float(np.nanquantile(values, 0.025))
        output[f"{key}_ci_high"] = float(np.nanquantile(values, 0.975))
        if key == "sensitivity":
            output["sensitivity_one_sided_95_lower"] = float(np.nanquantile(values, 0.05))
    return output


def load_scores(path, allowed_splits):
    df = pd.read_csv(path)
    required = {"split", "record", "event", "y_true", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.copy()
    df["split"] = df["split"].astype(str).str.lower().str.strip().replace({"val": "validation"})
    found = set(df["split"])
    if not found or not found.issubset(allowed_splits):
        raise ValueError(f"Allowed splits are {sorted(allowed_splits)}; found {sorted(found)}")
    if df[["record", "event", "y_true", "score"]].isna().any().any():
        raise ValueError("record, event, y_true, and score must not be missing")
    if (df["record"].astype(str).str.strip() == "").any() or (df["event"].astype(str).str.strip() == "").any():
        raise ValueError("record and event must not be blank")
    if not set(pd.unique(df["y_true"])).issubset({0, 1, False, True}):
        raise ValueError("y_true must be binary")
    if not np.isfinite(df["score"]).all():
        raise ValueError("score contains non-finite values")
    if ((df["score"] < 0) | (df["score"] > 1)).any():
        raise ValueError("score must be a probability in [0, 1]")
    if df.duplicated(["split", "event"]).any():
        raise ValueError("Duplicate split/event rows detected")
    return df


def check_vtac_counts(df, split):
    part = df[df["split"] == split]
    observed = {"n": len(part), "records": part["record"].nunique(), "true_vt": int(part["y_true"].sum())}
    if observed != VTAC_EXPECTED[split]:
        raise ValueError(f"{split} counts do not match frozen split: expected {VTAC_EXPECTED[split]}, observed {observed}")


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")


def metric_rows(df, thresholds, bootstrap, seed, external_cluster_mode=None):
    rows = []
    splits = [name for name in ("test", "external") if name in set(df["split"])]
    for threshold_row in thresholds.itertuples(index=False):
        target = float(threshold_row.target_sensitivity)
        threshold = float(threshold_row.threshold)
        for split in splits:
            part = df[df["split"] == split]
            cluster = "record" if split == "test" else external_cluster_mode
            result = metrics(part["y_true"], part["score"], threshold)
            row = {"target_sensitivity": target, "threshold": threshold, "split": split,
                   **result, "bootstrap_cluster": cluster}
            if bootstrap > 0:
                row.update(cluster_bootstrap(part, threshold, cluster, bootstrap,
                                             seed + int(round(target * 1000))))
            rows.append(row)
    return pd.DataFrame(rows)


def lock_threshold(args):
    df = load_scores(args.scores_csv, {"validation"})
    if set(df["split"]) != {"validation"}:
        raise ValueError("lock-threshold requires validation scores only")
    if not args.skip_vtac_count_check:
        check_vtac_counts(df, "validation")
    if not args.model_artifact:
        raise ValueError("At least one --model-artifact is required for the lock manifest")
    artifacts = [path.resolve() for path in args.model_artifact]
    if any(not path.is_file() for path in artifacts):
        raise ValueError("Every --model-artifact must be an existing file")

    args.outdir.mkdir(parents=True, exist_ok=True)
    threshold_rows, validation_rows = [], []
    for target in TARGETS:
        threshold, result = select_threshold(df, target)
        threshold_rows.append({"target_sensitivity": target, "threshold": threshold,
                               **{f"validation_{key}": value for key, value in result.items()}})
        validation_row = {"target_sensitivity": target, "threshold": threshold,
                          "split": "validation", **result, "bootstrap_cluster": "record"}
        if args.bootstrap > 0:
            validation_row.update(cluster_bootstrap(df, threshold, "record", args.bootstrap,
                                                    args.seed + int(round(target * 1000))))
        validation_rows.append(validation_row)

    thresholds_path = args.outdir / "locked_thresholds.csv"
    pd.DataFrame(threshold_rows).to_csv(thresholds_path, index=False)
    pd.DataFrame(validation_rows).to_csv(args.outdir / "validation_safety_utility_metrics.csv", index=False)
    manifest = {
        "schema": "vtac-threshold-lock-v1",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_scores": {"path": str(args.scores_csv.resolve()), "sha256": sha256_file(args.scores_csv)},
        "model_artifacts": [{"path": str(path), "sha256": sha256_file(path)} for path in artifacts],
        "thresholds_file": {"path": str(thresholds_path.resolve()), "sha256": sha256_file(thresholds_path)},
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
    }
    write_json(args.outdir / "threshold_lock_manifest.json", manifest)
    report = pd.DataFrame(threshold_rows)
    print(report[["target_sensitivity", "threshold", "validation_sensitivity", "validation_fasr"]].to_string(index=False))
    print(f"\nThreshold lock written to {args.outdir.resolve()}")


def load_frozen_thresholds(path):
    thresholds = pd.read_csv(path)
    required = {"target_sensitivity", "threshold"}
    missing = required - set(thresholds.columns)
    if missing:
        raise ValueError(f"Threshold file missing columns: {sorted(missing)}")
    found = set(thresholds["target_sensitivity"].astype(float))
    if thresholds["target_sensitivity"].duplicated().any() or found != set(TARGETS):
        raise ValueError(f"Threshold targets must be exactly {TARGETS}; found {sorted(found)}")
    if thresholds["threshold"].isna().any():
        raise ValueError("Threshold file contains missing thresholds")
    return thresholds


def verify_lock_manifest(thresholds_path, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if manifest.get("schema") != "vtac-threshold-lock-v1":
        raise ValueError("Unrecognized threshold lock manifest schema")
    expected = manifest.get("thresholds_file", {}).get("sha256")
    if not expected or sha256_file(thresholds_path) != expected:
        raise ValueError("Frozen threshold file SHA-256 does not match the lock manifest")
    return manifest


def apply_frozen(args):
    df = load_scores(args.scores_csv, {"test", "external"})
    if "test" not in set(df["split"]):
        raise ValueError("apply-frozen requires the VTaC test split")
    if not args.skip_vtac_count_check:
        check_vtac_counts(df, "test")
    if "external" in set(df["split"]) and args.external_cluster_mode is None:
        raise ValueError("--external-cluster-mode is required when external scores are present")

    thresholds = load_frozen_thresholds(args.thresholds)
    manifest_path = args.lock_manifest or args.thresholds.with_name("threshold_lock_manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"Threshold lock manifest not found: {manifest_path}")
    lock_manifest = verify_lock_manifest(args.thresholds, manifest_path)

    args.outdir.mkdir(parents=True, exist_ok=True)
    results = metric_rows(df, thresholds, args.bootstrap, args.seed, args.external_cluster_mode)
    results_path = args.outdir / "safety_utility_metrics.csv"
    results.to_csv(results_path, index=False)
    transport = []
    if {"test", "external"}.issubset(set(results["split"])):
        for target in TARGETS:
            test = results[(results.target_sensitivity == target) & (results.split == "test")].iloc[0]
            external = results[(results.target_sensitivity == target) & (results.split == "external")].iloc[0]
            transport.append({"target_sensitivity": target, "threshold": test.threshold,
                              "delta_sensitivity_external_minus_test": external.sensitivity - test.sensitivity,
                              "delta_fasr_external_minus_test": external.fasr - test.fasr,
                              "external_fn_per_100_true_vt": 100.0 * (1.0 - external.sensitivity)})
    transport_path = args.outdir / "transport_deltas.csv"
    pd.DataFrame(transport).to_csv(transport_path, index=False)
    application_manifest = {
        "schema": "vtac-frozen-application-v1",
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scores": {"path": str(args.scores_csv.resolve()), "sha256": sha256_file(args.scores_csv)},
        "thresholds_file": {"path": str(args.thresholds.resolve()), "sha256": sha256_file(args.thresholds)},
        "threshold_lock_manifest_sha256": sha256_file(manifest_path),
        "locked_validation_scores_sha256": lock_manifest["validation_scores"]["sha256"],
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "external_cluster_mode": args.external_cluster_mode,
        "outputs": {"safety_utility_metrics_sha256": sha256_file(results_path),
                    "transport_deltas_sha256": sha256_file(transport_path)},
    }
    write_json(args.outdir / "frozen_application_manifest.json", application_manifest)
    print(results[["target_sensitivity", "threshold", "split", "sensitivity", "fasr"]].to_string(index=False))
    print(f"\nFrozen-threshold results written to {args.outdir.resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock = subparsers.add_parser("lock-threshold", help="Select thresholds from validation scores only")
    lock.add_argument("scores_csv", type=Path)
    lock.add_argument("--model-artifact", type=Path, action="append", default=[])
    lock.add_argument("--outdir", type=Path, default=Path("vtac_threshold_lock"))
    lock.add_argument("--bootstrap", type=int, default=10000)
    lock.add_argument("--seed", type=int, default=20260810)
    lock.add_argument("--skip-vtac-count-check", action="store_true", help=argparse.SUPPRESS)
    lock.set_defaults(func=lock_threshold)
    apply = subparsers.add_parser("apply-frozen", help="Apply a locked threshold to test/external scores")
    apply.add_argument("scores_csv", type=Path)
    apply.add_argument("--thresholds", type=Path, required=True)
    apply.add_argument("--lock-manifest", type=Path)
    apply.add_argument("--outdir", type=Path, default=Path("vtac_threshold_results"))
    apply.add_argument("--bootstrap", type=int, default=10000)
    apply.add_argument("--seed", type=int, default=20260810)
    apply.add_argument("--external-cluster-mode", choices=("record", "event"))
    apply.add_argument("--skip-vtac-count-check", action="store_true", help=argparse.SUPPRESS)
    apply.set_defaults(func=apply_frozen)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
