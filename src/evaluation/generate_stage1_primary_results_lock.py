"""Render the Stage 1 manuscript-facing v2 lock from one canonical JSON source."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "00_protocol/frozen/STAGE1_PRIMARY_RESULTS_CANONICAL_v2.json"


def fmt(value):
    return format(float(value), ".4f")


def render_yaml(c):
    lines = [
        "schema: stage1-primary-results-lock-v2",
        "status: frozen_manuscript_reconciliation",
        "rounding: Python format specification :.4f (round-half-even)",
        f"primary_threshold: {c['primary_threshold']}",
        "secondary_thresholds: [0.065468185, 0.022276472]",
    ]
    for key, cohort in zip(("validation", "test", "external"), c["cohorts"]):
        line = (
            f"{key}: {{N: {cohort['n']}, true_vt: {cohort['true_vt']}, "
            f"false_vt: {cohort['false_vt']}, auroc: {fmt(cohort['auroc'])}, "
            f"auroc_ci: [{fmt(cohort['auroc_ci'][0])}, {fmt(cohort['auroc_ci'][1])}], "
            f"tau95: {{TP: {cohort['tp']}, FN: {cohort['fn']}, TN: {cohort['tn']}, "
            f"FP: {cohort['fp']}, sensitivity: {fmt(cohort['sensitivity'])}, "
            f"sensitivity_ci: [{fmt(cohort['sensitivity_ci'][0])}, {fmt(cohort['sensitivity_ci'][1])}], "
            f"fasr: {fmt(cohort['fasr'])}, fasr_ci: [{fmt(cohort['fasr_ci'][0])}, "
            f"{fmt(cohort['fasr_ci'][1])}]}}}}"
        )
        lines.append(line)
    lines.append("transport_deltas:")
    for d in c["transport_deltas"]:
        slug = d["comparison"].lower().replace(" ", "_").replace("-", "minus")
        lines.extend([
            f"  {slug}:",
            f"    sensitivity: {{estimate: {fmt(d['delta_sensitivity'])}, ci: [{fmt(d['sensitivity_ci'][0])}, {fmt(d['sensitivity_ci'][1])}]}}",
            f"    fasr: {{estimate: {fmt(d['delta_fasr'])}, ci: [{fmt(d['fasr_ci'][0])}, {fmt(d['fasr_ci'][1])}]}}",
        ])
    return "\n".join(lines) + "\n"


def render_markdown(c):
    payload = {
        "schema": c["schema"],
        "status": "frozen_manuscript_reconciliation",
        "rounding": c["rounding"],
        "primary_results": c["cohorts"],
        "transport_deltas": c["transport_deltas"],
        "source_artifacts": c["source_artifacts"],
    }
    table = ["| Cohort | Sensitivity CI | FASR CI |", "|---|---:|---:|"]
    for cohort in c["cohorts"]:
        table.append(
            f"| {cohort['name']} | {fmt(cohort['sensitivity_ci'][0])}-{fmt(cohort['sensitivity_ci'][1])} | "
            f"{fmt(cohort['fasr_ci'][0])}-{fmt(cohort['fasr_ci'][1])} |"
        )
    return (
        "# Stage 1 Primary Results Lock v2\n\n"
        "This version supersedes v1 for manuscript-facing use. It is rendered "
        "deterministically from the canonical JSON source.\n\n"
        + "\n".join(table)
        + "\n\n```json\n"
        + json.dumps(payload, indent=2)
        + "\n```\n"
    )


if __name__ == "__main__":
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    (ROOT / "00_protocol/frozen/STAGE1_PRIMARY_RESULTS_LOCK_v2.yaml").write_text(render_yaml(canonical), encoding="utf-8")
    (ROOT / "00_protocol/frozen/STAGE1_PRIMARY_RESULTS_LOCK_v2.md").write_text(render_markdown(canonical), encoding="utf-8")