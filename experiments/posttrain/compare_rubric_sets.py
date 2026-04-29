# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generalized rubric-set comparator.

Compares two cross-tier rubric JSONL files and emits a diff report.
Optionally takes edits dirs to compute the verbatim-citation signal.

Usage:
    # v2 baseline vs v3 cross-cutting-alwayson
    uv run python experiments/posttrain/compare_rubric_sets.py \\
        --baseline experiments/posttrain/stage3_output/cross_tier_rubrics_v2.jsonl \\
        --treatment experiments/posttrain/stage3_output/cross_tier_rubrics_v3_alwayson_flash.jsonl \\
        --label-baseline "v2_baseline" \\
        --label-treatment "v3_alwayson" \\
        --out experiments/posttrain/stage3_output/compare_v3_vs_v2_flash.md

    # with-self-edits vs with-union-edits
    uv run python experiments/posttrain/compare_rubric_sets.py \\
        --baseline experiments/posttrain/stage3_output/cross_tier_rubrics_v2_flash_with_self_edits.jsonl \\
        --treatment experiments/posttrain/stage3_output/cross_tier_rubrics_v2_flash_with_union_edits.jsonl \\
        --label-baseline "self_edits" \\
        --label-treatment "union_edits" \\
        --out experiments/posttrain/stage3_output/compare_union_vs_self_flash.md
"""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

CHANGE_THRESHOLD = 0.3


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def index_by_pair(rows: list[dict]) -> dict[tuple, dict]:
    return {(r["pair_id"], r["tension_point_idx"]): r for r in rows}


def sim_change(a: str, b: str) -> float:
    return 1.0 - SequenceMatcher(None, a or "", b or "").ratio()


def field(rubric_row: dict, top: str, sub: str) -> str:
    p = rubric_row.get("parsed", {})
    return (p.get(top) or {}).get(sub, "") or ""


def compute_per_pair_diffs(baseline: dict, treatment: dict) -> dict:
    """Compute text-change deltas across all rubric fields."""
    fields = [
        ("dominant_rubric", "GOOD"),
        ("dominant_rubric", "BAD"),
        ("dominant_rubric", "KEY_TENSION"),
        ("non_leakage_rubric", "GOOD"),
        ("non_leakage_rubric", "BAD"),
        ("non_leakage_rubric", "KEY_TENSION"),
        ("worked_example", "spec_compliant"),
        ("worked_example", "subordinate_over_satisfaction"),
        ("worked_example", "pathological_refusal"),
        ("rationale", "interpretive_choices_made"),
        ("rationale", "alternative_readings_rejected"),
    ]
    deltas = {}
    for top, sub in fields:
        b_text = field(baseline, top, sub)
        t_text = field(treatment, top, sub)
        deltas[f"{top}.{sub}"] = round(sim_change(b_text, t_text), 3)
    # Rationale clauses: how many citations changed?
    b_clauses = (baseline.get("parsed") or {}).get("rationale", {}).get("spec_clauses_anchored_on", [])
    t_clauses = (treatment.get("parsed") or {}).get("rationale", {}).get("spec_clauses_anchored_on", [])
    b_set = set(c.strip().lower() for c in b_clauses)
    t_set = set(c.strip().lower() for c in t_clauses)
    deltas["rationale.clauses_count_baseline"] = len(b_clauses)
    deltas["rationale.clauses_count_treatment"] = len(t_clauses)
    deltas["rationale.clauses_added"] = len(t_set - b_set)
    deltas["rationale.clauses_removed"] = len(b_set - t_set)
    return deltas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--treatment", type=Path, required=True)
    ap.add_argument("--label-baseline", default="baseline")
    ap.add_argument("--label-treatment", default="treatment")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    baseline = index_by_pair(load_jsonl(args.baseline))
    treatment = index_by_pair(load_jsonl(args.treatment))
    common = sorted(set(baseline.keys()) & set(treatment.keys()))

    out_lines = [
        f"# Rubric set comparison: {args.label_baseline} vs {args.label_treatment}",
        "",
        f"**Baseline**: `{args.baseline.relative_to(Path('.').resolve()) if args.baseline.is_absolute() else args.baseline}` ({len(baseline)} rows)",
        f"**Treatment**: `{args.treatment.relative_to(Path('.').resolve()) if args.treatment.is_absolute() else args.treatment}` ({len(treatment)} rows)",
        f"**Common pairs**: {len(common)}",
        "",
        "## Per-field aggregate change",
        "",
        "Mean text change (1 - SequenceMatcher.ratio) across all common pairs. Higher = more changed. Threshold for 'significant change' = 0.3.",
        "",
    ]

    fields = [
        "dominant_rubric.GOOD",
        "dominant_rubric.BAD",
        "dominant_rubric.KEY_TENSION",
        "non_leakage_rubric.GOOD",
        "non_leakage_rubric.BAD",
        "non_leakage_rubric.KEY_TENSION",
        "worked_example.spec_compliant",
        "worked_example.subordinate_over_satisfaction",
        "worked_example.pathological_refusal",
        "rationale.interpretive_choices_made",
        "rationale.alternative_readings_rejected",
    ]

    all_deltas = []
    for key in common:
        all_deltas.append(compute_per_pair_diffs(baseline[key], treatment[key]))

    out_lines.append("| field | mean Δ | median Δ | max Δ | n significant (>0.3) |")
    out_lines.append("|---|---:|---:|---:|---:|")
    for f in fields:
        vals = [d[f] for d in all_deltas]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        median = sorted(vals)[len(vals) // 2]
        mx = max(vals)
        n_sig = sum(1 for v in vals if v > CHANGE_THRESHOLD)
        out_lines.append(f"| {f} | {mean:.3f} | {median:.3f} | {mx:.3f} | {n_sig}/{len(vals)} |")

    # Rationale clause changes
    out_lines.append("")
    out_lines.append("## Rationale clause changes")
    out_lines.append("")
    avg_baseline_clauses = sum(d["rationale.clauses_count_baseline"] for d in all_deltas) / len(all_deltas)
    avg_treatment_clauses = sum(d["rationale.clauses_count_treatment"] for d in all_deltas) / len(all_deltas)
    avg_added = sum(d["rationale.clauses_added"] for d in all_deltas) / len(all_deltas)
    avg_removed = sum(d["rationale.clauses_removed"] for d in all_deltas) / len(all_deltas)
    out_lines.append(f"- Avg `spec_clauses_anchored_on` count in {args.label_baseline}: {avg_baseline_clauses:.1f}")
    out_lines.append(f"- Avg `spec_clauses_anchored_on` count in {args.label_treatment}: {avg_treatment_clauses:.1f}")
    out_lines.append(f"- Avg new clauses (in treatment, not baseline): {avg_added:.1f}")
    out_lines.append(f"- Avg removed clauses (in baseline, not treatment): {avg_removed:.1f}")
    out_lines.append("")

    # Per-pair table
    out_lines.append("## Per-pair changes")
    out_lines.append("")
    out_lines.append("| pair | dom.BAD Δ | non_leak.BAD Δ | spec_compliant Δ | alt_read Δ | clauses Δ |")
    out_lines.append("|---|---:|---:|---:|---:|---:|")
    for key in common:
        d = compute_per_pair_diffs(baseline[key], treatment[key])
        clauses_delta = d["rationale.clauses_count_treatment"] - d["rationale.clauses_count_baseline"]
        out_lines.append(
            f"| `{key[0]} tp={key[1]}` | "
            f"{d['dominant_rubric.BAD']:.2f} | {d['non_leakage_rubric.BAD']:.2f} | "
            f"{d['worked_example.spec_compliant']:.2f} | {d['rationale.alternative_readings_rejected']:.2f} | "
            f"{clauses_delta:+d} |"
        )

    args.out.write_text("\n".join(out_lines))
    print(f"Wrote {args.out}")

    # Headline
    mean_dom_bad = sum(d["dominant_rubric.BAD"] for d in all_deltas) / len(all_deltas)
    mean_we = sum(d["worked_example.spec_compliant"] for d in all_deltas) / len(all_deltas)
    print(f"Headline: mean dominant.BAD Δ={mean_dom_bad:.3f}, mean worked_example.spec_compliant Δ={mean_we:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
