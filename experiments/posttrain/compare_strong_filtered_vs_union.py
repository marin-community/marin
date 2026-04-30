# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare quality-filtered (STRONG-only) spec vs full union spec vs baseline.

Per-judge metrics:
- Schema validity, avg clauses, avg interp/alt/spec_compliant chars (already
  in master_comparison.py)
- Per-rubric text-change delta from baseline (same as compare_rubric_sets.py)

This script focuses on the headline question: does the 19-edit STRONG-only spec
give cleaner improvements than the 29-edit union spec?

Usage:
    uv run python experiments/posttrain/compare_strong_filtered_vs_union.py
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
STAGE3 = WORKTREE / "experiments/posttrain/stage3_output"
OUT = STAGE3 / "exp_strong_filtered_vs_union.md"

JUDGES = ["flash", "gpt51", "pro", "glm51"]


def load(p):
    if not p.exists():
        return None
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def baseline_path(judge):
    return STAGE3 / ("cross_tier_rubrics_v2.jsonl" if judge == "flash" else f"cross_tier_rubrics_v2_{judge}.jsonl")


def sim_change(a, b):
    return 1.0 - SequenceMatcher(None, a or "", b or "").ratio()


def per_rubric_change(baseline, variant, field_path):
    """Avg sim_change between baseline and variant rubrics on a JSON-path field."""
    b_idx = {(r["pair_id"], r["tension_point_idx"]): r for r in baseline}
    changes = []
    for v in variant:
        k = (v["pair_id"], v["tension_point_idx"])
        if k not in b_idx:
            continue
        b_p, v_p = b_idx[k]["parsed"], v["parsed"]
        b_val = b_p
        v_val = v_p
        for fp in field_path:
            b_val = (b_val or {}).get(fp, "") if isinstance(b_val, dict) else ""
            v_val = (v_val or {}).get(fp, "") if isinstance(v_val, dict) else ""
        changes.append(sim_change(str(b_val), str(v_val)))
    return sum(changes) / max(len(changes), 1) if changes else None


def main():
    out = ["# STRONG-only filtered spec vs all-edits union vs baseline", ""]
    out.append(
        "**Hypothesis**: filtering R1 edits to keep only STRONG-classified ones (19/29) "
        "should give cleaner per-rubric improvements than the full union (29 edits, of "
        "which 10 didn't propagate or only propagated weakly)."
    )
    out.append("")
    out.append(
        "**Method**: ran 4 writers on the 19-edit STRONG-only spec; compute mean "
        "per-rubric text change from baseline on key fields."
    )
    out.append("")
    out.append("## Per-judge mean text-change from baseline")
    out.append("")
    out.append(
        "Higher = more rewritten by the spec edits. Both variants should have positive "
        "deltas; the question is whether STRONG-only is cleaner."
    )
    out.append("")
    out.append("| field | judge | strong_only | union (all 29) |")
    out.append("|---|---|---:|---:|")

    fields = [
        ("dominant_rubric.BAD", ["dominant_rubric", "BAD"]),
        ("rationale.alt_readings", ["rationale", "alternative_readings_rejected"]),
        ("worked_example.spec_compliant", ["worked_example", "spec_compliant"]),
    ]
    for field_label, field_path in fields:
        for judge in JUDGES:
            base = load(baseline_path(judge))
            so = load(STAGE3 / f"cross_tier_rubrics_v2_{judge}_with_strong_only_edits.jsonl")
            un = load(STAGE3 / f"cross_tier_rubrics_v2_{judge}_with_union_edits.jsonl")
            if base is None or so is None or un is None:
                out.append(f"| {field_label} | {judge} | MISSING | MISSING |")
                continue
            so_d = per_rubric_change(base, so, field_path)
            un_d = per_rubric_change(base, un, field_path)
            so_str = f"{so_d:.3f}" if so_d is not None else "—"
            un_str = f"{un_d:.3f}" if un_d is not None else "—"
            out.append(f"| `{field_label}` | {judge} | {so_str} | {un_str} |")
        out.append("|---|---|---|---|")
    out.append("")

    out.append("## Schema validity")
    out.append("")
    out.append("| judge | strong_only | union |")
    out.append("|---|---:|---:|")
    for judge in JUDGES:
        so = load(STAGE3 / f"cross_tier_rubrics_v2_{judge}_with_strong_only_edits.jsonl")
        un = load(STAGE3 / f"cross_tier_rubrics_v2_{judge}_with_union_edits.jsonl")
        so_ok = sum(1 for r in so or [] if r["diag"].get("schema_ok"))
        un_ok = sum(1 for r in un or [] if r["diag"].get("schema_ok"))
        out.append(f"| {judge} | {so_ok}/{len(so or [])} | {un_ok}/{len(un or [])} |")
    out.append("")

    out.append("## Interpretation")
    out.append("")
    out.append(
        "- If strong_only deltas are CLOSE to union deltas, the 10 non-STRONG edits in union were mostly noise (filterable)"
    )
    out.append(
        "- If strong_only deltas are MUCH SMALLER than union deltas, the non-STRONG edits were also driving rewrites (just not in the propagation-detector's signal)"
    )
    out.append(
        "- If schema validity is HIGHER in strong_only, the saturation hypothesis is supported (fewer edits = more stable writer)"
    )

    OUT.write_text("\n".join(out))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
