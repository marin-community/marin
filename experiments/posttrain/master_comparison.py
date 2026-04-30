# ruff: noqa: E501, E731, RUF001, RUF002
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Master comparison: aggregates all experimental results into one matrix.

For each judge × architecture variant, computes:
- Schema validity
- Verbatim audit pass rate (case-insensitive substring, with caveat about fabrication)
- Avg rationale clauses per rubric
- Avg interpretive_choices_made / alternative_readings_rejected length
- Avg worked_example.spec_compliant length

Variants compared:
- v2_baseline: original v2 architecture, no edits
- v2_self_edits: v2 + each judge's own R1 edits
- v2_union_edits: v2 + all 29 R1 edits across judges
- v2_r1r2_edits: v2 + each judge's R1 + R2 edits cumulatively
- v3_alwayson: v3 architecture (4 cross-cutting always-loaded), no edits

Output: master_comparison_matrix.md
"""

from __future__ import annotations

import json
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
STAGE3 = WORKTREE / "experiments/posttrain/stage3_output"
SPEC_PATH = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"

JUDGES = ["flash", "gpt51", "pro", "glm51"]
VARIANTS = [
    (
        "v2_baseline",
        lambda j: STAGE3 / ("cross_tier_rubrics_v2.jsonl" if j == "flash" else f"cross_tier_rubrics_v2_{j}.jsonl"),
    ),
    ("v2_self_edits", lambda j: STAGE3 / f"cross_tier_rubrics_v2_{j}_with_self_edits.jsonl"),
    ("v2_union_edits", lambda j: STAGE3 / f"cross_tier_rubrics_v2_{j}_with_union_edits.jsonl"),
    ("v2_r1r2_edits", lambda j: STAGE3 / f"cross_tier_rubrics_v2_{j}_with_r1r2_edits.jsonl"),
    ("v2_r1r2r3_edits", lambda j: STAGE3 / f"cross_tier_rubrics_v2_{j}_with_r1r2r3_edits.jsonl"),
    ("v2_compiler_edits", lambda j: STAGE3 / f"cross_tier_rubrics_v2_{j}_with_compiler_edits.jsonl"),
    ("v2_strong_only_edits", lambda j: STAGE3 / f"cross_tier_rubrics_v2_{j}_with_strong_only_edits.jsonl"),
    ("v3_alwayson", lambda j: STAGE3 / f"cross_tier_rubrics_v3_alwayson_{j}.jsonl"),
]


def load_jsonl(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_spec_per_id(spec_path: Path) -> dict[str, str]:
    """Map statement_id → concatenated lower-case text (text + all examples).

    Used for STRICT per-pair verbatim audit: a clause is valid only if it
    appears in the dominant or subordinate statement of the rubric being
    audited, not anywhere in the spec.

    DO NOT replace this with full-spec concatenation. The earlier loose
    version (substring-match against the whole spec) was producing false
    passes — fabricated paraphrases would partial-match unrelated spec text
    by accident. See `.agents/projects/executable_specifications.md` §
    "MAJOR LEARNING: audit primitives must be per-statement, never full-spec".
    """
    out = {}
    for line in spec_path.open():
        s = json.loads(line)
        chunks = [s.get("text", "")]
        for ex in (s.get("metadata") or {}).get("examples") or []:
            chunks.append(ex.get("description", ""))
            chunks.append(ex.get("user_query", ""))
            chunks.append(ex.get("good_response", ""))
            chunks.append(ex.get("bad_response", ""))
        out[s["id"]] = "\n".join(chunks).lower()
    return out


def variant_to_spec_path(variant_label: str, judge: str) -> Path:
    """Which forked spec was used for this variant. Defaults to base spec."""
    if variant_label == "v2_self_edits":
        return WORKTREE / f"experiments/posttrain/specs/openai_model_spec_{judge}_self_edits.jsonl"
    if variant_label == "v2_union_edits":
        return WORKTREE / "experiments/posttrain/specs/openai_model_spec_union_round1_edits.jsonl"
    if variant_label == "v2_r1r2_edits":
        return WORKTREE / f"experiments/posttrain/specs/openai_model_spec_{judge}_r1r2_edits.jsonl"
    if variant_label == "v2_r1r2r3_edits":
        return WORKTREE / f"experiments/posttrain/specs/openai_model_spec_{judge}_r1r2r3_edits.jsonl"
    if variant_label == "v2_compiler_edits":
        return WORKTREE / f"experiments/posttrain/specs/openai_model_spec_{judge}_compiler_edits.jsonl"
    if variant_label == "v2_strong_only_edits":
        return WORKTREE / "experiments/posttrain/specs/openai_model_spec_strong_r1_only.jsonl"
    return SPEC_PATH


def compute_metrics(rows: list[dict], spec_per_id: dict[str, str]) -> dict:
    """STRICT per-pair verbatim audit.

    For each rubric, cited clauses must appear verbatim in the dominant or
    subordinate statement of THAT rubric (not anywhere in the spec).
    """
    n = len(rows)
    schema_ok = sum(1 for r in rows if r["diag"].get("schema_ok"))
    parse_ok = sum(1 for r in rows if r["diag"].get("parse_ok", True))

    clauses_per = []
    interp_chars = []
    alt_chars = []
    sc_chars = []
    verbatim_pass = 0
    total_clauses = 0
    for r in rows:
        rationale = r["parsed"].get("rationale", {})
        clauses = rationale.get("spec_clauses_anchored_on", []) or []
        clauses_per.append(len(clauses))
        dom_id = r.get("dominant_id")
        sub_id = r.get("subordinate_id")
        relevant = (spec_per_id.get(dom_id, "") + "\n" + spec_per_id.get(sub_id, "")).lower()
        for c in clauses:
            total_clauses += 1
            if (c or "").lower().strip() in relevant:
                verbatim_pass += 1
        interp_chars.append(len(rationale.get("interpretive_choices_made", "") or ""))
        alt_chars.append(len(rationale.get("alternative_readings_rejected", "") or ""))
        sc = (r["parsed"].get("worked_example") or {}).get("spec_compliant", "") or ""
        sc_chars.append(len(sc))

    avg = lambda vs: sum(vs) / max(len(vs), 1)
    return {
        "n": n,
        "schema_ok": f"{schema_ok}/{n}",
        "parse_ok": f"{parse_ok}/{n}",
        "verbatim_audit_strict": f"{verbatim_pass}/{total_clauses} ({100*verbatim_pass/max(total_clauses,1):.0f}%)",
        "avg_clauses": f"{avg(clauses_per):.1f}",
        "avg_interp_chars": f"{avg(interp_chars):.0f}",
        "avg_alt_chars": f"{avg(alt_chars):.0f}",
        "avg_spec_compliant_chars": f"{avg(sc_chars):.0f}",
    }


def main() -> int:
    base_spec_per_id = load_spec_per_id(SPEC_PATH)
    out_lines = [
        "# Master comparison matrix — all experiments",
        "",
        "Generated by `master_comparison.py`. ",
        "",
        "Variants:",
    ]
    for label, _ in VARIANTS:
        out_lines.append(f"- `{label}`")
    out_lines.append("")
    out_lines.append(
        'Note: `verbatim_audit_strict` checks each cited clause against ONLY the rubric\'s dominant + subordinate statements (in the FORKED spec used for that variant), not the full spec. The earlier loose audit (full-spec substring match) was producing false passes via accidental partial matches across unrelated statements; see `.agents/projects/executable_specifications.md` § "MAJOR LEARNING". Always use strict per-pair, per-statement.'
    )
    out_lines.append("")

    metrics_table: dict[tuple[str, str], dict | None] = {}
    for variant_label, path_fn in VARIANTS:
        for judge in JUDGES:
            path = path_fn(judge)
            rows = load_jsonl(path)
            if rows is None:
                metrics_table[(variant_label, judge)] = None
            else:
                forked_spec = variant_to_spec_path(variant_label, judge)
                spec_per_id = load_spec_per_id(forked_spec) if forked_spec.exists() else base_spec_per_id
                metrics_table[(variant_label, judge)] = compute_metrics(rows, spec_per_id)

    # One table per metric
    for metric in [
        "schema_ok",
        "verbatim_audit_strict",
        "avg_clauses",
        "avg_interp_chars",
        "avg_alt_chars",
        "avg_spec_compliant_chars",
    ]:
        out_lines.append(f"## {metric}")
        out_lines.append("")
        header = "| variant | " + " | ".join(JUDGES) + " |"
        sep = "|---|" + "|".join("---:" for _ in JUDGES) + "|"
        out_lines.append(header)
        out_lines.append(sep)
        for variant_label, _ in VARIANTS:
            row = f"| `{variant_label}` |"
            for judge in JUDGES:
                m = metrics_table.get((variant_label, judge))
                row += f" {m[metric] if m else 'MISSING'} |"
            out_lines.append(row)
        out_lines.append("")

    out_path = STAGE3 / "master_comparison_matrix.md"
    out_path.write_text("\n".join(out_lines))
    print(f"Wrote {out_path}")
    print("Variant × judge availability:")
    for variant_label, _ in VARIANTS:
        avail = sum(1 for judge in JUDGES if metrics_table.get((variant_label, judge)))
        print(f"  {variant_label}: {avail}/{len(JUDGES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
