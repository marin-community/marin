# ruff: noqa: RUF001
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict verbatim audit: checks spec_clauses_anchored_on against ONLY
the dominant + subordinate statements' text + examples for that rubric.

This catches the fabrication pattern where rubric writers cite quotes that
look spec-spirit-congruent and partial-match somewhere in the full spec
but DON'T appear in the actual statements being reasoned about.

Compare against `master_comparison.py`'s loose audit (substring anywhere
in any spec text).

Usage:
    uv run python experiments/posttrain/strict_verbatim_audit.py
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


def load_spec_per_id() -> dict[str, str]:
    """Map statement_id → concatenated lower-case text (text + all examples)."""
    out = {}
    for line in SPEC_PATH.open():
        s = json.loads(line)
        chunks = [s.get("text", "")]
        for ex in (s.get("metadata") or {}).get("examples") or []:
            chunks.append(ex.get("description", ""))
            chunks.append(ex.get("user_query", ""))
            chunks.append(ex.get("good_response", ""))
            chunks.append(ex.get("bad_response", ""))
        out[s["id"]] = "\n".join(chunks).lower()
    return out


def load_forked_spec_per_id(forked_path: Path) -> dict[str, str]:
    """Same but for a forked spec (has additional examples on edited statements)."""
    if not forked_path.exists():
        return {}
    out = {}
    for line in forked_path.open():
        s = json.loads(line)
        chunks = [s.get("text", "")]
        for ex in (s.get("metadata") or {}).get("examples") or []:
            chunks.append(ex.get("description", ""))
            chunks.append(ex.get("user_query", ""))
            chunks.append(ex.get("good_response", ""))
            chunks.append(ex.get("bad_response", ""))
        out[s["id"]] = "\n".join(chunks).lower()
    return out


def variant_to_spec_path(variant_label: str, judge: str) -> Path | None:
    """Which forked spec (if any) was used for this variant?"""
    if variant_label == "v2_baseline" or variant_label == "v3_alwayson":
        return None
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
    return None


def strict_audit(rubric_row: dict, spec_per_id: dict[str, str]) -> tuple[int, int, list]:
    """Audit clauses against ONLY the dominant + subordinate statements.

    Returns (verbatim_pass_count, total_clauses, list_of_failures).
    """
    parsed = rubric_row.get("parsed", {})
    rationale = parsed.get("rationale", {})
    clauses = rationale.get("spec_clauses_anchored_on", []) or []
    dom_id = rubric_row.get("dominant_id")
    sub_id = rubric_row.get("subordinate_id")

    relevant = (spec_per_id.get(dom_id, "") + "\n" + spec_per_id.get(sub_id, "")).lower()
    pass_count = 0
    failures = []
    for clause in clauses:
        if not clause:
            continue
        c_lower = (clause or "").lower().strip()
        if c_lower in relevant:
            pass_count += 1
        else:
            failures.append(
                {
                    "clause": clause[:200],
                    "pair_id": rubric_row.get("pair_id"),
                    "tp": rubric_row.get("tension_point_idx"),
                }
            )
    return pass_count, len(clauses), failures


def main() -> int:
    base_spec_per_id = load_spec_per_id()
    out_lines = [
        "# Strict verbatim audit",
        "",
        "Checks spec_clauses_anchored_on against ONLY the dominant + subordinate statements (not the whole spec).",
        "",
    ]
    out_lines.append(
        "Per-variant: forked spec is used (so edits' new examples are eligible for verbatim match if cited)."
    )
    out_lines.append("")
    out_lines.append("## Results table")
    out_lines.append("")
    out_lines.append("| variant | flash | gpt51 | pro | glm51 |")
    out_lines.append("|---|---:|---:|---:|---:|")

    fabrication_examples_by_variant: dict[tuple[str, str], list] = {}

    for variant_label, path_fn in VARIANTS:
        row = f"| `{variant_label}` |"
        for judge in JUDGES:
            rubric_path = path_fn(judge)
            rows = load_jsonl(rubric_path)
            if rows is None:
                row += " MISSING |"
                continue
            spec_path = variant_to_spec_path(variant_label, judge)
            spec_per_id = load_forked_spec_per_id(spec_path) if spec_path else base_spec_per_id

            total_pass = 0
            total_clauses = 0
            judge_failures = []
            for r in rows:
                pass_count, n_clauses, failures = strict_audit(r, spec_per_id)
                total_pass += pass_count
                total_clauses += n_clauses
                judge_failures.extend(failures)
            pct = 100 * total_pass / max(total_clauses, 1)
            row += f" {total_pass}/{total_clauses} ({pct:.0f}%) |"
            fabrication_examples_by_variant[(variant_label, judge)] = judge_failures
        out_lines.append(row)
    out_lines.append("")

    # Sample fabrication examples for the worst case
    out_lines.append("## Sample fabrication evidence (clauses NOT in dominant or subordinate statements)")
    out_lines.append("")
    out_lines.append("Showing up to 3 examples per (variant, judge):")
    out_lines.append("")
    for (variant_label, judge), failures in sorted(fabrication_examples_by_variant.items()):
        if not failures:
            continue
        out_lines.append(f"### `{variant_label}` × `{judge}` ({len(failures)} fabrications)")
        out_lines.append("")
        for f in failures[:3]:
            out_lines.append(f"- `{f['pair_id']} tp={f['tp']}`: {f['clause']!r}")
        out_lines.append("")

    out_path = STAGE3 / "exp_strict_verbatim_audit.md"
    out_path.write_text("\n".join(out_lines))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
