# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render baseline / R1 / R1+R2 rubric progressions for gpt51 (temp=0.2 re-run) across 5 pairs.

Same pairs as render_edit_progression_gpt51_glm51.py, but:
  - reads the *_temp02.jsonl outputs (with the temperature=0.2 fix)
  - gpt51 only (no glm51 column)

Output: stage3_output/edit_progression_gpt51_temp02_only.md
"""

from __future__ import annotations

import json
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
STAGE3 = WORKTREE / "experiments/posttrain/stage3_output"
PAIRED_RUBRICS = STAGE3 / "paired_rubrics_seed_40.jsonl"
OUT = STAGE3 / "edit_progression_gpt51_temp02_only.md"


def load_user_prompts() -> dict[tuple, str]:
    out = {}
    for line in PAIRED_RUBRICS.open():
        r = json.loads(line)
        tp = r.get("tension_point", {}) or {}
        out[(r["pair_id"], r["tension_point_idx"])] = tp.get("example_prompt", "") or ""
    return out


PAIRS = [
    ("avoid_targeted_political_manipulation__be_kind", 0),  # the "I am programmed" case
    ("assume_objective_pov__do_not_encourage_self_harm", 0),  # philosophical-framing self-harm
    ("avoid_hateful_content__present_perspectives", 3),  # dogwhistles
    ("assume_objective_pov__do_not_facilitate_illicit_behavior", 5),  # pen-test / darknet
    ("protect_privacy__support_mental_health", 4),  # suicide notification
]

# (label, file-suffix-after gpt51)  — empty string = baseline
VARIANTS = [
    ("Baseline (no edits, temp=0.2)", "_temp02"),
    ("After Edit 1 (R1 self-edits, temp=0.2)", "_with_self_edits_temp02"),
    ("After Edit 2 (R1+R2 cumulative, temp=0.2)", "_with_r1r2_edits_temp02"),
]


def load(p: Path) -> dict[tuple, dict]:
    return {(r["pair_id"], r["tension_point_idx"]): r for r in (json.loads(line) for line in p.open() if line.strip())}


def render_rubric(row: dict | None, label: str) -> list[str]:
    if row is None:
        return [f"**{label}**: *(missing)*", ""]
    parsed = row.get("parsed", {})
    dom = parsed.get("dominant_rubric", {})
    we = parsed.get("worked_example", {})
    rationale = parsed.get("rationale", {})
    out = [f"**{label}**", ""]
    out.append(f"- *dominant_rubric.GOOD*: {(dom.get('GOOD', '') or '').strip()}")
    out.append(f"- *dominant_rubric.BAD*: {(dom.get('BAD', '') or '').strip()}")
    out.append(f"- *worked_example.spec_compliant*: {(we.get('spec_compliant', '') or '').strip()}")
    out.append(
        f"- *rationale.interpretive_choices_made*: {(rationale.get('interpretive_choices_made', '') or '').strip()[:600]}"
    )
    out.append("")
    return out


def main() -> None:
    user_prompts = load_user_prompts()
    out_lines = ["# Rubric edit progression — gpt51 @ temp=0.2 — 5 representative pairs", ""]
    out_lines.append(
        "Same 5 pairs as `edit_progression_gpt51_glm51.md`, " "but using the temperature=0.2 re-runs (`*_temp02.jsonl`)."
    )
    out_lines.append("For each pair: shows baseline → R1 self-edits → R1+R2 cumulative for gpt51.")
    out_lines.append("")

    for _, suffix in VARIANTS:
        path = STAGE3 / f"cross_tier_rubrics_v2_gpt51{suffix}.jsonl"
        assert path.exists(), f"missing {path}"

    for i, (pair_id, tp) in enumerate(PAIRS, 1):
        out_lines.append("---\n")
        out_lines.append(f"## Pair {i}: `{pair_id}` tp={tp}")
        out_lines.append("")

        any_row = None
        for _, suffix in VARIANTS:
            path = STAGE3 / f"cross_tier_rubrics_v2_gpt51{suffix}.jsonl"
            rows = load(path)
            if (pair_id, tp) in rows:
                any_row = rows[(pair_id, tp)]
                break

        if any_row:
            tension = any_row.get("tension_name", "")
            dom_id = any_row.get("dominant_id", "")
            sub_id = any_row.get("subordinate_id", "")
            user_query = user_prompts.get((pair_id, tp), "")
            out_lines.append(f"**Tension**: {tension}")
            out_lines.append(f"**Dominant**: `{dom_id}` | **Subordinate**: `{sub_id}`")
            out_lines.append("")
            out_lines.append("**The user prompt:**")
            out_lines.append("")
            out_lines.append("> " + (user_query or "*(missing)*").replace("\n", "\n> "))
            out_lines.append("")

        out_lines.append("### GPT51 (temp=0.2)")
        out_lines.append("")
        for variant_label, suffix in VARIANTS:
            path = STAGE3 / f"cross_tier_rubrics_v2_gpt51{suffix}.jsonl"
            rows = load(path)
            row = rows.get((pair_id, tp))
            out_lines.extend(render_rubric(row, variant_label))

    OUT.write_text("\n".join(out_lines))
    print(f"Wrote {OUT} ({len(out_lines)} lines)")


if __name__ == "__main__":
    main()
