# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Second batch of 5 pairs for gpt51 vs glm51 spot-check.

Different domains than `render_edit_progression_gpt51_glm51.py`:
- identity / fairness (white pride; persistent pathology in both judges)
- technical safety (bomb recipe / info hazards)
- interpersonal tone (named-individual roast / satire)
- low-stakes formatting (markdown vs plaintext)
- safety + uncertainty (chemical exposure)

Output: stage3_output/edit_progression_gpt51_glm51_set2.md
"""

from __future__ import annotations

import json
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
STAGE3 = WORKTREE / "experiments/posttrain/stage3_output"
PAIRED_RUBRICS = STAGE3 / "paired_rubrics_seed_40.jsonl"
OUT = STAGE3 / "edit_progression_gpt51_glm51_set2.md"

# Five fresh pairs distinct from set 1
PAIRS = [
    ("be_kind__uphold_fairness", 1),  # white pride symmetric celebration (persistent pathology)
    ("avoid_errors__avoid_info_hazards", 2),  # bomb recipe correction (info hazards)
    ("avoid_abuse__letter_and_spirit", 2),  # named-individual satire/roast
    ("formatting__letter_and_spirit", 4),  # markdown vs plaintext (low stakes)
    ("express_uncertainty__prevent_imminent_harm", 5),  # chemical exposure (safety + uncertainty)
]

VARIANTS = [
    ("Baseline (no edits)", "{judge}"),
    ("After Edit 1 (R1 self-edits)", "{judge}_with_self_edits"),
    ("After Edit 2 (R1+R2 cumulative)", "{judge}_with_r1r2_edits"),
]


def load_user_prompts() -> dict[tuple, str]:
    out = {}
    for line in PAIRED_RUBRICS.open():
        r = json.loads(line)
        tp = r.get("tension_point", {}) or {}
        out[(r["pair_id"], r["tension_point_idx"])] = tp.get("example_prompt", "") or ""
    return out


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
    out_lines = ["# gpt51 vs glm51 — set 2 (5 new pairs)", ""]
    out_lines.append("Companion to `edit_progression_gpt51_glm51.md` (the original 5 pairs).")
    out_lines.append("Same 3-stage progression: baseline → R1 self-edits → R1+R2 cumulative.")
    out_lines.append("")
    out_lines.append("Note: gpt51 outputs here used the default API temperature (~1.0) — the temp=0.2 fix")
    out_lines.append("rerun is in flight separately (`_temp02` suffix files).")
    out_lines.append("")

    for i, (pair_id, tp) in enumerate(PAIRS, 1):
        out_lines.append("---\n")
        out_lines.append(f"## Pair {i}: `{pair_id}` tp={tp}")
        out_lines.append("")

        any_row = None
        for judge in ["gpt51", "glm51"]:
            for _, variant_pat in VARIANTS:
                path = STAGE3 / f"cross_tier_rubrics_v2_{variant_pat.format(judge=judge)}.jsonl"
                if path.exists():
                    rows = load(path)
                    if (pair_id, tp) in rows:
                        any_row = rows[(pair_id, tp)]
                        break
            if any_row:
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

        for judge in ["gpt51", "glm51"]:
            out_lines.append(f"### {judge.upper()}")
            out_lines.append("")
            for variant_label, variant_pat in VARIANTS:
                path = STAGE3 / f"cross_tier_rubrics_v2_{variant_pat.format(judge=judge)}.jsonl"
                rows = load(path) if path.exists() else {}
                row = rows.get((pair_id, tp))
                out_lines.extend(render_rubric(row, variant_label))

    OUT.write_text("\n".join(out_lines))
    print(f"Wrote {OUT} ({len(out_lines)} lines)")


if __name__ == "__main__":
    main()
