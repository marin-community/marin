# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment 1: self-edit propagation analysis.

For each of the 29 in-set proposed edits, compares the baseline rubric
(no edits) to the with-self-edits rubric on the edit's target test_pair,
computing 4 propagation signals:

1. **Citation**: does new rubric.rationale.spec_clauses_anchored_on cite
   any text from the new_example (case-insensitive substring match)?
2. **BAD criterion change**: 1 - SequenceMatcher.ratio on dominant_rubric.BAD
3. **Alternative-readings shift**: 1 - SequenceMatcher.ratio on
   rationale.alternative_readings_rejected
4. **Worked-example change**: 1 - SequenceMatcher.ratio on
   worked_example.spec_compliant

Per-edit classification:
- **strong propagation**: signal 1 fires AND at least 1 of {2,3,4} > 0.3
- **weak propagation**: signal 1 fires but no significant text change
- **no propagation**: signal 1 doesn't fire

Aggregate per judge + cross-judge rate.

Usage: uv run python experiments/posttrain/exp1_self_edit_propagation_analysis.py
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
STAGE3 = WORKTREE / "experiments/posttrain/stage3_output"
LM_JUDGE_EDITS = WORKTREE / "experiments/posttrain/lm_judge_edits"
OUT = STAGE3 / "exp1_self_edit_propagation_analysis.md"

JUDGES = ["flash", "gpt51", "pro", "glm51"]

CHANGE_THRESHOLD = 0.3  # 1 - similarity > 0.3 counts as significant change


def sim_change(a: str, b: str) -> float:
    """1 - SequenceMatcher.ratio. Higher = more changed."""
    return 1.0 - SequenceMatcher(None, a or "", b or "").ratio()


def citation_fires(rationale_clauses: list[str], new_example: dict) -> tuple[bool, list[str]]:
    """Does any spec_clauses_anchored_on entry contain text from the new_example?

    We check case-insensitive substring on each rationale clause vs each of the
    4 new_example fields. Returns (fired, list_of_matched_evidence)."""
    fields = [
        new_example.get("description", "") or "",
        new_example.get("user_query", "") or "",
        new_example.get("good_response", "") or "",
        new_example.get("bad_response", "") or "",
    ]
    matched: list[str] = []
    for clause in rationale_clauses:
        c_lower = (clause or "").lower()
        # require clause length > 20 chars to avoid trivial matches
        if len(c_lower) < 20:
            continue
        for field in fields:
            f_lower = field.lower()
            if not f_lower:
                continue
            # check if clause is substring of new_example field, OR
            # if any 40+ char window of the clause is in the new_example field
            if c_lower in f_lower:
                matched.append(clause[:120])
                break
            # window check: if a substring of the clause >= 40 chars matches
            for start in range(0, max(0, len(c_lower) - 40), 20):
                window = c_lower[start : start + 60]
                if window and len(window) >= 40 and window in f_lower:
                    matched.append(clause[:120] + " [partial]")
                    break
            else:
                continue
            break
    return (len(matched) > 0, matched)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def index_by_pair(rows: list[dict]) -> dict[tuple, dict]:
    return {(r["pair_id"], r["tension_point_idx"]): r for r in rows}


def load_edits(judge: str) -> list[dict]:
    edits_dir = LM_JUDGE_EDITS / judge / "proposed_edits"
    return [json.loads(f.read_text()) for f in sorted(edits_dir.glob("*.json"))]


def analyze_edit(edit: dict, baseline: dict, withedit: dict) -> dict:
    if baseline is None or withedit is None:
        return {"error": "missing baseline or with-edits rubric for this pair"}

    b_p = baseline["parsed"]
    w_p = withedit["parsed"]

    # Signal 1: citation
    new_clauses = w_p.get("rationale", {}).get("spec_clauses_anchored_on", []) or []
    cited, evidence = citation_fires(new_clauses, edit["new_example"])

    # Signal 2: BAD criterion change
    b_bad = (b_p.get("dominant_rubric") or {}).get("BAD", "") or ""
    w_bad = (w_p.get("dominant_rubric") or {}).get("BAD", "") or ""
    bad_change = sim_change(b_bad, w_bad)

    # Signal 3: alternative-readings shift
    b_alt = (b_p.get("rationale") or {}).get("alternative_readings_rejected", "") or ""
    w_alt = (w_p.get("rationale") or {}).get("alternative_readings_rejected", "") or ""
    alt_change = sim_change(b_alt, w_alt)

    # Signal 4: worked_example.spec_compliant change
    b_we = (b_p.get("worked_example") or {}).get("spec_compliant", "") or ""
    w_we = (w_p.get("worked_example") or {}).get("spec_compliant", "") or ""
    we_change = sim_change(b_we, w_we)

    # Classification
    significant_text_changes = sum(1 for x in [bad_change, alt_change, we_change] if x > CHANGE_THRESHOLD)
    if cited and significant_text_changes >= 1:
        classification = "STRONG"
    elif cited:
        classification = "WEAK (cited but no significant text change)"
    elif significant_text_changes >= 2:
        classification = "AMBIGUOUS (text changed without verbatim citation)"
    else:
        classification = "NONE"

    return {
        "cited": cited,
        "citation_evidence": evidence[:3],
        "bad_change": round(bad_change, 3),
        "alt_change": round(alt_change, 3),
        "we_change": round(we_change, 3),
        "significant_text_changes": significant_text_changes,
        "classification": classification,
    }


def main() -> None:
    out_lines: list[str] = []
    out_lines.append("# Experiment 1: Self-edit propagation analysis")
    out_lines.append("")
    out_lines.append("Per-edit propagation signals: did each of the 29 in-set edits actually move its target rubric?")
    out_lines.append("")
    out_lines.append("**Signals**:")
    out_lines.append(
        "- **Citation** (binary): does the new rubric's `rationale.spec_clauses_anchored_on` cite text from the new_example?"
    )
    out_lines.append("- **BAD change** (0-1): how much dominant_rubric.BAD changed (1 - difflib similarity)")
    out_lines.append("- **Alt change** (0-1): how much rationale.alternative_readings_rejected changed")
    out_lines.append("- **WE change** (0-1): how much worked_example.spec_compliant changed")
    out_lines.append("")
    out_lines.append(f"**Change threshold**: any signal > {CHANGE_THRESHOLD} = significant change.")
    out_lines.append("")
    out_lines.append("**Classification**:")
    out_lines.append("- **STRONG**: cited AND ≥1 significant text change")
    out_lines.append("- **WEAK**: cited but no significant text change (writer saw the example but didn't change much)")
    out_lines.append("- **AMBIGUOUS**: text changed but no verbatim citation (changes may be unrelated to edit)")
    out_lines.append("- **NONE**: no propagation signal fired")
    out_lines.append("")
    out_lines.append("---")
    out_lines.append("")

    judge_summaries: dict[str, dict] = {}

    def baseline_path_for(judge: str) -> Path:
        # flash's baseline is `cross_tier_rubrics_v2.jsonl` (no _flash suffix)
        return STAGE3 / ("cross_tier_rubrics_v2.jsonl" if judge == "flash" else f"cross_tier_rubrics_v2_{judge}.jsonl")

    for judge in JUDGES:
        baseline = index_by_pair(load_jsonl(baseline_path_for(judge)))
        withedits = index_by_pair(load_jsonl(STAGE3 / f"cross_tier_rubrics_v2_{judge}_with_self_edits.jsonl"))
        edits = load_edits(judge)

        out_lines.append(f"## Judge: `{judge}` ({len(edits)} edits)")
        out_lines.append("")
        out_lines.append("| edit_id | target_statement | test_pair | cited | BAD Δ | alt Δ | WE Δ | classification |")
        out_lines.append("|---|---|---|:---:|---:|---:|---:|---|")

        per_edit_results = []
        for e in edits:
            tp = (e["test_pair"]["pair_id"], e["test_pair"]["tension_point_idx"])
            result = analyze_edit(e, baseline.get(tp), withedits.get(tp))
            per_edit_results.append((e, result))
            if "error" in result:
                out_lines.append(
                    f"| `{e['edit_id']}` | `{e['target_statement_id']}` | `{tp[0]} tp={tp[1]}` | ERROR | - | - | - | {result['error']} |"
                )
                continue
            cited_mark = "✓" if result["cited"] else ""
            out_lines.append(
                f"| `{e['edit_id']}` | `{e['target_statement_id']}` | `{tp[0]} tp={tp[1]}` | "
                f"{cited_mark} | {result['bad_change']:.2f} | {result['alt_change']:.2f} | {result['we_change']:.2f} | "
                f"{result['classification']} |"
            )

        # Aggregate per judge
        n = len(per_edit_results)
        strong = sum(1 for _, r in per_edit_results if r.get("classification") == "STRONG")
        weak = sum(1 for _, r in per_edit_results if r.get("classification", "").startswith("WEAK"))
        ambig = sum(1 for _, r in per_edit_results if r.get("classification", "").startswith("AMBIGUOUS"))
        none = sum(1 for _, r in per_edit_results if r.get("classification") == "NONE")
        cited_total = sum(1 for _, r in per_edit_results if r.get("cited"))

        out_lines.append("")
        out_lines.append(f"**Summary**: STRONG={strong}, WEAK={weak}, AMBIG={ambig}, NONE={none}")
        out_lines.append(f"**Citation rate**: {cited_total}/{n} ({100*cited_total/n:.0f}%)")
        out_lines.append(f"**Strong-propagation rate**: {strong}/{n} ({100*strong/n:.0f}%)")
        out_lines.append(f"**Effective propagation (STRONG + WEAK)**: {strong + weak}/{n} ({100*(strong+weak)/n:.0f}%)")
        out_lines.append("")

        judge_summaries[judge] = {
            "n": n,
            "strong": strong,
            "weak": weak,
            "ambig": ambig,
            "none": none,
            "cited": cited_total,
        }

    # Cross-judge aggregate
    out_lines.append("---")
    out_lines.append("")
    out_lines.append("## Cross-judge aggregate")
    out_lines.append("")
    out_lines.append("| judge | n | citation | STRONG | WEAK | AMBIG | NONE | strong rate | citation rate |")
    out_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    total_n = total_strong = total_weak = total_ambig = total_none = total_cited = 0
    for judge in JUDGES:
        s = judge_summaries[judge]
        out_lines.append(
            f"| {judge} | {s['n']} | {s['cited']} | {s['strong']} | {s['weak']} | {s['ambig']} | {s['none']} | "
            f"{100*s['strong']/s['n']:.0f}% | {100*s['cited']/s['n']:.0f}% |"
        )
        total_n += s["n"]
        total_strong += s["strong"]
        total_weak += s["weak"]
        total_ambig += s["ambig"]
        total_none += s["none"]
        total_cited += s["cited"]
    out_lines.append(
        f"| **all** | **{total_n}** | **{total_cited}** | **{total_strong}** | **{total_weak}** | "
        f"**{total_ambig}** | **{total_none}** | **{100*total_strong/total_n:.0f}%** | "
        f"**{100*total_cited/total_n:.0f}%** |"
    )
    out_lines.append("")

    # Decision gate
    overall_strong = total_strong / total_n
    out_lines.append("## Headline interpretation")
    out_lines.append("")
    if overall_strong >= 0.7:
        out_lines.append(
            f"**HYPOTHESIS CONFIRMED**: {100*overall_strong:.0f}% of edits show strong propagation. "
            f"M5 spec-as-source-of-truth thesis empirically validated for the rubric layer."
        )
    elif overall_strong >= 0.5:
        out_lines.append(
            f"**PARTIAL VALIDATION**: {100*overall_strong:.0f}% strong propagation. "
            f"Some edit categories propagate cleanly; others don't. Investigate failure modes."
        )
    else:
        out_lines.append(
            f"**HYPOTHESIS NOT MET**: only {100*overall_strong:.0f}% strong propagation. "
            f"Spec-edit channel doesn't reliably move rubrics; M5 design needs revision."
        )
    out_lines.append("")

    # Failure mode investigation
    out_lines.append("## Edits that didn't propagate")
    out_lines.append("")
    for judge in JUDGES:
        edits = load_edits(judge)
        baseline = index_by_pair(load_jsonl(baseline_path_for(judge)))
        withedits = index_by_pair(load_jsonl(STAGE3 / f"cross_tier_rubrics_v2_{judge}_with_self_edits.jsonl"))
        for e in edits:
            tp = (e["test_pair"]["pair_id"], e["test_pair"]["tension_point_idx"])
            r = analyze_edit(e, baseline.get(tp), withedits.get(tp))
            if r.get("classification") == "NONE":
                out_lines.append(
                    f"- **{judge}/{e['edit_id']}** → `{e['target_statement_id']}` on `{tp[0]} tp={tp[1]}`: no propagation (BAD Δ={r['bad_change']:.2f}, alt Δ={r['alt_change']:.2f}, WE Δ={r['we_change']:.2f}, cited=False)"
                )
    out_lines.append("")

    OUT.write_text("\n".join(out_lines))
    print(f"Wrote {OUT}")
    print(
        f"\nHeadline: {total_strong}/{total_n} ({100*overall_strong:.0f}%) strong propagation, {total_cited}/{total_n} ({100*total_cited/total_n:.0f}%) cited."
    )


if __name__ == "__main__":
    main()
