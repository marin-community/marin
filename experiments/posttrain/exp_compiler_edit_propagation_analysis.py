# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment: COMPILER-edit propagation analysis.

Mirrors exp1_self_edit_propagation_analysis.py but applies to LM-compiler-proposed
edits (in `lm_compiler_proposed_edits/<judge>/`) and the resulting
`cross_tier_rubrics_v2_<judge>_with_compiler_edits.jsonl`.

For each compiler-proposed edit, computes 4 propagation signals vs baseline:

1. **Citation**: does new rubric.rationale.spec_clauses_anchored_on cite text
   from the compiler's new_example?
2. **BAD criterion change**: 1 - SequenceMatcher.ratio on dominant_rubric.BAD
3. **Alternative-readings shift**: similarity change on
   rationale.alternative_readings_rejected
4. **Worked-example change**: similarity change on worked_example.spec_compliant

Classification: STRONG / WEAK / AMBIGUOUS / NONE (same thresholds as exp1).

Writes per-judge tables + cross-judge aggregate to
`stage3_output/exp_compiler_edit_propagation_analysis.md`.

Usage:
    uv run python experiments/posttrain/exp_compiler_edit_propagation_analysis.py
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
STAGE3 = WORKTREE / "experiments/posttrain/stage3_output"
LM_COMPILER_EDITS = WORKTREE / "experiments/posttrain/lm_compiler_proposed_edits"
OUT = STAGE3 / "exp_compiler_edit_propagation_analysis.md"

JUDGES = ["flash", "gpt51", "pro", "glm51"]

CHANGE_THRESHOLD = 0.3


def sim_change(a: str, b: str) -> float:
    return 1.0 - SequenceMatcher(None, a or "", b or "").ratio()


def citation_fires(rationale_clauses: list[str], new_example: dict) -> tuple[bool, list[str]]:
    fields = [
        new_example.get("description", "") or "",
        new_example.get("user_query", "") or "",
        new_example.get("good_response", "") or "",
        new_example.get("bad_response", "") or "",
    ]
    matched: list[str] = []
    for clause in rationale_clauses:
        c_lower = (clause or "").lower()
        if len(c_lower) < 20:
            continue
        for field in fields:
            f_lower = field.lower()
            if not f_lower:
                continue
            if c_lower in f_lower:
                matched.append(clause[:120])
                break
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


def load_compiler_edits(judge: str) -> list[dict]:
    edits_dir = LM_COMPILER_EDITS / judge
    return [json.loads(f.read_text()) for f in sorted(edits_dir.glob("*.json"))]


def baseline_path_for(judge: str) -> Path:
    return STAGE3 / ("cross_tier_rubrics_v2.jsonl" if judge == "flash" else f"cross_tier_rubrics_v2_{judge}.jsonl")


def analyze_edit(edit: dict, baseline: dict | None, withedit: dict | None) -> dict:
    if baseline is None or withedit is None:
        return {"error": "missing baseline or with-compiler-edits rubric for this pair"}

    b_p = baseline["parsed"]
    w_p = withedit["parsed"]

    new_clauses = w_p.get("rationale", {}).get("spec_clauses_anchored_on", []) or []
    cited, evidence = citation_fires(new_clauses, edit["new_example"])

    b_bad = (b_p.get("dominant_rubric") or {}).get("BAD", "") or ""
    w_bad = (w_p.get("dominant_rubric") or {}).get("BAD", "") or ""
    bad_change = sim_change(b_bad, w_bad)

    b_alt = (b_p.get("rationale") or {}).get("alternative_readings_rejected", "") or ""
    w_alt = (w_p.get("rationale") or {}).get("alternative_readings_rejected", "") or ""
    alt_change = sim_change(b_alt, w_alt)

    b_we = (b_p.get("worked_example") or {}).get("spec_compliant", "") or ""
    w_we = (w_p.get("worked_example") or {}).get("spec_compliant", "") or ""
    we_change = sim_change(b_we, w_we)

    significant = sum(1 for x in [bad_change, alt_change, we_change] if x > CHANGE_THRESHOLD)
    if cited and significant >= 1:
        cls = "STRONG"
    elif cited:
        cls = "WEAK (cited but no significant text change)"
    elif significant >= 2:
        cls = "AMBIGUOUS (text changed without verbatim citation)"
    else:
        cls = "NONE"

    return {
        "cited": cited,
        "citation_evidence": evidence[:3],
        "bad_change": round(bad_change, 3),
        "alt_change": round(alt_change, 3),
        "we_change": round(we_change, 3),
        "significant_text_changes": significant,
        "classification": cls,
    }


def main() -> None:
    out: list[str] = []
    out.append("# Experiment: LM-compiler edit propagation analysis")
    out.append("")
    out.append(
        "For each compiler-proposed edit (in `lm_compiler_proposed_edits/<judge>/`), do the writer's "
        "outputs change in the predicted way when that edit is applied to the spec?"
    )
    out.append("")
    out.append(
        "This is the **second-order** test of the compiler primitive: the first-order test "
        "(target_statement_id match against agent ground truth) was 85% (46/54) with all 8 mismatches "
        "reasonable. This test asks: even if the compiler picks a defensible target, does the new_example "
        "it generates actually move the rubric the way it's supposed to?"
    )
    out.append("")
    out.append("**Signals** (same as exp1 self-edit propagation):")
    out.append("- **Citation**: rubric's `rationale.spec_clauses_anchored_on` cites text from the new_example")
    out.append("- **BAD/Alt/WE Δ**: 1 - difflib similarity between baseline and with-compiler-edits rubric")
    out.append(f"- **Change threshold**: any signal > {CHANGE_THRESHOLD} = significant change")
    out.append("")
    out.append("**Classification**:")
    out.append("- **STRONG**: cited AND ≥1 significant text change")
    out.append("- **WEAK**: cited but no significant text change")
    out.append("- **AMBIGUOUS**: text changed but no verbatim citation")
    out.append("- **NONE**: no propagation signal fired")
    out.append("")
    out.append("---")
    out.append("")

    summaries: dict[str, dict] = {}
    failures: list[tuple[str, dict, dict]] = []

    for judge in JUDGES:
        baseline_path = baseline_path_for(judge)
        with_path = STAGE3 / f"cross_tier_rubrics_v2_{judge}_with_compiler_edits.jsonl"
        if not baseline_path.exists() or not with_path.exists():
            out.append(f"## Judge `{judge}` — MISSING ({baseline_path.name} or {with_path.name})")
            out.append("")
            continue
        baseline = index_by_pair(load_jsonl(baseline_path))
        withedits = index_by_pair(load_jsonl(with_path))
        edits = load_compiler_edits(judge)

        out.append(f"## Judge: `{judge}` ({len(edits)} compiler edits)")
        out.append("")
        out.append(
            "| edit_id | target | test_pair | source-agent-target-match | cited | BAD Δ | alt Δ | WE Δ | classification |"
        )
        out.append("|---|---|---|:---:|:---:|---:|---:|---:|---|")

        per: list[tuple[dict, dict]] = []
        for e in edits:
            tp_id = (e["test_pair"]["pair_id"], e["test_pair"]["tension_point_idx"])
            r = analyze_edit(e, baseline.get(tp_id), withedits.get(tp_id))
            per.append((e, r))
            agent_match = "✓" if e.get("_target_match_with_agent") else ""
            if "error" in r:
                out.append(
                    f"| `{e['edit_id']}` | `{e['target_statement_id']}` | `{tp_id[0]} tp={tp_id[1]}` | {agent_match} | ERROR | - | - | - | {r['error']} |"
                )
                continue
            cited_mark = "✓" if r["cited"] else ""
            out.append(
                f"| `{e['edit_id']}` | `{e['target_statement_id']}` | `{tp_id[0]} tp={tp_id[1]}` | "
                f"{agent_match} | {cited_mark} | {r['bad_change']:.2f} | {r['alt_change']:.2f} | {r['we_change']:.2f} | "
                f"{r['classification']} |"
            )
            if r.get("classification") == "NONE":
                failures.append((judge, e, r))

        n = len(per)
        strong = sum(1 for _, r in per if r.get("classification") == "STRONG")
        weak = sum(1 for _, r in per if r.get("classification", "").startswith("WEAK"))
        ambig = sum(1 for _, r in per if r.get("classification", "").startswith("AMBIGUOUS"))
        none = sum(1 for _, r in per if r.get("classification") == "NONE")
        cited = sum(1 for _, r in per if r.get("cited"))

        out.append("")
        out.append(f"**Summary**: STRONG={strong}, WEAK={weak}, AMBIG={ambig}, NONE={none}")
        out.append(f"**Citation rate**: {cited}/{n} ({100*cited/max(n,1):.0f}%)")
        out.append(f"**Strong propagation rate**: {strong}/{n} ({100*strong/max(n,1):.0f}%)")
        out.append(f"**Effective propagation (STRONG+WEAK)**: {strong+weak}/{n} ({100*(strong+weak)/max(n,1):.0f}%)")
        out.append("")
        summaries[judge] = {"n": n, "strong": strong, "weak": weak, "ambig": ambig, "none": none, "cited": cited}

    out.append("---")
    out.append("")
    out.append("## Cross-judge aggregate")
    out.append("")
    out.append("| judge | n | citation | STRONG | WEAK | AMBIG | NONE | strong rate |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    tn = ts = tw = ta = tnone = tc = 0
    for judge in JUDGES:
        if judge not in summaries:
            continue
        s = summaries[judge]
        out.append(
            f"| {judge} | {s['n']} | {s['cited']} | {s['strong']} | {s['weak']} | {s['ambig']} | {s['none']} | "
            f"{100*s['strong']/max(s['n'],1):.0f}% |"
        )
        tn += s["n"]
        ts += s["strong"]
        tw += s["weak"]
        ta += s["ambig"]
        tnone += s["none"]
        tc += s["cited"]
    if tn:
        out.append(
            f"| **all** | **{tn}** | **{tc}** | **{ts}** | **{tw}** | **{ta}** | **{tnone}** | **{100*ts/tn:.0f}%** |"
        )
    out.append("")

    out.append("## Reference: agent-proposed self-edit baseline (exp1)")
    out.append("")
    out.append("- 29 round-1 edits, 19/29 (66%) STRONG, 23/29 (79%) cited.")
    out.append("- This is what compiler edits should approximate to validate the M5 primitive end-to-end.")
    out.append("")

    if failures:
        out.append("## Compiler edits that didn't propagate (NONE class)")
        out.append("")
        for judge, e, r in failures:
            out.append(
                f"- **{judge}/{e['edit_id']}** → `{e['target_statement_id']}` on "
                f"`{e['test_pair']['pair_id']} tp={e['test_pair']['tension_point_idx']}`: "
                f"BAD Δ={r['bad_change']:.2f}, alt Δ={r['alt_change']:.2f}, WE Δ={r['we_change']:.2f}, cited=False"
            )
        out.append("")

    OUT.write_text("\n".join(out))
    print(f"Wrote {OUT}")
    if tn:
        print(f"\nHeadline: {ts}/{tn} ({100*ts/tn:.0f}%) STRONG, {tc}/{tn} ({100*tc/tn:.0f}%) cited.")


if __name__ == "__main__":
    main()
