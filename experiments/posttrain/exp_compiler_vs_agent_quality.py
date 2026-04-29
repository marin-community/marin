# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare compiler-edit propagation vs agent-edit propagation, paired by source.

Each compiler edit carries `_source_agent_edit_id`, linking it to the agent edit
that produced its NL diagnosis input. So for each (judge, source_round, source_id):
- agent edit has a propagation classification (STRONG/WEAK/AMBIG/NONE) from
  exp1 (round-1 self-edits) or implicitly the round-2/round-3 propagation runs.
- compiler edit has a classification computed against
  cross_tier_rubrics_v2_<judge>_with_compiler_edits.jsonl.

This script restricts to ROUND-1 source edits (where we have a clean baseline
vs with-self-edits comparison) and pairs them.

Output: a 4x4 confusion matrix per judge + cross-judge aggregate showing
how often the compiler matches/exceeds/falls-short-of agent edit quality on
the same test_pair.

Usage:
    uv run python experiments/posttrain/exp_compiler_vs_agent_quality.py
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
STAGE3 = WORKTREE / "experiments/posttrain/stage3_output"
LM_JUDGE_EDITS = WORKTREE / "experiments/posttrain/lm_judge_edits"
LM_COMPILER_EDITS = WORKTREE / "experiments/posttrain/lm_compiler_proposed_edits"
OUT = STAGE3 / "exp_compiler_vs_agent_quality.md"

JUDGES = ["flash", "gpt51", "pro", "glm51"]
CHANGE_THRESHOLD = 0.3
CLASSES = ["STRONG", "WEAK", "AMBIG", "NONE"]


def sim_change(a: str, b: str) -> float:
    return 1.0 - SequenceMatcher(None, a or "", b or "").ratio()


def citation_fires(rationale_clauses: list[str], new_example: dict) -> bool:
    fields = [
        new_example.get("description", "") or "",
        new_example.get("user_query", "") or "",
        new_example.get("good_response", "") or "",
        new_example.get("bad_response", "") or "",
    ]
    for clause in rationale_clauses:
        c_lower = (clause or "").lower()
        if len(c_lower) < 20:
            continue
        for field in fields:
            f_lower = field.lower()
            if not f_lower:
                continue
            if c_lower in f_lower:
                return True
            for start in range(0, max(0, len(c_lower) - 40), 20):
                window = c_lower[start : start + 60]
                if window and len(window) >= 40 and window in f_lower:
                    return True
    return False


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def index_by_pair(rows: list[dict]) -> dict[tuple, dict]:
    return {(r["pair_id"], r["tension_point_idx"]): r for r in rows}


def baseline_path_for(judge: str) -> Path:
    return STAGE3 / ("cross_tier_rubrics_v2.jsonl" if judge == "flash" else f"cross_tier_rubrics_v2_{judge}.jsonl")


def classify(edit: dict, baseline: dict | None, withedit: dict | None) -> str:
    if baseline is None or withedit is None:
        return "MISSING"
    b_p = baseline["parsed"]
    w_p = withedit["parsed"]

    new_clauses = w_p.get("rationale", {}).get("spec_clauses_anchored_on", []) or []
    cited = citation_fires(new_clauses, edit["new_example"])

    def s(a, b):
        return sim_change(a, b)

    bad_d = s((b_p.get("dominant_rubric") or {}).get("BAD", ""), (w_p.get("dominant_rubric") or {}).get("BAD", ""))
    alt_d = s(
        (b_p.get("rationale") or {}).get("alternative_readings_rejected", ""),
        (w_p.get("rationale") or {}).get("alternative_readings_rejected", ""),
    )
    we_d = s(
        (b_p.get("worked_example") or {}).get("spec_compliant", ""),
        (w_p.get("worked_example") or {}).get("spec_compliant", ""),
    )
    sig = sum(1 for x in [bad_d, alt_d, we_d] if x > CHANGE_THRESHOLD)
    if cited and sig >= 1:
        return "STRONG"
    if cited:
        return "WEAK"
    if sig >= 2:
        return "AMBIG"
    return "NONE"


def load_agent_edits_round1(judge: str) -> list[dict]:
    d = LM_JUDGE_EDITS / judge / "proposed_edits"
    return [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]


def load_compiler_edits_with_round1_source(judge: str) -> list[dict]:
    d = LM_COMPILER_EDITS / judge
    out = []
    for f in sorted(d.glob("*.json")):
        e = json.loads(f.read_text())
        if e.get("_source_round") == "round1":
            out.append(e)
    return out


def main() -> None:
    out: list[str] = []
    out.append("# Experiment: compiler-edit vs agent-edit propagation quality (paired)")
    out.append("")
    out.append(
        "For each round-1 agent edit and the compiler edit that took it as NL-diagnosis input, "
        "compare propagation classification on the same test_pair. Tests whether the compiler "
        "matches agent-quality edit *output* (independent of the 85% target_statement match)."
    )
    out.append("")
    out.append(
        "**Classes**: STRONG (cited + ≥1 text change), WEAK (cited only), AMBIG (text changed, no citation), NONE."
    )
    out.append("")
    out.append("---")
    out.append("")

    grand: dict[tuple[str, str], int] = {(a, c): 0 for a in CLASSES for c in CLASSES}

    for judge in JUDGES:
        baseline_path = baseline_path_for(judge)
        self_path = STAGE3 / f"cross_tier_rubrics_v2_{judge}_with_self_edits.jsonl"
        comp_path = STAGE3 / f"cross_tier_rubrics_v2_{judge}_with_compiler_edits.jsonl"
        if not (baseline_path.exists() and self_path.exists() and comp_path.exists()):
            out.append(f"## Judge `{judge}` — MISSING files")
            out.append("")
            continue
        baseline = index_by_pair(load_jsonl(baseline_path))
        self_idx = index_by_pair(load_jsonl(self_path))
        comp_idx = index_by_pair(load_jsonl(comp_path))

        agent_edits = {e["edit_id"]: e for e in load_agent_edits_round1(judge)}
        compiler_edits = load_compiler_edits_with_round1_source(judge)

        out.append(f"## Judge: `{judge}` ({len(compiler_edits)} round-1-sourced compiler edits)")
        out.append("")
        out.append(
            "| source_id | test_pair | agent_target | compiler_target | match? | agent class | compiler class | verdict |"
        )
        out.append("|---|---|---|---|:---:|---|---|---|")

        cell: dict[tuple[str, str], int] = {(a, c): 0 for a in CLASSES for c in CLASSES}
        for ce in compiler_edits:
            src_id = ce.get("_source_agent_edit_id")
            ae = agent_edits.get(src_id)
            if ae is None:
                continue
            tp = (ae["test_pair"]["pair_id"], ae["test_pair"]["tension_point_idx"])
            agent_cls = classify(ae, baseline.get(tp), self_idx.get(tp))
            comp_cls = classify(ce, baseline.get(tp), comp_idx.get(tp))
            target_match_mark = "✓" if ce.get("_target_match_with_agent") else ""

            if agent_cls in CLASSES and comp_cls in CLASSES:
                cell[(agent_cls, comp_cls)] += 1
                grand[(agent_cls, comp_cls)] += 1

            order = {"STRONG": 3, "WEAK": 2, "AMBIG": 1, "NONE": 0, "MISSING": -1}
            if order[comp_cls] > order[agent_cls]:
                verdict = "**↑ exceeds**"
            elif order[comp_cls] == order[agent_cls]:
                verdict = "= matches"
            else:
                verdict = "↓ short"

            out.append(
                f"| `{src_id}` | `{tp[0]} tp={tp[1]}` | `{ae['target_statement_id']}` | "
                f"`{ce['target_statement_id']}` | {target_match_mark} | {agent_cls} | {comp_cls} | {verdict} |"
            )

        out.append("")
        out.append("**Confusion matrix** (rows = agent class, cols = compiler class):")
        out.append("")
        out.append("| agent → / compiler ↓ | " + " | ".join(CLASSES) + " | total |")
        out.append("|---|" + "|".join("---:" for _ in CLASSES) + "|---:|")
        for a in CLASSES:
            row_total = sum(cell[(a, c)] for c in CLASSES)
            out.append(f"| **{a}** | " + " | ".join(str(cell[(a, c)]) for c in CLASSES) + f" | {row_total} |")
        col_totals = [sum(cell[(a, c)] for a in CLASSES) for c in CLASSES]
        out.append("| **total** | " + " | ".join(str(t) for t in col_totals) + " | |")
        out.append("")
        n = sum(cell.values())
        diag = sum(cell[(c, c)] for c in CLASSES)
        better = sum(
            cell[(a, c)]
            for a in CLASSES
            for c in CLASSES
            if {"STRONG": 3, "WEAK": 2, "AMBIG": 1, "NONE": 0}[c] > {"STRONG": 3, "WEAK": 2, "AMBIG": 1, "NONE": 0}[a]
        )
        worse = sum(
            cell[(a, c)]
            for a in CLASSES
            for c in CLASSES
            if {"STRONG": 3, "WEAK": 2, "AMBIG": 1, "NONE": 0}[c] < {"STRONG": 3, "WEAK": 2, "AMBIG": 1, "NONE": 0}[a]
        )
        if n:
            out.append(f"- exact match: {diag}/{n} ({100*diag/n:.0f}%)")
            out.append(f"- compiler exceeds agent: {better}/{n} ({100*better/n:.0f}%)")
            out.append(f"- compiler falls short: {worse}/{n} ({100*worse/n:.0f}%)")
        out.append("")

    out.append("---")
    out.append("")
    out.append("## Cross-judge aggregate confusion matrix")
    out.append("")
    out.append("| agent → / compiler ↓ | " + " | ".join(CLASSES) + " | total |")
    out.append("|---|" + "|".join("---:" for _ in CLASSES) + "|---:|")
    for a in CLASSES:
        rt = sum(grand[(a, c)] for c in CLASSES)
        out.append(f"| **{a}** | " + " | ".join(str(grand[(a, c)]) for c in CLASSES) + f" | {rt} |")
    col_totals = [sum(grand[(a, c)] for a in CLASSES) for c in CLASSES]
    out.append("| **total** | " + " | ".join(str(t) for t in col_totals) + " | |")
    out.append("")
    n = sum(grand.values())
    if n:
        diag = sum(grand[(c, c)] for c in CLASSES)
        order = {"STRONG": 3, "WEAK": 2, "AMBIG": 1, "NONE": 0}
        better = sum(grand[(a, c)] for a in CLASSES for c in CLASSES if order[c] > order[a])
        worse = sum(grand[(a, c)] for a in CLASSES for c in CLASSES if order[c] < order[a])
        out.append(f"- exact match: {diag}/{n} ({100*diag/n:.0f}%)")
        out.append(f"- compiler exceeds agent: {better}/{n} ({100*better/n:.0f}%)")
        out.append(f"- compiler falls short: {worse}/{n} ({100*worse/n:.0f}%)")
        out.append("")
        agent_strong = sum(grand[("STRONG", c)] for c in CLASSES)
        comp_strong = sum(grand[(a, "STRONG")] for a in CLASSES)
        out.append(f"- agent-STRONG: {agent_strong}/{n} ({100*agent_strong/n:.0f}%)")
        out.append(f"- compiler-STRONG: {comp_strong}/{n} ({100*comp_strong/n:.0f}%)")
        out.append("")
        out.append("**Headline interpretation**:")
        if comp_strong >= agent_strong:
            out.append(
                f"- Compiler matches or exceeds agent on STRONG-propagation rate ({comp_strong} vs {agent_strong})."
            )
            out.append("- M5 primitive viable: an LM-compiled edit propagates as well as a hand-curated one.")
        else:
            out.append(f"- Compiler underperforms agent on STRONG-propagation ({comp_strong} vs {agent_strong}).")
            out.append("- M5 primitive needs work: examples need richer style or sharper focus.")

    OUT.write_text("\n".join(out))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
