"""Recompute ensemble agreement after re-judging Gemini + Claude with v2 rubrics
on the 3 previously-unfixed statements.

Compares 3 conditions per statement:
  - mix_v1: all 3 judges with v1 rubric (existing data)
  - mix_v2_partial: GPT v2 + Gemini v1 + Claude v1 (the original v2.0 experiment)
  - mix_v2_full: all 3 judges with v2 rubric (the v2.5 experiment, this script)

Question: does mix_v2_full close the gap left by mix_v2_partial?
  - If yes: the "frozen judges" hypothesis is correct; Gemini/Claude SHIFT under
    v2 prompting and the rubric IS fixable
  - If no: the disagreement is genuinely irreducible spec-text ambiguity; rubric
    cannot resolve it; drop the rubric for these statements

Also reports: per-judge score distribution shift v1 → v2 (did Gemini and Claude
actually change their scores in response to v2 anchors?).

No API calls. Pure stdlib.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from e9_kappa_alternatives import binary_cat, fleiss_kappa, krippendorff_alpha_interval, threeway_cat, to_table
from e9_build_qualitative_inputs import load_judgments

DIR = Path("experiments/posttrain/disagreement_primitive")
PER_JUDGMENT_V2 = DIR / "per_judgment_v2.jsonl"

UNFIXED = ["no_agenda", "comply_with_laws", "sexual_content_involving_minors"]
JUDGES = ("gpt", "gemini", "claude")


def load_v2_phase4_by_judge():
    """Load all v2 phase_4 judgments. Returns:
       (sid, scen, gen) -> {judge: {score, reasoning}}"""
    out = defaultdict(dict)
    if not PER_JUDGMENT_V2.exists():
        return out
    for line in PER_JUDGMENT_V2.open():
        r = json.loads(line)
        s = r.get("score")
        if not isinstance(s, int) or not 1 <= s <= 5: continue
        if r.get("condition") != "rubric_plus_spec": continue
        if r.get("rubric_version") != "v2": continue
        out[(r["statement_id"], r["scenario_idx"], r["generator"])][r["judge"]] = {
            "score": s, "reasoning": r.get("reasoning") or "",
        }
    return out


def stmt_metrics(by_cell, sid: str, condition: str) -> dict[str, Any]:
    tuples = []
    for (s, c, scen, gen), jd in by_cell.items():
        if s != sid or c != condition: continue
        if not all(j in jd for j in JUDGES): continue
        tuples.append((jd["gpt"]["score"], jd["gemini"]["score"], jd["claude"]["score"]))
    if len(tuples) < 2:
        return {"n": len(tuples), "k2": None, "k3": None, "alpha": None}
    return {
        "n": len(tuples),
        "alpha": krippendorff_alpha_interval(tuples),
        "k3": fleiss_kappa(to_table(tuples, threeway_cat), 3),
        "k2": fleiss_kappa(to_table(tuples, binary_cat), 2),
    }


def fmt(v):
    return "n/a   " if v is None else f"{v:+.3f}"


def main():
    print("Loading v1 (existing) judgments...")
    by_cell_v1 = load_judgments()
    print(f"  v1 cells: {len(by_cell_v1)}")

    print("Loading v2 phase_4 judgments...")
    v2_by_cell = load_v2_phase4_by_judge()
    print(f"  v2 phase_4 cells: {len(v2_by_cell)}")

    # mix_v2_partial: replace gpt under phase_4 with v2 if available, keep gemini+claude as v1
    by_cell_v2_partial = dict(by_cell_v1)
    for (sid, cond, scen, gen), jd in by_cell_v1.items():
        if cond != "rubric_plus_spec": continue
        v2_judges = v2_by_cell.get((sid, scen, gen), {})
        if "gpt" in v2_judges:
            new_jd = dict(jd)
            new_jd["gpt"] = v2_judges["gpt"]
            by_cell_v2_partial[(sid, cond, scen, gen)] = new_jd

    # mix_v2_full: replace ALL judges under phase_4 with v2 if available
    by_cell_v2_full = dict(by_cell_v1)
    for (sid, cond, scen, gen), jd in by_cell_v1.items():
        if cond != "rubric_plus_spec": continue
        v2_judges = v2_by_cell.get((sid, scen, gen), {})
        if not v2_judges: continue
        new_jd = dict(jd)
        for j in JUDGES:
            if j in v2_judges:
                new_jd[j] = v2_judges[j]
        by_cell_v2_full[(sid, cond, scen, gen)] = new_jd

    print("\n# Agreement on 3 unfixed statements: v1 vs v2_partial vs v2_full\n")
    print("| statement | n | α (v1) | α (v2_partial: GPT only) | α (v2_full: all 3) | Δ (v2_full − v1) |")
    print("|---|--:|--:|--:|--:|--:|")

    for sid in UNFIXED:
        m_v1 = stmt_metrics(by_cell_v1, sid, "rubric_plus_spec")
        m_partial = stmt_metrics(by_cell_v2_partial, sid, "rubric_plus_spec")
        m_full = stmt_metrics(by_cell_v2_full, sid, "rubric_plus_spec")
        d_full = m_full["alpha"] - m_v1["alpha"] if (m_v1["alpha"] is not None and m_full["alpha"] is not None) else None
        print(f"| {sid} | {m_full['n']} | {fmt(m_v1['alpha'])} | {fmt(m_partial['alpha'])} | {fmt(m_full['alpha'])} | {fmt(d_full)} |")

    # Same for k2, k3
    print("\n## Fleiss κ binary (k2) — same comparison\n")
    print("| statement | k2 (v1) | k2 (v2_partial) | k2 (v2_full) |")
    print("|---|--:|--:|--:|")
    for sid in UNFIXED:
        m_v1 = stmt_metrics(by_cell_v1, sid, "rubric_plus_spec")
        m_partial = stmt_metrics(by_cell_v2_partial, sid, "rubric_plus_spec")
        m_full = stmt_metrics(by_cell_v2_full, sid, "rubric_plus_spec")
        print(f"| {sid} | {fmt(m_v1['k2'])} | {fmt(m_partial['k2'])} | {fmt(m_full['k2'])} |")

    print("\n## Fleiss κ 3-way (k3) — same comparison\n")
    print("| statement | k3 (v1) | k3 (v2_partial) | k3 (v2_full) |")
    print("|---|--:|--:|--:|")
    for sid in UNFIXED:
        m_v1 = stmt_metrics(by_cell_v1, sid, "rubric_plus_spec")
        m_partial = stmt_metrics(by_cell_v2_partial, sid, "rubric_plus_spec")
        m_full = stmt_metrics(by_cell_v2_full, sid, "rubric_plus_spec")
        print(f"| {sid} | {fmt(m_v1['k3'])} | {fmt(m_partial['k3'])} | {fmt(m_full['k3'])} |")

    # Per-judge score-shift analysis: did Gemini and Claude actually change scores?
    print("\n# Per-judge: did Gemini and Claude actually shift under v2?\n")
    print("Counts of cells where each judge's score shifted v1 → v2 (rubric only changed; bare judgments unchanged).\n")
    print("| statement | judge | n shifted | n unchanged | mean Δscore | range Δscore |")
    print("|---|---|--:|--:|--:|---|")

    for sid in UNFIXED:
        for judge in JUDGES:
            shifted = unchanged = 0
            score_diffs = []
            for (s, c, scen, gen), jd in by_cell_v1.items():
                if s != sid or c != "rubric_plus_spec": continue
                v2 = v2_by_cell.get((sid, scen, gen), {}).get(judge)
                if v2 is None: continue
                if judge not in jd: continue  # v1 missing this judge
                v1_score = jd[judge]["score"]
                v2_score = v2["score"]
                if v1_score == v2_score:
                    unchanged += 1
                else:
                    shifted += 1
                    score_diffs.append(v2_score - v1_score)
            if score_diffs:
                mean_d = sum(score_diffs) / len(score_diffs)
                rng_d = f"[{min(score_diffs)}, {max(score_diffs)}]"
            else:
                mean_d = 0
                rng_d = "n/a"
            print(f"| {sid} | {judge} | {shifted} | {unchanged} | {mean_d:+.2f} | {rng_d} |")

    # Sanity: bare unchanged
    print("\n# Sanity check — bare condition (should be identical, v2 only affects phase_4)\n")
    print("| statement | α_bare (v1) | α_bare (v2_full) |")
    print("|---|--:|--:|")
    for sid in UNFIXED:
        b_v1 = stmt_metrics(by_cell_v1, sid, "variant_A")
        b_v2 = stmt_metrics(by_cell_v2_full, sid, "variant_A")
        print(f"| {sid} | {fmt(b_v1['alpha'])} | {fmt(b_v2['alpha'])} |")

    return 0


if __name__ == "__main__":
    main()
