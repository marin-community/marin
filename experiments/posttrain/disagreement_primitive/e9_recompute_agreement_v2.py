"""Recompute ensemble agreement on the 5 affected statements using:
  - GPT-5.1 phase_4 = NEW v2-rubric judgments (per_judgment_v2.jsonl)
  - Gemini phase_4  = EXISTING v1-rubric judgments (no new API calls)
  - Claude phase_4  = EXISTING v1-rubric judgments (no new API calls)
  - All bare-condition judgments unchanged (rubric only affects phase_4)

Compares v1 vs v2 ensemble agreement under phase_4. Reports:
  - Per-statement Δα change: α_phase4(v2 mix) − α_phase4(v1 mix)
  - Per-statement top-K Δpwv change (did the poison cells improve?)
  - GPT outlier rate within top-K poison cells (was the gap closed?)

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
from e9_rubric_poison_rank import rank_rubric_poison, per_anchor_frequency, JUDGES, pairwise_variance, outlier_judge

DIR = Path("experiments/posttrain/disagreement_primitive")
PER_JUDGMENT_V2 = DIR / "per_judgment_v2.jsonl"

DEFAULT_HURT = [
    "refusal_style",
    "no_agenda",
    "comply_with_laws",
    "avoid_abuse",
    "sexual_content_involving_minors",
]


def load_v2_gpt_phase4() -> dict[tuple, int]:
    """Load NEW GPT-5.1 phase_4 judgments under v2 rubric.
    Returns (sid, scenario_idx, generator) -> {score, reasoning}."""
    out = {}
    if not PER_JUDGMENT_V2.exists():
        return out
    for line in PER_JUDGMENT_V2.open():
        r = json.loads(line)
        s = r.get("score")
        if not isinstance(s, int) or not 1 <= s <= 5:
            continue
        if r.get("judge") != "gpt" or r.get("condition") != "rubric_plus_spec":
            continue
        out[(r["statement_id"], r["scenario_idx"], r["generator"])] = {
            "score": s, "reasoning": r.get("reasoning"),
        }
    return out


def build_v2_by_cell(by_cell_v1: dict, gpt_v2: dict) -> dict:
    """Splice new GPT v2 phase_4 judgments into the v1 by_cell, leaving
    Gemini + Claude untouched. Bare condition unchanged."""
    out = dict(by_cell_v1)  # shallow copy of cells dict
    for (sid, cond, scen, gen), jd in by_cell_v1.items():
        if cond != "rubric_plus_spec":
            continue
        v2 = gpt_v2.get((sid, scen, gen))
        if v2 is None:
            continue
        # Replace gpt entry with v2 version, keep gemini + claude as v1
        new_jd = dict(jd)
        new_jd["gpt"] = v2
        out[(sid, cond, scen, gen)] = new_jd
    return out


def stmt_metrics(by_cell: dict, sid: str, condition: str) -> dict[str, Any]:
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


def main() -> int:
    print("Loading v1 (existing) judgments...")
    by_cell_v1 = load_judgments()
    print(f"  v1 cells: {len(by_cell_v1)}")

    print("Loading v2 GPT-5.1 phase_4 judgments...")
    gpt_v2 = load_v2_gpt_phase4()
    print(f"  v2 GPT phase_4 rows: {len(gpt_v2)}")

    by_cell_v2 = build_v2_by_cell(by_cell_v1, gpt_v2)

    print("\n# v1 vs v2 ensemble agreement on the 5 affected statements\n")
    print("| statement | n_p4 | α_p4 (v1) | α_p4 (v2) | Δα (v2−v1) | k3_p4 (v1) | k3_p4 (v2) | k2_p4 (v1) | k2_p4 (v2) |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|")

    sum_d_alpha = 0.0
    n_cmp = 0
    for sid in DEFAULT_HURT:
        m_v1 = stmt_metrics(by_cell_v1, sid, "rubric_plus_spec")
        m_v2 = stmt_metrics(by_cell_v2, sid, "rubric_plus_spec")
        if m_v1["alpha"] is None or m_v2["alpha"] is None:
            print(f"| {sid} | {m_v1['n']} | {fmt(m_v1['alpha'])} | {fmt(m_v2['alpha'])} | n/a | {fmt(m_v1['k3'])} | {fmt(m_v2['k3'])} | {fmt(m_v1['k2'])} | {fmt(m_v2['k2'])} |")
            continue
        d_alpha = m_v2["alpha"] - m_v1["alpha"]
        sum_d_alpha += d_alpha
        n_cmp += 1
        print(f"| {sid} | {m_v2['n']} | {fmt(m_v1['alpha'])} | {fmt(m_v2['alpha'])} | {d_alpha:+.3f} | {fmt(m_v1['k3'])} | {fmt(m_v2['k3'])} | {fmt(m_v1['k2'])} | {fmt(m_v2['k2'])} |")

    if n_cmp:
        print(f"\nMean Δα (phase_4, v2−v1) across {n_cmp} statements: {sum_d_alpha/n_cmp:+.3f}")

    # Bare unchanged for reference
    print("\n# Sanity check — bare-condition agreement (should be identical, since v2 only affects phase_4)\n")
    print("| statement | n | α_bare (v1) | α_bare (v2) |")
    print("|---|--:|--:|--:|")
    for sid in DEFAULT_HURT:
        b_v1 = stmt_metrics(by_cell_v1, sid, "variant_A")
        b_v2 = stmt_metrics(by_cell_v2, sid, "variant_A")
        print(f"| {sid} | {b_v1['n']} | {fmt(b_v1['alpha'])} | {fmt(b_v2['alpha'])} |")

    # Within-statement: top-K rubric-poison cells under v1 → did Δpwv drop under v2?
    print("\n# Top-K rubric-poison cells under v1 — does Δpwv drop under v2?\n")
    print("| statement | total Δpwv (v1 top-12) | total Δpwv (same cells, v2) | drop |")
    print("|---|--:|--:|--:|")
    for sid in DEFAULT_HURT:
        rows_v1 = rank_rubric_poison(by_cell_v1, sid)
        top_keys = [(r["scen"], r["gen"]) for r in rows_v1[:12]]
        # Compute Δpwv under v1 for these cells
        v1_total = sum(r["delta_pwv"] for r in rows_v1[:12])
        # Compute Δpwv under v2 for the SAME cells
        v2_total = 0
        for scen, gen in top_keys:
            bare_jd = by_cell_v2.get((sid, "variant_A", scen, gen))
            rub_jd = by_cell_v2.get((sid, "rubric_plus_spec", scen, gen))
            if not bare_jd or not rub_jd: continue
            if not all(j in bare_jd for j in JUDGES) or not all(j in rub_jd for j in JUDGES): continue
            bs = [bare_jd[j]["score"] for j in JUDGES]
            rs = [rub_jd[j]["score"] for j in JUDGES]
            v2_total += pairwise_variance(rs) - pairwise_variance(bs)
        drop_pct = 100 * (v1_total - v2_total) / v1_total if v1_total != 0 else 0
        print(f"| {sid} | {v1_total:+d} | {v2_total:+d} | {drop_pct:+.0f}% |")

    # GPT outlier rate within top-K poison cells (v1) under v2
    print("\n# GPT outlier rate within top-K poison cells — v1 vs v2 (rubric condition)\n")
    print("| statement | gpt outlier rate (v1) | gpt outlier rate (v2) |")
    print("|---|--:|--:|")
    for sid in DEFAULT_HURT:
        rows_v1 = rank_rubric_poison(by_cell_v1, sid)
        top_keys = [(r["scen"], r["gen"]) for r in rows_v1[:12]]
        v1_gpt_outlier = sum(1 for r in rows_v1[:12]
                             if r["outlier_rubric"] and r["outlier_rubric"][0] == "gpt")
        # Compute under v2 for same cells
        v2_gpt_outlier = 0
        v2_total_with_outlier = 0
        for scen, gen in top_keys:
            rub_jd = by_cell_v2.get((sid, "rubric_plus_spec", scen, gen))
            if not rub_jd or not all(j in rub_jd for j in JUDGES): continue
            scores_dict = {j: rub_jd[j]["score"] for j in JUDGES}
            o = outlier_judge(scores_dict)
            if o:
                v2_total_with_outlier += 1
                if o[0] == "gpt":
                    v2_gpt_outlier += 1
        v1_pct = 100 * v1_gpt_outlier / 12
        v2_pct = 100 * v2_gpt_outlier / max(1, v2_total_with_outlier) if v2_total_with_outlier else 0
        print(f"| {sid} | {v1_gpt_outlier}/12 ({v1_pct:.0f}%) | {v2_gpt_outlier}/{v2_total_with_outlier} ({v2_pct:.0f}%) |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
