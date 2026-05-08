"""Analyze the 0-6 pilot results and compare to the 1-5 baseline.

Loads:
  - per_judgment_0_6.jsonl (gpt + gemini under 0-6, both conditions, 2 statements)
  - claude_judge_v0_0_6/<sid>/{bare,phase_4}_0_6_claude.jsonl
  - For comparison: per_judgment.jsonl (1-5 baseline) + claude_judge_v0/<sid>/

Computes:
  - Per-judge marginal distributions under 0-6 (each statement, each condition)
  - Trivial-clean rate (% all 3 judges in {5,6}) and ceiling-watch (% all 3 in {6})
  - Krippendorff α (interval) on 0-6 ratings
  - k2 binary collapse: {0,1,2} | {3,4,5,6}
  - k4 4-way collapse: {0,1,2} | {3} | {4,5} | {6}
  - Side-by-side vs 1-5 baseline metrics on the same 2 statements

Output: markdown to stdout.
"""

from __future__ import annotations

import json
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from e9_kappa_alternatives import fleiss_kappa, krippendorff_alpha_interval, to_table

DIR = Path("experiments/posttrain/disagreement_primitive")
PER_JUDGMENT_06 = DIR / "per_judgment_0_6.jsonl"
PER_JUDGMENT_15 = DIR / "grounding" / "per_judgment.jsonl"
CLAUDE_06_DIR = DIR / "claude_judge_v0_0_6"
CLAUDE_15_DIR = DIR / "claude_judge_v0"

PILOT_SIDS = ["no_erotica_or_gore", "no_topic_off_limits"]

# Score collapses for the 0-6 scale
def cat_k2_06(s: int) -> int:
    return 1 if s in (0, 1, 2) else 0  # 1 = problematic, 0 = borderline-or-clean

def cat_k4_06(s: int) -> int:
    if s in (0, 1, 2): return 0
    if s == 3: return 1
    if s in (4, 5): return 2
    return 3  # s == 6

# 1-5 collapses (matching e9_kappa_alternatives)
def cat_k2_15(s: int) -> int:
    return 1 if s in (1, 2) else 0

def cat_k3_15(s: int) -> int:
    if s in (1, 2): return 0
    if s == 3: return 1
    return 2


def load_three_judge_06(sid: str, condition: str) -> list[tuple[int, int, int]]:
    """Return (gpt, gemini, claude) tuples for cells where all 3 are scored under 0-6."""
    cond_internal = "variant_A_0_6" if condition == "bare" else "rubric_plus_spec_0_6"
    by_cell: dict[tuple, dict[str, int]] = defaultdict(dict)
    for line in PER_JUDGMENT_06.open():
        r = json.loads(line)
        if r.get("statement_id") != sid or r.get("condition") != cond_internal:
            continue
        s = r.get("score")
        if s is None or not 0 <= s <= 6:
            continue
        cell = (r["scenario_idx"], r["generator"])
        by_cell[cell][r["judge"]] = s
    out = []
    for cell, jd in by_cell.items():
        if all(j in jd for j in ("gpt", "gemini", "claude")):
            out.append((jd["gpt"], jd["gemini"], jd["claude"]))
    return out


def load_three_judge_15(sid: str, condition: str) -> list[tuple[int, int, int]]:
    """1-5 baseline: load (gpt, gemini, claude) tuples on the same statements."""
    cond_internal = "variant_A" if condition == "bare" else "rubric_plus_spec"
    by_cell: dict[tuple, dict[str, int]] = defaultdict(dict)
    for line in PER_JUDGMENT_15.open():
        r = json.loads(line)
        if r.get("statement_id") != sid or r.get("condition") != cond_internal:
            continue
        if r.get("judge") not in ("gpt", "gemini"):
            continue
        s = r.get("score")
        try:
            s = int(s) if s is not None else None
        except (TypeError, ValueError):
            s = None
        if s is None:
            continue
        cell = (r["scenario_idx"], r["generator"])
        by_cell[cell][r["judge"]] = s

    cond_short = "bare" if condition == "bare" else "phase_4"
    p = CLAUDE_15_DIR / sid / f"{cond_short}_claude.jsonl"
    if p.exists():
        for line in p.open():
            r = json.loads(line)
            if r.get("score") is None:
                continue
            s = r["score"]
            if not isinstance(s, int) or not 1 <= s <= 5:
                continue
            cell = (r["scenario_idx"], r["generator"])
            by_cell[cell]["claude"] = s

    out = []
    for cell, jd in by_cell.items():
        if all(j in jd for j in ("gpt", "gemini", "claude")):
            out.append((jd["gpt"], jd["gemini"], jd["claude"]))
    return out


def per_judge_distribution_06(sid: str, condition: str) -> dict[str, Counter]:
    cond_internal = "variant_A_0_6" if condition == "bare" else "rubric_plus_spec_0_6"
    out = defaultdict(Counter)
    for line in PER_JUDGMENT_06.open():
        r = json.loads(line)
        if r.get("statement_id") != sid or r.get("condition") != cond_internal:
            continue
        s = r.get("score")
        if s is None:
            continue
        out[r["judge"]][s] += 1
    return out


def per_judge_distribution_15(sid: str, condition: str) -> dict[str, Counter]:
    cond_internal = "variant_A" if condition == "bare" else "rubric_plus_spec"
    out = defaultdict(Counter)
    for line in PER_JUDGMENT_15.open():
        r = json.loads(line)
        if r.get("statement_id") != sid or r.get("condition") != cond_internal:
            continue
        if r.get("judge") not in ("gpt", "gemini"):
            continue
        try:
            s = int(r["score"]) if r["score"] is not None else None
        except (TypeError, ValueError):
            s = None
        if s is not None:
            out[r["judge"]][s] += 1

    cond_short = "bare" if condition == "bare" else "phase_4"
    p = CLAUDE_15_DIR / sid / f"{cond_short}_claude.jsonl"
    if p.exists():
        for line in p.open():
            r = json.loads(line)
            if r.get("score") is not None and isinstance(r["score"], int):
                out["claude"][r["score"]] += 1
    return out


def fmt(v):
    return "n/a   " if v is None else f"{v:+.3f}"


def compute_metrics_06(tuples: list[tuple[int, int, int]]) -> dict:
    if len(tuples) < 2:
        return {"n": len(tuples), "k2": None, "k4": None, "alpha": None}
    return {
        "n": len(tuples),
        "k2": fleiss_kappa(to_table(tuples, cat_k2_06), k=2),
        "k4": fleiss_kappa(to_table(tuples, cat_k4_06), k=4),
        "alpha": krippendorff_alpha_interval(tuples),
    }


def compute_metrics_15(tuples: list[tuple[int, int, int]]) -> dict:
    if len(tuples) < 2:
        return {"n": len(tuples), "k2": None, "k3": None, "alpha": None}
    return {
        "n": len(tuples),
        "k2": fleiss_kappa(to_table(tuples, cat_k2_15), k=2),
        "k3": fleiss_kappa(to_table(tuples, cat_k3_15), k=3),
        "alpha": krippendorff_alpha_interval(tuples),
    }


def trivial_clean_06(tuples: list[tuple[int, int, int]]) -> tuple[float, float]:
    if not tuples:
        return (0.0, 0.0)
    n = len(tuples)
    n_5_6 = sum(1 for t in tuples if all(s in (5, 6) for s in t))
    n_6 = sum(1 for t in tuples if all(s == 6 for s in t))
    return (100 * n_5_6 / n, 100 * n_6 / n)


def trivial_clean_15(tuples: list[tuple[int, int, int]]) -> float:
    if not tuples:
        return 0.0
    n_4_5 = sum(1 for t in tuples if all(s in (4, 5) for s in t))
    return 100 * n_4_5 / len(tuples)


def main():
    print("# Option A 0-6 pilot — analysis vs 1-5 baseline\n")
    print(f"Pilot statements: {PILOT_SIDS}")
    print(f"Conditions: bare-0-6 (var_A_0_6), phase_4-0-6 (rubric_plus_spec_0_6)")
    print(f"Judges: gpt-5.1, gemini-3-flash, claude-sonnet-4-6 (tool-use forced)\n")

    print("## Per-judge marginal distributions\n")
    for sid in PILOT_SIDS:
        for cond in ("bare", "phase_4"):
            print(f"### {sid} — {cond}")
            print()
            print("0-6:")
            d06 = per_judge_distribution_06(sid, cond)
            print("```")
            for j in ("gpt", "gemini", "claude"):
                if j in d06:
                    bins = "  ".join(f"{s}:{d06[j].get(s, 0):>2}" for s in range(7))
                    print(f"  {j:7s}  [{bins}]")
            print("```")
            print()
            print("1-5 (baseline):")
            d15 = per_judge_distribution_15(sid, cond)
            print("```")
            for j in ("gpt", "gemini", "claude"):
                if j in d15:
                    bins = "  ".join(f"{s}:{d15[j].get(s, 0):>2}" for s in range(1, 6))
                    print(f"  {j:7s}  [{bins}]")
            print("```")
            print()

    print("## Headline diagnostics — did Option A unstack the ceiling?\n")
    print("| statement | cond | trivial-clean (0-6: all in 5-6) | ceiling-watch (0-6: all = 6) | trivial-clean (1-5: all in 4-5) |")
    print("|---|---|--:|--:|--:|")
    for sid in PILOT_SIDS:
        for cond in ("bare", "phase_4"):
            t06 = load_three_judge_06(sid, cond)
            t15 = load_three_judge_15(sid, cond)
            tc06, ceiling = trivial_clean_06(t06)
            tc15 = trivial_clean_15(t15)
            print(f"| {sid} | {cond} | {tc06:.1f}% | {ceiling:.1f}% | {tc15:.1f}% |")

    print()
    print("## Agreement metrics — 0-6 vs 1-5 (same statements)\n")
    print("| statement | cond | n_06 | α_06 | k2_06 | k4_06 | n_15 | α_15 | k2_15 | k3_15 |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    rows_06 = {}
    rows_15 = {}
    for sid in PILOT_SIDS:
        for cond in ("bare", "phase_4"):
            t06 = load_three_judge_06(sid, cond)
            t15 = load_three_judge_15(sid, cond)
            m06 = compute_metrics_06(t06)
            m15 = compute_metrics_15(t15)
            rows_06[(sid, cond)] = m06
            rows_15[(sid, cond)] = m15
            print(f"| {sid} | {cond} | {m06['n']} | {fmt(m06['alpha'])} | {fmt(m06['k2'])} | {fmt(m06['k4'])} | "
                  f"{m15['n']} | {fmt(m15['alpha'])} | {fmt(m15['k2'])} | {fmt(m15['k3'])} |")

    print()
    print("## Δ within each scale (phase_4 − bare)\n")
    print("| statement | Δα (0-6) | Δk2 (0-6) | Δk4 (0-6) | Δα (1-5) | Δk2 (1-5) | Δk3 (1-5) |")
    print("|---|--:|--:|--:|--:|--:|--:|")
    for sid in PILOT_SIDS:
        b06 = rows_06[(sid, "bare")]
        p06 = rows_06[(sid, "phase_4")]
        b15 = rows_15[(sid, "bare")]
        p15 = rows_15[(sid, "phase_4")]
        def d(a, b, k):
            va, vb = a.get(k), b.get(k)
            return None if va is None or vb is None else vb - va
        print(f"| {sid} | {fmt(d(b06,p06,'alpha'))} | {fmt(d(b06,p06,'k2'))} | {fmt(d(b06,p06,'k4'))} | "
              f"{fmt(d(b15,p15,'alpha'))} | {fmt(d(b15,p15,'k2'))} | {fmt(d(b15,p15,'k3'))} |")

    print()
    print("## Verdict criteria\n")
    print("Option A is working if:")
    print("- ceiling-watch (% all=6) is **low** (target < 20%) — 6 isn't the new ceiling")
    print("- trivial-clean (0-6: % all in 5-6) is **lower** than the 1-5 baseline trivial-clean (% all in 4-5)")
    print("- Δk4 (0-6) ≥ 0 on both pilot statements — the rubric is no longer introducing fine-grained disagreement")
    print()
    print("If ceiling-watch is high (≥ 50%): Option A failed; 6 became the new 5.")
    print("If trivial-clean (0-6) is similar to 1-5: anchors aren't differentiating cases enough.")
    print("If Δk4 (0-6) < 0: rubric still introduces disagreement on borderline cases — bigger problem than scale.")


if __name__ == "__main__":
    main()
