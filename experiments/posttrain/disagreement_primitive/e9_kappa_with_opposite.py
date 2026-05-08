"""Analyze the impact of adding Grok-opposite as a 4th generator on the
8 target statements at 1-5 scale.

Loads:
  - per_judgment.jsonl (existing 3 generators × 3 judges × 2 conditions)
  - per_judgment_opposite.jsonl (Grok-opposite × 3 judges × 2 conditions)
  - claude_judge_v0/<sid>/{bare,phase_4}_claude.jsonl (Claude on existing 3 generators)
  - claude_judge_v0_opposite/<sid>/{bare,phase_4}_opposite_claude.jsonl

Compares:
  1. Score-distribution shift: % ≤ 3 per (generator, condition)
  2. Trivial-clean rate per (generator, condition)
  3. α / k2 / k3 on existing-3-generators only vs all-4-generators
  4. Per-statement: does the picture change?
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from e9_kappa_alternatives import binary_cat, fleiss_kappa, krippendorff_alpha_interval, threeway_cat, to_table

DIR = Path("experiments/posttrain/disagreement_primitive")
PER_JUDGMENT_OLD = DIR / "grounding" / "per_judgment.jsonl"
PER_JUDGMENT_NEW = DIR / "per_judgment_opposite.jsonl"
CLAUDE_OLD_DIR = DIR / "claude_judge_v0"
CLAUDE_NEW_DIR = DIR / "claude_judge_v0_opposite"

TARGET_SIDS = [
    "do_not_make_unprompted_personal_comments",
    "be_professional",
    "no_erotica_or_gore",
    "present_perspectives",
    "be_thorough_but_efficient",
    "avoid_hateful_content",
    "no_topic_off_limits",
    "be_clear",
]
GROK_LABEL = "grok-4-1-fast-non-reasoning-opposite"


def load_all_scores() -> dict:
    """Returns by_cell[(sid, cond_internal, scen, gen)] = {judge: score}."""
    by_cell: dict[tuple, dict[str, int]] = defaultdict(dict)

    # Old per_judgment (gpt + gemini)
    for line in PER_JUDGMENT_OLD.open():
        r = json.loads(line)
        if r.get("statement_id") not in TARGET_SIDS: continue
        if r.get("condition") not in {"variant_A", "rubric_plus_spec"}: continue
        if r.get("judge") not in {"gpt", "gemini"}: continue
        try:
            s = int(r["score"]) if r.get("score") is not None else None
        except (TypeError, ValueError):
            s = None
        if s is None or not 1 <= s <= 5: continue
        cell = (r["statement_id"], r["condition"], r["scenario_idx"], r["generator"])
        by_cell[cell][r["judge"]] = s

    # New per_judgment (Grok-opposite × gpt+gemini+claude)
    for line in PER_JUDGMENT_NEW.open():
        r = json.loads(line)
        if r.get("statement_id") not in TARGET_SIDS: continue
        if r.get("condition") not in {"variant_A", "rubric_plus_spec"}: continue
        s = r.get("score")
        if s is None or not isinstance(s, int) or not 1 <= s <= 5: continue
        cell = (r["statement_id"], r["condition"], r["scenario_idx"], r["generator"])
        by_cell[cell][r["judge"]] = s

    # Old Claude
    for sid in TARGET_SIDS:
        for cond_short, cond_internal in [("bare", "variant_A"), ("phase_4", "rubric_plus_spec")]:
            p = CLAUDE_OLD_DIR / sid / f"{cond_short}_claude.jsonl"
            if not p.exists(): continue
            for line in p.open():
                r = json.loads(line)
                s = r.get("score")
                if s is None or not isinstance(s, int) or not 1 <= s <= 5: continue
                cell = (sid, cond_internal, r["scenario_idx"], r["generator"])
                by_cell[cell]["claude"] = s

    # New Claude (already in per_judgment_new but also have per-statement files)
    # The flat per_judgment_new already contains claude rows; redundant load skipped.

    return by_cell


def per_generator_distribution(by_cell, condition_internal: str) -> dict:
    """For each generator under one condition, return per-judge score histogram."""
    out: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for (sid, cond, scen, gen), jd in by_cell.items():
        if cond != condition_internal: continue
        for judge, s in jd.items():
            out[gen][judge][s] += 1
    return out


def trivial_clean_per_gen(by_cell, condition_internal: str) -> dict:
    """For each generator under one condition: (n_3judge_cells, % all 4-5, % ≤3 by any judge)."""
    out: dict[str, dict[str, float | int]] = {}
    by_gen: dict[str, list[dict]] = defaultdict(list)
    for (sid, cond, scen, gen), jd in by_cell.items():
        if cond != condition_internal: continue
        if len(jd) < 3: continue  # need 3 judges
        by_gen[gen].append(jd)
    for gen, cells in by_gen.items():
        n = len(cells)
        if n == 0:
            continue
        n_clean = sum(1 for c in cells if all(s in (4, 5) for s in c.values()))
        n_anylow = sum(1 for c in cells if any(s <= 3 for s in c.values()))
        n_alllow = sum(1 for c in cells if all(s <= 3 for s in c.values()))
        out[gen] = {
            "n": n,
            "pct_all_4_5": 100 * n_clean / n,
            "pct_any_lo3": 100 * n_anylow / n,
            "pct_all_lo3": 100 * n_alllow / n,
        }
    return out


def metrics_on_subset(by_cell, condition_internal: str, generators: set[str] | None = None) -> dict:
    """Compute Fleiss-2/Fleiss-3/Krippendorff-α on (sid, scen, gen) cells from given gens."""
    tuples = []
    for (sid, cond, scen, gen), jd in by_cell.items():
        if cond != condition_internal: continue
        if generators is not None and gen not in generators: continue
        if len(jd) < 3: continue
        tuples.append((jd["gpt"], jd["gemini"], jd["claude"]))
    if len(tuples) < 2:
        return {"n": len(tuples), "k2": None, "k3": None, "alpha": None}
    return {
        "n": len(tuples),
        "k2": fleiss_kappa(to_table(tuples, binary_cat), 2),
        "k3": fleiss_kappa(to_table(tuples, threeway_cat), 3),
        "alpha": krippendorff_alpha_interval(tuples),
    }


def metrics_per_statement(by_cell, sid: str, condition_internal: str,
                          generators: set[str] | None = None) -> dict:
    tuples = []
    for (s, cond, scen, gen), jd in by_cell.items():
        if s != sid or cond != condition_internal: continue
        if generators is not None and gen not in generators: continue
        if len(jd) < 3: continue
        tuples.append((jd["gpt"], jd["gemini"], jd["claude"]))
    if len(tuples) < 2:
        return {"n": len(tuples), "k2": None, "k3": None, "alpha": None}
    return {
        "n": len(tuples),
        "k2": fleiss_kappa(to_table(tuples, binary_cat), 2),
        "k3": fleiss_kappa(to_table(tuples, threeway_cat), 3),
        "alpha": krippendorff_alpha_interval(tuples),
    }


def fmt(v): return "n/a   " if v is None else f"{v:+.3f}"


def main():
    by_cell = load_all_scores()
    print(f"# Generator-diversity analysis: existing-3 vs +Grok-opposite\n")
    print(f"Loaded {len(by_cell)} cells across (sid × cond × scen × gen).\n")

    EXISTING_GENS = {"gpt-5.1", "Qwen/Qwen2.5-7B-Instruct-Turbo", "gemini-3-flash-preview"}
    ALL_GENS = EXISTING_GENS | {GROK_LABEL}
    GROK_ONLY = {GROK_LABEL}

    # 1. Score distribution per generator per judge (under bare)
    print("## Score distribution per generator under bare condition (variant_A, 1-5)\n")
    print("Per-judge marginal distribution. We expect Grok-opposite to push mass left.\n")
    for cond_internal, label in [("variant_A", "bare"), ("rubric_plus_spec", "phase_4")]:
        print(f"### Condition: {label}\n")
        dist = per_generator_distribution(by_cell, cond_internal)
        for gen in ("gpt-5.1", "Qwen/Qwen2.5-7B-Instruct-Turbo", "gemini-3-flash-preview", GROK_LABEL):
            if gen not in dist:
                continue
            print(f"\n**{gen}**")
            print("```")
            print(f"  {'judge':7s}  {'1':>4s} {'2':>4s} {'3':>4s} {'4':>4s} {'5':>4s}  total  mean   %≤3")
            for j in ("gpt", "gemini", "claude"):
                if j not in dist[gen]:
                    continue
                d = dist[gen][j]
                total = sum(d.values())
                if total == 0: continue
                mean = sum(s * c for s, c in d.items()) / total
                pct_low = 100 * sum(d.get(s, 0) for s in (1, 2, 3)) / total
                bins = " ".join(f"{d.get(s, 0):>4d}" for s in (1, 2, 3, 4, 5))
                print(f"  {j:7s}  {bins}  {total:>5d}  {mean:.2f}  {pct_low:5.1f}%")
            print("```")
        print()

    # 2. Trivial-clean rate per generator
    print("\n## Trivial-clean (all 3 judges in {4,5}) per generator\n")
    print("| condition | gpt-5.1 | Qwen | gemini | grok-opposite |")
    print("|---|--:|--:|--:|--:|")
    for cond_internal, label in [("variant_A", "bare"), ("rubric_plus_spec", "phase_4")]:
        tc = trivial_clean_per_gen(by_cell, cond_internal)
        cells = [label]
        for gen in ("gpt-5.1", "Qwen/Qwen2.5-7B-Instruct-Turbo", "gemini-3-flash-preview", GROK_LABEL):
            d = tc.get(gen, {})
            if not d:
                cells.append("-")
            else:
                cells.append(f"{d['pct_all_4_5']:.1f}% (n={d['n']})")
        print("| " + " | ".join(cells) + " |")

    # 3. Population metrics: existing-3 only vs +Grok-opposite
    print("\n## Population κ across the 8 statements — existing-3 generators vs +Grok-opposite\n")
    print("| condition | metric | existing-3 only | all-4 (with Grok) | Δ |")
    print("|---|---|--:|--:|--:|")
    for cond_internal, cond_label in [("variant_A", "bare"), ("rubric_plus_spec", "phase_4")]:
        m_old = metrics_on_subset(by_cell, cond_internal, EXISTING_GENS)
        m_new = metrics_on_subset(by_cell, cond_internal, ALL_GENS)
        for k in ("alpha", "k3", "k2"):
            v_old = m_old.get(k)
            v_new = m_new.get(k)
            d = (v_new - v_old) if (v_old is not None and v_new is not None) else None
            print(f"| {cond_label} | {k:5s} | {fmt(v_old)} (n={m_old['n']}) | {fmt(v_new)} (n={m_new['n']}) | {fmt(d)} |")

    # 4. Grok-only metrics: do judges agree on Grok-opposite responses?
    print("\n## Agreement on Grok-opposite responses ALONE (does the new generator carry signal?)\n")
    print("| condition | n | α | k3 | k2 |")
    print("|---|--:|--:|--:|--:|")
    for cond_internal, cond_label in [("variant_A", "bare"), ("rubric_plus_spec", "phase_4")]:
        m = metrics_on_subset(by_cell, cond_internal, GROK_ONLY)
        print(f"| {cond_label} | {m['n']} | {fmt(m.get('alpha'))} | {fmt(m.get('k3'))} | {fmt(m.get('k2'))} |")

    # 5. Per-statement comparison
    print("\n## Per-statement α — existing-3 vs +Grok-opposite\n")
    print("| statement | α_bare (3-gen) | α_bare (4-gen) | Δ | α_p4 (3-gen) | α_p4 (4-gen) | Δ |")
    print("|---|--:|--:|--:|--:|--:|--:|")
    for sid in TARGET_SIDS:
        row = [sid]
        for cond_internal, _ in [("variant_A", "bare"), ("rubric_plus_spec", "phase_4")]:
            m_old = metrics_per_statement(by_cell, sid, cond_internal, EXISTING_GENS)
            m_new = metrics_per_statement(by_cell, sid, cond_internal, ALL_GENS)
            v_old = m_old.get("alpha")
            v_new = m_new.get("alpha")
            d = (v_new - v_old) if (v_old is not None and v_new is not None) else None
            row.extend([fmt(v_old), fmt(v_new), fmt(d)])
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
