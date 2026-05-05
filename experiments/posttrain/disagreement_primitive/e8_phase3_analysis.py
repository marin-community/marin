# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E8 phase 3 analysis — compare to the paper's published numbers.

Computes:
  1. Per-judge flag rate (problematic = non-compliant ∨ ambiguous; paper convention)
  2. Pairwise judge agreement % (paper got 67-77% across pairs)
  3. Cohen's κ on the 2-judge ensemble (binary collapse of 3-way decisions)
  4. Per-statement Cohen's κ (group by scenario seed statement)
  5. Behavioral D from existing variant-A scores → bin → frequent-non-compliance rate
     → multiplier (paper got 5–13× across bins)

All pure analysis — $0 cost.
"""

from __future__ import annotations
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import DIR, load_jsonl


# Paper convention: "problematic" = non-compliant ∨ ambiguous (binary collapse for κ + bin analysis)
def is_problematic(decision: str) -> bool:
    return decision in {"non-compliant", "ambiguous"}


def cohens_kappa_binary(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's κ for 2 raters with binary labels (problematic vs not)."""
    if len(pairs) < 5:
        return None
    n = len(pairs)
    a_pos = sum(1 for a, _ in pairs if a)
    b_pos = sum(1 for _, b in pairs if b)
    p_a_pos = a_pos / n; p_b_pos = b_pos / n
    p_a_neg = 1 - p_a_pos; p_b_neg = 1 - p_b_pos
    p_e = p_a_pos * p_b_pos + p_a_neg * p_b_neg
    p_o = sum(1 for a, b in pairs if a == b) / n
    if p_e >= 1.0:
        return None
    return (p_o - p_e) / (1 - p_e)


def fleiss_kappa(rows: list[list[int]]) -> float | None:
    """Fleiss' κ for n raters with k categories.

    `rows[i]` is a list of length k giving the count of raters who assigned
    item i to each category. Each row must sum to the same number (n raters).

    Returns None if degenerate (single category, no items).
    """
    if not rows:
        return None
    N = len(rows)
    n = sum(rows[0])
    if n < 2:
        return None
    k = len(rows[0])
    # Per-item agreement: P_i = (sum c_ij^2 - n) / (n * (n - 1))
    P_i = []
    for r in rows:
        if sum(r) != n:
            return None
        s = sum(c * c for c in r) - n
        P_i.append(s / (n * (n - 1)))
    P_bar = sum(P_i) / N
    # Category proportions
    p_j = [sum(r[j] for r in rows) / (N * n) for j in range(k)]
    P_e = sum(p * p for p in p_j)
    if P_e >= 1.0:
        return None
    return (P_bar - P_e) / (1 - P_e)


def load_phase3(path: Path) -> dict:
    return {(r["statement_id"], r["scenario_idx"], r["generator"]): r
            for r in load_jsonl(path)
            if "error" not in r and r.get("decision") in {"compliant", "non-compliant", "ambiguous"}}


def main() -> int:
    # --- Load phase-3 judgments (GLM may not exist yet) ---
    j_gpt = load_phase3(DIR / "phase3_gpt/judgments.jsonl")
    j_gem = load_phase3(DIR / "phase3_gemini/judgments.jsonl")
    j_glm = load_phase3(DIR / "phase3_glm/judgments.jsonl")
    print(f"  phase 3 GPT judgments:    {len(j_gpt)} valid")
    print(f"  phase 3 Gemini judgments: {len(j_gem)} valid")
    print(f"  phase 3 GLM judgments:    {len(j_glm)} valid")
    have_glm = len(j_glm) > 100  # threshold: enough to be analyzable
    shared = sorted(set(j_gpt.keys()) & set(j_gem.keys()))
    if have_glm:
        shared_3 = sorted(set(j_gpt.keys()) & set(j_gem.keys()) & set(j_glm.keys()))
        print(f"  shared (3 judges valid): {len(shared_3)}")
    print(f"  shared (≥2 judges valid): {len(shared)}")

    # --- 1. Per-judge flag rate (problematic) ---
    print("\n=== Per-judge flag rate (paper convention: problematic = non-compliant ∨ ambiguous) ===")
    print(f"  PAPER reference: Claude 4 Sonnet 48.1%, o3 35.5%, Gemini 2.5 Pro 36.5%")
    judge_set = [("GPT-5.1", j_gpt), ("Gemini-3-Flash", j_gem)]
    if have_glm:
        judge_set.append(("GLM-5.1", j_glm))
    for label, jmap in judge_set:
        flagged = sum(1 for r in jmap.values() if is_problematic(r["decision"]))
        print(f"  {label}: {flagged}/{len(jmap)} = {100*flagged/len(jmap):.1f}%")

    # Decision breakdown
    print("\n=== 3-way decision breakdown ===")
    for label, jmap in judge_set:
        c = Counter(r["decision"] for r in jmap.values())
        total = sum(c.values())
        print(f"  {label}: " + ", ".join(f"{d}={n} ({100*n/total:.1f}%)" for d, n in c.most_common()))

    # --- 2. Pairwise agreement on shared keys ---
    print("\n=== Pairwise judge agreement (binary problematic vs not, on shared keys) ===")
    print(f"  PAPER reference pairs: Claude↔o3 67.5%, Claude↔Gem 72.4%, o3↔Gem 76.8% — avg ~72%")
    pairs_gpt_gem = [(is_problematic(j_gpt[k]["decision"]), is_problematic(j_gem[k]["decision"])) for k in shared]
    same = sum(1 for a, b in pairs_gpt_gem if a == b)
    print(f"  GPT ↔ Gemini: {same}/{len(pairs_gpt_gem)} = {100*same/len(pairs_gpt_gem):.1f}% agreement")

    if have_glm:
        # 3-judge pairwise
        sh3 = sorted(set(j_gpt.keys()) & set(j_gem.keys()) & set(j_glm.keys()))
        pairs_gpt_glm = [(is_problematic(j_gpt[k]["decision"]), is_problematic(j_glm[k]["decision"])) for k in sh3]
        pairs_gem_glm = [(is_problematic(j_gem[k]["decision"]), is_problematic(j_glm[k]["decision"])) for k in sh3]
        for label, pp in [("GPT ↔ GLM", pairs_gpt_glm), ("Gemini ↔ GLM", pairs_gem_glm)]:
            s = sum(1 for a, b in pp if a == b)
            print(f"  {label}: {s}/{len(pp)} = {100*s/len(pp):.1f}% agreement")

    # 3-way exact agreement
    same_3way = sum(1 for k in shared if j_gpt[k]["decision"] == j_gem[k]["decision"])
    print(f"  GPT ↔ Gemini exact 3-way: {same_3way}/{len(shared)} = {100*same_3way/len(shared):.1f}%")
    if have_glm:
        s_all3 = sum(1 for k in sh3 if j_gpt[k]["decision"] == j_gem[k]["decision"] == j_glm[k]["decision"])
        print(f"  All 3 judges agree exactly: {s_all3}/{len(sh3)} = {100*s_all3/len(sh3):.1f}%")

    # --- 3. κ (Cohen for 2-judge, Fleiss for 3-judge) ---
    if have_glm:
        print("\n=== Fleiss' κ (3 judges, binary problematic) — replicate paper's 0.42 ===")
        # Build N × 2 count matrix: rows[i] = [count_not_problematic, count_problematic]
        # Each row sums to n_judges = 3
        rows_3judge = []
        for k in sh3:
            flags = [is_problematic(j_gpt[k]["decision"]),
                     is_problematic(j_gem[k]["decision"]),
                     is_problematic(j_glm[k]["decision"])]
            n_pos = sum(flags); n_neg = 3 - n_pos
            rows_3judge.append([n_neg, n_pos])
        fk = fleiss_kappa(rows_3judge)
        print(f"  Fleiss' κ (3 judges, binary problematic): {fk:.4f}" if fk is not None else "  (insufficient data)")
        print(f"  PAPER reference: Fleiss' κ = 0.42")
    else:
        print("\n=== Cohen's κ (2-judge binary) — paper's 3-judge Fleiss κ = 0.42 ===")
        kappa = cohens_kappa_binary(pairs_gpt_gem)
        print(f"  Cohen's κ (GPT vs Gemini): {kappa:.4f}" if kappa is not None else "  (insufficient data)")

    # --- 4. Per-statement κ (group by seed statement) ---
    if have_glm:
        print("\n=== Per-statement Fleiss' κ (paper does NOT compute this; 3-judge) ===")
        by_stmt_3: dict[str, list[list[int]]] = defaultdict(list)
        for k in sh3:
            flags = [is_problematic(j_gpt[k]["decision"]),
                     is_problematic(j_gem[k]["decision"]),
                     is_problematic(j_glm[k]["decision"])]
            n_pos = sum(flags); n_neg = 3 - n_pos
            by_stmt_3[k[0]].append([n_neg, n_pos])
        per_stmt_kappa = []
        for sid, rows in by_stmt_3.items():
            k = fleiss_kappa(rows) if len(rows) >= 5 else None
            per_stmt_kappa.append((sid, k, len(rows)))
    else:
        print("\n=== Per-statement Cohen's κ (paper does NOT compute this) ===")
        by_stmt_pairs: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        for k in shared:
            by_stmt_pairs[k[0]].append(
                (is_problematic(j_gpt[k]["decision"]), is_problematic(j_gem[k]["decision"]))
            )
        per_stmt_kappa = []
        for sid, plist in by_stmt_pairs.items():
            k = cohens_kappa_binary(plist)
            per_stmt_kappa.append((sid, k, len(plist)))

    finite = [(s, k, n) for s, k, n in per_stmt_kappa if k is not None]
    print(f"  {len(finite)}/{len(per_stmt_kappa)} statements have computable κ")
    if finite:
        ks = sorted(k for _, k, _ in finite)
        print(f"  κ distribution: min={min(ks):.3f}, p25={ks[len(ks)//4]:.3f}, median={ks[len(ks)//2]:.3f}, p75={ks[3*len(ks)//4]:.3f}, max={max(ks):.3f}")
        print(f"  κ < 0:   {sum(1 for k in ks if k < 0)} statements (judges anti-agree)")
        print(f"  κ < 0.2: {sum(1 for k in ks if k < 0.2)} statements (slight)")
        print(f"  κ ≥ 0.4: {sum(1 for k in ks if k >= 0.4)} statements (moderate or above)")
        print(f"\n  Top 10 lowest-κ statements (judges disagree most):")
        for sid, k, n in sorted(finite, key=lambda x: x[1])[:10]:
            print(f"    {sid:40s}  κ={k:+.3f}  n={n}")
        print(f"\n  Top 10 highest-κ statements (judges agree most):")
        for sid, k, n in sorted(finite, key=lambda x: -x[1])[:10]:
            print(f"    {sid:40s}  κ={k:+.3f}  n={n}")

    # --- 5. Behavioral D from existing phase-1 + phase-2 variant-A scores → bin multiplier ---
    # D(x) per (statement, scenario) = max over judge of [STD across generators of variant-A scores]
    print("\n=== Behavioral D + bin multiplier (paper's headline 5-13× ratio) ===")
    print("  PAPER reference: 13.9× bin ratio across 5 disagreement bins (Fig 2, S_OpenAI)")

    # Reload variant-A scores from phase 1 + phase 2
    va_paths = [
        ("gpt",    DIR / "e8_va_judgments.jsonl"),
        ("gemini", DIR / "phase2_gemini/va_judgments.jsonl"),
        ("glm",    DIR / "phase2_glm/va_judgments.jsonl"),
    ]
    va: dict[str, dict[tuple, int]] = {}
    for label, p in va_paths:
        va[label] = {(r["statement_id"], r["scenario_idx"], r["generator"]): r["score"]
                     for r in load_jsonl(p)
                     if "error" not in r and r.get("score") in {1, 2, 3, 4, 5}}

    # Per (statement, scenario): collect 3 generator scores per judge, compute STD, take max over judges
    GENERATORS = ["gpt-5.1", "Qwen/Qwen2.5-7B-Instruct-Turbo", "gemini-3-flash-preview"]
    scenario_d: dict[tuple, float] = {}  # (statement, scenario) → D
    for sid_idx in {(k[0], k[1]) for j in va.values() for k in j}:
        sid, idx = sid_idx
        per_judge_std = []
        for jl in ("gpt", "gemini", "glm"):
            scores = [va[jl].get((sid, idx, gen)) for gen in GENERATORS]
            valid = [s for s in scores if s is not None]
            if len(valid) >= 2:
                per_judge_std.append(statistics.stdev(valid))
        if per_judge_std:
            scenario_d[(sid, idx)] = max(per_judge_std)

    print(f"  scenarios with computable D: {len(scenario_d)}")

    # Per scenario: did all 3 generators get flagged "problematic" by majority of judges? (frequent non-compliance, paper convention)
    by_scenario_p3: dict[tuple, dict] = defaultdict(lambda: {"gpt": [], "gem": [], "glm": []})
    for k, r in j_gpt.items():
        by_scenario_p3[(k[0], k[1])]["gpt"].append((k[2], is_problematic(r["decision"])))
    for k, r in j_gem.items():
        by_scenario_p3[(k[0], k[1])]["gem"].append((k[2], is_problematic(r["decision"])))
    if have_glm:
        for k, r in j_glm.items():
            by_scenario_p3[(k[0], k[1])]["glm"].append((k[2], is_problematic(r["decision"])))

    def freq_noncomp(scenario_data, mode):
        """For each generator, is it flagged by `mode` of the available judges?
        Then: are ALL 3 generators flagged?"""
        per_gen = {}
        for j_label, items in scenario_data.items():
            for gen, flag in items:
                per_gen.setdefault(gen, []).append(flag)
        if len(per_gen) < 3:
            return None
        n_judges_per_gen = max(len(v) for v in per_gen.values())
        if n_judges_per_gen == 0:
            return None
        # For each generator, judge it problematic by mode
        gen_flagged = {}
        for gen, flags in per_gen.items():
            if mode == "any":
                gen_flagged[gen] = any(flags)
            elif mode == "majority":
                gen_flagged[gen] = sum(flags) > len(flags) / 2
            elif mode == "all":
                gen_flagged[gen] = all(flags)
        return all(gen_flagged.get(g, False) for g in GENERATORS)

    # Bin scenarios by D
    BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]
    bin_data = defaultdict(lambda: {"total": 0, "fnc_any": 0, "fnc_majority": 0, "fnc_all": 0})
    for (sid, idx), D in scenario_d.items():
        for low, high in BINS:
            if low <= D < high:
                bin_label = f"[{low}, {high})"
                break
        else:
            bin_label = f"[≥{BINS[-1][1]}]"
        sd = by_scenario_p3.get((sid, idx))
        if sd is None:
            continue
        bin_data[bin_label]["total"] += 1
        for mode in ("any", "majority", "all"):
            if freq_noncomp(sd, mode):
                bin_data[bin_label][f"fnc_{mode}"] += 1

    mode_label = "majority" if have_glm else "any"  # paper uses majority-of-3
    print(f"\n  Frequent non-compliance criterion: '{mode_label}' (paper used majority-of-3)")
    print(f"  Bin           | n    | fnc-any | fnc-majority | fnc-all")
    for low, high in BINS:
        label = f"[{low}, {high})"
        d = bin_data.get(label, {})
        n = d.get("total", 0)
        if n == 0:
            continue
        a, m, al = d["fnc_any"], d["fnc_majority"], d["fnc_all"]
        print(f"  {label:14s} | n={n:4d} | {100*a/n:5.1f}% | {100*m/n:5.1f}%      | {100*al/n:5.1f}%")

    # Compute the headline multiplier (paper's metric) using majority-vote mode
    rates_majority = [d[f"fnc_{mode_label}"] / d["total"] for d in bin_data.values() if d["total"] > 0]
    rates_majority_nz = [r for r in rates_majority if r > 0]
    if rates_majority_nz and min(rates_majority_nz) > 0:
        ratio = max(rates_majority_nz) / min(rates_majority_nz)
        print(f"\n  Bin ratio (max / min freq-noncomp, '{mode_label}' mode): {ratio:.1f}×")
        print(f"  PAPER reference: 13.9× (Fig 2, S_OpenAI majority-vote of 3 judges)")

    # --- 6. Write final report ---
    out_jsonl = DIR / "phase3_per_statement_kappa.jsonl"
    with out_jsonl.open("w") as fh:
        for sid, k, n in per_stmt_kappa:
            fh.write(json.dumps({"statement_id": sid, "cohens_kappa_binary": k, "n_pairs": n}) + "\n")
    print(f"\n  wrote {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
