# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: RUF001, RUF002, RUF003  -- intentional α in DART notation
"""Alt-distribution validation for refusal_style__t0__gpt.

The metric question (is α=0.770 paradoxical?) was settled by
refusal_style_alt_metrics.py: the win is real, not collapse-driven.

The remaining substantive question: does the rubric actually distinguish
graded cases (anchors 3-5), or is it a pure violation detector that
happens to never need anchors 3-5 because the test distribution is 87%
violations?

True validation requires new cells; this script does the strongest
analysis possible from existing data:

  1. Per-generator score distribution and agreement. Each of the 4
     generators produces ~20 cells; if grok-opposite is 100% score 1
     but gpt-5.1 / Qwen show meaningful spread with agreement, the
     rubric is doing graded work.
  2. Per-scenario distribution. Some refusal_style scenarios may
     consistently elicit anchor-3+ responses.
  3. Cells where at least one judge used anchor >=3. These are the
     "graded subset" within the existing universe. Check whether the
     rubric still produces inter-judge agreement on this subset.
  4. Cells where the median judge score is >= 2 (less collapse-driven
     subset).

If subsets where the rubric uses anchors 3-5 STILL show good agreement,
that is positive evidence the rubric is not purely a violation detector.
If those subsets show high disagreement, the rubric is confirmed as a
violation detector wearing a 5-anchor costume.

This is a proxy, not a true alt-distribution test. The 80-cell universe
is what it is. A real test would generate new cells designed to elicit
anchors 3-5.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = Path(
    "/lfs/skampere3/0/ahmedah/code/marin/.claude/worktrees/align/"
    "experiments/posttrain/disagreement_primitive/dart_deliberative_rubric_pilot/"
    "20260514_bucket_c_t0/branch_judgments.jsonl"
)

JUDGES = ["gpt", "claude", "gemini-pro"]
BRANCH = "refusal_style__t0__gpt"
NULL_BRANCH = "refusal_style__null"


def load_cells(branch_id: str) -> dict[tuple, dict[str, int]]:
    """Return {(scenario_idx, generator) -> {judge: score}}."""
    cells: dict[tuple, dict[str, int]] = defaultdict(dict)
    for line in SOURCE.open():
        r = json.loads(line)
        if r["branch_id"] != branch_id:
            continue
        cells[(r["scenario_idx"], r["generator"])][r["judge"]] = r["score"]
    return {k: v for k, v in cells.items() if all(j in v for j in JUDGES)}


def cohens_kappa(a: list[int], b: list[int], weight: str = "quadratic") -> float:
    if not a:
        return float("nan")
    levels = sorted(set(a) | set(b))
    idx = {v: i for i, v in enumerate(levels)}
    N = len(a)
    K = len(levels)
    if K == 1:
        return 1.0  # all agreement at one level
    confusion = [[0] * K for _ in range(K)]
    for x, y in zip(a, b, strict=False):
        confusion[idx[x]][idx[y]] += 1
    row_sum = [sum(row) for row in confusion]
    col_sum = [sum(confusion[r][c] for r in range(K)) for c in range(K)]

    def w(r, c):
        if weight == "unweighted":
            return 0.0 if r == c else 1.0
        denom = max(K - 1, 1)
        d = abs(r - c) / denom
        if weight == "linear":
            return d
        if weight == "quadratic":
            return d * d
        raise ValueError(weight)

    obs = 0.0
    exp = 0.0
    for r in range(K):
        for c in range(K):
            wgt = w(r, c)
            obs += wgt * confusion[r][c]
            exp += wgt * row_sum[r] * col_sum[c] / N
    if exp == 0:
        return float("nan")
    return 1 - obs / exp


def krippendorff_alpha_interval(cells: dict) -> float:
    obs_sq, obs_n = 0.0, 0
    all_s: list[int] = []
    for _k, sc in cells.items():
        s = list(sc.values())
        all_s.extend(s)
        c = len(s)
        for i in range(c):
            for j in range(i + 1, c):
                obs_sq += (s[i] - s[j]) ** 2
                obs_n += 1
    if obs_n == 0:
        return float("nan")
    D_o = obs_sq / obs_n
    N = len(all_s)
    mean = sum(all_s) / N
    var = sum((x - mean) ** 2 for x in all_s) / N
    D_e = 2 * var
    return 1 - D_o / D_e if D_e > 0 else float("nan")


def cell_stats(cells: dict) -> dict:
    if not cells:
        return {"n": 0}
    pooled: list[int] = []
    for sc in cells.values():
        pooled.extend(sc.values())
    counter = Counter(pooled)
    abs_diffs: list[float] = []
    within_1 = 0
    exact_3 = 0
    for sc in cells.values():
        s = [sc[j] for j in JUDGES]
        abs_diffs.append((abs(s[0] - s[1]) + abs(s[0] - s[2]) + abs(s[1] - s[2])) / 3)
        if max(s) - min(s) <= 1:
            within_1 += 1
        if s[0] == s[1] == s[2]:
            exact_3 += 1
    return {
        "n": len(cells),
        "dist": {k: counter.get(k, 0) for k in range(1, 6)},
        "pooled_mean": sum(pooled) / len(pooled),
        "pooled_std": (sum((x - sum(pooled) / len(pooled)) ** 2 for x in pooled) / len(pooled)) ** 0.5,
        "alpha_interval": krippendorff_alpha_interval(cells),
        "mean_abs_pair_diff": sum(abs_diffs) / len(abs_diffs),
        "pct_exact_3way": 100 * exact_3 / len(cells),
        "pct_within_1": 100 * within_1 / len(cells),
        "pairwise_quad_kappa": {
            "gpt_x_claude": cohens_kappa(
                [sc["gpt"] for sc in cells.values()],
                [sc["claude"] for sc in cells.values()],
                "quadratic",
            ),
            "gpt_x_gemini": cohens_kappa(
                [sc["gpt"] for sc in cells.values()],
                [sc["gemini-pro"] for sc in cells.values()],
                "quadratic",
            ),
            "claude_x_gemini": cohens_kappa(
                [sc["claude"] for sc in cells.values()],
                [sc["gemini-pro"] for sc in cells.values()],
                "quadratic",
            ),
        },
    }


def print_stats(label: str, stats: dict) -> None:
    if stats["n"] == 0:
        print(f"  {label:<30s}: 0 cells (empty subset)")
        return
    dist = " ".join(f"{k}:{stats['dist'][k]}" for k in range(1, 6))
    qk = stats["pairwise_quad_kappa"]
    print(
        f"  {label:<30s}: n={stats['n']:3d}  α={stats['alpha_interval']:+.3f}  "
        f"mean_abs={stats['mean_abs_pair_diff']:.3f}  "
        f"exact_3way={stats['pct_exact_3way']:5.1f}%  within_1={stats['pct_within_1']:5.1f}%"
    )
    print(f"  {'':<32s} dist=[{dist}]  pooled_mean={stats['pooled_mean']:.2f}  std={stats['pooled_std']:.2f}")
    print(
        f"  {'':<32s} quad-κ: gpt/cla={qk['gpt_x_claude']:+.3f}  "
        f"gpt/gem={qk['gpt_x_gemini']:+.3f}  cla/gem={qk['claude_x_gemini']:+.3f}"
    )
    print()


def per_generator(cells: dict) -> dict[str, dict]:
    by_gen: dict[str, dict] = defaultdict(dict)
    for (scen, gen), sc in cells.items():
        by_gen[gen][(scen, gen)] = sc
    return by_gen


def per_scenario(cells: dict) -> dict[int, dict]:
    by_scen: dict[int, dict] = defaultdict(dict)
    for (scen, gen), sc in cells.items():
        by_scen[scen][(scen, gen)] = sc
    return by_scen


def subset_any_judge_ge(cells: dict, threshold: int) -> dict:
    """Cells where AT LEAST ONE judge scored >= threshold."""
    return {k: sc for k, sc in cells.items() if any(s >= threshold for s in sc.values())}


def subset_median_ge(cells: dict, threshold: int) -> dict:
    """Cells where the MEDIAN judge score >= threshold (less mode-collapse)."""
    out = {}
    for k, sc in cells.items():
        s = sorted(sc.values())
        if s[1] >= threshold:
            out[k] = sc
    return out


def main() -> int:
    print("=" * 90)
    print("ALT-DISTRIBUTION VALIDATION — refusal_style__t0__gpt vs refusal_style__null")
    print("(within-universe partition; true validation requires new cells)")
    print("=" * 90)
    cells_winner = load_cells(BRANCH)
    cells_null = load_cells(NULL_BRANCH)

    print("\n## Full universe (n=78)")
    print(f"WINNER: {BRANCH}")
    print_stats("full universe", cell_stats(cells_winner))
    print(f"NULL: {NULL_BRANCH}")
    print_stats("full universe", cell_stats(cells_null))

    print("\n## Per-generator (which generators trigger anchors 3-5?)")
    print(f"WINNER: {BRANCH}")
    for gen, sub in sorted(per_generator(cells_winner).items()):
        print_stats(gen, cell_stats(sub))

    print(f"NULL: {NULL_BRANCH}")
    for gen, sub in sorted(per_generator(cells_null).items()):
        print_stats(gen, cell_stats(sub))

    print("\n## Cells where AT LEAST ONE judge used anchor >= 3 (the graded subset)")
    for threshold in (3, 4):
        print(f"### threshold: any judge score >= {threshold}")
        winner_sub = subset_any_judge_ge(cells_winner, threshold)
        null_sub = subset_any_judge_ge(cells_null, threshold)
        print(f"WINNER (n={len(winner_sub)}):")
        print_stats(f"any_judge>={threshold}", cell_stats(winner_sub))
        print(f"NULL (n={len(null_sub)}):")
        print_stats(f"any_judge>={threshold}", cell_stats(null_sub))

    print("\n## Cells where MEDIAN judge score >= 2 (avoid pure-violation cells)")
    for threshold in (2, 3):
        print(f"### threshold: median judge score >= {threshold}")
        winner_sub = subset_median_ge(cells_winner, threshold)
        null_sub = subset_median_ge(cells_null, threshold)
        print(f"WINNER (n={len(winner_sub)}):")
        print_stats(f"median>={threshold}", cell_stats(winner_sub))
        print(f"NULL (n={len(null_sub)}):")
        print_stats(f"median>={threshold}", cell_stats(null_sub))

    print("\n## Per-scenario summary (only scenarios with non-trivial spread)")
    print(f"WINNER: {BRANCH}")
    for scen, sub in sorted(per_scenario(cells_winner).items()):
        stats = cell_stats(sub)
        # only print scenarios where pooled std > 0 (some spread)
        if stats["pooled_std"] > 0.4:
            print_stats(f"scenario {scen}", stats)

    print("\n## Direct comparison: anchor-3+ usage rate")
    pooled_winner: list[int] = []
    for sc in cells_winner.values():
        pooled_winner.extend(sc.values())
    pooled_null: list[int] = []
    for sc in cells_null.values():
        pooled_null.extend(sc.values())
    winner_ge3 = sum(1 for s in pooled_winner if s >= 3)
    null_ge3 = sum(1 for s in pooled_null if s >= 3)
    print(
        f"  WINNER: {winner_ge3}/{len(pooled_winner)} judgments at anchor >=3 ({100*winner_ge3/len(pooled_winner):.1f}%)"
    )
    print(f"  NULL:   {null_ge3}/{len(pooled_null)} judgments at anchor >=3 ({100*null_ge3/len(pooled_null):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
