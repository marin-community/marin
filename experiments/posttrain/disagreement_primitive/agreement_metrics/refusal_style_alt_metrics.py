# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: RUF001, RUF002, RUF003  -- intentional α in DART notation

"""Alternative ordinal-agreement metrics for refusal_style.

Question: is the α=0.770 win for `refusal_style__t0__gpt` driven by genuine
inter-judge convergence, or by mode-collapse at score 1 (which interval
Krippendorff α tends to flatter)?

Compute for both `refusal_style__null` (statement-only baseline) and
`refusal_style__t0__gpt` (winning branch):
  - Krippendorff α (interval) -- verify reported 0.770
  - Pairwise Cohen's κ: unweighted, linear, quadratic
  - % exact 3-way agreement
  - % all-judges-within-1
  - Mean absolute pairwise diff per cell
  - Score distributions

If the win is mode-collapse, weighted κ (which is less sensitive to range
restriction than interval α) should be substantially LOWER than α for the
winner. % exact agreement should be very high (consensus at 1). Mean abs
diff should be small. Pattern would tell us the rubric is detecting "is
this a violation y/n" rather than scoring on a graded scale.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = Path(
    "/lfs/skampere3/0/ahmedah/code/marin/.claude/worktrees/align/"
    "experiments/posttrain/disagreement_primitive/dart_deliberative_rubric_pilot/"
    "20260514_bucket_c_t0/branch_judgments.jsonl"
)

JUDGES = ["gpt", "claude", "gemini-pro"]
BRANCHES_TO_COMPARE = ["refusal_style__null", "refusal_style__t0__gpt"]


def load_cells(branch_id: str) -> dict[tuple, dict[str, int]]:
    """Return {(scenario_idx, generator) -> {judge: score}}."""
    cells: dict[tuple, dict[str, int]] = defaultdict(dict)
    for line in SOURCE.open():
        r = json.loads(line)
        if r["branch_id"] != branch_id:
            continue
        cells[(r["scenario_idx"], r["generator"])][r["judge"]] = r["score"]
    # Drop cells missing any judge
    return {k: v for k, v in cells.items() if all(j in v for j in JUDGES)}


# --- Krippendorff α (interval) ---


def krippendorff_alpha_interval(cells: dict) -> float:
    """Krippendorff's α with interval (squared) distance.

    For interval data with c coders per unit, the formula reduces to:
        α = 1 - (n-1) * D_o / D_e
    where D_o is observed mean squared pairwise difference within units and
    D_e is over the marginal distribution of all observations.
    """
    obs_pair_sq = 0.0
    obs_pair_n = 0
    all_scores: list[int] = []
    for _key, judge_scores in cells.items():
        scores = list(judge_scores.values())
        all_scores.extend(scores)
        c = len(scores)
        for i in range(c):
            for j in range(i + 1, c):
                obs_pair_sq += (scores[i] - scores[j]) ** 2
                obs_pair_n += 1
    if obs_pair_n == 0:
        return float("nan")
    D_o = obs_pair_sq / obs_pair_n
    # D_e: variance across all observations (E[squared pairwise diff] in marginal)
    N = len(all_scores)
    if N < 2:
        return float("nan")
    mean = sum(all_scores) / N
    var_pop = sum((x - mean) ** 2 for x in all_scores) / N
    # E[(X-Y)^2] = 2*Var when X, Y iid from marginal
    D_e = 2 * var_pop
    return 1 - D_o / D_e if D_e > 0 else float("nan")


# --- Cohen's κ, weighted variants ---


def cohens_kappa(a: list[int], b: list[int], weight: str = "unweighted") -> float:
    """Pairwise Cohen's κ. weight in {unweighted, linear, quadratic}."""
    assert len(a) == len(b)
    levels = sorted(set(a) | set(b))
    idx = {v: i for i, v in enumerate(levels)}
    N = len(a)
    K = len(levels)
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
            exp += wgt * row_sum[r] * col_sum[c] / N if N > 0 else 0.0
    if exp == 0:
        return float("nan")
    return 1 - obs / exp


# --- Other simple metrics ---


def per_cell_summary(cells: dict) -> dict:
    """Return mean abs pair diff, % exact 3-way agreement, % all-within-1."""
    if not cells:
        return {"n_cells": 0}
    abs_diffs: list[float] = []
    exact_3way = 0
    within_1 = 0
    for _k, scores in cells.items():
        s = [scores[j] for j in JUDGES]
        # mean abs pairwise diff over 3 pairs
        d = abs(s[0] - s[1]) + abs(s[0] - s[2]) + abs(s[1] - s[2])
        abs_diffs.append(d / 3)
        if s[0] == s[1] == s[2]:
            exact_3way += 1
        if max(s) - min(s) <= 1:
            within_1 += 1
    n = len(abs_diffs)
    return {
        "n_cells": n,
        "mean_abs_pairdiff_per_cell": sum(abs_diffs) / n,
        "pct_exact_3way": 100 * exact_3way / n,
        "pct_within_1": 100 * within_1 / n,
    }


def score_distribution(cells: dict) -> dict:
    pooled: list[int] = []
    per_judge: dict[str, list[int]] = {j: [] for j in JUDGES}
    for _k, scores in cells.items():
        for j in JUDGES:
            pooled.append(scores[j])
            per_judge[j].append(scores[j])
    pooled_counter = Counter(pooled)
    if not pooled:
        return {}
    mean = sum(pooled) / len(pooled)
    var = sum((x - mean) ** 2 for x in pooled) / len(pooled)
    std = var**0.5
    return {
        "pooled_dist": {k: pooled_counter.get(k, 0) for k in range(1, 6)},
        "pooled_mean": mean,
        "pooled_std": std,
        "per_judge_mean": {j: sum(per_judge[j]) / len(per_judge[j]) for j in JUDGES},
    }


def pairwise_metrics(cells: dict) -> dict:
    """For each unique pair of judges, compute κ variants and Spearman-ish stats."""
    out: dict[str, dict] = {}
    judge_pairs = [("gpt", "claude"), ("gpt", "gemini-pro"), ("claude", "gemini-pro")]
    for j1, j2 in judge_pairs:
        a = [scores[j1] for scores in cells.values()]
        b = [scores[j2] for scores in cells.values()]
        out[f"{j1}_x_{j2}"] = {
            "kappa_unweighted": cohens_kappa(a, b, "unweighted"),
            "kappa_linear": cohens_kappa(a, b, "linear"),
            "kappa_quadratic": cohens_kappa(a, b, "quadratic"),
            "mean_abs_diff": sum(abs(x - y) for x, y in zip(a, b, strict=False)) / len(a) if a else float("nan"),
            "pct_exact": 100 * sum(1 for x, y in zip(a, b, strict=False) if x == y) / len(a) if a else float("nan"),
        }
    return out


def report_branch(branch_id: str) -> dict:
    cells = load_cells(branch_id)
    return {
        "branch": branch_id,
        "alpha_interval": krippendorff_alpha_interval(cells),
        "per_cell": per_cell_summary(cells),
        "score_dist": score_distribution(cells),
        "pairwise": pairwise_metrics(cells),
    }


def format_report(reports: list[dict]) -> str:
    out: list[str] = []
    out.append("=" * 78)
    out.append(f"{'metric':<35s} " + " ".join(f"{r['branch']:>22s}" for r in reports))
    out.append("=" * 78)

    def row(label: str, fmt: str, get):
        vals = []
        for r in reports:
            v = get(r)
            if v is None:
                vals.append(f"{'—':>22s}")
            else:
                vals.append(f"{fmt.format(v):>22s}")
        out.append(f"{label:<35s} " + " ".join(vals))

    out.append("--- Krippendorff α (interval) ---")
    row("α_interval (3-judge)", "{:.4f}", lambda r: r["alpha_interval"])

    out.append("--- Cell-level agreement ---")
    row("n cells (complete)", "{:d}", lambda r: r["per_cell"]["n_cells"])
    row("mean abs pair-diff / cell", "{:.4f}", lambda r: r["per_cell"]["mean_abs_pairdiff_per_cell"])
    row("% exact 3-way agreement", "{:.1f}%", lambda r: r["per_cell"]["pct_exact_3way"])
    row("% all judges within 1", "{:.1f}%", lambda r: r["per_cell"]["pct_within_1"])

    out.append("--- Score distribution (pooled across 3 judges) ---")
    for k in range(1, 6):
        row(f"  score {k}", "{:d}", lambda r, k=k: r["score_dist"]["pooled_dist"].get(k, 0))
    row("pooled mean", "{:.3f}", lambda r: r["score_dist"]["pooled_mean"])
    row("pooled std", "{:.3f}", lambda r: r["score_dist"]["pooled_std"])

    out.append("--- Pairwise Cohen's κ (unweighted / linear / quadratic) ---")
    for pair in ["gpt_x_claude", "gpt_x_gemini-pro", "claude_x_gemini-pro"]:
        out.append(f"  pair: {pair}")
        row("    κ unweighted", "{:.4f}", lambda r, p=pair: r["pairwise"][p]["kappa_unweighted"])
        row("    κ linear", "{:.4f}", lambda r, p=pair: r["pairwise"][p]["kappa_linear"])
        row("    κ quadratic", "{:.4f}", lambda r, p=pair: r["pairwise"][p]["kappa_quadratic"])
        row("    mean abs diff", "{:.4f}", lambda r, p=pair: r["pairwise"][p]["mean_abs_diff"])
        row("    % exact (pair)", "{:.1f}%", lambda r, p=pair: r["pairwise"][p]["pct_exact"])

    out.append("=" * 78)
    return "\n".join(out)


def main() -> int:
    reports = [report_branch(b) for b in BRANCHES_TO_COMPARE]
    print(format_report(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
