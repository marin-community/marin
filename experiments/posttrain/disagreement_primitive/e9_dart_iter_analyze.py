# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: RUF001, RUF002, RUF003  -- long LM-prompt strings + intentional unicode (α, ×, −, –) used in DART notation

"""DART iterative validation — Phase 4 analysis.

Reads new judgment data for a round (per_judgment_iter_round_{N}.jsonl) plus
existing baseline data (per_judgment_opposite.jsonl), computes per-statement ×
per-condition α (Krippendorff interval), Δα vs baseline, Δα vs prior round
(rounds > 1), and bootstrap CI. Per-cell Δpwv on top-K poison cells.

Updates each statement's history.json with the round's empirical results and
assigns a verdict: CONVERGED / IMPROVING / STUCK.

Usage:
    .venv/bin/python e9_dart_iter_analyze.py --round 1
    .venv/bin/python e9_dart_iter_analyze.py --round 2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e9_kappa_alternatives import krippendorff_alpha_interval

DIR = Path("experiments/posttrain/disagreement_primitive")
ITER_DIR = DIR / "dart_iteration"
BASELINE_PER_JUDGMENT = DIR / "per_judgment_opposite.jsonl"

T1 = 0.5
EPSILON_IMPROVING = 0.05
DECELERATION_THRESHOLD = 0.025  # for round 3


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def cell_key(r: dict) -> tuple:
    """Identifier for a single (statement, scenario, generator) triple."""
    return (r["statement_id"], r["scenario_idx"], r["generator"])


def build_rater_tuples(per_judgment_rows: list[dict]) -> list[tuple]:
    """Group per-judgment rows by (statement, scenario, generator) and emit
    (gpt_score, gemini_score, claude_score) tuples for α computation."""
    by_cell: dict[tuple, dict[str, int | None]] = defaultdict(dict)
    for r in per_judgment_rows:
        if r.get("score") is None:
            continue
        by_cell[cell_key(r)][r["judge"]] = r["score"]
    tuples = []
    for _ck, scores in by_cell.items():
        # canonical order: gpt, gemini, claude — pad missing with None and skip if all missing
        triple = (scores.get("gpt"), scores.get("gemini"), scores.get("claude"))
        if all(t is None for t in triple):
            continue
        tuples.append(triple)
    return tuples


def alpha_for_subset(per_judgment_rows: list[dict], statement_id: str, condition: str | None) -> float | None:
    rows = [r for r in per_judgment_rows if r["statement_id"] == statement_id]
    if condition is not None:
        rows = [r for r in rows if r.get("condition") == condition]
    triples = build_rater_tuples(rows)
    return krippendorff_alpha_interval(triples)


def bootstrap_alpha_ci(
    per_judgment_rows: list[dict],
    statement_id: str,
    condition: str | None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float | None, float | None] | None:
    rows = [r for r in per_judgment_rows if r["statement_id"] == statement_id]
    if condition is not None:
        rows = [r for r in rows if r.get("condition") == condition]
    triples = build_rater_tuples(rows)
    if len(triples) < 5:
        return None
    rng = random.Random(seed)
    boots = []
    n = len(triples)
    for _ in range(n_boot):
        sample = [triples[rng.randrange(n)] for _ in range(n)]
        a = krippendorff_alpha_interval(sample)
        if a is not None:
            boots.append(a)
    if not boots:
        return None
    boots.sort()
    lo = boots[int(alpha / 2 * len(boots))]
    hi = boots[int((1 - alpha / 2) * len(boots)) - 1]
    return lo, hi


def pwv_top_k(per_judgment_rows: list[dict], statement_id: str, condition: str | None, k: int = 10) -> float:
    """Sum of top-K per-cell pairwise variance.

    pwv = sum over judge pairs of (s_i - s_j)^2.
    """
    rows = [r for r in per_judgment_rows if r["statement_id"] == statement_id]
    if condition is not None:
        rows = [r for r in rows if r.get("condition") == condition]
    by_cell: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        if r.get("score") is None:
            continue
        by_cell[cell_key(r)][r["judge"]] = r["score"]

    pwvs = []
    for _ck, scores in by_cell.items():
        ss = [s for s in scores.values() if s is not None]
        v = 0.0
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                v += (ss[i] - ss[j]) ** 2
        pwvs.append(v)
    pwvs.sort(reverse=True)
    return sum(pwvs[:k])


def classify_verdict(
    alpha_after: float | None, delta_alpha: float | None, prior_round_alpha: float | None = None, round_n: int = 1
) -> str:
    if alpha_after is None or delta_alpha is None:
        return "stuck"
    if alpha_after >= T1:
        return "converged"
    # Deceleration check for round 3
    if round_n >= 3 and prior_round_alpha is not None:
        if alpha_after - prior_round_alpha < DECELERATION_THRESHOLD:
            return "stuck"  # forced — diminishing returns
    if delta_alpha >= EPSILON_IMPROVING:
        return "improving"
    return "stuck"


def baseline_per_judgment_for_statement(statement_id: str) -> list[dict]:
    """Load the baseline (C0) per-judgment rows for this statement.

    Baseline = original v1 rubric + original v1 spec + 3-judge ensemble. Comes from
    per_judgment_opposite.jsonl filtered to phase_4 (rubric+spec) condition.
    """
    rows = []
    for r in load_jsonl(BASELINE_PER_JUDGMENT):
        if r.get("statement_id") != statement_id:
            continue
        # phase_4 = rubric + spec ; the column name differs across files
        cond = r.get("condition", "")
        if cond not in ("rubric_plus_spec", "phase_4"):
            continue
        # Tag as C0 baseline so it's clearly distinguished
        rows.append({**r, "condition": "C0"})
    return rows


def analyze_round(round_n: int, conditions: list[str]) -> dict:
    """Compute per-statement, per-condition α and verdicts; update history.json."""
    iter_judgment_path = ITER_DIR / f"per_judgment_iter_round_{round_n}.jsonl"
    if not iter_judgment_path.exists():
        raise SystemExit(f"missing {iter_judgment_path} — run e9_dart_iter_judge.py first")
    iter_rows = load_jsonl(iter_judgment_path)
    print(f"Round {round_n}: loaded {len(iter_rows)} judgment rows")

    statements = sorted({s.name for s in ITER_DIR.iterdir() if s.is_dir() and (s / "history.json").exists()})
    print(f"Statements: {len(statements)}")

    summary = {"round": round_n, "per_statement": {}}

    for sid in statements:
        sid_dir = ITER_DIR / sid
        history_path = sid_dir / "history.json"
        history = json.loads(history_path.read_text())
        # Skip statements that don't have an entry for this round
        # (e.g. CONVERGED in a prior round, no need to re-analyze)
        if len(history) < round_n or history[round_n - 1].get("verdict") in ("converged", "stuck", "skipped"):
            continue
        baseline_rows = baseline_per_judgment_for_statement(sid)
        # Combine baseline + iter rows for this statement
        stmt_iter_rows = [r for r in iter_rows if r.get("statement_id") == sid]
        combined = baseline_rows + stmt_iter_rows

        # Compute per-condition α
        alpha_by_cond = {}
        for c in ["C0", *conditions]:
            a = alpha_for_subset(combined, sid, c)
            alpha_by_cond[c] = a

        # Bootstrap CI for the operative condition (use C3 if available, else last in conditions)
        operative = "C3" if "C3" in conditions else (conditions[-1] if conditions else "C0")
        ci = bootstrap_alpha_ci(combined, sid, operative)

        # Top-10 pwv per condition
        pwv_top10_by_cond = {c: pwv_top_k(combined, sid, c, k=10) for c in ["C0", *conditions]}

        # Determine alpha_before (round_n-1's after, or C0 baseline if round 1)
        if round_n == 1:
            alpha_before = alpha_by_cond.get("C0")
        else:
            prior = history[round_n - 2]  # 0-indexed
            alpha_before = prior.get("alpha_after_round")

        alpha_after = alpha_by_cond.get(operative)
        delta_alpha = (alpha_after - alpha_before) if (alpha_after is not None and alpha_before is not None) else None

        # Δpwv: percent drop from C0 to operative
        pwv_c0 = pwv_top10_by_cond.get("C0", 0)
        pwv_op = pwv_top10_by_cond.get(operative, 0)
        pwv_pct_drop = (pwv_c0 - pwv_op) / pwv_c0 if pwv_c0 > 0 else None

        prior_alpha = history[round_n - 2]["alpha_after_round"] if round_n > 1 and len(history) >= round_n - 1 else None
        verdict = classify_verdict(alpha_after, delta_alpha, prior_alpha, round_n)

        # Update history entry
        entry = history[round_n - 1]
        entry.update(
            {
                "alpha_before_round": alpha_before,
                "alpha_after_round": alpha_after,
                "alpha_by_condition": alpha_by_cond,
                "alpha_ci_95_operative_condition": ci,
                "operative_condition": operative,
                "delta_alpha": delta_alpha,
                "delta_pwv_top10_pct_drop": pwv_pct_drop,
                "pwv_top10_by_condition": pwv_top10_by_cond,
                "verdict": verdict,
                "analyzed_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        history_path.write_text(json.dumps(history, indent=2))

        summary["per_statement"][sid] = {
            "alpha_before": alpha_before,
            "alpha_after": alpha_after,
            "delta_alpha": delta_alpha,
            "verdict": verdict,
            "operative": operative,
        }
        ab = f"{alpha_before:.3f}" if alpha_before is not None else "?"
        aa = f"{alpha_after:.3f}" if alpha_after is not None else "?"
        da = f"{delta_alpha:+.3f}" if delta_alpha is not None else "?"
        print(f"  {sid:35s} α: {ab} → {aa}  Δ={da}  verdict={verdict}")

    # Tally verdicts
    from collections import Counter

    verdict_counts = Counter(s["verdict"] for s in summary["per_statement"].values())
    summary["verdict_counts"] = dict(verdict_counts)

    summary_path = ITER_DIR / f"round_{round_n}_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nVerdict counts: {dict(verdict_counts)}")
    print(f"Wrote {summary_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--conditions", default="C1,C2,C3")
    args = ap.parse_args()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    analyze_round(args.round, conditions)


if __name__ == "__main__":
    main()
