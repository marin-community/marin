# ruff: noqa: E501, RUF001
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stretch goal: measure rubric stability across runs.

Compares two runs of `build_target_pair_rubrics.py` (different model
samples but same prompt + temperature) on the same 65 target pairs.
For each pair, compute:

- Jaccard overlap of `spec_clauses_anchored_on` clauses across runs
- Token-level Jaccard between good_criterion / bad_criterion / key_tension
- Total verbatim-clause counts per run

Renders `rubric_stability_report.md`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[3]
DEFAULT_DIR = WORKTREE / "experiments/posttrain/disagreement_primitive"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tokenize(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def render(run1: list[dict[str, Any]], run2: list[dict[str, Any]]) -> str:
    by_pair_1 = {(r["statement_a_id"], r["statement_b_id"]): r["rubric"] for r in run1}
    by_pair_2 = {(r["statement_a_id"], r["statement_b_id"]): r["rubric"] for r in run2}
    common = sorted(set(by_pair_1) & set(by_pair_2))

    rows: list[dict[str, Any]] = []
    for pair in common:
        r1 = by_pair_1[pair]
        r2 = by_pair_2[pair]
        c1 = set(r1.get("rationale", {}).get("spec_clauses_anchored_on", []))
        c2 = set(r2.get("rationale", {}).get("spec_clauses_anchored_on", []))
        good_j = jaccard(tokenize(r1.get("good_criterion", "")), tokenize(r2.get("good_criterion", "")))
        bad_j = jaccard(tokenize(r1.get("bad_criterion", "")), tokenize(r2.get("bad_criterion", "")))
        key_j = jaccard(tokenize(r1.get("key_tension", "")), tokenize(r2.get("key_tension", "")))
        # Verbatim clause-set Jaccard
        vc_j = jaccard(c1, c2)
        rows.append(
            {
                "pair": pair,
                "good_token_jaccard": good_j,
                "bad_token_jaccard": bad_j,
                "key_token_jaccard": key_j,
                "verbatim_clause_jaccard": vc_j,
                "n_clauses_run1": len(c1),
                "n_clauses_run2": len(c2),
            }
        )

    lines = ["# Rubric stability across two runs", ""]
    lines.append(
        f"Compared {len(common)} pairs. Run 1 = `target_pair_rubrics.jsonl`, run 2 = `target_pair_rubrics_run2.jsonl`. Both use `gpt-5.1` reasoning_effort=none, temperature=0.2."
    )
    lines.append("")
    n = len(rows)
    avg_good = sum(r["good_token_jaccard"] for r in rows) / max(1, n)
    avg_bad = sum(r["bad_token_jaccard"] for r in rows) / max(1, n)
    avg_key = sum(r["key_token_jaccard"] for r in rows) / max(1, n)
    avg_vc = sum(r["verbatim_clause_jaccard"] for r in rows) / max(1, n)
    lines.append("## Mean Jaccard across runs")
    lines.append("")
    lines.append("| field | mean Jaccard | meaning |")
    lines.append("|---|--:|---|")
    lines.append(f"| good_criterion (token) | {avg_good:.3f} | how similar the GOOD criterion phrasing is across runs |")
    lines.append(f"| bad_criterion (token) | {avg_bad:.3f} | how similar the BAD criterion phrasing is across runs |")
    lines.append(f"| key_tension (token) | {avg_key:.3f} | how similar the explanatory paragraph is |")
    lines.append(
        f"| spec_clauses_anchored_on (verbatim set) | {avg_vc:.3f} | fraction of anchored verbatim clauses shared between runs |"
    )
    lines.append("")
    # Pairs with worst stability
    rows.sort(key=lambda r: r["verbatim_clause_jaccard"])
    lines.append("## Top 10 pairs with LOWEST verbatim-clause overlap (least stable)")
    lines.append("")
    lines.append("| pair | verbatim Jaccard | good Jaccard | bad Jaccard | n clauses (run1, run2) |")
    lines.append("|---|--:|--:|--:|---|")
    for r in rows[:10]:
        a, b = r["pair"]
        lines.append(
            f"| `{a}` × `{b}` | {r['verbatim_clause_jaccard']:.2f} | {r['good_token_jaccard']:.2f} | "
            f"{r['bad_token_jaccard']:.2f} | ({r['n_clauses_run1']}, {r['n_clauses_run2']}) |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run1", type=Path, default=DEFAULT_DIR / "target_pair_rubrics.jsonl")
    parser.add_argument("--run2", type=Path, default=DEFAULT_DIR / "target_pair_rubrics_run2.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR / "rubric_stability_report.md")
    args = parser.parse_args()
    r1 = load_jsonl(args.run1)
    r2 = load_jsonl(args.run2)
    if not r1 or not r2:
        raise SystemExit("Need both run1 and run2 rubric files.")
    args.output.write_text(render(r1, r2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
