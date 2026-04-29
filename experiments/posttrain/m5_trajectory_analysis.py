# ruff: noqa: B905, F841
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze per-rubric flag trajectories across M5 closed-loop rounds.

Reads `m5_closed_loop_<judge>{_<tag>}_r{i}_diagnoses.jsonl` files for
i=0..N and produces:

- Per-pair flag pattern (e.g., "FFPF" for round 0=flag, 1=flag, 2=pass, 3=flag)
- Pattern frequency table
- Convergence interpretation: stable fix rate, oscillation rate, persistent
  flagging rate
- Per-round flag count line chart (ASCII)
- Diagnosis evolution: for each FXP→FXF (re-flag after fix), is the diagnosis
  the same as before the fix?

Usage:
    uv run python experiments/posttrain/m5_trajectory_analysis.py --judge gpt51 --tag ""
    uv run python experiments/posttrain/m5_trajectory_analysis.py --judge flash
"""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
STAGE3 = WORKTREE / "experiments/posttrain/stage3_output"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def discover_rounds(judge: str, tag: str) -> list[int]:
    suffix = f"_{tag}" if tag else ""
    rounds = []
    for p in STAGE3.glob(f"m5_closed_loop_{judge}{suffix}_r*_diagnoses.jsonl"):
        try:
            rn = int(p.stem.split("_r")[-1].split("_")[0])
            rounds.append(rn)
        except ValueError:
            continue
    return sorted(set(rounds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=True)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    rounds = discover_rounds(args.judge, args.tag)
    if not rounds:
        print(f"No diagnoses files found for judge={args.judge}, tag={args.tag}")
        return

    diagnoses_by_round: dict[int, dict[tuple, dict]] = {}
    for r in rounds:
        path = STAGE3 / f"m5_closed_loop_{args.judge}{suffix}_r{r}_diagnoses.jsonl"
        rows = load_jsonl(path)
        diagnoses_by_round[r] = {(row["pair_id"], row["tension_point_idx"]): row for row in rows}

    # Get all pair keys
    all_keys = set()
    for d in diagnoses_by_round.values():
        all_keys.update(d.keys())
    keys = sorted(all_keys)

    # Per-pair pattern
    patterns: dict[str, list[tuple]] = {}
    rounds_sorted = sorted(rounds)
    for k in keys:
        pat = ""
        for r in rounds_sorted:
            row = diagnoses_by_round[r].get(k)
            if row is None:
                pat += "?"
            elif row["review"].get("has_pathology"):
                pat += "F"
            else:
                pat += "P"
        patterns.setdefault(pat, []).append(k)

    out: list[str] = []
    out.append(f"# M5 closed-loop trajectory analysis — judge={args.judge}{suffix}")
    out.append("")
    out.append(f"Rounds: {rounds_sorted}. Total cross-tier pairs: {len(keys)}.")
    out.append("")
    out.append("## Per-round flag count")
    out.append("")
    out.append("| round | flagged | passed | total |")
    out.append("|---:|---:|---:|---:|")
    for r in rounds_sorted:
        flagged = sum(1 for row in diagnoses_by_round[r].values() if row["review"].get("has_pathology"))
        passed = sum(1 for row in diagnoses_by_round[r].values() if not row["review"].get("has_pathology"))
        total = flagged + passed
        out.append(f"| r{r} | {flagged} | {passed} | {total} |")
    out.append("")

    # ASCII chart
    out.append("```")
    max_flagged = max(
        sum(1 for row in d.values() if row["review"].get("has_pathology")) for d in diagnoses_by_round.values()
    )
    for r in rounds_sorted:
        flagged = sum(1 for row in diagnoses_by_round[r].values() if row["review"].get("has_pathology"))
        bar = "█" * flagged + "·" * (max_flagged - flagged)
        out.append(f"r{r}: {bar} {flagged:>3}")
    out.append("```")
    out.append("")

    out.append("## Pattern frequency")
    out.append("")
    out.append("| pattern | n | interpretation |")
    out.append("|---|---:|---|")

    def interpret(pat: str) -> str:
        last = pat[-1]
        if pat == "F" * len(pat):
            return "always flagged (no progress)"
        if pat == "P" * len(pat):
            return "never flagged"
        # check if monotone (all flips in one direction)
        flips_to_p = sum(1 for i in range(1, len(pat)) if pat[i - 1] == "F" and pat[i] == "P")
        flips_to_f = sum(1 for i in range(1, len(pat)) if pat[i - 1] == "P" and pat[i] == "F")
        if flips_to_p == 1 and flips_to_f == 0:
            return f"converged at r{pat.index('P')}"
        if flips_to_f == 1 and flips_to_p == 0:
            return f"regressed at r{pat.index('F') if pat.startswith('P') else pat.find('F', 1)}"
        if flips_to_p > 0 and flips_to_f > 0:
            return f"oscillates ({flips_to_p}P, {flips_to_f}F flips)"
        return ""

    for pat in sorted(patterns.keys()):
        n = len(patterns[pat])
        out.append(f"| `{pat}` | {n} | {interpret(pat)} |")
    out.append("")

    out.append("## Per-pair trajectories")
    out.append("")
    out.append("| pair | tp | trajectory |")
    out.append("|---|---|---|")
    for k in keys:
        pat = ""
        for r in rounds_sorted:
            row = diagnoses_by_round[r].get(k)
            if row is None:
                pat += "?"
            elif row["review"].get("has_pathology"):
                pat += "F"
            else:
                pat += "P"
        out.append(f"| `{k[0]}` | {k[1]} | `{pat}` |")
    out.append("")

    # Diagnosis evolution: for pairs that go F→P→F, are diagnoses similar?
    fpf_keys = [
        k
        for k, pats in zip(
            keys,
            [
                (diagnoses_by_round.get(r, {}).get(k, {}).get("review", {}).get("has_pathology") for r in rounds_sorted)
                for k in keys
            ],
        )
    ]

    out.append("## Diagnosis evolution for re-flagged-after-fix pairs (FXP→FXF)")
    out.append("")
    n_same = 0
    n_diff = 0
    sim_scores = []
    for k in keys:
        pat = ""
        last_diag = []
        for r in rounds_sorted:
            row = diagnoses_by_round[r].get(k)
            if row is None:
                pat += "?"
                last_diag.append("")
            else:
                pat += "F" if row["review"].get("has_pathology") else "P"
                last_diag.append(row["review"].get("diagnosis", "") or "")
        # find F segments after a P
        for i in range(1, len(pat) - 1):
            if pat[i] == "P" and i + 1 < len(pat) and pat[i + 1] == "F":
                # find prior F before this P
                prior_f_idx = None
                for j in range(i - 1, -1, -1):
                    if pat[j] == "F":
                        prior_f_idx = j
                        break
                if prior_f_idx is not None:
                    sim = SequenceMatcher(None, last_diag[prior_f_idx][:300], last_diag[i + 1][:300]).ratio()
                    sim_scores.append(sim)
                    if sim > 0.5:
                        n_same += 1
                    else:
                        n_diff += 1
                    out.append(f"- `{k[0]} tp={k[1]}` r{prior_f_idx} F → r{i} P → r{i+1} F (text-sim={sim:.2f})")

    out.append("")
    if sim_scores:
        avg = sum(sim_scores) / len(sim_scores)
        out.append(f"- {n_same} same-pathology re-flags (sim>0.5), {n_diff} different-pathology re-flags (sim≤0.5).")
        out.append(f"- Avg diagnosis text similarity for re-flagged pairs: {avg:.2f}")

    out_path = STAGE3 / f"m5_trajectory_analysis_{args.judge}{suffix}.md"
    out_path.write_text("\n".join(out))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
