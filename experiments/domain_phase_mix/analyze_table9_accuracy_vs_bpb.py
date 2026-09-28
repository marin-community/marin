# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Do the Table-9 components whose BPB improved also improve in accuracy?

Joins a `report_table9_accuracy` coverage report (accuracy per scored component and checkpoint) with the native
Table-9 BPB summaries of the same checkpoints, for one baseline and one candidate mixture, and writes a per-component
table with both deltas, sign agreement, and rank correlation, overall and by component group.

usage: uv run --no-sync python -m experiments.domain_phase_mix.analyze_table9_accuracy_vs_bpb \\
    --coverage COVERAGE.json --native-summary SUMMARIES.json --baseline proportional --candidate unimax8 --output DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from scipy import stats

NATIVE_PREFIX = "olmo_base_easy/table9/"
GROUPS = (("minerva_math_", "math"), ("mt_mbpp_", "code"), ("codex_humaneval", "code"), ("mbpp", "code"))
BASIC_SKILLS = "basic_skills_"
MMLU = "mmlu_"


@dataclass(frozen=True)
class ComponentRow:
    component: str
    group: str
    baseline_bpb: float
    candidate_bpb: float
    delta_bpb: float
    baseline_accuracy_pct: float
    candidate_accuracy_pct: float
    delta_accuracy_pp: float
    agree: bool | None
    """True when lower BPB came with higher accuracy (or higher with lower); None when either delta is zero."""


def component_group(name: str) -> str:
    if name.startswith(BASIC_SKILLS):
        return "basic skills"
    if name.startswith(MMLU):
        return "mmlu"
    for prefix, group in GROUPS:
        if name == prefix or name.startswith(prefix):
            return group
    return "qa"


def report_for(coverage: list[dict], checkpoint_uri: str) -> dict:
    matching = [r for r in coverage if r["checkpoint"]["checkpoint_uri"] == checkpoint_uri]
    if len(matching) != 1:
        raise ValueError(f"{len(matching)} coverage reports for {checkpoint_uri}")
    return matching[0]


def component_rows(coverage: list[dict], native: dict, baseline: str, candidate: str) -> list[ComponentRow]:
    base_report = report_for(coverage, native[baseline]["config"]["checkpoint_path"])
    cand_report = report_for(coverage, native[candidate]["config"]["checkpoint_path"])
    scored = sorted(set(base_report["coverage"]["components"]) & set(cand_report["coverage"]["components"]))
    rows = []
    for name in scored:
        b_bpb = float(native[baseline]["summary"][NATIVE_PREFIX + name + "/bpb"])
        c_bpb = float(native[candidate]["summary"][NATIVE_PREFIX + name + "/bpb"])
        b_acc = 100 * float(base_report["coverage"]["components"][name])
        c_acc = 100 * float(cand_report["coverage"]["components"][name])
        d_bpb, d_acc = c_bpb - b_bpb, c_acc - b_acc
        agree = None if d_bpb == 0 or d_acc == 0 else (d_bpb < 0) == (d_acc > 0)
        rows.append(ComponentRow(name, component_group(name), b_bpb, c_bpb, d_bpb, b_acc, c_acc, d_acc, agree))
    return rows


def agreement_summary(rows: list[ComponentRow]) -> dict:
    decided = [r for r in rows if r.agree is not None]
    summary: dict = {
        "scored": len(rows),
        "decided": len(decided),
        "agree": sum(r.agree for r in decided),
        "bpb_improved": sum(r.delta_bpb < 0 for r in rows),
        "bpb_improved_and_accuracy_improved": sum(r.delta_bpb < 0 and r.delta_accuracy_pp > 0 for r in rows),
    }
    if len(rows) >= 3:
        improvement = [-r.delta_bpb for r in rows]
        gains = [r.delta_accuracy_pp for r in rows]
        summary["spearman"] = float(stats.spearmanr(improvement, gains).statistic)
        summary["pearson"] = float(stats.pearsonr(improvement, gains).statistic)
    return summary


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregate(rows: list[ComponentRow], native: dict, baseline: str, candidate: str) -> dict:
    """Equal-weight means over scored components, by group, plus the BPB change on unscored components."""
    groups = sorted({r.group for r in rows})
    scored = {r.component for r in rows}
    out: dict = {"scored_components": len(rows)}
    for label, subset in [("all_scored", rows), *[(g, [r for r in rows if r.group == g]) for g in groups]]:
        out[label] = {
            "components": len(subset),
            "accuracy_pct": {
                baseline: mean([r.baseline_accuracy_pct for r in subset]),
                candidate: mean([r.candidate_accuracy_pct for r in subset]),
            },
            "bpb": {
                baseline: mean([r.baseline_bpb for r in subset]),
                candidate: mean([r.candidate_bpb for r in subset]),
            },
        }
    unscored = sorted(
        key.removeprefix(NATIVE_PREFIX).removesuffix("/bpb")
        for key in native[baseline]["summary"]
        if key.startswith(NATIVE_PREFIX)
        and key.endswith("/bpb")
        and key.removeprefix(NATIVE_PREFIX).removesuffix("/bpb") not in scored
    )
    if unscored:
        out["unscored_bpb_only"] = {
            "components": unscored,
            "bpb": {
                name: mean([float(native[name]["summary"][NATIVE_PREFIX + c + "/bpb"]) for c in unscored])
                for name in (baseline, candidate)
            },
        }
    return out


def markdown(rows: list[ComponentRow], baseline: str, candidate: str) -> str:
    lines = [
        f"| component | group | BPB {baseline} | BPB {candidate} | ΔBPB | acc {baseline} | acc {candidate} | Δacc pp "
        "| agree |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in sorted(rows, key=lambda r: r.delta_bpb):
        mark = "" if r.agree is None else ("yes" if r.agree else "no")
        lines.append(
            f"| {r.component} | {r.group} | {r.baseline_bpb:.4f} | {r.candidate_bpb:.4f} | {r.delta_bpb:+.4f} | "
            f"{r.baseline_accuracy_pct:.2f} | {r.candidate_accuracy_pct:.2f} | {r.delta_accuracy_pp:+.2f} | {mark} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coverage", type=Path, required=True, help="coverage.json written by report_table9_accuracy")
    parser.add_argument("--native-summary", type=Path, required=True, help="native Table-9 W&B summaries by name")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coverage = json.loads(args.coverage.read_text())
    native = json.loads(args.native_summary.read_text())
    rows = component_rows(coverage, native, args.baseline, args.candidate)
    if not rows:
        raise ValueError("No component is scored for both checkpoints")
    groups = sorted({r.group for r in rows})
    summary = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "overall": agreement_summary(rows),
        "by_group": {g: agreement_summary([r for r in rows if r.group == g]) for g in groups},
        "native_macro_bpb": {
            name: native[name]["summary"]["olmo_base_easy/table9_macro_bpb"] for name in (args.baseline, args.candidate)
        },
        "aggregate": aggregate(rows, native, args.baseline, args.candidate),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "components.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    table = markdown(rows, args.baseline, args.candidate)
    (args.output / "components.md").write_text(table + "\n")
    print(table)
    overall = summary["overall"]
    print(
        f"\nscored {overall['scored']}; BPB improved on {overall['bpb_improved']}, of which accuracy also improved on "
        f"{overall['bpb_improved_and_accuracy_improved']}; sign agreement {overall['agree']}/{overall['decided']}; "
        f"Spearman(-dBPB, dAcc) {overall.get('spearman', math.nan):.2f}"
    )
    agg = summary["aggregate"]
    print(f"\nequal-weight means over scored components ({agg['scored_components']} of 51):")
    for label in ["all_scored", *groups]:
        a = agg[label]
        print(
            f"  {label:13s} n={a['components']:2d}  accuracy {a['accuracy_pct'][args.baseline]:.2f} -> "
            f"{a['accuracy_pct'][args.candidate]:.2f} "
            f"({a['accuracy_pct'][args.candidate] - a['accuracy_pct'][args.baseline]:+.2f} pp)"
            f"  bpb {a['bpb'][args.baseline]:.4f} -> {a['bpb'][args.candidate]:.4f}"
            f" ({100 * (a['bpb'][args.candidate] / a['bpb'][args.baseline] - 1):+.1f}%)"
        )
    if "unscored_bpb_only" in agg:
        u = agg["unscored_bpb_only"]
        print(
            f"  unscored (BPB only) n={len(u['components']):2d}  bpb {u['bpb'][args.baseline]:.4f} -> "
            f"{u['bpb'][args.candidate]:.4f} ({100 * (u['bpb'][args.candidate] / u['bpb'][args.baseline] - 1):+.1f}%)"
        )
    for g in groups:
        s = summary["by_group"][g]
        spearman = s.get("spearman", math.nan)
        print(f"  {g:13s} scored {s['scored']:2d}  agree {s['agree']}/{s['decided']}  spearman {spearman:.2f}")


if __name__ == "__main__":
    main()
