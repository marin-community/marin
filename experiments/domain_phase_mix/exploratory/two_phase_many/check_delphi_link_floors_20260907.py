# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sanity check of the log-deficit link's floor rules against the held-out bank's per-task minima.

A floor rule maps the fit swarm to a per-task lower bound phi_t; a valid rule never places phi_t above the
lowest value any held-out run has reached on task t. Rules checked: the current 0.95 x swarm minimum, and the
proportional-anchored family phi_t = prop_t - kappa (prop_t - swarm_min_t) for several kappa. The empirical
kappa of each task, (prop_t - bank_min_t) / (prop_t - swarm_min_t), is reported by task group. Inputs: the
frozen panel of `delphi_offline_selection_20260906` (swarm minima, proportional row) and the corrected held-out
registry `single_phase_heldout_round3_corrected_20260903` (per-coordinate components), restricted to the
frozen bank's coordinates. Nothing is fitted or launched.

With --selection (a selection package) and --method, the fitted floors of that method's full-panel fits
(fold -1 shards) are checked the same way, and the chosen link and kappa are tabulated by task group.

usage: uv run python check_delphi_link_floors_20260907.py [--selection DIR --method ID]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE = SCRIPT_DIR / "reference_outputs"
BENCHMARK = REFERENCE / "delphi_offline_selection_20260906"
REGISTRY = REFERENCE / "single_phase_heldout_round3_corrected_20260903" / "heldout_coordinate_components.csv"
OUTPUT = REFERENCE / "delphi_link_floor_check_20260907"
PANEL = "delphi_3e18_39bucket"
ANCHORS = REFERENCE / "delphi_floor_anchors_20260907" / "anchors.csv"
KAPPAS = (1.0, 1.5, 2.0, 2.5)
CURRENT_FRACTION = 0.95
NEXT_RUNG_PROPORTIONAL = {"uncheatable": 0.9022, "table9": 0.9715}


def task_group(component: str) -> str:
    name = component.split("/")[2] if component.count("/") >= 3 else component
    if name.startswith("mt_mbpp") or name.startswith("basic_skills_coding"):
        return "code"
    if name.startswith("minerva") or "math" in name or "arithmetic" in name:
        return "math"
    if name.startswith("mmlu"):
        return "mmlu"
    if "github" in name:
        return "code"
    if "arxiv" in name:
        return "science"
    return "qa/other"


def fitted_floors(selection: Path, method: str, panel: dict) -> dict[tuple[str, str], tuple[float, float, str]]:
    """(target, component) -> (floor, kappa, link) from a method's full-panel shards in a selection package."""
    out = {}
    for target in ("uncheatable", "table9"):
        components = [str(c) for c in panel[f"{target}_components"]]
        for index, component in enumerate(components):
            shard = np.load(selection / "baseline_shards" / method / target / f"r0_f-1_c{index}.npz", allow_pickle=True)
            diagnostics = json.loads(str(shard["diagnostics_json"]))
            out[(target, component)] = (
                float(diagnostics.get("floor", float("nan"))),
                float(diagnostics.get("kappa", float("nan"))),
                str(diagnostics["link"]),
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", type=Path, default=None, help="a selection package with fold -1 shards")
    parser.add_argument("--method", default="weibull_softplus_unscaled@fitted_floor_link")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = np.load(BENCHMARK / "inputs" / "panel.npz", allow_pickle=True)
    fitted = fitted_floors(args.selection, args.method, panel) if args.selection is not None else {}
    # The proportional anchor is the reliability mean the fitted-floor link uses, not the panel's single row.
    table = pd.read_csv(ANCHORS)
    # The anchors file also carries the Llama panels since 2026-09-07; this check is about the Delphi bank.
    anchors = table[table.panel == "delphi_3e18_39bucket"].set_index("component").proportional_bpb
    registry = pd.read_csv(REGISTRY)
    registry = registry[registry.panel.eq(PANEL)]
    rows = []
    for target in ("uncheatable", "table9"):
        labels = pd.read_csv(BENCHMARK / "inputs" / f"{target}_bank_labels.csv")
        bank = registry[registry.target.eq(target) & registry.coordinate_id.isin(labels.coordinate_id)]
        bank_min = bank.groupby("component").bpb_mean.min()
        bank_count = bank.groupby("component").coordinate_id.nunique()
        components = [str(c) for c in panel[f"{target}_components"]]
        outcomes = panel[f"{target}_outcomes"]
        for index, component in enumerate(components):
            swarm_min = float(outcomes[:, index].min())
            prop = float(anchors[component])
            bmin = float(bank_min[component])
            row = {
                "target": target,
                "component": component.split("/")[2] if component.count("/") >= 3 else component,
                "group": task_group(component),
                "proportional": prop,
                "swarm_min": swarm_min,
                "swarm_gap": prop - swarm_min,
                "bank_min": bmin,
                "bank_coordinates": int(bank_count[component]),
                "kappa_bank": (prop - bmin) / (prop - swarm_min) if prop > swarm_min else np.nan,
                "floor_0.95": CURRENT_FRACTION * swarm_min,
                "floor_0.95_valid": CURRENT_FRACTION * swarm_min <= bmin,
            }
            for kappa in KAPPAS:
                floor = prop - kappa * (prop - swarm_min)
                row[f"floor_kappa{kappa:g}"] = floor
                row[f"floor_kappa{kappa:g}_valid"] = floor <= bmin
            if fitted:
                floor, kappa, link = fitted[(target, component)]
                row["fitted_link"] = link
                row["fitted_kappa"] = kappa
                row["floor_fitted"] = floor if link != "identity" else np.nan
                row["floor_fitted_valid"] = bool(link == "identity" or floor <= bmin)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "floor_check.csv", index=False)
    lines = ["# Floor rules against the held-out bank's per-task minima", ""]
    for target in ("uncheatable", "table9"):
        sub = table[table.target.eq(target)]
        lines.append(f"## {target} ({len(sub)} tasks)")
        lines.append("")
        lines.append("| rule | tasks with floor above the bank minimum | worst excess (BPB) |")
        lines.append("|---|---:|---:|")
        rules = [("0.95 x swarm min", "floor_0.95")]
        rules += [(f"kappa {kappa:g}", f"floor_kappa{kappa:g}") for kappa in KAPPAS]
        if fitted:
            rules.append(("fitted floor (identity-link tasks pass)", "floor_fitted"))
        for name, column in rules:
            invalid = sub[~sub[f"{column}_valid"]]
            worst = float((invalid[column] - invalid.bank_min).max()) if len(invalid) else 0.0
            lines.append(f"| {name} | {len(invalid)} | {worst:.4f} |")
        if fitted:
            lines.append("")
            lines.append(
                "Fitted-floor choices by task group (identity-link count; kappa median [min, max] of floor tasks):"
            )
            lines.append("")
            for group, frame in sub.groupby("group"):
                floors = frame[frame.fitted_link.ne("identity")]
                if len(floors):
                    kappas = floors.fitted_kappa
                    text = f"{kappas.median():.2f} [{kappas.min():.2f}, {kappas.max():.2f}]"
                else:
                    text = "none"
                identity = int(frame.fitted_link.eq("identity").sum())
                lines.append(f"- {group} ({len(frame)}): identity {identity}, floor kappa {text}")
        lines.append("")
        lines.append("Empirical kappa of the bank minimum by task group (median [min, max]):")
        lines.append("")
        for group, frame in sub.groupby("group"):
            k = frame.kappa_bank.dropna()
            lines.append(f"- {group} ({len(frame)}): {k.median():.2f} [{k.min():.2f}, {k.max():.2f}]")
        lines.append("")
        agg = panel[f"{target}_aggregate"]
        prop_agg = float(agg[[str(r) for r in panel["runs"]].index("singleavg_fit_000_baseline_proportional")])
        swarm_agg = float(agg.min())
        bank_agg = float(pd.read_csv(BENCHMARK / "inputs" / f"{target}_bank_labels.csv").measured_mean_bpb.min())
        lines.append(
            f"Aggregate: proportional {prop_agg:.4f}, swarm min {swarm_agg:.4f}, bank min {bank_agg:.4f}, "
            f"empirical kappa {(prop_agg - bank_agg) / (prop_agg - swarm_agg):.2f}; next-rung proportional "
            f"{NEXT_RUNG_PROPORTIONAL[target]:.4f}; kappa-2 aggregate floor "
            f"{prop_agg - 2 * (prop_agg - swarm_agg):.4f}."
        )
        lines.append("")
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    pd.set_option("display.width", 220)
    worst = table.sort_values("kappa_bank", ascending=False).head(12)
    print(
        worst[
            ["target", "component", "group", "proportional", "swarm_min", "bank_min", "kappa_bank", "bank_coordinates"]
        ]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
