# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["gcsfs>=2025.5.1", "numpy>=2.0", "pandas>=2.2", "scipy>=1.14"]
# ///
"""Score the H4 multi-target rollouts against the ATOM-031 gates.

The question is whether an optimizer-aware utility computed at a parent state can choose the continuation
weight a 512-update rollout would actually prefer -- because if it can, the ATOM-030 prefix search can
prune branches with rollouts at about a tenth of the cost of training them.

The previous H4 analysis could not answer it. Its readout was `paloma_programming_languages` alone, which
falls monotonically in the StarCoder weight, so every parent's argmin sat at the boundary q = 1.0 and any
monotone utility scored a perfect match for free. Three things are done differently here, all fixed in the
preregistration before this data existed:

  the objective  is the equal-weight macro over all four targets, so the code/text trade-off is visible
                 and an interior optimum can exist at all;
  the gates      are scored in order, and gate A can fail the whole question before any mapping is fitted;
  the baselines  are explicit. A rule that cannot beat "always pick the grid centre" carries no
                 state-conditional information however high its correlation.

Usage: ``uv run python ... [--results <gs://...>] [--utility <optimizer_utility.csv>]``
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_UTILITY = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_gradient_probe_full_results_20260818"
DEFAULT_OUTPUT = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_h4_macro_rollout_results_20260819"
PRIMARY_UPDATES = 512
CALIBRATION_ROLE = "h4_independent_calibration"
VALIDATION_ROLE = "h4_primary_validation"
PRACTICAL_NULL = 0.0001  # BPB over 512 updates, pro-rata from the 0.001 endpoint null
GRID_CENTRE = 0.45
BLOCK_WEIGHTS = {
    "paloma_programming_languages": 64,
    "paloma_c4_en": 7,
    "uncheatable_github_python": 7,
    "uncheatable_wikipedia_english": 4,
}


def load_rollouts(result_root: str) -> pd.DataFrame:
    """One row per (rollout row, readout, target)."""
    fs = gcsfs.GCSFileSystem()
    paths = fs.glob(f"{result_root.removeprefix('gs://')}/full/rollout/*/*/rows/*.json")
    rows = []
    for path in paths:
        with fs.open(path, "rb") as handle:
            document = json.load(handle)
        if document.get("kind") != "h4_macro_rollout":
            raise RuntimeError(f"Unexpected document kind in {path}")
        row = document["row"]
        for readout in document["readouts"]:
            for target, measurement in sorted(readout["targets"].items()):
                rows.append(
                    {
                        "parent_trajectory_id": row["parent_trajectory_id"],
                        "parent_checkpoint_label": row["parent_checkpoint_label"],
                        "starcoder_weight": float(row["starcoder_weight"]),
                        "analysis_role": row["analysis_role"],
                        "rollout_order_seed": int(row["rollout_order_seed"]),
                        "updates": int(readout["updates"]),
                        "target": target,
                        "bpb": float(measurement["bpb"]),
                        "bpb_standard_error": float(measurement["bpb_standard_error"]),
                    }
                )
    if not rows:
        raise RuntimeError(f"No H4 macro rollout documents under {result_root}")
    return pd.DataFrame(rows)


def macro(frame: pd.DataFrame, weights: Mapping[str, float] | None) -> pd.DataFrame:
    """Collapse targets into one objective per (parent, checkpoint, seed, q, updates)."""
    keys = [
        "parent_trajectory_id",
        "parent_checkpoint_label",
        "analysis_role",
        "rollout_order_seed",
        "starcoder_weight",
        "updates",
    ]
    working = frame.copy()
    working["weight"] = 1.0 if weights is None else working["target"].map(weights).astype(float)
    working["weighted"] = working["bpb"] * working["weight"]
    grouped = working.groupby(keys, as_index=False)[["weighted", "weight"]].sum()
    grouped["bpb"] = grouped["weighted"] / grouped["weight"]
    return grouped.drop(columns=["weighted", "weight"])


def curves(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per parent-checkpoint-seed cell, with its q grid as an array."""
    keys = ["parent_trajectory_id", "parent_checkpoint_label", "analysis_role", "rollout_order_seed"]
    rows = []
    for key, group in frame.groupby(keys):
        ordered = group.sort_values("starcoder_weight")
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "q": ordered["starcoder_weight"].to_numpy(),
                "bpb": ordered["bpb"].to_numpy(),
            }
        )
    return pd.DataFrame(rows)


def gate_a(cell_curves: pd.DataFrame) -> dict:
    """Does an interior optimum exist behaviourally? Scored before any mapping is fitted.

    Interior means interior to the q GRID -- the argmin is not at q=0 or q=1 -- because that is what
    decides whether a utility has anything to get right. Scoring it against the spread of the observed
    argmins instead would call a unanimous interior optimum "0% interior", which is the opposite of the
    truth and is what a first version of this function did.
    """
    argmins = np.array([row.q[int(np.argmin(row.bpb))] for row in cell_curves.itertuples()])
    grid = np.concatenate([row.q for row in cell_curves.itertuples()])
    low, high = float(grid.min()), float(grid.max())
    interior = (argmins > low) & (argmins < high)
    return {
        "cells": len(argmins),
        "interior_fraction": float(interior.mean()),
        "argmin_histogram": {float(q): int(c) for q, c in zip(*np.unique(argmins, return_counts=True), strict=True)},
        "passes": bool(interior.mean() >= 0.5),
    }


def fit_mapping(calibration: pd.DataFrame) -> float:
    """Zero-intercept slope from utility to macro BPB, both centred within parent. Frozen after this."""
    denominator = float(np.square(calibration["delta_utility"]).sum())
    if denominator == 0.0:
        raise RuntimeError("calibration utility has zero spread")
    return float((calibration["delta_utility"] * calibration["delta_bpb"]).sum() / denominator)


def centred(merged: pd.DataFrame) -> pd.DataFrame:
    """Centre utility and BPB within each parent cell, so only the SHAPE in q is compared."""
    keys = ["parent_trajectory_id", "parent_checkpoint_label", "rollout_order_seed"]
    working = merged.copy()
    working["delta_utility"] = working["utility_dot"] - working.groupby(keys)["utility_dot"].transform("mean")
    working["delta_bpb"] = working["bpb"] - working.groupby(keys)["bpb"].transform("mean")
    return working


def selection_regret(merged: pd.DataFrame, chooser) -> dict:
    """What a practitioner loses in macro BPB by following `chooser` instead of the cell's own optimum."""
    keys = ["parent_trajectory_id", "parent_checkpoint_label", "rollout_order_seed"]
    regrets, exact, within_one = [], [], []
    for _key, group in merged.groupby(keys):
        ordered = group.sort_values("starcoder_weight")
        q = ordered["starcoder_weight"].to_numpy()
        bpb = ordered["bpb"].to_numpy()
        best = int(np.argmin(bpb))
        picked = int(np.argmin(np.abs(q - chooser(ordered))))
        regrets.append(float(bpb[picked] - bpb[best]))
        exact.append(picked == best)
        within_one.append(abs(picked - best) <= 1)
    return {
        "cells": len(regrets),
        "mean_regret_bpb": float(np.mean(regrets)),
        "max_regret_bpb": float(np.max(regrets)),
        "exact_match": float(np.mean(exact)),
        "within_one_step": float(np.mean(within_one)),
        "beats_practical_null": bool(np.mean(regrets) <= PRACTICAL_NULL),
    }


def utility_chooser(slope: float):
    """The frozen mapping's pick: the q whose predicted BPB change is lowest."""

    def choose(group: pd.DataFrame) -> float:
        predicted = slope * group["delta_utility"].to_numpy()
        return float(group["starcoder_weight"].to_numpy()[int(np.argmin(predicted))])

    return choose


def report(name: str, result: Mapping) -> str:
    return (
        f"  {name:34s} regret {result['mean_regret_bpb']:+.6f}  max {result['max_regret_bpb']:+.6f}  "
        f"exact {result['exact_match']:.0%}  within-1 {result['within_one_step']:.0%}  cells {result['cells']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="gs://marin-us-central1/analysis/pinlin_calvin_xu/data_mixture/"
        "starcoder_wsd80_h4_macro_rollout_v1_20260819",
    )
    parser.add_argument("--utility", type=Path, default=DEFAULT_UTILITY / "optimizer_utility.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    raw = load_rollouts(args.results)
    raw = raw[raw["updates"].eq(PRIMARY_UPDATES)]
    utility = pd.read_csv(args.utility)
    utility = utility[utility["component"].eq("trunk")]
    # Utility is per target; the objective is a macro, so the utility side is aggregated the same way.
    utility = utility.groupby(["parent_trajectory_id", "parent_checkpoint_label", "starcoder_weight"], as_index=False)[
        "utility_dot"
    ].mean()

    summaries: dict[str, dict] = {}
    for label, weights in (("equal (primary)", None), ("block counts (secondary)", BLOCK_WEIGHTS)):
        objective = macro(raw, weights)
        cell_curves = curves(objective)
        gate = gate_a(cell_curves)
        print(f"\n=== {label} ===")
        print(
            f"GATE A interior-optimum fraction {gate['interior_fraction']:.0%} over {gate['cells']} cells "
            f"-> {'PASS' if gate['passes'] else 'FAIL'}"
        )
        print(f"  argmin histogram {gate['argmin_histogram']}")
        summaries[label] = {"gate_a": gate}
        if not gate["passes"]:
            print("  gate A failed: the readout cannot pose the question, so no mapping is fitted.")
            continue

        merged = centred(
            objective.merge(
                utility,
                on=["parent_trajectory_id", "parent_checkpoint_label", "starcoder_weight"],
                how="inner",
                validate="many_to_one",
            )
        )
        slope = fit_mapping(merged[merged["analysis_role"].eq(CALIBRATION_ROLE)])
        chooser = utility_chooser(slope)
        print(f"GATE B frozen slope {slope:+.4f} fitted on {CALIBRATION_ROLE} only")
        validation = merged[merged["analysis_role"].eq(VALIDATION_ROLE)]
        # Correlation is reported but is NOT the gate, because a utility can track the shape of the curve
        # closely and still put its optimum in the wrong place -- and only the place is used to select.
        # The original H4 analysis reported R2 0.825 on a curve whose argmin selection was worthless.
        pearson = stats.pearsonr(validation["delta_utility"], validation["delta_bpb"])
        spearman = stats.spearmanr(validation["delta_utility"], validation["delta_bpb"])
        print(
            f"  shape agreement (not the gate): Pearson {pearson.statistic:+.3f} "
            f"(p={pearson.pvalue:.1g}), Spearman {spearman.statistic:+.3f}, n={len(validation)}"
        )
        results = {"utility (frozen mapping)": selection_regret(validation, chooser)}
        results["null: always q=1.0"] = selection_regret(validation, lambda g: 1.0)
        results[f"null: always q={GRID_CENTRE}"] = selection_regret(validation, lambda g: GRID_CENTRE)
        for name, result in results.items():
            print(report(name, result))
        summaries[label]["gate_b"] = results

        print("GATE C transport, same frozen mapping")
        transport = {}
        for role, group in merged[~merged["analysis_role"].isin((CALIBRATION_ROLE, VALIDATION_ROLE))].groupby(
            "analysis_role"
        ):
            transport[str(role)] = selection_regret(group, chooser)
            print(report(str(role), transport[str(role)]))
        for label_ckpt, group in merged.groupby("parent_checkpoint_label"):
            transport[f"checkpoint:{label_ckpt}"] = selection_regret(group, chooser)
            print(report(f"checkpoint:{label_ckpt}", transport[f"checkpoint:{label_ckpt}"]))
        summaries[label]["gate_c"] = transport

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "gates.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    raw.to_csv(args.output / "macro_rollout_readouts.csv", index=False)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
