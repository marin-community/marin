# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run equal LR tuning and paired optimizer repeats after batch/layout selection."""

import argparse
import json
from dataclasses import asdict, replace

import draccus
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.snowball_isoflop.budget import budgeted_map
from experiments.post_training.snowball_isoflop.config import INPUTS, PREFIX, SIZES, VERSION
from experiments.post_training.snowball_isoflop.measurements import read_json
from experiments.post_training.snowball_isoflop.probes import Probe, at_budget
from experiments.post_training.snowball_isoflop.run_probes import run_probe

OPTIMIZERS = ("grug_moe_muonh_v1", "adam")


def lr_candidates(size: int, optimizer: str) -> tuple[float, ...]:
    pretrain = json.loads((INPUTS / f"d{size}-pretrain-settings.json").read_text())["optimizer"]
    center = 5e-5 if optimizer == "grug_moe_muonh_v1" else 5e-5 * 3 / 13
    upper = pretrain["learning_rate"] if optimizer == "grug_moe_muonh_v1" else pretrain["adam_lr"]
    return center / 3, center, upper


def selected_layout(size: int, selection_path: str) -> Probe:
    selection = read_json(selection_path)
    if selection["version"] != VERSION:
        raise ValueError("Batch/layout selection is from another configuration version")
    loss_report = read_json(f"{PREFIX}/{VERSION}/manager-C-loss/completed-{size}.json")["result"]
    matches = [
        report
        for report in loss_report["probes"]
        if draccus.decode(Probe, report["probe"]).cell.name == selection["selected_cells"][str(size)]
    ]
    if len(matches) != 1 or matches[0]["status"] != "complete":
        raise ValueError("Selected batch/layout must have completed its C loss comparison")
    return draccus.decode(Probe, matches[0]["probe"])


def final_loss(report: dict) -> float:
    steps = report["probe"]["steps"]
    final = [row for row in report["loss_curve"] if row.get("global_step") == steps - 1 and "eval/loss" in row]
    if len(final) != 1:
        raise ValueError("Optimizer comparison lacks one final assistant-target validation")
    return float(final[0]["eval/loss"])


def sweep(size: int, selection_path: str) -> dict:
    selected = selected_layout(size, selection_path)
    results = []
    winners = {}
    for optimizer in OPTIMIZERS:
        reports = [
            run_probe(
                at_budget(
                    replace(selected, optimizer=optimizer, learning_rate=lr, seed=0),
                    3e18,
                    "D",
                )
            )
            for lr in lr_candidates(size, optimizer)
        ]
        if any(report["status"] != "complete" for report in reports):
            raise ValueError("Every optimizer requires three valid equal-budget LR trials")
        # The primary metric is assistant-token micro cross entropy; preserve
        # macro/source curves alongside it so tradeoffs remain inspectable.
        winner = min(reports, key=final_loss)
        winners[optimizer] = {"probe": winner["probe"], "final_loss": final_loss(winner)}
        results.extend(reports)
    return {"size": size, "selection": selection_path, "trials": results, "winners": winners}


def q5_probes() -> tuple[Probe, ...]:
    probes = []
    for size in SIZES:
        tuning = read_json(f"{PREFIX}/{VERSION}/manager-D/completed-{size}.json")["result"]
        if len(tuning["trials"]) != 6 or any(trial["status"] != "complete" for trial in tuning["trials"]):
            raise ValueError("Q5 requires equal completed LR tuning for both optimizers")
        for optimizer in OPTIMIZERS:
            selected = draccus.decode(Probe, tuning["winners"][optimizer]["probe"])
            probes.extend(at_budget(replace(selected, seed=seed), 1e19, "Q5") for seed in (0, 1, 2))
    return tuple(probes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("D", "Q5"), required=True)
    parser.add_argument("--selection")
    parser.add_argument("--h100-hour-limit", type=float, required=True)
    parser.add_argument("--max-concurrent", type=int, choices=range(1, 9), default=3)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    ceiling = 384 if args.phase == "D" else 1152
    if not 0 < args.h100_hour_limit <= ceiling:
        parser.error("Optimizer phase exceeds its approved H100-hour ceiling")
    manager = StoragePath(f"{PREFIX}/{VERSION}/manager-{args.phase}")
    if args.phase == "D":
        if args.selection is None:
            parser.error("D requires a recorded per-size batch/layout selection from C loss results")
        selections = {str(size): selected_layout(size, args.selection) for size in SIZES}
        if not args.run:
            print(json.dumps({"selected": {key: asdict(p) for key, p in selections.items()}}))
            return
        (manager / "registered-grid.json").write_text(
            json.dumps(
                {
                    "version": VERSION,
                    "selection": args.selection,
                    "reference_flops_per_trial": 3e18,
                    "lr_candidates": {
                        str(size): {optimizer: lr_candidates(size, optimizer) for optimizer in OPTIMIZERS}
                        for size in SIZES
                    },
                },
                indent=2,
            )
        )
        budgeted_map(
            {str(size): size for size in SIZES},
            lambda size: sweep(size, args.selection),
            manager,
            args.h100_hour_limit,
            min(3, args.max_concurrent),
        )
        return
    probes = q5_probes()
    if not args.run:
        print(json.dumps([asdict(probe) for probe in probes]))
        return
    (manager / "registered-grid.json").write_text(json.dumps([asdict(probe) for probe in probes], indent=2))
    budgeted_map(
        {probe.cell.name: probe for probe in probes}, run_probe, manager, args.h100_hour_limit, args.max_concurrent
    )


if __name__ == "__main__":
    main()
