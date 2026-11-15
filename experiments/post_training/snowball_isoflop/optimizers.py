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
from experiments.post_training.snowball_isoflop.probes import Probe, at_budget, build_probe_pod
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


def completed_trial_report(selected: Probe, optimizer: str, lr: float, trial_phase: str) -> dict | None:
    """Return a completed equal-budget trial report from an earlier attempt, or None.

    Preempted D attempts leave dispatched trial directories behind; a rerun with
    fresh identities would redo that work. When an earlier attempt's trial for
    this geometry completed, reuse its probe-result report instead.
    """
    probe = at_budget(replace(selected, optimizer=optimizer, learning_rate=lr, seed=0), 3e18, trial_phase)
    root = StoragePath(build_probe_pod(probe).output_path)
    if not (root / "probe-result.json").exists():
        return None
    report = read_json(str(root / "probe-result.json"))
    if report.get("status") != "complete":
        return None
    return report


def trial_attempt_phase(trial_phase: str, attempt: int) -> str:
    """Fresh probe identity for each stall retry of one LR candidate."""
    return trial_phase if attempt == 0 else f"{trial_phase}-retry{attempt}"


def run_lr_trials(
    selected: Probe,
    optimizer: str,
    trial_phase: str,
    prior_trial_phases: tuple[str, ...] = (),
    max_stalls: int = 3,
) -> tuple[list[dict], list[dict]]:
    """Run one optimizer's LR candidates at the selected batch, with stall retries.

    A watchdog stall is an infrastructure failure, not an experimental result:
    each stalled candidate retries under a fresh identity up to ``max_stalls``
    total attempts, and only then becomes a recorded exclusion. A memory limit
    is deterministic at the trial geometry and excludes immediately. Returns
    ``(completed_reports, exclusion_records)``.
    """
    reports: list[dict] = []
    exclusions: list[dict] = []
    for lr in lr_candidates(selected.size, optimizer):
        reused = next(
            (
                report
                for phase in prior_trial_phases
                if (report := completed_trial_report(selected, optimizer, lr, phase)) is not None
            ),
            None,
        )
        if reused is not None:
            reports.append(reused)
            continue
        attempts = []
        report = None
        for attempt in range(max_stalls):
            probe = at_budget(
                replace(selected, optimizer=optimizer, learning_rate=lr, seed=0),
                3e18,
                trial_attempt_phase(trial_phase, attempt),
            )
            report = run_probe(probe)
            if report["status"] == "complete":
                reports.append(report)
                report = None
                break
            attempts.append({"phase": probe.cell.phase, "status": report["status"], "error": report.get("error")})
            if report["status"] != "stall_limit":
                break
        if report is not None and report["status"] != "complete":
            exclusions.append({"optimizer": optimizer, "learning_rate": lr, "attempts": attempts})
    return reports, exclusions


def sweep(size: int, selection_path: str, trial_phase: str = "D", prior_trial_phases: tuple[str, ...] = ()) -> dict:
    selected = selected_layout(size, selection_path)
    results = []
    excluded = []
    winners = {}
    for optimizer in OPTIMIZERS:
        reports, exclusions = run_lr_trials(selected, optimizer, trial_phase, prior_trial_phases)
        if not reports:
            raise ValueError(f"Every LR candidate failed for {optimizer} on d{size}")
        # The primary metric is assistant-token micro cross entropy; preserve
        # macro/source curves alongside it so tradeoffs remain inspectable.
        winner = min(reports, key=final_loss)
        winners[optimizer] = {
            "probe": winner["probe"],
            "final_loss": final_loss(winner),
            "excluded": exclusions,
        }
        results.extend(reports)
        excluded.extend(exclusions)
    result = {
        "size": size,
        "selection": selection_path,
        "trials": results,
        "excluded": excluded,
        "winners": winners,
    }
    if excluded:
        # Prominent flag: the LR sweep ran with unrunnable candidates, so the
        # winners cover only the geometries that completed.
        result["degenerate_lr_tuning"] = True
    return result


def q5_probes() -> tuple[Probe, ...]:
    probes = []
    for size in SIZES:
        tuning = read_json(f"{PREFIX}/{VERSION}/manager-D/completed-{size}.json")["result"]
        if any(trial["status"] != "complete" for trial in tuning["trials"]):
            raise ValueError("Q5 requires equal completed LR tuning for both optimizers")
        excluded = {(record["optimizer"], record["learning_rate"]) for record in tuning.get("excluded", [])}
        if tuning.get("degenerate_lr_tuning"):
            # Prominent degenerate flag: some LR candidates were unrunnable
            # (three stall attempts each) and are absent from the sweep; the
            # winners cover only the candidates that completed.
            print(
                f"WARNING d{size}: degenerate LR tuning - excluded candidates "
                f"{sorted(excluded)}; winners cover runnable candidates only"
            )
        for optimizer in OPTIMIZERS:
            if optimizer not in tuning["winners"]:
                raise ValueError(f"Q5 requires a winner for {optimizer} on d{size}")
            accounted = {(optimizer, lr) for lr in lr_candidates(size, optimizer) if (optimizer, lr) not in excluded}
            matched = {
                (decoded.optimizer, decoded.learning_rate)
                for trial in tuning["trials"]
                if trial["status"] == "complete" and (decoded := draccus.decode(Probe, trial["probe"]))
            }
            if not accounted <= matched:
                raise ValueError(f"Q5 accounting mismatch for {optimizer} on d{size}")
            selected = draccus.decode(Probe, tuning["winners"][optimizer]["probe"])
            probes.extend(at_budget(replace(selected, seed=seed), 1e19, "Q5") for seed in (0, 1, 2))
    return tuple(probes)


def persist_sweep(size: int, result: dict, manager: StoragePath) -> dict:
    """Write a completed sweep result where Q5's runner reads it, flagging degeneracy."""
    (manager / f"completed-{size}.json").write_text(json.dumps({"result": result}, indent=2))
    if result.get("degenerate_lr_tuning"):
        (manager / "excluded-lr-candidates.json").write_text(
            json.dumps({"sizes": [size], "excluded": result["excluded"]}, indent=2)
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("D", "Q5"), required=True)
    parser.add_argument("--selection")
    parser.add_argument(
        "--trial-phase",
        default=None,
        # Attempt-scoped identity prefix: preempted trials keep their dispatch
        # markers, so each D attempt needs fresh probe identities.
        help="Attempt-scoped identity prefix for D trial probes; required to run D",
    )
    parser.add_argument(
        "--prior-trial-phases",
        default="",
        help="Comma-separated earlier trial phases whose completed trials are reused instead of rerun",
    )
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
        if args.trial_phase is None:
            parser.error("Running D requires --trial-phase: prior attempt identities keep their dispatch markers")
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
            lambda size: persist_sweep(
                size,
                sweep(size, args.selection, args.trial_phase, tuple(filter(None, args.prior_trial_phases.split(",")))),
                manager,
            ),
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
