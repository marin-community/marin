# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute bounded, size-specific batch and layout searches through SFTSpec."""

import argparse
import json
import math
from dataclasses import asdict, replace

import draccus
from iris.client.client import JobFailedError, iris_ctx
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.snowball_isoflop.budget import budgeted_map
from experiments.post_training.snowball_isoflop.config import (
    PREFIX,
    SEQ_LEN,
    SIZES,
    VERSION,
    training_flops_per_position,
)
from experiments.post_training.snowball_isoflop.measurements import read_json, training_history, warm_measurement
from experiments.post_training.snowball_isoflop.probes import (
    Probe,
    at_budget,
    batch_probes,
    build_probe_pod,
    layout_probes,
)
from experiments.post_training.snowball_isoflop.run import run_training_pod


def run_probe(probe: Probe) -> dict:
    pod = build_probe_pod(probe)
    root = StoragePath(pod.output_path)
    if (root / "consumed-config.json").exists():
        raise FileExistsError(f"Probe artifacts already exist: {root}")
    result = {
        "probe": asdict(probe),
        "output": str(root),
        "positions": probe.steps * probe.batch_size * SEQ_LEN,
        "reference_flops": probe.steps * probe.batch_size * SEQ_LEN * training_flops_per_position(probe.size),
    }
    try:
        run_training_pod(probe.cell, pod)
    except JobFailedError as error:
        context = iris_ctx()
        assert context.job_id is not None and context.client is not None
        if not error.job_id.to_wire().startswith(f"{context.job_id.to_wire()}/"):
            raise ValueError("Probe failure belongs to another job tree") from error
        job = context.client.job(error.job_id)
        oom_logs = job.logs(substring="RESOURCE_EXHAUSTED", max_lines=8, tail=True)
        if not oom_logs:
            oom_logs = job.logs(substring="out of memory", max_lines=8, tail=True)
        if not oom_logs:
            raise
        result.update(status="memory_limit", error=str(error))
        (root / "probe-result.json").write_text(json.dumps(result, indent=2))
        return result
    gate = read_json(str(root / "initial-eval-match.json"))
    if not gate["passed"]:
        raise ValueError("Probe completed without passing its initial evaluation gate")
    state, rows = training_history(probe.cell.name, probe.steps)
    if state != "finished":
        raise ValueError(f"Completed probe has unfinished W&B state: {state}")
    (root / "bounded-history.json").write_text(json.dumps(rows, indent=2))
    training = [row for row in rows if "train/loss" in row]
    if {int(row["global_step"]) for row in training} != set(range(probe.steps)):
        raise ValueError("Completed probe has an incomplete training loss history")
    if any(not math.isfinite(float(row["train/loss"])) for row in training):
        raise ValueError("Probe loss history is nonfinite")
    result.update(status="complete", initial_eval=gate)
    if probe.phase.startswith("B") or probe.phase == "C-throughput":
        result["measurement"] = asdict(warm_measurement(rows, probe.size, probe.batch_size))
    if probe.phase.startswith("C"):
        keys = ("train/moe/sender_dropped", "train/moe/receiver_dropped")
        if any(any(key not in row for key in keys) for row in training):
            result["status"] = "missing_dispatch_counts"
        elif any(float(row[key]) != 0 for row in training for key in keys):
            result["status"] = "routing_drops"
    result["loss_curve"] = [
        {
            key: value
            for key, value in row.items()
            if key in ("global_step", "train/loss", "eval/loss", "eval/macro_loss")
        }
        for row in rows
    ]
    (root / "probe-result.json").write_text(json.dumps(result, indent=2))
    return result


def batch_saturation(size: int) -> dict:
    candidates = batch_probes(size)
    diagnostic = run_probe(candidates[0])
    reports = []
    valid = []
    small_gains = 0
    selected = None
    reason = "search_limit_before_bracketed_saturation"
    pending = list(candidates[1:])
    accumulating = False
    while pending and len(reports) < 5:
        probe = pending.pop(0)
        report = run_probe(probe)
        reports.append(report)
        memory_limited = report["status"] == "memory_limit"
        headroom = None if memory_limited else report["measurement"]["minimum_memory_headroom_fraction"]
        if not memory_limited and headroom is None:
            reason = "missing_memory_statistics"
            break
        if memory_limited or headroom < 0.1:
            reason = "memory_limit_before_bracketed_saturation"
            if accumulating or not valid:
                break
            # Spend the remaining bounded probes on accumulation at the last
            # safe microbatch, which amortizes optimizer/offload cost without
            # increasing per-microbatch activation memory.
            accumulating = True
            safe = draccus.decode(Probe, valid[-1]["probe"])
            pending = [replace(safe, accumulation=safe.accumulation * 2**i) for i in range(1, 6 - len(reports))]
            continue
        valid.append(report)
        selected = report["probe"]
        if len(valid) < 2:
            continue
        current = valid[-1]["measurement"]["inclusive_positions_per_second_per_gpu"]
        previous = valid[-2]["measurement"]["inclusive_positions_per_second_per_gpu"]
        small_gains = small_gains + 1 if current / previous < 1.05 else 0
        if small_gains == 2:
            selected = valid[-3]["probe"]
            reason = "two_consecutive_gains_below_5_percent"
            break
    return {
        "size": size,
        "selected": selected,
        "reason": reason,
        "accumulation_growth_used": accumulating,
        "accumulation_diagnostic": diagnostic,
        "probes": reports,
    }


def layout_throughput(size: int) -> dict:
    batch_report = read_json(f"{PREFIX}/{VERSION}/manager-B/completed-{size}.json")["result"]
    if batch_report["selected"] is None:
        raise ValueError(f"B did not select a safe batch for d{size}")
    batch = draccus.decode(Probe, batch_report["selected"])
    reports = [run_probe(probe) for probe in layout_probes(batch)]
    valid = [
        report
        for report in reports
        if report["status"] == "complete"
        and report["measurement"]["sender_dropped"] == 0
        and report["measurement"]["receiver_dropped"] == 0
        and report["measurement"]["minimum_memory_headroom_fraction"] is not None
        and report["measurement"]["minimum_memory_headroom_fraction"] >= 0.1
    ]
    if reports[0] not in valid:
        raise ValueError("The baseline layout failed the C correctness/headroom gate")
    ring = [report for report in valid if report["probe"]["layout"]["dispatch"] == "ring_gather_combine"]
    if not ring:
        raise ValueError("C has no valid ring_gather_combine layout for its required loss comparison")
    finalists = [
        reports[0],
        max(ring, key=lambda report: report["measurement"]["inclusive_positions_per_second_per_gpu"]),
    ]
    others = [report for report in valid if report not in finalists]
    if not others:
        raise ValueError("C requires a baseline and two valid finalists")
    finalists.append(max(others, key=lambda report: report["measurement"]["inclusive_positions_per_second_per_gpu"]))
    return {"size": size, "probes": reports, "finalists": [report["probe"] for report in finalists]}


def layout_loss(size: int) -> dict:
    throughput = read_json(f"{PREFIX}/{VERSION}/manager-C-throughput/completed-{size}.json")["result"]
    reports = [run_probe(at_budget(draccus.decode(Probe, probe), 3e18, "C-loss")) for probe in throughput["finalists"]]
    if any(report["status"] != "complete" for report in reports):
        raise ValueError("A C loss finalist did not complete its matched budget")
    return {"size": size, "probes": reports}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("B", "C-throughput", "C-loss"), required=True)
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--h100-hour-limit", type=float, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    calibration = read_json(args.calibration_report)
    if calibration["version"] != VERSION or not calibration["all_six_cells_valid"]:
        raise ValueError("B/C requires valid baseline calibration")
    prior = 0.0
    if args.phase == "C-loss":
        prior = read_json(f"{PREFIX}/{VERSION}/manager-C-throughput/allocation-ledger.json")["h100_hours"]
    ceiling = 144 if args.phase == "B" else 384
    if not 0 < args.h100_hour_limit <= ceiling - prior:
        raise ValueError("The requested remaining phase budget exceeds its approved ceiling")
    operations = {"B": batch_saturation, "C-throughput": layout_throughput, "C-loss": layout_loss}
    if not args.run:
        print(
            json.dumps({"phase": args.phase, "sizes": SIZES, "prior_h100_hours": prior, "limit": args.h100_hour_limit})
        )
        return
    budgeted_map(
        {str(size): size for size in SIZES},
        operations[args.phase],
        StoragePath(f"{PREFIX}/{VERSION}/manager-{args.phase}"),
        args.h100_hour_limit,
        3,
    )


if __name__ == "__main__":
    main()
