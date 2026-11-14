# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bound one native calibration control that serializes GPU collectives."""

import argparse
import json
from dataclasses import replace

import draccus
from marin.training.training import TrainLmOnPodConfig
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.snowball_isoflop.budget import budgeted_map
from experiments.post_training.snowball_isoflop.config import PREFIX, VERSION, Cell, Mask, build_pod
from experiments.post_training.snowball_isoflop.run import run_training_pod


def diagnostic_pod(cell: Cell) -> TrainLmOnPodConfig:
    pod = build_pod(cell)
    environment = dict(pod.env_vars or {})
    environment["XLA_FLAGS"] += " --xla_gpu_disable_async_collectives=ALLCOLLECTIVES"
    return replace(pod, env_vars=environment)


def run_control(cell: Cell) -> str:
    return run_training_pod(cell, diagnostic_pod(cell))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(512, 768, 1024), required=True)
    parser.add_argument("--mask", type=Mask, choices=tuple(Mask), required=True)
    parser.add_argument("--h100-hour-limit", type=float, default=3)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.h100_hour_limit <= 3:
        parser.error("The diagnostic control is capped at 3 H100-hours")
    cell = Cell(args.size, args.mask, 0, 20, (20,), "diagnostic-serial-collectives")
    pod = diagnostic_pod(cell)
    if not args.run:
        print(json.dumps(draccus.encode(pod)))
        return
    gate = pod.train_config.initial_eval_match
    if gate is None or gate.pre_reseed_reference_path is None:
        raise ValueError("The control requires both initialization references")
    for path in (gate.reference_path, gate.pre_reseed_reference_path):
        if not StoragePath(path).exists():
            raise ValueError(f"Missing initialization reference: {path}")
    manager = StoragePath(f"{PREFIX}/{VERSION}/manager-{cell.name}")
    budgeted_map({cell.name: cell}, run_control, manager, args.h100_hour_limit, 1)


if __name__ == "__main__":
    main()
