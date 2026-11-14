# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bound a node-level first-update kernel trace of the native SFTSpec recipe."""

import argparse
import json

import draccus
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.hooks.nsys import NsysHook
from iris.rpc.proto_display import priority_band_value
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env, run_levanter_train_lm
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.post_training.snowball_isoflop.budget import budgeted_map
from experiments.post_training.snowball_isoflop.config import (
    PREFIX,
    TASK_TIMEOUT,
    VERSION,
    Cell,
    Mask,
    build_pod,
    resources,
)

PROFILE_DELAY = 360
PROFILE_DURATION = 300


def profile_cell(size: int) -> Cell:
    return Cell(size, Mask.ASSISTANT, 0, 20, (20,), "diagnostic-kernel-profile")


def profile_hook(size: int) -> NsysHook:
    cell = profile_cell(size)
    return NsysHook(
        output_uri=f"{PREFIX}/{VERSION}/{cell.name}/nsys",
        tasks="first",
        trace="cuda,nvtx",
        delay=PROFILE_DELAY,
        duration=PROFILE_DURATION,
        graph_trace="node",
    )


def submit_profile(size: int) -> str:
    cell = profile_cell(size)
    pod = build_pod(cell)
    root = StoragePath(pod.output_path)
    if (root / "consumed-config.json").exists():
        raise FileExistsError(f"Profile artifacts already exist: {root}")
    (root / "consumed-config.json").write_text(json.dumps(draccus.encode(pod), indent=2))
    environment = resolve_training_env({**(pod.env_vars or {}), "JAX_PLATFORMS": "cuda,cpu"}, resources())
    command = profile_hook(size).wrap(
        [
            "python",
            "-m",
            "iris.jax.multigpu_main",
            "--nproc",
            "8",
            "--devices-per-proc",
            "1",
            "--",
            "python",
            "-u",
            "-m",
            "experiments.post_training.snowball_isoflop.profile_stall",
            "--size",
            str(size),
            "--train",
        ]
    )
    (root / "profile-command.json").write_text(json.dumps(command, indent=2))
    job = current_client().submit(
        JobRequest(
            name=f"profile-{cell.name}-{VERSION}",
            entrypoint=Entrypoint.from_binary(command[0], command[1:]),
            resources=resources(),
            environment=create_environment(extras=extras_for_resources(resources()), env_vars=environment),
            timeout=Duration.from_seconds(TASK_TIMEOUT),
            processes_per_task=1,
            priority=priority_band_value("interactive"),
            max_retries_preemption=0,
        )
    )
    (root / "dispatch.json").write_text(json.dumps({"job_id": job.job_id}))
    # A bounded trace intentionally terminates its target; this is not a valid
    # calibration trajectory. The outer wrapper traces all eight rank children.
    job.wait(raise_on_failure=False)
    reports = (root / "nsys/*.nsys-rep").glob()
    (root / "profile-reports.json").write_text(json.dumps([str(report) for report in reports], indent=2))
    if len(reports) != 1:
        raise ValueError(f"Expected one eight-rank profile, found {len(reports)}")
    return str(root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(512, 768, 1024), default=1024)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--h100-hour-limit", type=float, default=3)
    args = parser.parse_args()
    cell = profile_cell(args.size)
    pod = build_pod(cell)
    if args.train:
        run_levanter_train_lm(pod)
        return
    if not args.run:
        print(json.dumps({"pod": draccus.encode(pod), "profile": draccus.encode(profile_hook(args.size))}))
        return
    if not 0 < args.h100_hour_limit <= 3:
        parser.error("The profile must fit its 3-H100-hour diagnostic envelope inside A")
    gate = pod.train_config.initial_eval_match
    if gate is None or gate.pre_reseed_reference_path is None:
        raise ValueError("The profile requires both initialization references")
    for path in (gate.reference_path, gate.pre_reseed_reference_path):
        if not StoragePath(path).exists():
            raise ValueError(f"Missing initialization reference: {path}")
    budgeted_map(
        {str(args.size): args.size},
        submit_profile,
        StoragePath(f"{PREFIX}/{VERSION}/manager-{cell.name}"),
        args.h100_hour_limit,
        1,
    )


if __name__ == "__main__":
    main()
