# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the standalone MoE reproducer in NVIDIA's JAX container."""

import argparse
import runpy
import sys
from dataclasses import dataclass

from fray.cluster import ResourceConfig

from experiments.grug.dispatch import dispatch_grug_training_run

NVIDIA_JAX_IMAGE = "nvcr.io/nvidia/jax:26.06-py3"


@dataclass(frozen=True)
class StandaloneTrial:
    arguments: tuple[str, ...]


def _run_trial(trial: StandaloneTrial) -> None:
    sys.argv = ["grug_moe_mfu.py", *trial.arguments]
    runpy.run_module("experiments.grug.moe.standalone.grug_moe_mfu", run_name="__main__")


def launch_trial(
    *,
    run_id: str,
    arguments: tuple[str, ...],
    replicas: int,
    gpus_per_node: int,
) -> None:
    resources = ResourceConfig.with_gpu(
        "GB200",
        count=gpus_per_node,
        cpu=32,
        ram="256g",
        disk="256g",
        replicas=replicas,
        image=NVIDIA_JAX_IMAGE,
    )
    dispatch_grug_training_run(
        run_id=run_id,
        config=StandaloneTrial(arguments),
        local_entrypoint=_run_trial,
        resources=resources,
        max_retries_failure=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--replicas", type=int, default=16)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = tuple(args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments)
    if not arguments:
        raise ValueError("Pass standalone grug_moe_mfu.py arguments after '--'.")
    launch_trial(
        run_id=args.run_id,
        arguments=arguments,
        replicas=args.replicas,
        gpus_per_node=args.gpus_per_node,
    )


if __name__ == "__main__":
    main()
