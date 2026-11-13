# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate pinned pretrained bases through Levanter's independent eval entrypoint."""

import argparse
import json
from dataclasses import replace
from enum import StrEnum
from functools import partial

import draccus
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.rpc.proto_display import priority_band_value
from levanter.compat.hf_checkpoints import RepoRef
from levanter.main.eval_lm import EvalLmConfig
from levanter.main.eval_lm import main as eval_main
from levanter.tracker.json_file import JsonFileTrackerConfig
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.post_training.snowball_isoflop.budget import budgeted_map
from experiments.post_training.snowball_isoflop.config import (
    PREFIX,
    REFERENCE_VERSION,
    SEQ_LEN,
    SIZES,
    SPECIAL_IDS,
    TASK_TIMEOUT,
    Cell,
    Mask,
    build_pod,
    model_receipt,
    reference_data_config,
    resources,
)


class ReferenceVariant(StrEnum):
    RAW_RING = "raw-ring"
    RAW_SONIC = "raw-sonic"
    RESEEDED_SONIC = "reseeded-sonic"


def reference_config(size: int, variant: ReferenceVariant = ReferenceVariant.RAW_RING) -> EvalLmConfig:
    cell = Cell(size, Mask.FULL, 0, 20, (20,), "reference")
    base = build_pod(cell).train_config
    suffix = "" if variant == ReferenceVariant.RAW_RING else f"-{variant}"
    root = f"{PREFIX}/{REFERENCE_VERSION}/reference-d{size}{suffix}"
    receipt = model_receipt(size)
    trainer = replace(
        base.trainer,
        id=f"{REFERENCE_VERSION}-reference-d{size}{suffix}",
        max_eval_batches=1,
        tracker=JsonFileTrackerConfig(output_path=root),
    )
    return EvalLmConfig(
        hf_checkpoint=RepoRef.from_string(f"{receipt['repo']}@{receipt['revision']}"),
        trainer=trainer,
        data=reference_data_config(),
        max_eval_length=SEQ_LEN,
        model=replace(
            base.model, moe_implementation="ring_gather_combine" if variant == ReferenceVariant.RAW_RING else "sonic"
        ),
        snowball_sft=base.snowball_sft if variant == ReferenceVariant.RESEEDED_SONIC else None,
        audit_token_ids=SPECIAL_IDS,
        result_path=f"{root}/result.json",
    )


def evaluate_reference(size: int, variant: ReferenceVariant) -> None:
    eval_main(reference_config(size, variant))


def submit_reference(size: int, variant: ReferenceVariant) -> None:
    config = reference_config(size, variant)
    assert config.result_path is not None
    root = StoragePath(config.result_path).parent
    if (root / "result.json").exists():
        raise FileExistsError(f"Reference result already exists for d{size}")
    (root / "consumed-config.json").write_text(json.dumps(draccus.encode(config), indent=2))
    environment = resolve_training_env(
        {
            "XLA_FLAGS": "--xla_gpu_shard_autotuning=false --xla_gpu_nccl_termination_timeout_seconds=600",
            "JAX_PLATFORMS": "cuda,cpu",
        },
        resources(),
    )
    job = current_client().submit(
        JobRequest(
            name=config.trainer.id,
            entrypoint=Entrypoint.from_callable(evaluate_reference, args=[size, variant]),
            resources=resources(),
            environment=create_environment(extras=extras_for_resources(resources()), env_vars=environment),
            timeout=Duration.from_seconds(TASK_TIMEOUT),
            processes_per_task=8,
            priority=priority_band_value("interactive"),
            max_retries_preemption=0,
        )
    )
    (root / "dispatch.json").write_text(json.dumps({"job_id": job.job_id}))
    job.wait(raise_on_failure=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--size", type=int, choices=SIZES)
    parser.add_argument(
        "--variant", type=ReferenceVariant, choices=tuple(ReferenceVariant), default=ReferenceVariant.RAW_RING
    )
    parser.add_argument("--h100-hour-limit", type=float, default=24)
    args = parser.parse_args()
    sizes = (args.size,) if args.size is not None else SIZES
    if not args.run:
        print(json.dumps({size: draccus.encode(reference_config(size, args.variant)) for size in sizes}))
        return
    if not 0 < args.h100_hour_limit <= 24:
        parser.error("References require a cap within the approved 24-H100-hour envelope")
    root = StoragePath(f"{PREFIX}/{REFERENCE_VERSION}/reference-manager-{args.variant}-{args.size or 'all'}")
    budgeted_map(
        {str(size): size for size in sizes},
        partial(submit_reference, variant=args.variant),
        root,
        args.h100_hour_limit,
        3,
    )


if __name__ == "__main__":
    main()
