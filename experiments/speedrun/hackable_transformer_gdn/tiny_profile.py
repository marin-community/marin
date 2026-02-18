# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lightweight profiled training entrypoint for GDN TPU kernel iteration.

This script intentionally reuses `build_run(...)` from
`hackable_transformer_gdn.py` so we profile the same architecture code path,
while overriding runtime knobs via environment variables for short feedback
loops.

Environment overrides:
- GDN_PROFILE_SIZE: one of 130m, 300m, 520m, 1_2b (default: 130m)
- GDN_PROFILE_TPU_VARIANT: TPU resource variant (default: v5p-8)
- GDN_PROFILE_NUM_STEPS: train steps (default: 20)
- GDN_PROFILE_PROFILE_START_STEP: profiler start step (default: 2)
- GDN_PROFILE_PROFILE_NUM_STEPS: profiler duration in steps (default: 6)
- GDN_PROFILE_BATCH_SIZE: optional global batch size override (default if unset: size-specific safe tiny-profile batch)
- GDN_PROFILE_CHUNK_SIZE: optional GDN chunk size override
- GDN_PROFILE_SEGMENT_SIZE: optional GDN segment size override
- GDN_PROFILE_RUN_NAME_PREFIX: run-name prefix (default: gdn_tinyprof)
- GDN_PROFILE_RUN_NAME_SUFFIX: optional run-name suffix
"""

import dataclasses
import os

from fray.cluster import ResourceConfig

from experiments.defaults import default_train
from experiments.speedrun.hackable_transformer_gdn.hackable_transformer_gdn import build_run
from marin.execution.executor import executor_main


_SAFE_BATCH_SIZE_BY_SIZE: dict[str, int] = {
    "130m": 8,
    "300m": 4,
    "520m": 2,
    "1_2b": 1,
}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _run_name(size: str, steps: int, chunk_size: int, segment_size: int) -> str:
    prefix = os.environ.get("GDN_PROFILE_RUN_NAME_PREFIX", "gdn_tinyprof")
    suffix = os.environ.get("GDN_PROFILE_RUN_NAME_SUFFIX", "").strip()

    name = f"{prefix}_{size}_ch{chunk_size}_seg{segment_size}_{steps}steps"
    if suffix:
        name = f"{name}_{suffix}"
    return name


if __name__ == "__main__":
    size = os.environ.get("GDN_PROFILE_SIZE", "130m")
    tpu_variant = os.environ.get("GDN_PROFILE_TPU_VARIANT", "v5p-8")
    num_steps = _env_int("GDN_PROFILE_NUM_STEPS", 20)
    profile_start_step = _env_int("GDN_PROFILE_PROFILE_START_STEP", 2)
    profile_num_steps = _env_int("GDN_PROFILE_PROFILE_NUM_STEPS", 6)
    batch_size_override = _env_optional_int("GDN_PROFILE_BATCH_SIZE")
    chunk_size_override = _env_optional_int("GDN_PROFILE_CHUNK_SIZE")
    segment_size_override = _env_optional_int("GDN_PROFILE_SEGMENT_SIZE")

    if batch_size_override is None:
        batch_size_override = _SAFE_BATCH_SIZE_BY_SIZE.get(size, 8)

    _, base_cfg = build_run(size, use_gpu=False)

    model_cfg = base_cfg.model_config
    if chunk_size_override is not None:
        model_cfg = dataclasses.replace(model_cfg, gdn_chunk_size=chunk_size_override)
    if segment_size_override is not None:
        model_cfg = dataclasses.replace(model_cfg, gdn_segment_size=segment_size_override)

    train_cfg = dataclasses.replace(
        base_cfg.train_config,
        resources=ResourceConfig.with_tpu(tpu_variant),
        num_train_steps=num_steps,
        profiler=True,
        profiler_start_step=profile_start_step,
        profiler_num_steps=profile_num_steps,
        steps_per_hf_export=-1,
    )
    train_cfg = dataclasses.replace(train_cfg, train_batch_size=batch_size_override)

    run_name = _run_name(size, num_steps, model_cfg.gdn_chunk_size, model_cfg.gdn_segment_size)

    step = default_train(
        name=f"speedrun/{run_name}",
        tokenized=base_cfg.tokenized_dataset,
        model_config=model_cfg,
        train_config=train_cfg,
        tags=["speedrun", "gdn", "gdn_tiny_profile", "kernel_optimization"],
        use_default_validation=False,
        eval_harness_tasks=[],
        wandb_group=os.environ.get("WANDB_GROUP", "gdn-tiny-profile"),
    )

    executor_main(
        steps=[step],
        description="Lightweight profiled GDN training run for TPU kernel optimization.",
    )
