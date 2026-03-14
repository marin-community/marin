# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
- GDN_PROFILE_GDN_LAYERS_PER_BLOCK: optional GDN layer-count-per-block override
- GDN_PROFILE_GDN_BLOCK_SIZE: optional GDN block-size override
- GDN_PROFILE_GDN_KERNEL_ENTRY_BRANCH_CORE_SHARDING_DIAGNOSTIC: if `1`, enable
  the opt-in prepared-array leaf-call branch-core sharding diagnostic
- GDN_PROFILE_GDN_BRANCH_BOUNDARY_PROTOTYPE: if `1`, enable the opt-in array-only GDN branch boundary prototype
- GDN_PROFILE_DECODER_BLOCK_BOUNDARY_PROTOTYPE: if `1`, enable the opt-in fixed-`3/4` decoder-block boundary prototype
- GDN_PROFILE_GRADIENT_CHECKPOINTING: optional transformer-layer checkpointing override
  (`true`, `false`, `offload`, `recompute`, `full`, `save_all`, `nested`)
- GDN_PROFILE_ALL_TRANSFORMER: if `1`, disable all GDN layers and benchmark an all-transformer stack
- GDN_PROFILE_RUN_NAME_PREFIX: run-name prefix (default: gdn_tinyprof)
- GDN_PROFILE_RUN_NAME_SUFFIX: optional run-name suffix
"""

import dataclasses
import os

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig

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


def _env_optional_gradient_checkpointing(name: str) -> bool | str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"offload", "recompute", "full", "save_all", "nested"}:
        return normalized
    raise ValueError(f"{name} must be one of true/false/offload/recompute/full/save_all/nested, got {value!r}")


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _run_name(
    size: str,
    steps: int,
    chunk_size: int,
    segment_size: int,
    *,
    all_transformer: bool,
    gdn_layers_per_block: int,
    gdn_block_size: int,
) -> str:
    prefix = os.environ.get("GDN_PROFILE_RUN_NAME_PREFIX", "gdn_tinyprof")
    suffix = os.environ.get("GDN_PROFILE_RUN_NAME_SUFFIX", "").strip()

    arch = "attnonly" if all_transformer else f"gdn{gdn_layers_per_block}of{gdn_block_size}"
    name = f"{prefix}_{arch}_{size}_ch{chunk_size}_seg{segment_size}_{steps}steps"
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
    gdn_layers_per_block_override = _env_optional_int("GDN_PROFILE_GDN_LAYERS_PER_BLOCK")
    gdn_block_size_override = _env_optional_int("GDN_PROFILE_GDN_BLOCK_SIZE")
    gdn_kernel_entry_branch_core_sharding_diagnostic = _env_flag(
        "GDN_PROFILE_GDN_KERNEL_ENTRY_BRANCH_CORE_SHARDING_DIAGNOSTIC"
    )
    gdn_branch_boundary_prototype = _env_flag("GDN_PROFILE_GDN_BRANCH_BOUNDARY_PROTOTYPE")
    decoder_block_boundary_prototype = _env_flag("GDN_PROFILE_DECODER_BLOCK_BOUNDARY_PROTOTYPE")
    gradient_checkpointing_override = _env_optional_gradient_checkpointing("GDN_PROFILE_GRADIENT_CHECKPOINTING")
    all_transformer = _env_flag("GDN_PROFILE_ALL_TRANSFORMER")

    if batch_size_override is None:
        batch_size_override = _SAFE_BATCH_SIZE_BY_SIZE.get(size, 8)

    _, base_cfg = build_run(size, use_gpu=False)

    model_cfg = base_cfg.model_config
    if chunk_size_override is not None:
        model_cfg = dataclasses.replace(model_cfg, gdn_chunk_size=chunk_size_override)
    if segment_size_override is not None:
        model_cfg = dataclasses.replace(model_cfg, gdn_segment_size=segment_size_override)
    if gdn_layers_per_block_override is not None:
        model_cfg = dataclasses.replace(model_cfg, gdn_layers_per_block=gdn_layers_per_block_override)
    if gdn_block_size_override is not None:
        model_cfg = dataclasses.replace(model_cfg, gdn_block_size=gdn_block_size_override)
    if gdn_kernel_entry_branch_core_sharding_diagnostic:
        model_cfg = dataclasses.replace(model_cfg, gdn_use_kernel_entry_branch_core_sharding_diagnostic=True)
    if gdn_branch_boundary_prototype:
        model_cfg = dataclasses.replace(model_cfg, gdn_use_branch_boundary_prototype=True)
    if decoder_block_boundary_prototype:
        model_cfg = dataclasses.replace(model_cfg, gdn_use_decoder_block_boundary_prototype=True)
    if gradient_checkpointing_override is not None:
        model_cfg = dataclasses.replace(model_cfg, gradient_checkpointing=gradient_checkpointing_override)
    if all_transformer:
        model_cfg = dataclasses.replace(
            model_cfg,
            use_gated_deltanet=False,
            gdn_layers_per_block=0,
        )

    base_profiler = base_cfg.train_config.profiler
    profiler_cfg = (
        dataclasses.replace(
            base_profiler,
            enabled=True,
            start_step=profile_start_step,
            num_steps=profile_num_steps,
            perfetto_link=False,
        )
        if isinstance(base_profiler, ProfilerConfig)
        else ProfilerConfig(
            enabled=True, start_step=profile_start_step, num_steps=profile_num_steps, perfetto_link=False
        )
    )

    train_cfg = dataclasses.replace(
        base_cfg.train_config,
        resources=ResourceConfig.with_tpu(tpu_variant),
        num_train_steps=num_steps,
        profiler=profiler_cfg,
        steps_per_hf_export=-1,
    )
    train_cfg = dataclasses.replace(train_cfg, train_batch_size=batch_size_override)

    run_name = _run_name(
        size,
        num_steps,
        model_cfg.gdn_chunk_size,
        model_cfg.gdn_segment_size,
        all_transformer=all_transformer,
        gdn_layers_per_block=model_cfg.gdn_layers_per_block,
        gdn_block_size=model_cfg.gdn_block_size,
    )

    tags = ["speedrun", "gdn", "gdn_tiny_profile", "kernel_optimization"]
    if all_transformer:
        tags.append("attn_only_baseline")
    else:
        tags.append(f"gdn_fraction_{model_cfg.gdn_layers_per_block}of{model_cfg.gdn_block_size}")

    gdn_layer_fraction = 0.0
    if model_cfg.use_gated_deltanet and model_cfg.gdn_block_size > 0:
        gdn_layer_fraction = model_cfg.num_gdn_layers / model_cfg.num_layers
    resolved_all_transformer = int(model_cfg.num_gdn_layers == 0)
    print(
        "[gdnctl] GDN profile model: "
        f"all_transformer={resolved_all_transformer} "
        f"gdn_layers_per_block={model_cfg.gdn_layers_per_block} "
        f"gdn_block_size={model_cfg.gdn_block_size} "
        f"kernel_entry_branch_core_sharding_diagnostic="
        f"{int(model_cfg.gdn_use_kernel_entry_branch_core_sharding_diagnostic)} "
        f"branch_boundary_prototype={int(model_cfg.gdn_use_branch_boundary_prototype)} "
        f"decoder_block_boundary_prototype={int(model_cfg.gdn_use_decoder_block_boundary_prototype)} "
        f"gradient_checkpointing={model_cfg.gradient_checkpointing} "
        f"gdn_layer_fraction={gdn_layer_fraction:.6f}",
        flush=True,
    )

    step = default_train(
        name=f"speedrun/{run_name}",
        tokenized=base_cfg.tokenized_dataset,
        model_config=model_cfg,
        train_config=train_cfg,
        tags=tags,
        use_default_validation=False,
        eval_harness_tasks=[],
        wandb_group=os.environ.get("WANDB_GROUP", "gdn-tiny-profile"),
    )

    executor_main(
        steps=[step],
        description="Lightweight profiled GDN training run for TPU kernel optimization.",
    )
