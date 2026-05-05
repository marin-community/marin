# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic-data CoreWeave Grug MoE accelerator smoke.

This launcher intentionally avoids remote dataset dependencies so it can test
the current Grug MoE training/profiler path on a specific CoreWeave accelerator
without moving data across regions.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.data import AsyncDataset
from levanter.data.text import DirectDatasetComponent, GrugLmExample, LmDataConfig
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.tracker import NoopConfig, TrackerConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.ferries.canary_ferry import CANARY_OPTIMIZER, CANARY_TRAINER
from experiments.grug.moe.launch import GRUG_MOE_TRIAL_MODEL, GrugMoeLaunchConfig, run_grug_moe_trial

DEFAULT_STEPS = 50
DEFAULT_PROFILE_STEPS = 10
DEFAULT_SYNTHETIC_EXAMPLES = 1_000_000


@dataclass(frozen=True)
class SyntheticGrugDataset(AsyncDataset[GrugLmExample]):
    """Deterministic token stream for Grug training smokes."""

    seq_len: int
    vocab_size: int
    size: int = DEFAULT_SYNTHETIC_EXAMPLES
    seed: int = 0

    async def async_len(self) -> int:
        return self.size

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[GrugLmExample]:
        position = np.arange(self.seq_len, dtype=np.int32)
        examples: list[GrugLmExample] = []
        for index in indices:
            tokens = (position + self.seed + index * 9973) % self.vocab_size
            loss_weight = np.ones(self.seq_len, dtype=np.float32)
            loss_weight[-1] = 0.0
            examples.append(
                GrugLmExample(
                    tokens=tokens,
                    loss_weight=loss_weight,
                    attn_mask=GrugAttentionMask.causal(),
                )
            )
        return examples


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    return int(raw) if raw else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "")
    return float(raw) if raw else default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "")
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _run_id() -> str:
    return (
        os.environ.get("CW_GRUG_RUN_ID")
        or os.environ.get("RUN_ID")
        or datetime.datetime.now(datetime.timezone.utc).strftime("cw-grug-smoke-%Y%m%d-%H%M%S")
    )


def _gpu_resources() -> ResourceConfig:
    gpu_type = os.environ.get("CW_GRUG_GPU_TYPE", "H100")
    gpu_count = _env_int("CW_GRUG_GPU_COUNT", 8)
    replicas = _env_int("CW_GRUG_GPU_REPLICAS", 1)
    cpu = _env_float("CW_GRUG_CPU", 32)
    ram = os.environ.get("CW_GRUG_RAM", "256g")
    disk = os.environ.get("CW_GRUG_DISK", "256g")
    return ResourceConfig.with_gpu(gpu_type, count=gpu_count, cpu=cpu, ram=ram, disk=disk, replicas=replicas)


def _synthetic_data() -> LmDataConfig:
    return LmDataConfig(
        tokenizer="passthrough",
        vocab_size=GRUG_MOE_TRIAL_MODEL.vocab_size,
        shuffle=False,
        components={
            "synthetic": DirectDatasetComponent(
                datasets={
                    "train": SyntheticGrugDataset(
                        seq_len=GRUG_MOE_TRIAL_MODEL.max_seq_len,
                        vocab_size=GRUG_MOE_TRIAL_MODEL.vocab_size,
                    )
                }
            )
        },
    )


def _tracker_from_env(tags: list[str]) -> TrackerConfig | tuple[TrackerConfig, ...]:
    mode = os.environ.get("CW_GRUG_TRACKER", "wandb").lower()
    json_logger = JsonLoggerConfig(logger_name=os.environ.get("CW_GRUG_JSON_LOGGER", "cw_grug.metrics"))
    wandb = WandbConfig(
        project=os.environ.get("WANDB_PROJECT", "marin"),
        tags=tags,
        group=os.environ.get("CW_GRUG_WANDB_GROUP", "cw-grug-accelerator-smoke"),
        mode=os.environ.get("WANDB_MODE", "offline"),
        name=None,
        replicate_path=this_output_path(),
    )
    if mode == "json_logger":
        return json_logger
    if mode == "both":
        return (wandb, json_logger)
    if mode == "noop":
        return NoopConfig()
    if mode != "wandb":
        raise ValueError(f"Unknown CW_GRUG_TRACKER={mode!r}; expected wandb, json_logger, both, or noop")
    return wandb


def _build_step_from_env() -> ExecutorStep:
    gpu_type = os.environ.get("CW_GRUG_GPU_TYPE", "H100")
    gpu_count = _env_int("CW_GRUG_GPU_COUNT", 8)
    run_id = _run_id()
    steps = _env_int("CW_GRUG_STEPS", DEFAULT_STEPS)
    batch_size = _env_int("CW_GRUG_BATCH_SIZE", 32)
    profiler_enabled = _env_bool("CW_GRUG_PROFILER_ENABLED", True)
    profiler_steps = _env_int("CW_GRUG_PROFILE_STEPS", DEFAULT_PROFILE_STEPS)
    log_every = _env_int("CW_GRUG_LOG_EVERY", 1)

    tags = [
        "cw-grug",
        "accelerator-smoke",
        "synthetic-data",
        gpu_type.lower(),
        f"{gpu_count}gpu",
    ]

    return ExecutorStep(
        name=f"cw-grug-accelerator-smoke-{gpu_type.lower()}x{gpu_count}-{run_id}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(GRUG_MOE_TRIAL_MODEL),
            data=_synthetic_data(),
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(_gpu_resources()),
            steps=versioned(steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=_tracker_from_env(tags),
            optimizer=versioned(CANARY_OPTIMIZER),
            grug_trainer=versioned(dataclasses.replace(CANARY_TRAINER, log_every=log_every)),
            eval=None,
            profiler=ProfilerConfig(enabled=profiler_enabled, start_step=5, num_steps=profiler_steps),
        ),
    )


cw_grug_accelerator_smoke = _build_step_from_env()


def main() -> None:
    executor_main(steps=[cw_grug_accelerator_smoke])


if __name__ == "__main__":
    main()
