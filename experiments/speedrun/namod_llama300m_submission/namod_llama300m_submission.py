# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Official speedrun submission attempts for LLaMA-300M using NAMO/NAMO-D.

This experiment is designed for local multi-GPU execution, with
NAMO-D as the primary optimizer and NAMO as a fallback.
"""

import dataclasses
import logging
import os
from typing import Any

from levanter.optim import NamoConfig, NamoDConfig, OptimizerConfig
from levanter.models import loss as loss_mod

from experiments.llama import llama_300m
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution import THIS_OUTPUT_PATH, ExecutorStep, ensure_versioned
from marin.execution.executor import executor_main
from marin.processing.tokenize.download_pretokenized import PretokenizedCacheDownloadConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

PARAM_COUNT_300M = 300_000_000
DEFAULT_SEQ_LEN = 4096

DEFAULT_BATCH_SIZE = int(os.environ.get("NAMO_SPEEDRUN_BATCH_SIZE", "144"))
DEFAULT_GPU_COUNT = int(os.environ.get("NAMO_SPEEDRUN_GPU_COUNT", "6"))
DEFAULT_GPU_TYPE = os.environ.get("NAMO_SPEEDRUN_GPU_TYPE", "A40")
DEFAULT_PER_DEVICE_PARALLELISM = int(os.environ.get("NAMO_SPEEDRUN_PER_DEVICE_PARALLELISM", "4"))
DEFAULT_STEPS_PER_EVAL = int(os.environ.get("NAMO_SPEEDRUN_STEPS_PER_EVAL", "1000"))
DEFAULT_STEPS_PER_EXPORT = int(os.environ.get("NAMO_SPEEDRUN_STEPS_PER_EXPORT", "5000"))
DEFAULT_NUM_TRAIN_STEPS_OVERRIDE = os.environ.get("NAMO_SPEEDRUN_NUM_TRAIN_STEPS")
DEFAULT_RUN_SUFFIX = os.environ.get("NAMO_SPEEDRUN_RUN_SUFFIX", "").strip()
DEFAULT_TOKENIZED_CACHE_PATH = os.environ.get(
    "NAMO_SPEEDRUN_TOKENIZED_INPUT", "tokenized/subcache/fineweb-edu-10B-ac65f6"
)

AUTHOR = Author(
    name="Suraj Ranganath",
    affiliation="UC San Diego",
    url="https://github.com/suraj-ranganath",
)


def get_num_train_steps(param_count: int, batch_size: int, max_seq_len: int) -> int:
    """Compute Chinchilla-optimal training steps for 20x parameter tokens."""
    total_tokens = param_count * 20
    tokens_per_step = batch_size * max_seq_len
    return total_tokens // tokens_per_step


def _reuse_local_pretokenized_cache(
    cfg: PretokenizedCacheDownloadConfig,
) -> PretokenizedCacheDownloadConfig:
    """Treat an existing local pretokenized cache as a tokenization step output."""
    if not os.path.exists(cfg.cache_path):
        raise FileNotFoundError(f"Expected pretokenized cache directory at {cfg.cache_path}")
    return cfg


def _existing_cache_step() -> ExecutorStep[PretokenizedCacheDownloadConfig]:
    return ExecutorStep(
        name="tokenized/subcache/fineweb-edu-10B-local",
        fn=_reuse_local_pretokenized_cache,
        config=PretokenizedCacheDownloadConfig(
            cache_path=THIS_OUTPUT_PATH,
            tokenizer=ensure_versioned("marin-community/marin-tokenizer"),  # type: ignore[arg-type]
            hf_repo_id=ensure_versioned("marin-community/fineweb-edu-pretokenized-10B"),  # type: ignore[arg-type]
        ),
        override_output_path=DEFAULT_TOKENIZED_CACHE_PATH,
    )


def _force_xla_fused_ce_if_requested() -> None:
    """Avoid pallas GPU fused-CE assertions on multi-GPU local runs."""
    if os.environ.get("NAMO_SPEEDRUN_FORCE_XLA_LOSS", "1") not in {"1", "true", "True"}:
        return

    original = loss_mod.fused_cross_entropy_loss_and_logsumexp_penalty_kernel

    def _xla_impl(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("implementation", "xla")
        return original(*args, **kwargs)

    loss_mod.fused_cross_entropy_loss_and_logsumexp_penalty_kernel = _xla_impl
    logger.info("Enabled XLA fallback for fused CE kernel (NAMO_SPEEDRUN_FORCE_XLA_LOSS=1).")


def _build_optimizer(profile: str) -> OptimizerConfig:
    if profile == "official_namod":
        return NamoDConfig(
            learning_rate=8e-3,
            adam_lr=2.4e-3,
            weight_decay=0.1,
            warmup=2000,
            min_lr_ratio=0.0,
            lr_schedule="linear",
            decay=1.0,
            momentum=0.98,
            mu2=0.99,
            adamnorm_eps=1e-8,
            muon_epsilon=1e-8,
            max_grad_norm=1.5,
            scale_coeff=0.2,
            coefficient_type="simple",
        )

    if profile == "official_namod_alt":
        return NamoDConfig(
            learning_rate=1e-2,
            adam_lr=3e-3,
            weight_decay=0.08,
            warmup=1200,
            min_lr_ratio=0.0,
            lr_schedule="linear",
            decay=1.0,
            momentum=0.95,
            mu2=0.99,
            adamnorm_eps=1e-8,
            muon_epsilon=1e-8,
            max_grad_norm=1.0,
            scale_coeff=0.2,
            coefficient_type="simple",
        )

    if profile == "fallback_namo":
        return NamoConfig(
            learning_rate=8e-3,
            adam_lr=2.4e-3,
            weight_decay=0.1,
            warmup=2000,
            min_lr_ratio=0.0,
            lr_schedule="linear",
            decay=1.0,
            momentum=0.98,
            mu2=0.99,
            adamnorm_eps=1e-8,
            muon_epsilon=1e-8,
            max_grad_norm=1.5,
            scale_coeff=0.2,
            coefficient_type="simple",
        )

    raise ValueError(f"Unknown NAMO speedrun profile: {profile}")


def build_config(profile: str) -> tuple[str, SpeedrunConfig]:
    optimizer = _build_optimizer(profile)

    model_config = dataclasses.replace(llama_300m, max_seq_len=DEFAULT_SEQ_LEN)
    train_steps = get_num_train_steps(PARAM_COUNT_300M, DEFAULT_BATCH_SIZE, model_config.max_seq_len)
    if DEFAULT_NUM_TRAIN_STEPS_OVERRIDE is not None:
        train_steps = int(DEFAULT_NUM_TRAIN_STEPS_OVERRIDE)

    train = SimpleTrainConfig(
        resources=ResourceConfig.with_gpu(DEFAULT_GPU_TYPE, count=DEFAULT_GPU_COUNT, cpu=96, ram="256g", disk="256g"),
        train_batch_size=DEFAULT_BATCH_SIZE,
        num_train_steps=train_steps,
        learning_rate=optimizer.learning_rate,
        optimizer_config=optimizer,
        steps_per_eval=DEFAULT_STEPS_PER_EVAL,
        steps_per_export=DEFAULT_STEPS_PER_EXPORT,
        per_device_parallelism=DEFAULT_PER_DEVICE_PARALLELISM,
    )

    suffix = f"_{DEFAULT_RUN_SUFFIX}" if DEFAULT_RUN_SUFFIX else ""
    run_name = f"llama_300m_{profile}_namo_submission{suffix}"
    description = f"LLaMA-300M speedrun using NAMO-family optimizer ({profile}) on local multi-GPU resources."

    return run_name, SpeedrunConfig(
        author=AUTHOR,
        description=description,
        model_config=model_config,
        train_config=train,
        tokenized_dataset=_existing_cache_step(),
    )


if __name__ == "__main__":
    _force_xla_fused_ce_if_requested()

    profiles_raw = os.environ.get("NAMO_SPEEDRUN_PROFILES", "official_namod_alt")
    profiles = [name.strip() for name in profiles_raw.split(",") if name.strip()]

    steps = []
    for profile in profiles:
        name, cfg = build_config(profile)
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg, tags=["namo", "namo-speedrun", "llama300m", profile]))

    executor_main(
        steps=steps,
        description="LLaMA-300M NAMO/NAMO-D speedrun submission attempts",
    )
