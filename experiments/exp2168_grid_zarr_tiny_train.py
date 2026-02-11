# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tiny training experiment for grid-token Zarr data.

This follows the same shape as `experiments/tutorials/train_tiny_model_cpu.py`,
but replaces text tokenization with a cache-building step from a grid-token Zarr export.
"""

from __future__ import annotations

import os
from typing import Literal

from fray.v2 import ResourceConfig
from marin.execution.executor import executor_main, versioned

from experiments.defaults import default_train
from experiments.llama import llama_nano
from experiments.marin_models import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig
from marin.tokenize.grid_zarr_cache import GridZarrTokenizeConfig, grid_zarr_to_pretokenized_cache
from marin.tokenize.grid_zarr_loader import GridTokenZarrSource


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def _env_opt_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return int(raw)


def _source_from_env() -> GridTokenZarrSource:
    local_path = os.environ.get("GRID_TOKENS_ZARR_PATH")
    if local_path:
        return GridTokenZarrSource(path=os.path.expanduser(local_path))

    hf_repo_id = os.environ.get("GRID_TOKENS_HF_REPO_ID")
    if hf_repo_id:
        return GridTokenZarrSource(
            hf_repo_id=hf_repo_id,
            hf_revision=os.environ.get("GRID_TOKENS_HF_REVISION") or "main",
            hf_subpath=os.environ.get("GRID_TOKENS_HF_SUBPATH"),
            hf_mode=os.environ.get("GRID_TOKENS_HF_MODE", "stage"),
            hf_token=os.environ.get("HF_TOKEN"),
        )

    raise ValueError(
        "Set either GRID_TOKENS_ZARR_PATH or GRID_TOKENS_HF_REPO_ID for exp2168_grid_zarr_tiny_train."
    )


sequence_ordering_env = os.environ.get("GRID_TRAIN_SEQUENCE_ORDERING", "prog_first")
if sequence_ordering_env not in {"prog_first", "storage"}:
    raise ValueError(
        "GRID_TRAIN_SEQUENCE_ORDERING must be one of {'prog_first', 'storage'}, "
        f"got {sequence_ordering_env!r}."
    )
sequence_ordering: Literal["prog_first", "storage"] = sequence_ordering_env
max_train_windows = _env_opt_int("GRID_TRAIN_MAX_WINDOWS")
max_validation_windows = _env_opt_int("GRID_TRAIN_MAX_VALIDATION_WINDOWS")

grid_tokenized = grid_zarr_to_pretokenized_cache(
    name="tokenized/grid-zarr-tiny-train-cache",
    config=GridZarrTokenizeConfig(
        source=_source_from_env(),
        tokenizer=marin_tokenizer,
        max_levels=_env_opt_int("GRID_TRAIN_MAX_LEVELS"),
        max_codebooks=_env_opt_int("GRID_TRAIN_MAX_CODEBOOKS"),
        sequence_ordering=sequence_ordering,
        n_history=_env_int("GRID_TRAIN_HISTORY_STEPS", 2),
        sequence_length=_env_int("GRID_TRAIN_SEQUENCE_LENGTH", 512),
        max_train_windows=max_train_windows if max_train_windows is not None else 4096,
        max_validation_windows=max_validation_windows if max_validation_windows is not None else 512,
        split_seed=_env_int("GRID_TRAIN_SPLIT_SEED", 0),
        tags=["grid-zarr", "prebuilt"],
    ),
)


tiny_grid_train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_cpu(),
    train_batch_size=_env_int("GRID_TRAIN_BATCH_SIZE", 4),
    num_train_steps=_env_int("GRID_TRAIN_NUM_STEPS", 100),
    train_seq_len=_env_int("GRID_TRAIN_SEQUENCE_LENGTH", 512),
    learning_rate=6e-4,
    weight_decay=0.1,
    max_eval_batches=4,
)


grid_nano_model = default_train(
    name="marin-nano-grid-zarr",
    tokenized=grid_tokenized,
    model_config=versioned(llama_nano),
    train_config=tiny_grid_train_config,
    tags=["llama", "nano", "grid-zarr", "tutorial-like"],
    eval_harness_tasks=[],
    use_default_validation=False,
)


if __name__ == "__main__":
    executor_main(
        steps=[grid_nano_model],
        description="Tiny Marin training experiment over grid-token Zarr data.",
    )
