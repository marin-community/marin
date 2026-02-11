# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test experiment for the grid-token Zarr dataloader."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Literal

import fsspec
import numpy as np
import torch

from marin.execution import THIS_OUTPUT_PATH, ExecutorStep, executor_main
from marin.tokenize.grid_zarr_loader import (
    GridTokenZarrSource,
    GridZarrBatcher,
    build_loss_masks,
    build_split_indices,
    load_grid_sequence_layout,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridZarrDataloaderSmokeConfig:
    """Configuration for a grid-token Zarr dataloader smoke run."""

    source: GridTokenZarrSource
    output_path: str = THIS_OUTPUT_PATH
    max_levels: int | None = 2
    max_codebooks: int | None = 1
    sequence_ordering: Literal["prog_first", "storage"] = "prog_first"
    n_history: int = 2
    batch_size: int = 4
    seed: int = 0
    device: str = "cpu"


def _run_grid_zarr_dataloader_smoke(cfg: GridZarrDataloaderSmokeConfig) -> dict[str, object]:
    logging.basicConfig(level=logging.INFO)

    layout, sequence = load_grid_sequence_layout(
        cfg.source,
        max_levels=cfg.max_levels,
        max_codebooks=cfg.max_codebooks,
        sequence_ordering=cfg.sequence_ordering,
    )
    steps = cfg.n_history + 1
    train_idx, val_idx, eval_idx = build_split_indices(int(sequence.shape[0]))
    train_mask, eval_mask = build_loss_masks(layout, steps)

    batcher = GridZarrBatcher(sequence=sequence, layout=layout, steps=steps)
    rng = np.random.default_rng(cfg.seed)
    x, y = batcher.sample_batch(train_idx, cfg.batch_size, rng, torch.device(cfg.device))
    token_metadata = layout.token_metadata()

    summary = {
        "sequence_shape": [int(sequence.shape[0]), int(sequence.shape[1])],
        "levels": [level.__dict__ for level in layout.levels],
        "seq_len": int(layout.seq_len),
        "step_len": int(layout.step_len),
        "codebook_vocab_size": int(layout.codebook_vocab_size),
        "batch_x_shape": [int(x.shape[0]), int(x.shape[1])],
        "batch_y_shape": [int(y.shape[0]), int(y.shape[1])],
        "train_split_size": int(train_idx.shape[0]),
        "val_split_size": int(val_idx.shape[0]),
        "eval_split_size": int(eval_idx.shape[0]),
        "train_mask_tokens": int(train_mask.sum()),
        "eval_mask_tokens": int(eval_mask.sum()),
        "metadata": {
            "level_ids_shape": list(token_metadata.level_ids.shape),
            "slot_ids_shape": list(token_metadata.slot_ids.shape),
            "codebook_ids_shape": list(token_metadata.codebook_ids.shape),
            "prognostic_tokens": int(token_metadata.prognostic_mask.sum()),
            "land_tokens": int(token_metadata.land_mask.sum()),
            "latlon_features_shape": list(token_metadata.latlon_features.shape),
        },
    }

    fs, output_root = fsspec.core.url_to_fs(cfg.output_path)
    fs.makedirs(output_root, exist_ok=True)
    summary_path = os.path.join(cfg.output_path, "grid_zarr_dataloader_summary.json")
    with fsspec.open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    logger.info("Grid Zarr dataloader smoke summary written to %s", summary_path)
    return summary


def _source_from_env() -> GridTokenZarrSource:
    local_path = os.environ.get("GRID_TOKENS_ZARR_PATH")
    if local_path:
        return GridTokenZarrSource(path=local_path)

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
        "Set either GRID_TOKENS_ZARR_PATH or GRID_TOKENS_HF_REPO_ID for exp2167_grid_zarr_dataloader."
    )


def grid_zarr_dataloader_smoke_step(
    source: GridTokenZarrSource | None = None,
) -> ExecutorStep[GridZarrDataloaderSmokeConfig]:
    """Create the experiment step that loads and batches a grid-token Zarr dataset."""
    return ExecutorStep(
        name="grid-zarr-dataloader-smoke",
        fn=_run_grid_zarr_dataloader_smoke,
        config=GridZarrDataloaderSmokeConfig(source=source or _source_from_env()),
    )


if __name__ == "__main__":
    executor_main(
        steps=[grid_zarr_dataloader_smoke_step()],
        description="Smoke-test experiment for the grid-token Zarr dataloader.",
    )
