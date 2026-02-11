# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert grid-token Zarr exports into Levanter prebuilt token caches."""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import zarr
from levanter.data.text import LmDatasetSourceConfigBase, PrebuiltLmDatasetFormat, UrlDatasetSourceConfig
from levanter.data.text.formats import PrebuiltCacheProcessor
from levanter.store.cache import CacheMetadata, SerialCacheWriter

from marin.execution import THIS_OUTPUT_PATH, ExecutorStep, InputName
from marin.processing.tokenize.tokenize import TokenizeConfigBase
from marin.tokenize.grid_zarr_loader import (
    GridSequenceLayout,
    GridTokenZarrSource,
    build_loss_masks,
    build_split_indices,
    load_grid_sequence_layout,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridZarrTokenizeConfig(TokenizeConfigBase):
    """Configuration for building a prebuilt token cache from a grid-token Zarr export."""

    source: GridTokenZarrSource
    cache_path: str = THIS_OUTPUT_PATH
    tokenizer: str = "gpt2"

    max_levels: int | None = None
    max_codebooks: int | None = None
    sequence_ordering: Literal["prog_first", "storage"] = "prog_first"

    n_history: int = 2
    sequence_length: int = 512
    max_train_windows: int | None = 4096
    max_validation_windows: int | None = 512
    split_seed: int = 0

    input_ids_key: str = "input_ids"
    loss_weights_key: str | None = "loss_weights"
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_history < 1:
            raise ValueError("n_history must be >= 1")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be >= 2")
        if self.max_train_windows is not None and self.max_train_windows <= 0:
            raise ValueError("max_train_windows must be positive when set")
        if self.max_validation_windows is not None and self.max_validation_windows <= 0:
            raise ValueError("max_validation_windows must be positive when set")

    def as_lm_dataset_source_config(
        self,
        actual_output_path: str | InputName | None,
        *,
        include_raw_paths: bool = True,
    ) -> LmDatasetSourceConfigBase:
        """Expose the generated cache through Levanter's URL dataset source config."""
        del include_raw_paths
        if actual_output_path is None:
            raise ValueError("actual_output_path must be provided for grid Zarr cache sources.")
        return UrlDatasetSourceConfig(
            tags=self.tags,
            train_urls=[],
            validation_urls=[],
            cache_dir=actual_output_path,
            format=PrebuiltLmDatasetFormat(
                input_ids_key=self.input_ids_key,
                loss_weights_key=self.loss_weights_key,
            ),
        )


def grid_zarr_to_pretokenized_cache(
    name: str,
    config: GridZarrTokenizeConfig,
) -> ExecutorStep[GridZarrTokenizeConfig]:
    """Create an executor step that materializes grid Zarr data into a prebuilt token cache."""
    return ExecutorStep(
        name=name,
        description="Build Levanter prebuilt train/validation caches from grid-token Zarr data.",
        fn=build_grid_zarr_pretokenized_cache,
        config=config,
    )


def build_grid_zarr_pretokenized_cache(config: GridZarrTokenizeConfig) -> GridZarrTokenizeConfig:
    """Build train/validation prebuilt caches from a grid-token Zarr export."""
    layout, sequence = load_grid_sequence_layout(
        config.source,
        max_levels=config.max_levels,
        max_codebooks=config.max_codebooks,
        sequence_ordering=config.sequence_ordering,
    )
    steps = config.n_history + 1
    full_window_len = steps * layout.step_len
    if config.sequence_length > full_window_len:
        raise ValueError(
            f"sequence_length={config.sequence_length} exceeds one window length={full_window_len}. "
            "Reduce sequence_length or increase n_history."
        )

    train_time_idx, val_time_idx, eval_time_idx = build_split_indices(int(sequence.shape[0]))
    validation_time_idx = np.concatenate([val_time_idx, eval_time_idx], axis=0)

    rng = np.random.default_rng(config.split_seed)
    train_starts = _window_starts(train_time_idx.shape[0], steps, config.max_train_windows, rng)
    validation_starts = _window_starts(validation_time_idx.shape[0], steps, config.max_validation_windows, rng)
    if validation_starts.shape[0] == 0 and train_starts.shape[0] > 0:
        logger.warning(
            "Holdout split does not provide enough timesteps for validation windows; "
            "falling back to sampled train windows for validation."
        )
        validation_time_idx = train_time_idx
        validation_starts = _window_starts(validation_time_idx.shape[0], steps, config.max_validation_windows, rng)

    train_prediction_mask, eval_prediction_mask = build_loss_masks(layout, steps)
    train_token_mask = np.concatenate(
        [np.zeros(1, dtype=np.float32), train_prediction_mask.astype(np.float32)],
        axis=0,
    )
    eval_token_mask = np.concatenate(
        [np.zeros(1, dtype=np.float32), eval_prediction_mask.astype(np.float32)],
        axis=0,
    )

    processor = PrebuiltCacheProcessor(config.input_ids_key, config.loss_weights_key)
    cache_metadata = CacheMetadata(preprocessor_metadata=processor.metadata)
    train_cache_dir = os.path.join(config.cache_path, "train")
    validation_cache_dir = os.path.join(config.cache_path, "validation")

    train_examples = _write_split_cache(
        cache_dir=train_cache_dir,
        split_name="train",
        sequence=sequence,
        layout=layout,
        time_indices=train_time_idx,
        starts=train_starts,
        steps=steps,
        sequence_length=config.sequence_length,
        token_loss_weight=train_token_mask,
        input_ids_key=config.input_ids_key,
        loss_weights_key=config.loss_weights_key,
        exemplar=processor.output_exemplar,
        metadata=cache_metadata,
    )

    validation_examples = _write_split_cache(
        cache_dir=validation_cache_dir,
        split_name="validation",
        sequence=sequence,
        layout=layout,
        time_indices=validation_time_idx,
        starts=validation_starts,
        steps=steps,
        sequence_length=config.sequence_length,
        token_loss_weight=eval_token_mask,
        input_ids_key=config.input_ids_key,
        loss_weights_key=config.loss_weights_key,
        exemplar=processor.output_exemplar,
        metadata=cache_metadata,
    )

    data_vocab_size = int(layout.codebook_vocab_size + 2)
    logger.info(
        "Built grid Zarr prebuilt cache at %s (train_examples=%d, validation_examples=%d, vocab=%d, seq_len=%d)",
        config.cache_path,
        train_examples,
        validation_examples,
        data_vocab_size,
        config.sequence_length,
    )
    return dataclasses.replace(config, cache_path=config.cache_path)


def _window_starts(
    total_indices: int,
    steps: int,
    max_windows: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    available = total_indices - steps + 1
    if available <= 0:
        return np.asarray([], dtype=np.int64)

    starts = np.arange(available, dtype=np.int64)
    if max_windows is not None and max_windows < available:
        starts = rng.choice(starts, size=max_windows, replace=False)
        starts.sort()
    return starts


def _encode_window_tokens(
    sequence: zarr.Array,
    layout: GridSequenceLayout,
    time_idx: np.ndarray,
) -> np.ndarray:
    seq = layout.select_sequence(sequence, time_idx).astype(np.int64)
    mapped = seq + layout.pos_offsets[None, :]
    mapped[seq < 0] = layout.codebook_vocab_size

    tokens = np.empty((seq.shape[0], layout.step_len), dtype=np.int32)
    tokens[:, 0] = int(layout.codebook_vocab_size + 1)
    tokens[:, 1:] = mapped.astype(np.int32)
    return tokens.reshape(-1)


def _write_split_cache(
    *,
    cache_dir: str,
    split_name: str,
    sequence: zarr.Array,
    layout: GridSequenceLayout,
    time_indices: np.ndarray,
    starts: np.ndarray,
    steps: int,
    sequence_length: int,
    token_loss_weight: np.ndarray,
    input_ids_key: str,
    loss_weights_key: str | None,
    exemplar: dict[str, np.ndarray],
    metadata: CacheMetadata,
) -> int:
    if token_loss_weight.shape[0] != steps * layout.step_len:
        raise ValueError("token_loss_weight length does not match encoded window length.")

    split_examples = 0
    with SerialCacheWriter(
        cache_dir,
        exemplar=exemplar,
        metadata=metadata,
        shard_name=split_name,
        mode="w",
    ) as writer:
        for start in starts:
            time_idx = time_indices[start : start + steps]
            flat_tokens = _encode_window_tokens(sequence, layout, time_idx)

            usable = (flat_tokens.shape[0] // sequence_length) * sequence_length
            if usable == 0:
                continue

            chunked_tokens = flat_tokens[:usable].reshape(-1, sequence_length)
            chunked_weights = token_loss_weight[:usable].reshape(-1, sequence_length)
            split_examples += int(chunked_tokens.shape[0])

            if loss_weights_key is None:
                writer.write_batch({input_ids_key: list(chunked_tokens)})
            else:
                writer.write_batch(
                    {
                        input_ids_key: list(chunked_tokens),
                        loss_weights_key: list(chunked_weights),
                    }
                )

    if split_examples == 0:
        logger.warning(
            "No examples written for split=%s at %s (starts=%d, seq_len=%d).",
            split_name,
            cache_dir,
            starts.shape[0],
            sequence_length,
        )
    return split_examples
