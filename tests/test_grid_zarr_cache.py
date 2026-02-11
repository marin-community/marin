# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from levanter.data.passthrough_tokenizer import PassthroughTokenizer
from levanter.data.text import PrebuiltLmDatasetFormat, UrlDatasetSourceConfig
from levanter.data.text.cache import load_lm_dataset_cache

from marin.tokenize.grid_zarr_cache import GridZarrTokenizeConfig, build_grid_zarr_pretokenized_cache
from marin.tokenize.grid_zarr_loader import GridLevelSpec, GridTokenZarrSource


def _make_tiny_grid_export(path: Path) -> list[GridLevelSpec]:
    levels = [
        GridLevelSpec(height=1, width=2, forcing_tokens=1, prognostic_tokens=1, rvq_depth=1, codebook_size=4),
        GridLevelSpec(height=2, width=2, forcing_tokens=1, prognostic_tokens=1, rvq_depth=2, codebook_size=3),
    ]

    root = zarr.open_group(str(path), mode="w")
    root.attrs["grid_levels"] = [level.__dict__ for level in levels]
    root.create_array("time", data=np.arange(8, dtype=np.float64), chunks=(1,))
    root.create_array("time_index", data=np.arange(8, dtype=np.int64), chunks=(1,))

    seq_len = sum(level.height * level.width * level.total_tokens * level.rvq_depth for level in levels)
    sequence = np.arange(8 * seq_len, dtype=np.int32).reshape(8, seq_len) % 5
    land_mask = np.zeros(seq_len, dtype=bool)
    land_mask[::9] = True
    sequence[:, land_mask] = -1
    root.create_array("sequence", data=sequence, chunks=(1, seq_len))
    root.create_array("land_mask", data=land_mask, chunks=(seq_len,))

    pixels = root.create_group("pixels")
    level0 = pixels.create_group("level_0")
    level0.create_array("lat", data=np.array([-10.0, 10.0], dtype=np.float32), chunks=(2,))
    level0.create_array("lon", data=np.array([0.0, 45.0], dtype=np.float32), chunks=(2,))

    level1 = pixels.create_group("level_1")
    level1.create_array("lat", data=np.array([-15.0, -5.0, 5.0, 15.0], dtype=np.float32), chunks=(4,))
    level1.create_array("lon", data=np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32), chunks=(4,))

    codebooks = root.create_group("codebooks")
    cb0 = codebooks.create_group("level_0")
    cb0.create_array("codebook_0", data=np.arange(8, dtype=np.float32).reshape(4, 2), chunks=(4, 2))

    cb1 = codebooks.create_group("level_1")
    cb1.create_array("codebook_0", data=np.arange(6, dtype=np.float32).reshape(3, 2), chunks=(3, 2))
    cb1.create_array("codebook_1", data=np.arange(6, dtype=np.float32).reshape(3, 2) + 100, chunks=(3, 2))

    return levels


def test_grid_zarr_cache_builder_writes_train_and_validation(tmp_path: Path) -> None:
    zarr_path = tmp_path / "tiny_grid_tokens.zarr"
    _make_tiny_grid_export(zarr_path)

    cache_path = tmp_path / "cache"
    config = GridZarrTokenizeConfig(
        source=GridTokenZarrSource(path=str(zarr_path)),
        cache_path=str(cache_path),
        tokenizer="gpt2",
        sequence_ordering="storage",
        n_history=2,
        sequence_length=8,
        max_train_windows=8,
        max_validation_windows=8,
        split_seed=0,
    )

    out = build_grid_zarr_pretokenized_cache(config)
    assert out.cache_path == str(cache_path)

    train_ledger = cache_path / "train" / "shard_ledger.json"
    validation_ledger = cache_path / "validation" / "shard_ledger.json"
    assert train_ledger.exists()
    assert validation_ledger.exists()

    fmt = PrebuiltLmDatasetFormat(input_ids_key=config.input_ids_key, loss_weights_key=config.loss_weights_key)
    passthrough = PassthroughTokenizer(vocab_size=10_000)
    train_cache = load_lm_dataset_cache(str(cache_path / "train"), fmt, passthrough)
    validation_cache = load_lm_dataset_cache(str(cache_path / "validation"), fmt, passthrough)

    assert len(train_cache) > 0
    assert len(validation_cache) > 0

    train_example = train_cache.store[0]
    validation_example = validation_cache.store[0]

    assert train_example[config.input_ids_key].shape == (config.sequence_length,)
    assert train_example[config.loss_weights_key].shape == (config.sequence_length,)
    assert validation_example[config.input_ids_key].shape == (config.sequence_length,)
    assert validation_example[config.loss_weights_key].shape == (config.sequence_length,)


def test_grid_zarr_config_exposes_prebuilt_source_config(tmp_path: Path) -> None:
    config = GridZarrTokenizeConfig(
        source=GridTokenZarrSource(path=str(tmp_path / "dummy.zarr")),
        cache_path=str(tmp_path / "cache"),
        tokenizer="gpt2",
    )

    source = config.as_lm_dataset_source_config(str(tmp_path / "cache"))
    assert isinstance(source, UrlDatasetSourceConfig)
    assert isinstance(source.format, PrebuiltLmDatasetFormat)
    assert source.cache_dir == str(tmp_path / "cache")
    assert source.train_urls == []
    assert source.validation_urls == []
