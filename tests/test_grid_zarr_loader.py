# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from marin.tokenize.grid_zarr_loader import (
    GridLevelSpec,
    GridTokenZarrSource,
    GridZarrBatcher,
    build_loss_masks,
    build_split_indices,
    load_grid_sequence_layout,
    resolve_grid_token_source,
)


def _make_tiny_grid_export(path: Path) -> list[GridLevelSpec]:
    levels = [
        GridLevelSpec(height=1, width=2, forcing_tokens=1, prognostic_tokens=1, rvq_depth=1, codebook_size=4),
        GridLevelSpec(height=2, width=2, forcing_tokens=1, prognostic_tokens=1, rvq_depth=2, codebook_size=3),
    ]

    root = zarr.open_group(str(path), mode="w")
    root.attrs["grid_levels"] = [level.__dict__ for level in levels]
    root.create_array("time", data=np.arange(6, dtype=np.float64), chunks=(1,))
    root.create_array("time_index", data=np.arange(6, dtype=np.int64), chunks=(1,))

    seq_len = sum(level.height * level.width * level.total_tokens * level.rvq_depth for level in levels)
    sequence = np.arange(6 * seq_len, dtype=np.int32).reshape(6, seq_len) % 3
    land_mask = np.zeros(seq_len, dtype=bool)
    land_mask[::7] = True
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


def _expected_indices_after_codebook_trim(levels: list[GridLevelSpec], max_codebooks: int) -> np.ndarray:
    indices = []
    offset = 0
    for level in levels:
        token_count = level.height * level.width * level.total_tokens
        keep_depth = min(level.rvq_depth, max_codebooks)
        for token_idx in range(token_count):
            base = offset + token_idx * level.rvq_depth
            for codebook_idx in range(keep_depth):
                indices.append(base + codebook_idx)
        offset += token_count * level.rvq_depth
    return np.asarray(indices, dtype=np.int64)


def test_load_grid_sequence_layout_with_spatial_metadata(tmp_path: Path) -> None:
    levels = _make_tiny_grid_export(tmp_path / "tiny_grid_tokens.zarr")
    layout, sequence = load_grid_sequence_layout(tmp_path / "tiny_grid_tokens.zarr", sequence_ordering="storage")

    expected_seq_len = sum(level.height * level.width * level.total_tokens * level.rvq_depth for level in levels)
    assert layout.seq_len == expected_seq_len
    assert layout.step_len == expected_seq_len + 1
    assert sequence.shape == (6, expected_seq_len)
    assert layout.latlon_features.shape == (expected_seq_len, 4)

    metadata = layout.token_metadata()
    assert metadata.level_ids.shape == (expected_seq_len,)
    assert metadata.slot_ids.shape == (expected_seq_len,)
    assert metadata.codebook_ids.shape == (expected_seq_len,)
    assert metadata.prognostic_mask.shape == (expected_seq_len,)
    assert metadata.land_mask.shape == (expected_seq_len,)


def test_layout_trims_codebooks_like_tokens_gpt(tmp_path: Path) -> None:
    levels = _make_tiny_grid_export(tmp_path / "tiny_grid_tokens.zarr")
    layout, sequence = load_grid_sequence_layout(
        tmp_path / "tiny_grid_tokens.zarr",
        max_codebooks=1,
        sequence_ordering="storage",
    )

    expected_indices = _expected_indices_after_codebook_trim(levels, max_codebooks=1)
    expected = np.asarray(sequence[0, expected_indices], dtype=np.int32)
    actual = np.asarray(layout.select_sequence(sequence, np.array([0], dtype=np.int64))[0], dtype=np.int32)
    np.testing.assert_array_equal(actual, expected)


def test_batcher_and_masks_work_end_to_end(tmp_path: Path) -> None:
    _make_tiny_grid_export(tmp_path / "tiny_grid_tokens.zarr")
    layout, sequence = load_grid_sequence_layout(tmp_path / "tiny_grid_tokens.zarr", sequence_ordering="prog_first")

    train_idx, val_idx, eval_idx = build_split_indices(sequence.shape[0])
    assert train_idx.shape[0] > 0
    assert eval_idx.shape[0] > 0
    assert train_idx.shape[0] + val_idx.shape[0] + eval_idx.shape[0] == sequence.shape[0]

    steps = 3
    train_mask, eval_mask = build_loss_masks(layout, steps)
    expected_len = layout.step_len * steps - 1
    assert train_mask.shape == (expected_len,)
    assert eval_mask.shape == (expected_len,)
    assert int(eval_mask.sum()) <= int(train_mask.sum())

    batcher = GridZarrBatcher(sequence=sequence, layout=layout, steps=steps)
    x, y = batcher.sample_batch(
        train_idx,
        batch_size=2,
        rng=np.random.default_rng(0),
        device=torch.device("cpu"),
    )
    assert x.shape == (2, expected_len)
    assert y.shape == (2, expected_len)
    assert batcher.vocab_size == layout.codebook_vocab_size + 2


def test_resolve_grid_source_hf_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    zarr_path = snapshot_dir / "exports" / "tokens_grid.zarr"
    zarr_path.mkdir(parents=True)

    def _fake_snapshot_download(**_: object) -> str:
        return str(snapshot_dir)

    monkeypatch.setattr("marin.tokenize.grid_zarr_loader._snapshot_download", _fake_snapshot_download)
    source = GridTokenZarrSource(
        hf_repo_id="org/repo",
        hf_revision="deadbee",
        hf_subpath="exports/tokens_grid.zarr",
        hf_mode="stage",
        hf_stage_dir=str(tmp_path / "stage"),
    )

    resolved = resolve_grid_token_source(source)
    assert resolved.path == str(zarr_path)
    assert resolved.store is None


def test_resolve_grid_source_hf_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}
    fake_store = {".zgroup": b"{}"}

    def _fake_get_mapper(uri: str) -> dict[str, bytes]:
        captured["uri"] = uri
        return fake_store

    monkeypatch.setattr("marin.tokenize.grid_zarr_loader.fsspec.get_mapper", _fake_get_mapper)
    source = GridTokenZarrSource(
        hf_repo_id="org/repo",
        hf_revision="main",
        hf_subpath="exports/tokens_grid.zarr",
        hf_mode="direct",
    )

    resolved = resolve_grid_token_source(source)
    assert captured["uri"] == "hf://datasets/org/repo@main/exports/tokens_grid.zarr"
    assert resolved.path is None
    assert resolved.store is fake_store
