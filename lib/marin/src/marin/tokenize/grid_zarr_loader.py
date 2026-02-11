# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grid-token Zarr loading and batching utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import fsspec
import numpy as np
import torch
import zarr


@dataclass(frozen=True)
class GridLevelSpec:
    """Metadata for one level in a grid-token hierarchy."""

    height: int
    width: int
    forcing_tokens: int
    prognostic_tokens: int
    rvq_depth: int
    codebook_size: int

    @property
    def total_tokens(self) -> int:
        """Total token slots per pixel for this level."""
        return int(self.forcing_tokens + self.prognostic_tokens)


@dataclass(frozen=True)
class GridTokenMetadata:
    """Per-token metadata arrays aligned with one timestep sequence layout."""

    level_ids: np.ndarray
    slot_ids: np.ndarray
    codebook_ids: np.ndarray
    prognostic_mask: np.ndarray
    land_mask: np.ndarray
    latlon_features: np.ndarray


@dataclass(frozen=True)
class GridTokenZarrSource:
    """How to locate a grid-token Zarr export."""

    path: str | None = None

    hf_repo_id: str | None = None
    hf_revision: str | None = None
    hf_subpath: str | None = None
    hf_repo_type: Literal["dataset", "model"] = "dataset"
    hf_token: str | None = None
    hf_mode: Literal["stage", "direct"] = "stage"
    hf_stage_dir: str | None = None

    def __post_init__(self) -> None:
        has_path = self.path is not None
        has_hf = self.hf_repo_id is not None
        if has_path == has_hf:
            raise ValueError("Specify exactly one of `path` or `hf_repo_id`.")
        if self.hf_mode not in {"stage", "direct"}:
            raise ValueError(f"Unsupported hf_mode: {self.hf_mode}")


@dataclass(frozen=True)
class _ResolvedGridTokenSource:
    path: str | None = None
    store: Any = None


@dataclass(frozen=True)
class GridSequenceLayout:
    """Sequence layout metadata derived from a grid-token Zarr export."""

    levels: list[GridLevelSpec]
    seq_len: int
    step_len: int
    codebook_table: np.ndarray
    codebook_dim: int
    codebook_vocab_size: int
    pos_offsets: np.ndarray
    land_mask: np.ndarray
    prognostic_mask: np.ndarray
    level_ids: np.ndarray
    slot_ids: np.ndarray
    codebook_ids: np.ndarray
    latlon_features: np.ndarray
    max_tokens_per_level: int
    max_rvq_depth: int
    sequence_slice: slice | None
    sequence_indices: np.ndarray | None
    sequence_order: np.ndarray | None
    sequence_order_inverse: np.ndarray | None

    @classmethod
    def from_group(
        cls,
        root: zarr.Group,
        *,
        max_levels: int | None = None,
        max_codebooks: int | None = None,
        sequence_ordering: Literal["prog_first", "storage"] = "prog_first",
    ) -> tuple[GridSequenceLayout, zarr.Array]:
        """Build a layout from a loaded grid-token Zarr group."""
        if "grid_levels" not in root.attrs:
            raise ValueError("Expected `grid_levels` metadata in Zarr root attrs.")

        levels_full = [GridLevelSpec(**_coerce_level_dict(level)) for level in root.attrs["grid_levels"]]
        sequence = root["sequence"]

        full_seq_len = _sequence_length(levels_full)
        if int(sequence.shape[1]) != full_seq_len:
            raise ValueError("Zarr `sequence` length does not match `grid_levels` metadata.")

        if max_levels is not None:
            max_levels = int(max_levels)
            if max_levels <= 0:
                raise ValueError("max_levels must be >= 1")
            max_levels = min(max_levels, len(levels_full))

        if max_codebooks is not None:
            max_codebooks = int(max_codebooks)
            if max_codebooks <= 0:
                raise ValueError("max_codebooks must be >= 1")

        levels_selected = levels_full[:max_levels] if max_levels is not None else levels_full
        levels = _trim_levels_codebooks(levels_selected, max_codebooks)

        land_mask = np.asarray(root["land_mask"][:], dtype=bool)
        sequence_slice = None
        sequence_indices = None
        if max_levels is not None or max_codebooks is not None:
            needs_codebook_trim = max_codebooks is not None and any(
                max_codebooks < level.rvq_depth for level in levels_selected
            )
            if needs_codebook_trim:
                sequence_indices = _build_sequence_indices(levels_selected, max_codebooks)
            else:
                selected_len = _sequence_length(levels_selected)
                if selected_len != full_seq_len:
                    sequence_slice = slice(0, selected_len)

        if sequence_indices is not None:
            land_mask = land_mask[sequence_indices]
        elif sequence_slice is not None:
            land_mask = land_mask[sequence_slice]

        codebook_table, codebook_offsets = _load_codebooks(root, levels)
        codebook_dim = int(codebook_table.shape[1])
        codebook_vocab_size = int(codebook_table.shape[0])

        (
            pos_offsets,
            prognostic_mask,
            level_ids,
            slot_ids,
            codebook_ids,
            latlon_features,
        ) = _build_layout_arrays(root, levels, codebook_offsets)

        seq_len = int(pos_offsets.shape[0])
        if land_mask.shape[0] != seq_len:
            raise ValueError("Land mask length mismatch with computed sequence length.")
        if prognostic_mask.shape[0] != seq_len:
            raise ValueError("Prognostic mask length mismatch with computed sequence length.")

        sequence_order = None
        sequence_order_inverse = None
        if sequence_ordering not in {"prog_first", "storage"}:
            raise ValueError("sequence_ordering must be `prog_first` or `storage`.")
        if sequence_ordering == "prog_first":
            sequence_order = _build_sequence_order(prognostic_mask)
            sequence_order_inverse = np.empty_like(sequence_order)
            sequence_order_inverse[sequence_order] = np.arange(sequence_order.shape[0], dtype=np.int64)
            pos_offsets = pos_offsets[sequence_order]
            prognostic_mask = prognostic_mask[sequence_order]
            level_ids = level_ids[sequence_order]
            slot_ids = slot_ids[sequence_order]
            codebook_ids = codebook_ids[sequence_order]
            latlon_features = latlon_features[sequence_order]
            land_mask = land_mask[sequence_order]

        max_tokens = max(level.total_tokens for level in levels)
        max_rvq = max(level.rvq_depth for level in levels)

        return (
            cls(
                levels=levels,
                seq_len=seq_len,
                step_len=seq_len + 1,
                codebook_table=codebook_table,
                codebook_dim=codebook_dim,
                codebook_vocab_size=codebook_vocab_size,
                pos_offsets=pos_offsets,
                land_mask=land_mask,
                prognostic_mask=prognostic_mask,
                level_ids=level_ids,
                slot_ids=slot_ids,
                codebook_ids=codebook_ids,
                latlon_features=latlon_features,
                max_tokens_per_level=max_tokens,
                max_rvq_depth=max_rvq,
                sequence_slice=sequence_slice,
                sequence_indices=sequence_indices,
                sequence_order=sequence_order,
                sequence_order_inverse=sequence_order_inverse,
            ),
            sequence,
        )

    def select_sequence(self, sequence: zarr.Array, time_idx: slice | np.ndarray) -> np.ndarray:
        """Read rows from a Zarr sequence and apply configured level/codebook/order selections."""
        if self.sequence_indices is not None:
            if hasattr(sequence, "oindex") and not isinstance(time_idx, slice):
                seq = sequence.oindex[time_idx, self.sequence_indices]
            else:
                seq = sequence[time_idx, :]
                if seq.ndim == 1:
                    seq = seq[self.sequence_indices]
                else:
                    seq = seq[:, self.sequence_indices]
        elif self.sequence_slice is not None:
            seq = sequence[time_idx, self.sequence_slice]
        else:
            seq = sequence[time_idx, :]

        if self.sequence_order is not None:
            if seq.ndim == 1:
                seq = seq[self.sequence_order]
            else:
                seq = seq[:, self.sequence_order]
        return seq

    def to_storage_order(self, seq: np.ndarray) -> np.ndarray:
        """Map a sequence from current ordering back to Zarr storage ordering."""
        if self.sequence_order_inverse is None:
            return seq
        if seq.ndim == 1:
            return seq[self.sequence_order_inverse]
        return seq[:, self.sequence_order_inverse]

    def token_metadata(self) -> GridTokenMetadata:
        """Return per-token spatial/semantic metadata aligned to the loaded layout ordering."""
        return GridTokenMetadata(
            level_ids=self.level_ids,
            slot_ids=self.slot_ids,
            codebook_ids=self.codebook_ids,
            prognostic_mask=self.prognostic_mask,
            land_mask=self.land_mask,
            latlon_features=self.latlon_features,
        )


class GridZarrBatcher:
    """Autoregressive batch sampler over timestep-token rows from a grid-token Zarr export."""

    def __init__(self, sequence: zarr.Array, layout: GridSequenceLayout, steps: int) -> None:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.sequence = sequence
        self.layout = layout
        self.steps = steps
        self.land_id = layout.codebook_vocab_size
        self.start_id = layout.codebook_vocab_size + 1
        self.vocab_size = layout.codebook_vocab_size + 2

    def sample_batch(
        self,
        indices: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an autoregressive batch and return `(x, y)` token tensors."""
        if indices.shape[0] < self.steps:
            raise ValueError("Not enough timesteps for requested history length.")

        tokens = np.empty((batch_size, self.steps, self.layout.step_len), dtype=np.int64)
        max_start = indices.shape[0] - self.steps

        for batch_idx in range(batch_size):
            start = int(rng.integers(0, max_start + 1))
            time_idx = indices[start : start + self.steps]
            seq = self.layout.select_sequence(self.sequence, time_idx)
            mapped = seq.astype(np.int64) + self.layout.pos_offsets[None, :]
            mapped[seq < 0] = self.land_id
            tokens[batch_idx, :, 0] = self.start_id
            tokens[batch_idx, :, 1:] = mapped

        flat = tokens.reshape(batch_size, -1)
        x = torch.from_numpy(flat[:, :-1]).to(device, non_blocking=True)
        y = torch.from_numpy(flat[:, 1:]).to(device, non_blocking=True)
        return x, y


def load_grid_sequence_layout(
    source: GridTokenZarrSource | str | Path,
    *,
    max_levels: int | None = None,
    max_codebooks: int | None = None,
    sequence_ordering: Literal["prog_first", "storage"] = "prog_first",
) -> tuple[GridSequenceLayout, zarr.Array]:
    """Resolve a source and return `(layout, sequence_array)`."""
    root = open_grid_token_group(source)
    return GridSequenceLayout.from_group(
        root,
        max_levels=max_levels,
        max_codebooks=max_codebooks,
        sequence_ordering=sequence_ordering,
    )


def open_grid_token_group(source: GridTokenZarrSource | str | Path) -> zarr.Group:
    """Open a grid-token Zarr group from local/fsspec/HF source settings."""
    resolved = resolve_grid_token_source(source)
    if resolved.path is not None:
        return zarr.open_group(resolved.path, mode="r")
    return zarr.open_group(store=resolved.store, mode="r")


def resolve_grid_token_source(source: GridTokenZarrSource | str | Path) -> _ResolvedGridTokenSource:
    """Resolve local path/HF source config into a Zarr path or store object."""
    if isinstance(source, Path):
        return _ResolvedGridTokenSource(path=str(source))
    if isinstance(source, str):
        return _ResolvedGridTokenSource(path=source)
    if source.path is not None:
        return _ResolvedGridTokenSource(path=source.path)

    if source.hf_mode == "direct":
        hf_uri = _build_hf_uri(source)
        return _ResolvedGridTokenSource(store=fsspec.get_mapper(hf_uri))

    staged_path = _stage_hf_snapshot(source)
    return _ResolvedGridTokenSource(path=staged_path)


def build_split_indices(total: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split timestep indices into train/val/eval with the same heuristic used in `~/tokens`."""
    if total < 2:
        raise ValueError("Need at least 2 timesteps to split.")
    holdout_start = int(total * 0.9)
    holdout_start = max(1, min(holdout_start, total - 1))
    holdout = np.arange(holdout_start, total, dtype=np.int64)
    split = holdout_start + max(1, int(len(holdout) * 0.2))
    split = min(split, total - 1)
    train = np.arange(0, holdout_start, dtype=np.int64)
    val = np.arange(holdout_start, split, dtype=np.int64)
    eval_idx = np.arange(split, total, dtype=np.int64)
    return train, val, eval_idx


def build_loss_masks(layout: GridSequenceLayout, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Build train/eval token masks for an autoregressive window with `steps` timesteps."""
    if steps < 1:
        raise ValueError("steps must be >= 1")

    step_train = np.zeros(layout.step_len, dtype=bool)
    step_train[1:] = ~layout.land_mask
    train_mask = np.tile(step_train, steps)[1:]

    step_eval = np.zeros(layout.step_len, dtype=bool)
    step_eval[1:] = layout.prognostic_mask & ~layout.land_mask
    eval_prefix = np.zeros(layout.step_len * (steps - 1), dtype=bool)
    eval_mask = np.concatenate([eval_prefix, step_eval])[1:]
    return train_mask, eval_mask


def _coerce_level_dict(raw_level: Any) -> dict[str, int]:
    return {
        "height": int(raw_level["height"]),
        "width": int(raw_level["width"]),
        "forcing_tokens": int(raw_level["forcing_tokens"]),
        "prognostic_tokens": int(raw_level["prognostic_tokens"]),
        "rvq_depth": int(raw_level["rvq_depth"]),
        "codebook_size": int(raw_level["codebook_size"]),
    }


def _trim_levels_codebooks(levels: list[GridLevelSpec], max_codebooks: int | None) -> list[GridLevelSpec]:
    if max_codebooks is None:
        return levels
    trimmed = []
    for level in levels:
        depth = min(int(level.rvq_depth), int(max_codebooks))
        trimmed.append(
            GridLevelSpec(
                height=int(level.height),
                width=int(level.width),
                forcing_tokens=int(level.forcing_tokens),
                prognostic_tokens=int(level.prognostic_tokens),
                rvq_depth=depth,
                codebook_size=int(level.codebook_size),
            )
        )
    return trimmed


def _sequence_length(levels: list[GridLevelSpec]) -> int:
    length = 0
    for level in levels:
        npix = int(level.height) * int(level.width)
        length += npix * int(level.total_tokens) * int(level.rvq_depth)
    return int(length)


def _build_sequence_indices(levels: list[GridLevelSpec], max_codebooks: int | None) -> np.ndarray:
    if max_codebooks is None:
        raise ValueError("max_codebooks is required to build sequence indices.")

    indices = []
    offset = 0
    for level in levels:
        npix = int(level.height) * int(level.width)
        token_count = npix * level.total_tokens
        depth = int(level.rvq_depth)
        keep_depth = min(depth, int(max_codebooks))
        level_size = token_count * depth

        if keep_depth == depth:
            indices.append(np.arange(offset, offset + level_size, dtype=np.int64))
        else:
            base = np.arange(token_count, dtype=np.int64) * depth + offset
            keep = base[:, None] + np.arange(keep_depth, dtype=np.int64)
            indices.append(keep.reshape(-1))
        offset += level_size

    if not indices:
        return np.array([], dtype=np.int64)
    return np.concatenate(indices, axis=0)


def _load_codebooks(root: zarr.Group, levels: list[GridLevelSpec]) -> tuple[np.ndarray, np.ndarray]:
    tables = []
    offsets = []
    offset = 0
    for level_idx, level in enumerate(levels):
        level_group = root["codebooks"][f"level_{level_idx}"]
        for codebook_idx in range(level.rvq_depth):
            data = np.asarray(level_group[f"codebook_{codebook_idx}"][:], dtype=np.float32)
            if data.shape[0] != level.codebook_size:
                raise ValueError(f"Codebook size mismatch at level={level_idx}, codebook={codebook_idx}.")
            tables.append(data)
            offsets.append(offset)
            offset += level.codebook_size

    table = np.concatenate(tables, axis=0)
    return table, np.asarray(offsets, dtype=np.int64)


def _build_layout_arrays(
    root: zarr.Group,
    levels: list[GridLevelSpec],
    codebook_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos_offsets = []
    prognostic_mask = []
    level_ids = []
    slot_ids = []
    codebook_ids = []
    latlon_features = []

    codebook_start = 0
    for level_idx, level in enumerate(levels):
        npix = int(level.height) * int(level.width)
        tokens_per_pixel = int(level.total_tokens) * int(level.rvq_depth)
        pixel_ids = np.repeat(np.arange(npix, dtype=np.int64), tokens_per_pixel)

        slot = np.repeat(np.arange(level.total_tokens, dtype=np.int32), level.rvq_depth)
        slot = np.tile(slot, npix)
        codebook_local = np.tile(np.arange(level.rvq_depth, dtype=np.int32), level.total_tokens)
        codebook_local = np.tile(codebook_local, npix)
        codebook_global = codebook_start + codebook_local

        pos_offsets.append(codebook_offsets[codebook_global])
        prognostic_mask.append(slot >= int(level.forcing_tokens))
        level_ids.append(np.full(slot.shape, level_idx, dtype=np.int32))
        slot_ids.append(slot)
        codebook_ids.append(codebook_local)

        lat = np.asarray(root["pixels"][f"level_{level_idx}"]["lat"][:], dtype=np.float32)
        lon = np.asarray(root["pixels"][f"level_{level_idx}"]["lon"][:], dtype=np.float32)
        if lat.shape[0] != npix or lon.shape[0] != npix:
            raise ValueError(f"Pixel metadata mismatch for level={level_idx}.")

        pixel_lat = lat[pixel_ids]
        pixel_lon = lon[pixel_ids]
        lat_rad = np.deg2rad(pixel_lat)
        lon_rad = np.deg2rad(pixel_lon)
        latlon = np.stack(
            [np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)],
            axis=-1,
        ).astype(np.float32)
        latlon_features.append(latlon)

        codebook_start += level.rvq_depth

    return (
        np.concatenate(pos_offsets, axis=0),
        np.concatenate(prognostic_mask, axis=0),
        np.concatenate(level_ids, axis=0),
        np.concatenate(slot_ids, axis=0),
        np.concatenate(codebook_ids, axis=0),
        np.concatenate(latlon_features, axis=0),
    )


def _build_sequence_order(prognostic_mask: np.ndarray) -> np.ndarray:
    prog = np.where(prognostic_mask)[0]
    forcing = np.where(~prognostic_mask)[0]
    return np.concatenate([prog, forcing], axis=0).astype(np.int64)


def _build_hf_uri(source: GridTokenZarrSource) -> str:
    if source.hf_repo_id is None:
        raise ValueError("hf_repo_id is required for HF source resolution.")
    revision = source.hf_revision or "main"
    prefix = "datasets" if source.hf_repo_type == "dataset" else "models"
    uri = f"hf://{prefix}/{source.hf_repo_id}@{revision}"
    if source.hf_subpath:
        uri = f"{uri}/{source.hf_subpath.strip('/')}"
    return uri


def _snapshot_download(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    local_dir: str,
    token: str | None,
    allow_patterns: list[str] | None,
) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=local_dir,
        token=token,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )


def _stage_hf_snapshot(source: GridTokenZarrSource) -> str:
    if source.hf_repo_id is None:
        raise ValueError("hf_repo_id is required for staged HF source resolution.")

    revision = source.hf_revision or "main"
    stage_root = source.hf_stage_dir or os.path.join(os.path.expanduser("~"), ".cache", "marin", "grid-zarr")
    local_dir = os.path.join(
        stage_root,
        source.hf_repo_type,
        source.hf_repo_id.replace("/", "__"),
        revision,
    )

    allow_patterns = None
    normalized_subpath = source.hf_subpath.strip("/") if source.hf_subpath is not None else None
    if normalized_subpath:
        allow_patterns = [normalized_subpath, f"{normalized_subpath}/**"]

    snapshot_dir = _snapshot_download(
        repo_id=source.hf_repo_id,
        repo_type=source.hf_repo_type,
        revision=revision,
        local_dir=local_dir,
        token=source.hf_token,
        allow_patterns=allow_patterns,
    )

    if normalized_subpath:
        staged_path = os.path.join(snapshot_dir, normalized_subpath)
    else:
        staged_path = snapshot_dir

    if not os.path.exists(staged_path):
        raise FileNotFoundError(f"Staged HF path does not exist: {staged_path}")
    return staged_path
