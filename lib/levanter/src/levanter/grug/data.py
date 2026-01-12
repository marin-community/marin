# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any

from jax.sharding import Mesh

from levanter.data.loader import DataLoader
from levanter.data.text import TokenSeqDataset
from levanter.store.cache import TreeCache

# Levanter's DataLoader expects an axis name for the batch dimension. We map it to
# the replica/data axes so each data shard loads only its local share.
DEFAULT_AXIS_MAPPING = {"batch": ("replica_dcn", "replica", "data")}


def make_token_dataset(cache: TreeCache[dict], *, seq_len: int) -> TokenSeqDataset:
    """Thin wrapper so callers don't touch Levanter internals directly."""

    return TokenSeqDataset(cache, seq_len)


def make_dataloader(
    dataset: TokenSeqDataset,
    *,
    batch_size: int,
    mesh: Mesh,
    axis_mapping: Mapping[str, tuple[str, ...]] | None = None,
    max_buffered_batches: int = 64,
    prefetch_size: int = 32,
    pad_final_batch: bool = True,
    allow_nondivisible_batch_size: bool = False,
) -> DataLoader:
    """Wraps a TokenSeqDataset with Levanter's sharding-aware DataLoader."""

    axis_resources = axis_mapping or DEFAULT_AXIS_MAPPING
    return DataLoader(
        dataset,
        batch_size=batch_size,
        mesh=mesh,
        axis_resources=axis_resources,
        batch_axis_name="batch",
        max_buffered_batches=max_buffered_batches,
        prefetch_size=prefetch_size,
        pad_final_batch=pad_final_batch,
        allow_nondivisible_batch_size=allow_nondivisible_batch_size,
    )


def build_token_loader(
    *,
    cache: TreeCache[dict],
    seq_len: int,
    batch_size: int,
    mesh: Mesh,
    axis_mapping: Mapping[str, tuple[str, ...]] | None = None,
    loader_kwargs: Mapping[str, Any] | None = None,
) -> DataLoader:
    """Convenience helper: cache -> TokenSeqDataset -> DataLoader."""

    dataset = make_token_dataset(cache, seq_len=seq_len)
    kwargs = dict(loader_kwargs or {})
    return make_dataloader(dataset, batch_size=batch_size, mesh=mesh, axis_mapping=axis_mapping, **kwargs)


__all__ = [
    "make_token_dataset",
    "make_dataloader",
    "build_token_loader",
]
