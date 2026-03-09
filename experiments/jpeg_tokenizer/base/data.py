# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fsspec
import jax
import numpy as np
from haliax import Axis

from levanter.data import AsyncDataset
from levanter.data.text import CausalLmDataset, DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample


@dataclass(frozen=True)
class TokenSequenceRecord:
    """Small manifest record for a precomputed token sequence."""

    example_id: str
    split: str
    num_tokens: int
    checksum: str
    source_index: int


@dataclass(frozen=True)
class TokenStoreSplitInfo:
    """Metadata for one split in a token store."""

    num_examples: int
    seq_len: int
    tokens_path: str
    manifest_path: str


@dataclass(frozen=True)
class TokenStoreMetadata:
    """Top-level metadata for a deterministic on-disk token store."""

    dataset: str
    dataset_config: str
    image_column: str
    vocab_size: int
    seq_len: int
    canonical_config: dict[str, Any]
    tokenizer_config: dict[str, Any]
    splits: dict[str, TokenStoreSplitInfo]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["splits"] = {name: asdict(split) for name, split in self.splits.items()}
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TokenStoreMetadata:
        return cls(
            dataset=payload["dataset"],
            dataset_config=payload["dataset_config"],
            image_column=payload["image_column"],
            vocab_size=payload["vocab_size"],
            seq_len=payload["seq_len"],
            canonical_config=payload["canonical_config"],
            tokenizer_config=payload["tokenizer_config"],
            splits={name: TokenStoreSplitInfo(**split) for name, split in payload["splits"].items()},
        )


class InMemoryTokenSequenceDataset(AsyncDataset[np.ndarray]):
    """Finite random-access dataset for precomputed token sequences."""

    def __init__(self, sequences: Sequence[np.ndarray]):
        self._sequences = tuple(np.asarray(sequence, dtype=np.int32) for sequence in sequences)

    async def async_len(self) -> int:
        return len(self._sequences)

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[np.ndarray]:
        return [self._sequences[index] for index in indices]


class TokenMatrixDataset(AsyncDataset[np.ndarray]):
    """File-backed fixed-length token dataset backed by a memory-mapped `.npy` matrix."""

    def __init__(self, tokens: np.ndarray):
        if tokens.ndim != 2:
            raise ValueError(f"Expected a rank-2 token matrix, got shape {tokens.shape}")
        if not np.issubdtype(tokens.dtype, np.integer):
            raise ValueError(f"Expected integer token matrix, got dtype {tokens.dtype}")
        self._tokens = tokens

    async def async_len(self) -> int:
        return int(self._tokens.shape[0])

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[np.ndarray]:
        if not indices:
            return []
        return [np.asarray(self._tokens[index], dtype=np.int32) for index in indices]


def as_causal_lm_dataset(
    dataset: AsyncDataset[np.ndarray],
    *,
    seq_len: int,
    eos_id: int | None = None,
    ignore_id: int | None = None,
    block_cross_document_attention: bool = True,
) -> AsyncDataset[GrugLmExample]:
    """Wrap raw token sequences as causal LM examples."""

    if ignore_id is not None:

        def _create_example(tokens: np.ndarray) -> GrugLmExample:
            return GrugLmExample.causal(
                tokens=jax.device_put(tokens.astype(np.int32, copy=False)),
                ignore_id=ignore_id,
                eos_id=eos_id,
                block_cross_document_attention=block_cross_document_attention,
            )

        return dataset.map(_create_example)

    return CausalLmDataset(
        dataset,
        Axis("position", seq_len),
        eos_id=eos_id,
        block_cross_document_attention=block_cross_document_attention,
    )


def build_passthrough_lm_data_config(
    *,
    train_dataset: AsyncDataset[GrugLmExample],
    validation_dataset: AsyncDataset[GrugLmExample],
    vocab_size: int,
    component_name: str = "jpeg_tokens",
) -> LmDataConfig:
    """Construct the minimal Levanter config for direct JPEG-token datasets."""

    return LmDataConfig(
        components={
            component_name: DirectDatasetComponent(
                datasets={
                    "train": train_dataset,
                    "validation": validation_dataset,
                }
            )
        },
        vocab_size=vocab_size,
        tokenizer="passthrough",
    )


def materialize_token_store(store_dir: str | Path, local_cache_dir: str | Path | None = None) -> Path:
    """Return a local path for a token store, downloading remote files if needed."""

    store_dir_str = str(store_dir)
    if "://" not in store_dir_str:
        return Path(store_dir_str)

    fs, remote_dir = fsspec.core.url_to_fs(store_dir_str)
    cache_root = Path(local_cache_dir or Path(tempfile.gettempdir()) / "jpeg_tokenizer_token_store_cache")
    local_store_dir = cache_root / hashlib.sha256(store_dir_str.encode("utf-8")).hexdigest()
    local_store_dir.mkdir(parents=True, exist_ok=True)

    metadata_local_path = local_store_dir / "metadata.json"
    _copy_remote_file(fs, f"{remote_dir}/metadata.json", metadata_local_path)
    metadata = read_token_store_metadata(local_store_dir)
    for split_info in metadata.splits.values():
        _copy_remote_file(fs, f"{remote_dir}/{split_info.tokens_path}", local_store_dir / split_info.tokens_path)
        _copy_remote_file(fs, f"{remote_dir}/{split_info.manifest_path}", local_store_dir / split_info.manifest_path)
    return local_store_dir


def read_token_store_metadata(store_dir: str | Path) -> TokenStoreMetadata:
    """Load token-store metadata from disk."""

    metadata_path = Path(store_dir) / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        return TokenStoreMetadata.from_dict(json.load(handle))


def read_token_store_manifest(store_dir: str | Path, split: str) -> list[TokenSequenceRecord]:
    """Load the manifest records for one token-store split."""

    metadata = read_token_store_metadata(store_dir)
    split_info = metadata.splits[split]
    manifest_path = Path(store_dir) / split_info.manifest_path
    with manifest_path.open("r", encoding="utf-8") as handle:
        return [TokenSequenceRecord(**json.loads(line)) for line in handle if line.strip()]


def open_token_matrix_dataset(store_dir: str | Path, split: str) -> TokenMatrixDataset:
    """Open one split of a token store as a memory-mapped AsyncDataset."""

    metadata = read_token_store_metadata(store_dir)
    split_info = metadata.splits[split]
    tokens_path = Path(store_dir) / split_info.tokens_path
    tokens = np.load(tokens_path, mmap_mode="r")
    if tokens.shape != (split_info.num_examples, split_info.seq_len):
        raise ValueError(
            f"Token matrix shape {tokens.shape} does not match metadata {(split_info.num_examples, split_info.seq_len)}"
        )
    return TokenMatrixDataset(tokens)


def build_passthrough_lm_data_config_from_store(
    *,
    store_dir: str | Path,
    train_split: str = "train",
    validation_split: str = "validation",
    component_name: str = "jpeg_tokens",
    local_cache_dir: str | Path | None = None,
) -> LmDataConfig:
    """Construct a passthrough Levanter config from a token store on disk."""

    local_store_dir = materialize_token_store(store_dir, local_cache_dir=local_cache_dir)
    metadata = read_token_store_metadata(local_store_dir)
    ignore_id = metadata.tokenizer_config.get("loss_mask_ignore_id")
    eos_id = metadata.tokenizer_config.get("eos_token_id")
    return build_passthrough_lm_data_config(
        train_dataset=as_causal_lm_dataset(
            open_token_matrix_dataset(local_store_dir, train_split),
            seq_len=metadata.seq_len,
            eos_id=eos_id,
            ignore_id=ignore_id,
        ),
        validation_dataset=as_causal_lm_dataset(
            open_token_matrix_dataset(local_store_dir, validation_split),
            seq_len=metadata.seq_len,
            eos_id=eos_id,
            ignore_id=ignore_id,
        ),
        vocab_size=metadata.vocab_size,
        component_name=component_name,
    )


def write_token_store(
    store_dir: str | Path,
    *,
    metadata: TokenStoreMetadata,
    split_tokens: dict[str, np.ndarray],
    split_records: dict[str, Sequence[TokenSequenceRecord]],
) -> None:
    """Write a deterministic fixed-length token store to disk."""

    store_path = Path(store_dir)
    store_path.mkdir(parents=True, exist_ok=True)

    metadata_path = store_path / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    for split, split_info in metadata.splits.items():
        tokens = np.asarray(split_tokens[split])
        records = list(split_records[split])
        expected_shape = (split_info.num_examples, split_info.seq_len)
        if tokens.shape != expected_shape:
            raise ValueError(f"Split {split} tokens have shape {tokens.shape}, expected {expected_shape}")
        if not np.issubdtype(tokens.dtype, np.integer):
            raise ValueError(f"Split {split} tokens must be integer typed, got {tokens.dtype}")
        if len(records) != split_info.num_examples:
            raise ValueError(f"Split {split} manifest has {len(records)} records, expected {split_info.num_examples}")

        np.save(store_path / split_info.tokens_path, tokens)
        with (store_path / split_info.manifest_path).open("w", encoding="utf-8") as handle:
            for record in records:
                json.dump(asdict(record), handle, sort_keys=True)
                handle.write("\n")


def _copy_remote_file(fs, remote_path: str, local_path: Path) -> None:
    if local_path.exists():
        return
    with fs.open(remote_path, "rb") as src, local_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)


__all__ = [
    "InMemoryTokenSequenceDataset",
    "TokenMatrixDataset",
    "TokenSequenceRecord",
    "TokenStoreMetadata",
    "TokenStoreSplitInfo",
    "as_causal_lm_dataset",
    "build_passthrough_lm_data_config",
    "build_passthrough_lm_data_config_from_store",
    "materialize_token_store",
    "open_token_matrix_dataset",
    "read_token_store_manifest",
    "read_token_store_metadata",
    "write_token_store",
]
