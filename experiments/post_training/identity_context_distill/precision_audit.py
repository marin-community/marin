# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regional, bounded-memory source and LR0 export audits; invoke only on the worker."""

import hashlib
import json
import time
from collections.abc import Mapping
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from levanter.compat.fsspec_safetensor import TensorRecord
from levanter.compat.hf_export import run_on_export_writer, write_export_json
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.identity_context_distill.precision_source_audit import (
    CHUNK_BYTES,
    check_export_auxiliary_metadata,
    check_precision_layout,
    check_source_metadata,
    checkpoint_records,
    open_audit_stream,
)
from experiments.post_training.identity_context_distill.sft import check_export_metadata


def check_master_hash(name: str, replicated: jax.Array, expected: str) -> None:
    host = np.asarray(replicated.addressable_data(0), order="C")
    digest = hashlib.sha256(memoryview(host).cast("B")).hexdigest()
    if digest != expected:
        raise ValueError(f"Initialized master bits differ: {name}")


def audit_initialized_masters(tensors: Mapping[str, jax.Array], source_receipt_path: str, receipt_path: str) -> None:
    """Check HF-mapped master bits collectively, one replicated tensor at a time.

    Pass the model's standard ``to_state_dict()`` mapping on every rank inside
    the trainer mesh. No dtype conversion is permitted on the initialized values.
    """
    started = time.monotonic()
    source = json.loads(StoragePath(source_receipt_path).read_bytes())["tensors"]
    if len(source) != 502 or tensors.keys() != source.keys():
        raise ValueError("Initialized master keyset differs from source")
    for name in sorted(source):
        tensor = tensors[name]
        if tensor.dtype != jnp.float32 or list(tensor.shape) != source[name]["shape"]:
            raise ValueError(f"Initialized master shape/dtype differs: {name}")
        replicated = jax.sharding.reshard(tensor, P())
        replicated.block_until_ready()

        run_on_export_writer(partial(check_master_hash, name, replicated, source[name]["master_sha256"]))
        del replicated
    run_on_export_writer(
        lambda: write_export_json(
            StoragePath(receipt_path),
            {
                "source_receipt": source_receipt_path,
                "tensor_count": 502,
                "tensors": source,
                "master_comparison_seconds": time.monotonic() - started,
            },
        )
    )


def compare_tensor_bytes(source: TensorRecord, exported: TensorRecord) -> str:
    """Compare all bytes and finite values with bounded source/export read buffers."""
    if source.shape != exported.shape or source.dtype != exported.dtype:
        raise ValueError(f"Tensor shape/dtype changed: {source.key}")
    digest = hashlib.sha256()
    remaining = source.byte_end - source.byte_start
    with (
        open_audit_stream(source.file_path) as original,
        open_audit_stream(exported.file_path) as written,
    ):
        original.seek(source.byte_start)
        written.seek(exported.byte_start)
        while remaining:
            count = min(CHUNK_BYTES, remaining)
            left = original.read(count)
            right = written.read(count)
            if len(left) != count or len(right) != count:
                raise ValueError(f"Truncated tensor payload: {source.key}")
            if left != right:
                raise ValueError(f"LR0 tensor bytes changed: {source.key}")
            if not np.isfinite(np.frombuffer(left, dtype=source.dtype)).all():
                raise ValueError(f"Nonfinite source/export tensor: {source.key}")
            digest.update(left)
            remaining -= count
    return digest.hexdigest()


def audit_precision_export(
    source_root: str, export_root: str, expected_hashes: dict[str, str], *, receipt_path: str
) -> None:
    """Verify all 502 LR0 tensors and metadata, writing a receipt only after full success."""
    started = time.monotonic()
    check_source_metadata(source_root, expected_hashes)
    source_base, export_base = StoragePath(source_root), StoragePath(export_root)
    source = checkpoint_records(source_root)
    exported = checkpoint_records(export_root)
    check_precision_layout(exported)
    if source.keys() != exported.keys():
        raise ValueError("Export tensor keyset differs from pinned source")
    source_config = json.loads((source_base / "config.json").read_bytes())
    exported_config = json.loads((export_base / "config.json").read_bytes())
    check_export_metadata(source_config, exported_config)
    metadata_additions = check_export_auxiliary_metadata(source_root, export_root)
    tensors = {}
    payload_started = time.monotonic()
    for name in sorted(source):
        tensors[name] = {
            "sha256": compare_tensor_bytes(source[name], exported[name]),
            "dtype": str(source[name].dtype),
            "shape": list(source[name].shape),
            "bytes": source[name].byte_end - source[name].byte_start,
        }
    payload_seconds = time.monotonic() - payload_started
    payload_bytes = 2 * sum(record.byte_end - record.byte_start for record in source.values())
    write_export_json(
        StoragePath(receipt_path),
        {
            "source_metadata_sha256": expected_hashes,
            "accepted_metadata_additions": metadata_additions,
            "tensor_count": 502,
            "tensors": tensors,
            "audit_io": {
                "chunk_bytes": CHUNK_BYTES,
                "payload_bytes_read": payload_bytes,
                "payload_read_and_hash_seconds": payload_seconds,
                "payload_bytes_per_second": payload_bytes / payload_seconds if payload_seconds > 0 else None,
                "total_seconds_before_receipt": time.monotonic() - started,
            },
        },
    )
