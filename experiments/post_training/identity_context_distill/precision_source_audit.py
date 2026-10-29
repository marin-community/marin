# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regional source checkpoint audit, independent of the SFT factory."""

import hashlib
import json
import math
import time
from pathlib import PurePosixPath
from typing import BinaryIO

import jax.numpy as jnp
import numpy as np
from levanter.compat.fsspec_safetensor import TensorRecord
from levanter.compat.hf_export import write_export_json
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath

INDEX = "model.safetensors.index.json"
SOURCE_FILES = (
    "config.json",
    "generation_config.json",
    INDEX,
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
ROUTER_KEYS = frozenset(f"model.layers.{layer}.mlp.router.bias" for layer in range(26))
DTYPES = {"F32": np.dtype("float32"), "BF16": np.dtype(jnp.bfloat16)}
CHUNK_BYTES = 64 * 1024**2
MAX_HEADER_BYTES = 16 * 1024**2


def open_audit_stream(url: str) -> BinaryIO:
    """Open a bounded, uncached read without passing file options to the S3 session."""
    filesystem, path = url_to_fs(url)
    return filesystem.open(path, "rb", cache_type="none", block_size=CHUNK_BYTES)


def check_export_auxiliary_metadata(source_root: str, export_root: str) -> list[str]:
    """Check tokenizer/generation metadata, allowing an exact embedded template copy.

    Tokenizer saving may add chat_template to tokenizer_config.json when the
    source stores it only in chat_template.jinja. Every other key and the
    separate template file must remain equal to the source. Returns the accepted
    added fields for the audit receipt.
    """
    source, exported = StoragePath(source_root), StoragePath(export_root)
    template = (source / "chat_template.jinja").read_bytes()
    if template != (exported / "chat_template.jinja").read_bytes():
        raise ValueError("Export chat template bytes changed")
    additions = []
    for filename in ("generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        original = json.loads((source / filename).read_bytes())
        saved = json.loads((exported / filename).read_bytes())
        if filename == "tokenizer_config.json" and "chat_template" not in original and "chat_template" in saved:
            embedded = saved.pop("chat_template")
            if not isinstance(embedded, str) or embedded.encode("utf-8") != template:
                raise ValueError("Export embedded chat template differs from source bytes")
            additions.append("tokenizer_config.json.chat_template")
        if original != saved:
            raise ValueError(f"Export metadata changed: {filename}")
    return additions


def checkpoint_records(root: str) -> dict[str, TensorRecord]:
    """Read headers and validate index coverage, tensor sizes and nonoverlapping bounds."""
    base = StoragePath(root)
    index = json.loads((base / INDEX).read_bytes())
    weight_map = index["weight_map"]
    if not weight_map:
        raise ValueError("Empty tensor index")
    records = {}
    total = 0
    for shard in sorted(set(weight_map.values())):
        if PurePosixPath(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError("Index shard must be a plain safetensors filename")
        path = base / shard
        size = path.size()
        with open_audit_stream(str(path)) as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated safetensors prefix: {shard}")
            length = int.from_bytes(prefix, "little")
            if not 0 < length <= MAX_HEADER_BYTES or 8 + length > size:
                raise ValueError(f"Invalid safetensors header length: {shard}")
            header = json.loads(stream.read(length))
        intervals = []
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if name in records or weight_map.get(name) != shard:
                raise ValueError(f"Shard/index disagreement: {name}")
            if spec["dtype"] not in DTYPES:
                raise ValueError(f"Unexpected precision dtype: {name}")
            dtype = DTYPES[spec["dtype"]]
            shape = tuple(spec["shape"])
            if any(type(dimension) is not int or dimension <= 0 for dimension in shape):
                raise ValueError(f"Invalid tensor shape: {name}")
            start, end = spec["data_offsets"]
            if type(start) is not int or type(end) is not int or start < 0:
                raise ValueError(f"Invalid tensor offsets: {name}")
            if end - start != math.prod(shape) * dtype.itemsize or 8 + length + end > size:
                raise ValueError(f"Tensor bounds/size mismatch: {name}")
            records[name] = TensorRecord(name, dtype, shape, str(path), 8 + length + start, 8 + length + end)
            intervals.append((start, end))
            total += end - start
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError(f"Overlapping or unindexed payload in {shard}")
            cursor = end
        if 8 + length + cursor != size:
            raise ValueError(f"Trailing unindexed payload in {shard}")
    if records.keys() != weight_map.keys():
        raise ValueError("Index references missing tensors")
    if index.get("metadata", {}).get("total_size") != total:
        raise ValueError("Index total_size differs from tensor payload")
    return records


def check_precision_layout(records: dict[str, TensorRecord]) -> None:
    """Require the frozen 502-key dtype layout; exact names are pinned by the source index."""
    found = {name for name in records if name.endswith(".mlp.router.bias")}
    if len(records) != 502 or found != ROUTER_KEYS:
        raise ValueError("Expected 502 tensors including exactly 26 router biases")
    for name, record in records.items():
        expected = DTYPES["F32"] if name in ROUTER_KEYS else DTYPES["BF16"]
        if record.dtype != expected:
            raise ValueError(f"Wrong storage dtype: {name}")
        if name in ROUTER_KEYS and record.shape != (256,):
            raise ValueError(f"Wrong router bias shape: {name}")


def check_source_metadata(root: str, expected_hashes: dict[str, str]) -> None:
    """Check pinned regional metadata bytes and every tensor header before loading."""
    if set(expected_hashes) != set(SOURCE_FILES):
        raise ValueError("Source audit requires all six frozen metadata hashes")
    for filename in SOURCE_FILES:
        contents = (StoragePath(root) / filename).read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected_hashes[filename]:
            raise ValueError(f"Pinned source metadata changed: {filename}")
    check_precision_layout(checkpoint_records(root))


def audit_precision_source(root: str, expected_hashes: dict[str, str], receipt_path: str) -> None:
    """Stream all source values before load, recording native and lossless FP32 hashes."""
    started = time.monotonic()
    check_source_metadata(root, expected_hashes)
    records = checkpoint_records(root)
    payload_started = time.monotonic()
    payload_bytes = 0
    tensors = {}
    for name, record in sorted(records.items()):
        native, master = hashlib.sha256(), hashlib.sha256()
        remaining = record.byte_end - record.byte_start
        with open_audit_stream(record.file_path) as stream:
            stream.seek(record.byte_start)
            while remaining:
                count = min(CHUNK_BYTES, remaining)
                chunk = stream.read(count)
                if len(chunk) != count:
                    raise ValueError(f"Truncated source tensor: {name}")
                values = np.frombuffer(chunk, dtype=record.dtype)
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite source tensor: {name}")
                native.update(chunk)
                master.update(values.astype(np.float32).tobytes())
                payload_bytes += count
                remaining -= count
        tensors[name] = {
            "native_sha256": native.hexdigest(),
            "master_sha256": master.hexdigest(),
            "shape": list(record.shape),
        }
    payload_seconds = time.monotonic() - payload_started
    write_export_json(
        StoragePath(receipt_path),
        {
            "source_metadata_sha256": expected_hashes,
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
