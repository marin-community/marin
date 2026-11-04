# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded checkpoint audits with caller-supplied tensor and metadata expectations.

All ranks enter ``audit_trained_export`` in matching tensor order. Only the
export writer reads saved bytes, holding at most one replicated master tensor.
Model-specific frozen names, layouts and config rules belong to the caller.
"""

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import PurePosixPath
from typing import BinaryIO

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath

from levanter.compat.fsspec_safetensor import TensorRecord
from levanter.compat.hf_export import run_on_export_writer, write_export_json
from levanter.trainer_state import TrainerState

INDEX = "model.safetensors.index.json"
DTYPES = {"F32": np.dtype("float32"), "BF16": np.dtype(jnp.bfloat16), "F16": np.dtype("float16")}
CHUNK_BYTES = 64 * 1024**2
MAX_HEADER_BYTES = 16 * 1024**2


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: np.dtype


def _open_audit_stream(url: str) -> BinaryIO:
    filesystem, path = url_to_fs(url)
    return filesystem.open(path, "rb", cache_type="none", block_size=CHUNK_BYTES)


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
        with _open_audit_stream(str(path)) as stream:
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


def _check_layout(records: Mapping[str, TensorRecord], expected: Mapping[str, TensorSpec]) -> None:
    if records.keys() != expected.keys():
        raise ValueError("Checkpoint tensor keysets differ")
    for name, spec in expected.items():
        if records[name].shape != spec.shape or records[name].dtype != spec.dtype:
            raise ValueError(f"Checkpoint shape or dtype differs: {name}")


def _check_metadata_hashes(root: str, expected: Mapping[str, str]) -> None:
    for filename, digest in expected.items():
        if hashlib.sha256((StoragePath(root) / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Pinned source metadata changed: {filename}")


def compare_export_metadata(
    source_root: str,
    export_root: str,
    *,
    filenames: Sequence[str],
    allow_embedded_chat_template: bool = False,
    config_validator: Callable[[dict, dict], None] | None = None,
) -> list[str]:
    """Compare selected HF metadata, optionally accepting a byte-identical embedded template.

    JSON objects compare by value; other files compare by bytes. A caller-supplied
    config validator replaces equality only for ``config.json``. This keeps
    architecture-specific allowances out of the library.
    """
    source, exported = StoragePath(source_root), StoragePath(export_root)
    additions = []
    if allow_embedded_chat_template and "chat_template.jinja" not in filenames:
        raise ValueError("Embedded-template acceptance requires the separate template comparison")
    for filename in filenames:
        original_bytes = (source / filename).read_bytes()
        saved_bytes = (exported / filename).read_bytes()
        if not filename.endswith(".json"):
            if original_bytes != saved_bytes:
                raise ValueError(f"Export metadata changed: {filename}")
            continue
        original, saved = json.loads(original_bytes), json.loads(saved_bytes)
        if filename == "config.json" and config_validator is not None:
            config_validator(original, saved)
            continue
        if (
            allow_embedded_chat_template
            and filename == "tokenizer_config.json"
            and "chat_template" not in original
            and "chat_template" in saved
        ):
            embedded = saved.pop("chat_template")
            if (
                not isinstance(embedded, str)
                or embedded.encode("utf-8") != (source / "chat_template.jinja").read_bytes()
            ):
                raise ValueError("Export embedded chat template differs from source bytes")
            additions.append("tokenizer_config.json.chat_template")
        if original != saved:
            raise ValueError(f"Export metadata changed: {filename}")
    return additions


def audit_source_checkpoint(
    root: str,
    *,
    expected_metadata_hashes: Mapping[str, str],
    expected_layout: Mapping[str, TensorSpec],
    receipt_path: str,
) -> None:
    """Hash finite source tensors in bounded reads and record their lossless FP32 views."""
    _check_metadata_hashes(root, expected_metadata_hashes)
    records = checkpoint_records(root)
    _check_layout(records, expected_layout)
    tensors = {}
    for name, record in sorted(records.items()):
        native, master = hashlib.sha256(), hashlib.sha256()
        remaining = record.byte_end - record.byte_start
        with _open_audit_stream(record.file_path) as stream:
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
                remaining -= count
        tensors[name] = {
            "native_sha256": native.hexdigest(),
            "master_sha256": master.hexdigest(),
            "shape": list(record.shape),
        }
    write_export_json(
        StoragePath(receipt_path),
        {"source_metadata_sha256": dict(expected_metadata_hashes), "tensors": tensors},
    )


def _compare_saved_master(
    replicated: jax.Array, record: TensorRecord, *, expected_frozen_master_sha256: str | None
) -> dict:
    host = np.asarray(replicated.addressable_data(0), order="C")
    if host.dtype != np.float32 or tuple(host.shape) != record.shape:
        raise ValueError(f"Trained master shape or dtype differs: {record.key}")
    master_sha = hashlib.sha256(memoryview(host).cast("B")).hexdigest()
    if expected_frozen_master_sha256 is not None and master_sha != expected_frozen_master_sha256:
        raise ValueError(f"Frozen master changed: {record.key}")
    flat = host.reshape(-1)
    chunk_elements = CHUNK_BYTES // record.dtype.itemsize
    digest = hashlib.sha256()
    with _open_audit_stream(record.file_path) as stream:
        stream.seek(record.byte_start)
        for start in range(0, flat.size, chunk_elements):
            master_chunk = flat[start : start + chunk_elements]
            expected = master_chunk.astype(record.dtype)
            if not np.isfinite(master_chunk).all() or not np.isfinite(expected).all():
                raise ValueError(f"Nonfinite trained/exported tensor: {record.key}")
            payload = stream.read(expected.nbytes)
            if payload != expected.tobytes(order="C"):
                raise ValueError(f"Saved tensor differs from trained master cast: {record.key}")
            digest.update(payload)
    return {
        "master_sha256": master_sha,
        "export_sha256": digest.hexdigest(),
        "frozen": expected_frozen_master_sha256 is not None,
    }


def audit_trained_export(
    state: TrainerState,
    export_path: str,
    *,
    source_root: str,
    source_receipt_path: str,
    source_receipt_sha256: str,
    expected_metadata_hashes: Mapping[str, str],
    expected_layout: Mapping[str, TensorSpec],
    frozen_keys: frozenset[str],
    metadata_files: Sequence[str],
    allow_embedded_chat_template: bool = False,
    config_validator: Callable[[dict, dict], None] | None = None,
) -> None:
    """Collectively compare an HF export to trained FP32 masters and pinned frozen values.

    The writer checks the pinned source receipt and exact caller-supplied key,
    shape and dtype layout before all ranks replicate one tensor at a time.
    Only the writer opens exported tensor payloads. Call after the native save;
    this audit does not save or delete any model checkpoint.
    """
    tensors = state.model.to_state_dict()

    def prepare():
        _check_metadata_hashes(source_root, expected_metadata_hashes)
        receipt_bytes = StoragePath(source_receipt_path).read_bytes()
        if hashlib.sha256(receipt_bytes).hexdigest() != source_receipt_sha256:
            raise ValueError("Source audit receipt hash differs from the launch pin")
        source = json.loads(receipt_bytes)["tensors"]
        exported = checkpoint_records(export_path)
        _check_layout(exported, expected_layout)
        if tensors.keys() != source.keys() or tensors.keys() != exported.keys() or not frozen_keys <= tensors.keys():
            raise ValueError("Trained/source/exported tensor keysets differ")
        for name, tensor in tensors.items():
            if tensor.dtype != jnp.float32 or list(tensor.shape) != source[name]["shape"]:
                raise ValueError(f"Trained master shape or dtype differs: {name}")
        additions = compare_export_metadata(
            source_root,
            export_path,
            filenames=metadata_files,
            allow_embedded_chat_template=allow_embedded_chat_template,
            config_validator=config_validator,
        )
        return source, exported, additions

    prepared = run_on_export_writer(prepare)
    results = {}
    for name in sorted(tensors):
        replicated = jax.sharding.reshard(tensors[name], P())
        replicated.block_until_ready()

        def compare(name, replicated):
            assert prepared is not None
            source, exported, _ = prepared
            results[name] = _compare_saved_master(
                replicated,
                exported[name],
                expected_frozen_master_sha256=source[name]["master_sha256"] if name in frozen_keys else None,
            )

        run_on_export_writer(partial(compare, name, replicated))
        del replicated

    def write_receipt():
        assert prepared is not None
        write_export_json(
            StoragePath(export_path) / "trained-precision-audit.json",
            {
                "completed_updates": int(state.step),
                "tensor_count": len(results),
                "source_receipt": source_receipt_path,
                "source_receipt_sha256": source_receipt_sha256,
                "source_metadata_sha256": dict(expected_metadata_hashes),
                "accepted_metadata_additions": prepared[2],
                "tensors": results,
            },
        )

    run_on_export_writer(write_receipt)
