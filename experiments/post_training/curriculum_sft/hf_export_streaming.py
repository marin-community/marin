# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream a stacked Grug TensorStore checkpoint into sharded HF safetensors."""

import itertools
import json
import logging
import os
import struct
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np
import tensorstore as ts
from levanter.checkpoint_manifest import CheckpointArray, read_manifest
from levanter.compat.hf_checkpoints import (
    DEFAULT_MAX_SHARD_SIZE,
    SAFE_TENSORS_INDEX_NAME,
    SAFE_TENSORS_MODEL,
    _save_tokenizer_pretrained,
)
from levanter.tensorstore_serialization import ARRAY_DRIVER, KVSTORE_DRIVER, _create_ocdbt_spec
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.moe.model import GrugModelConfig

logger = logging.getLogger(__name__)

_BF16_BYTES = 2
_READ_AHEAD = 4
_UPLOAD_WORKERS = 8


@dataclass(frozen=True)
class _Weight:
    name: str
    source: str
    shape: tuple[int, ...]
    layer: int | None = None
    transpose: bool = False
    router_bias: bool = False


@dataclass
class _ShardFiles:
    files: dict[str, np.memmap]
    views: dict[str, np.ndarray]
    index: dict
    source_shards: dict[str, list[str]]


def _weights(config: GrugModelConfig, arrays: dict[str, CheckpointArray]) -> list[_Weight]:
    root = (
        ("model.embed_tokens.weight", "token_embed", False),
        ("model.embed_norm.weight", "embed_norm/weight", False),
        ("model.embed_gated_norm.down_proj.weight", "embed_gated_norm/w_down", True),
        ("model.embed_gated_norm.up_proj.weight", "embed_gated_norm/w_up", True),
        ("model.norm.weight", "final_norm/weight", False),
        ("model.final_gated_norm.down_proj.weight", "final_gated_norm/w_down", True),
        ("model.final_gated_norm.up_proj.weight", "final_gated_norm/w_up", True),
        ("lm_head.weight", "output_proj", True),
    )
    block = (
        ("input_layernorm.weight", "rms_attn/weight", False),
        ("attn_gated_norm.down_proj.weight", "attn_gated_norm/w_down", True),
        ("attn_gated_norm.up_proj.weight", "attn_gated_norm/w_up", True),
        ("self_attn.q_proj.weight", "attn/w_q", True),
        ("self_attn.k_proj.weight", "attn/w_k", True),
        ("self_attn.v_proj.weight", "attn/w_v", True),
        ("self_attn.o_proj.weight", "attn/w_o", True),
        ("self_attn.attn_gate.weight", "attn/attn_gate", True),
        ("post_attention_layernorm.weight", "rms_mlp/weight", False),
        ("mlp_gated_norm.down_proj.weight", "mlp_gated_norm/w_down", True),
        ("mlp_gated_norm.up_proj.weight", "mlp_gated_norm/w_up", True),
        ("mlp.router.weight", "mlp/router", True),
        ("mlp.router.bias", "mlp/router_bias", False),
        ("mlp.experts.gate_proj.weight", "mlp/expert_mlp/w_gate", True),
        ("mlp.experts.up_proj.weight", "mlp/expert_mlp/w_up", True),
        ("mlp.experts.down_proj.weight", "mlp/expert_mlp/w_down", True),
    )
    shared = (
        ("shared_expert.gate_proj.weight", "shared/w_gate", True),
        ("shared_expert.up_proj.weight", "shared/w_up", True),
        ("shared_expert.down_proj.weight", "shared/w_down", True),
    )

    def weight(name: str, source: str, layer: int | None, transpose: bool, router_bias: bool = False) -> _Weight:
        entry = arrays[source]
        shape = entry.shape if layer is None else entry.shape[1:]
        if layer is not None and entry.shape[0] != config.num_layers:
            raise ValueError(f"{source} has {entry.shape[0]} layers, expected {config.num_layers}")
        if transpose:
            shape = (*shape[:-2], shape[-1], shape[-2])
        return _Weight(name, source, shape, layer, transpose, router_bias)

    weights = [weight(name, f"params/{path}", None, transpose) for name, path, transpose in root]
    for layer in range(config.num_layers):
        for name, path, transpose in (*block, *(shared if config.shared_expert_intermediate_dim > 0 else ())):
            router_bias = name == "mlp.router.bias"
            source = "pending_qb_betas" if router_bias else f"params/stacked_blocks/stacked/{path}"
            weights.append(weight(f"model.layers.{layer}.{name}", source, layer, transpose, router_bias))
    return weights


def _create_shards(directory: Path, weights: list[_Weight], max_shard_size: int) -> _ShardFiles:
    by_source: dict[str, list[_Weight]] = {}
    for weight in weights:
        by_source.setdefault(weight.source, []).append(weight)
    groups: list[tuple[str, list[_Weight]]] = []
    total_size = 0
    for source, source_weights in by_source.items():
        # Keep shard boundaries inside one native array so its files can upload
        # while later arrays are still being read.
        group: list[_Weight] = []
        group_size = 0
        for weight in source_weights:
            size = int(np.prod(weight.shape)) * _BF16_BYTES
            if group and group_size + size > max_shard_size:
                groups.append((source, group))
                group = []
                group_size = 0
            group.append(weight)
            group_size += size
            total_size += size
        if group:
            groups.append((source, group))

    shard_count = len(groups)
    views: dict[str, np.ndarray] = {}
    files: dict[str, np.memmap] = {}
    source_shards: dict[str, list[str]] = {}
    weight_map: dict[str, str] = {}
    for number, (source, tensors) in enumerate(groups, start=1):
        filename = SAFE_TENSORS_MODEL if shard_count == 1 else f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        source_shards.setdefault(source, []).append(filename)
        offsets: dict[str, tuple[int, int]] = {}
        size = 0
        for tensor in tensors:
            length = int(np.prod(tensor.shape)) * _BF16_BYTES
            offsets[tensor.name] = (size, size + length)
            size += length
            weight_map[tensor.name] = filename
        header = {
            "__metadata__": {"format": "pt"},
            **{
                tensor.name: {"dtype": "BF16", "shape": list(tensor.shape), "data_offsets": list(offsets[tensor.name])}
                for tensor in tensors
            },
        }
        # Safetensors stores an eight-byte header length, padded JSON, then raw tensor bytes.
        header_bytes = json.dumps(header, separators=(",", ":")).encode()
        header_bytes += b" " * (-len(header_bytes) % 8)
        path = directory / filename
        with path.open("wb") as output:
            output.write(struct.pack("<Q", len(header_bytes)))
            output.write(header_bytes)
            output.truncate(8 + len(header_bytes) + size)
        mapped = np.memmap(path, dtype=np.uint8, mode="r+")
        files[filename] = mapped
        for tensor in tensors:
            views[tensor.name] = np.ndarray(
                tensor.shape,
                dtype=ml_dtypes.bfloat16,
                buffer=mapped,
                offset=8 + len(header_bytes) + offsets[tensor.name][0],
            )
    return _ShardFiles(files, views, {"metadata": {"total_size": total_size}, "weight_map": weight_map}, source_shards)


def _read_chunks(checkpoint_path: str, entry: CheckpointArray):
    store = ts.open(_create_ocdbt_spec(checkpoint_path, entry.path), read=True).result()
    ranges = [range(0, dim, chunk) for dim, chunk in zip(entry.shape, entry.chunk_shape, strict=True)]
    pending = deque()
    for starts in itertools.product(*ranges):
        slices = tuple(
            slice(start, min(start + chunk, dim))
            for start, chunk, dim in zip(starts, entry.chunk_shape, entry.shape, strict=True)
        )
        pending.append((slices, store[slices].read()))
        if len(pending) >= _READ_AHEAD:
            ready_slices, future = pending.popleft()
            yield ready_slices, np.asarray(future.result())
    while pending:
        ready_slices, future = pending.popleft()
        yield ready_slices, np.asarray(future.result())


def _write_chunks(
    checkpoint_path: str,
    arrays: dict[str, CheckpointArray],
    weights: list[_Weight],
    views: dict[str, np.ndarray],
    source_complete: Callable[[str, list[_Weight]], None],
) -> None:
    by_source: dict[str, list[_Weight]] = {}
    for weight in weights:
        by_source.setdefault(weight.source, []).append(weight)

    for source, targets in by_source.items():
        started = time.monotonic()
        entry = arrays[source]
        if targets[0].router_bias:
            # The bias mean covers every expert even if a future checkpoint splits
            # that small array across TensorStore chunks.
            store = ts.open(_create_ocdbt_spec(checkpoint_path, source), read=True).result()
            bias = -np.asarray(store.read().result()).astype(np.float32)
            bias -= bias.mean(axis=-1, keepdims=True)
            chunks = [(tuple(slice(0, dim) for dim in entry.shape), bias)]
        else:
            chunks = _read_chunks(checkpoint_path, entry)
        for slices, data in chunks:
            for target in targets:
                if target.layer is None:
                    value = data
                    destination = slices
                elif target.layer in range(slices[0].start, slices[0].stop):
                    value = data[target.layer - slices[0].start]
                    destination = slices[1:]
                else:
                    continue
                if target.transpose:
                    value = np.swapaxes(value, -1, -2)
                    destination = (*destination[:-2], destination[-1], destination[-2])
                views[target.name][destination] = value.astype(ml_dtypes.bfloat16, copy=False)
        logger.info("Streamed %s in %.1fs", source, time.monotonic() - started)
        source_complete(source, targets)


def export_checkpoint_streaming(
    checkpoint_path: str,
    output_path: str,
    config: GrugModelConfig,
    tokenizer,
    directory: Path,
    *,
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
) -> None:
    """Read each native checkpoint chunk once and publish HF shards and metadata."""
    manifest = read_manifest(checkpoint_path)
    if manifest is None or manifest.array_driver != ARRAY_DRIVER or manifest.kvstore_driver != KVSTORE_DRIVER:
        raise ValueError(f"Expected an OCDBT zarr3 checkpoint with a manifest at {checkpoint_path}")
    arrays = {entry.path: entry for entry in manifest.arrays}
    weights = _weights(config, arrays)
    shards = _create_shards(directory, weights, max_shard_size)
    logger.info("Writing %d BF16 tensors into %d safetensors shards", len(weights), len(shards.files))

    def upload(filename: str) -> None:
        started = time.monotonic()
        StoragePath(prefix_join(output_path, filename)).upload_from(str(directory / filename))
        logger.info("Uploaded %s in %.1fs", filename, time.monotonic() - started)
        os.unlink(directory / filename)

    with ThreadPoolExecutor(max_workers=_UPLOAD_WORKERS) as pool:
        uploads = []

        def source_complete(source: str, targets: list[_Weight]) -> None:
            for filename in shards.source_shards[source]:
                shards.files[filename].flush()
            for target in targets:
                del shards.views[target.name]
            for filename in shards.source_shards[source]:
                del shards.files[filename]
                uploads.append(pool.submit(upload, filename))

        _write_chunks(checkpoint_path, arrays, weights, shards.views, source_complete)
        for future in uploads:
            future.result()

    _save_tokenizer_pretrained(tokenizer, str(directory))
    converter = (
        config.hf_checkpoint_converter().replaced(tokenizer=tokenizer).with_config_overrides({"dtype": "bfloat16"})
    )
    (directory / "config.json").write_text(json.dumps(converter.hf_config_dict(config, config.vocab_size)))
    if len(set(shards.index["weight_map"].values())) > 1:
        (directory / SAFE_TENSORS_INDEX_NAME).write_text(json.dumps(shards.index))
    for path in directory.iterdir():
        upload(path.name)
