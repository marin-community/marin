# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream a native Snowball compute policy into SkyRL's existing expert-block transport.

This module runs in the JAX child, without Torch, Ray or SkyRL imports. The parent
owns the existing NCCL communicators. Only one complete expert matrix or bounded
dense slice crosses the subprocess boundary at a time; no HF checkpoint is written.
"""

import math
from collections.abc import Generator
from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec, reshard
from levanter.models.snowball import (
    SnowballBlock,
    SnowballLMHeadModel,
    snowball_block_to_state_dict,
    snowball_embeddings_to_state_dict,
    snowball_final_to_state_dict,
    snowball_to_state_dict,
)

MAX_PUBLICATION_BYTES = 64 << 20
_ROUTERS = (".mlp.router.weight", ".mlp.router.bias")
_VOCABULARY = ("model.embed_tokens.weight", "lm_head.weight")
_DTYPE_BYTES = {"bfloat16": 2, "float32": 4}


def expert_payload_key(name: str) -> str:
    return f"expert:{name}"


def dense_payload_key(name: str, offset: int, numel: int, dtype: str) -> str:
    return f"dense:{name}:{offset}:{numel}:{dtype}"


@dataclass(frozen=True)
class _SourceSlice:
    rank: int
    key: str
    local_expert: int | None = None


@dataclass(frozen=True)
class _Stage:
    layer: int | None
    names: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    sharding: NamedSharding
    local_sources: tuple[_SourceSlice, ...]


def _owner_spans(sharding: NamedSharding, shape: tuple[int, ...], devices: dict[int, jax.Device], ranks):
    indices = sharding.devices_indices_map(shape)
    spans = []
    cursor = 0
    for rank in ranks:
        leading, *rest = indices[devices[rank]]
        if not isinstance(leading, slice) or leading.step not in (None, 1):
            raise ValueError("Publication requires contiguous leading-axis partitions")
        if any(index != slice(None) for index in rest):
            raise ValueError("Publication partitions must retain complete rows")
        start = leading.start or 0
        stop = shape[0] if leading.stop is None else leading.stop
        if start != cursor or not start < stop <= shape[0]:
            raise ValueError("Publication partitions must cover the leading axis exactly once")
        spans.append((rank, start, stop))
        cursor = stop
    if cursor != shape[0]:
        raise ValueError("Publication partitions do not cover the tensor")
    return tuple(spans)


def _stage_transform(names: tuple[str, ...], shape: tuple[int, ...], sharding: NamedSharding, dtype: str):
    def materialize(module):
        if isinstance(module, SnowballBlock):
            tensors = snowball_block_to_state_dict(module, 0)
        elif names[0].startswith("model.embed_"):
            tensors = snowball_embeddings_to_state_dict(module)
        else:
            tensors = snowball_final_to_state_dict(module)
        pieces = tuple(reshard(tensors[name], sharding) for name in names)
        value = jnp.concatenate(pieces, axis=1) if len(pieces) == 2 else pieces[0]
        if value.shape != shape:
            raise ValueError("Snowball publication shape changed since inventory")
        return value.astype(jnp.float32 if dtype == "float32" else jnp.bfloat16)

    return jax.jit(materialize, out_shardings=sharding)


class SnowballPublication:
    """One frozen publication plan, with a borrowed compute view during each transfer.

    Every JAX host derives the same global inventory and walks all staging
    collectives in the same order. ``next`` emits only this host's source slices.
    ``end`` must be called after consumption, including when the parent aborts.
    Trainer/receiver installation and byte-for-byte replay remain SkyRL's job.
    """

    def __init__(self, model: SnowballLMHeadModel, mesh: Mesh):
        # Keep compiled stage transforms scoped to this publisher.
        self._stage_transform = lru_cache(maxsize=64)(_stage_transform)
        self.mesh = mesh
        self.devices = dict(enumerate(mesh.devices.flat))
        self.local_devices = {
            rank: int(device.local_hardware_id) if device.local_hardware_id is not None else device.id
            for rank, device in self.devices.items()
            if device.process_index == jax.process_index()
        }
        count = len(self.devices)
        config = model.config
        if not count or config.num_experts % count:
            raise ValueError("Whole Snowball experts must partition evenly over publication devices")
        partitioned = NamedSharding(mesh, PartitionSpec(tuple(mesh.axis_names)))
        replicated = NamedSharding(mesh, PartitionSpec())
        with jax.set_mesh(mesh):
            abstract = jax.eval_shape(lambda compute: snowball_to_state_dict(compute.transformer), model)
        for name, tensor in abstract.items():
            allowed = (jnp.bfloat16, jnp.float32) if name.endswith(_ROUTERS) else (jnp.bfloat16,)
            if tensor.dtype not in allowed:
                raise ValueError(f"Publication expects the learner's BF16 compute view: {name} is {tensor.dtype}")

        inventories = {
            rank: {
                "trainer": {"rank": rank, "dp": 0, "pp": 0, "ep": rank},
                "expert_parallel_size": count,
                "layers": list(range(config.num_layers)),
                "model": {
                    "num_experts": config.num_experts,
                    "hidden_size": config.hidden_dim,
                    "expert_hidden_size": config.hidden_dim,
                    "intermediate_size": config.intermediate_dim,
                },
                "experts": [],
                "dense": [],
            }
            for rank in self.devices
        }
        stages = []
        for layer in sorted(range(config.num_layers), key=str):
            prefix = f"model.layers.{layer}.mlp.experts"
            parts = ("gate", "up", "down")
            names = tuple(f"{prefix}.{part}_proj.weight" for part in parts)
            for part, name in zip(parts, names, strict=True):
                dimensions = (
                    (config.hidden_dim, config.intermediate_dim)
                    if part == "down"
                    else (config.intermediate_dim, config.hidden_dim)
                )
                if abstract.pop(name).shape != (config.num_experts, *dimensions):
                    raise ValueError(f"Unexpected canonical Snowball expert shape: {name}")
            for projection, sources, dimensions in (
                ("fc1", names[:2], (2 * config.intermediate_dim, config.hidden_dim)),
                ("fc2", names[2:], (config.hidden_dim, config.intermediate_dim)),
            ):
                nbytes = math.prod(dimensions) * _DTYPE_BYTES["bfloat16"]
                if nbytes > MAX_PUBLICATION_BYTES:
                    raise ValueError("One complete expert exceeds the publication buffer budget")
                key = f"{prefix}.{projection}"
                shape = (config.num_experts, *dimensions)
                owners = _owner_spans(partitioned, shape, self.devices, tuple(self.devices))
                local_sources = []
                for rank, start, stop in owners:
                    for expert in range(start, stop):
                        name = f"{key}.expert{expert}"
                        inventories[rank]["experts"].append(
                            {
                                "name": name,
                                "layer": layer,
                                "pp": 0,
                                "expert": expert,
                                "projection": projection,
                                "nbytes": nbytes,
                            }
                        )
                        if rank in self.local_devices:
                            local_sources.append(_SourceSlice(rank, expert_payload_key(name), expert - start))
                ordered = tuple(sorted(local_sources, key=lambda source: source.key))
                stages.append(_Stage(layer, sources, shape, "bfloat16", partitioned, ordered))

        for index, (name, tensor) in enumerate(sorted(abstract.items())):
            layer = int(name.split(".")[2]) if name.startswith("model.layers.") else None
            dtype = "float32" if name.endswith(_ROUTERS) else "bfloat16"
            nbytes = math.prod(tensor.shape) * _DTYPE_BYTES[dtype]
            split = tensor.shape[0] % count == 0 and (name in _VOCABULARY or nbytes > MAX_PUBLICATION_BYTES)
            sharding = partitioned if split else replicated
            owners = _owner_spans(
                sharding, tensor.shape, self.devices, tuple(self.devices) if split else (index % count,)
            )
            local_sources = []
            row_elements = math.prod(tensor.shape[1:])
            for rank, start, stop in owners:
                numel = (stop - start) * row_elements
                if numel * _DTYPE_BYTES[dtype] > MAX_PUBLICATION_BYTES:
                    raise ValueError(f"Dense source shard exceeds the publication buffer budget: {name}")
                offset = start * row_elements
                if rank in self.local_devices:
                    local_sources.append(_SourceSlice(rank, dense_payload_key(name, offset, numel, dtype)))
                inventories[rank]["dense"].append(
                    {
                        "hf_name": name,
                        "hf_offset": offset,
                        "numel": numel,
                        "wire_dtype": dtype,
                        "source_key": name,
                        "source_offset": 0,
                        "pp": 0,
                    }
                )
            stages.append(_Stage(layer, (name,), tensor.shape, dtype, sharding, tuple(local_sources)))
        self.inventories = inventories
        self.stages = tuple(stages)
        self.session: Generator[dict, None, None] | None = None
        self.version: int | None = None
        self.exhausted = False

    def inventory(self, version: int) -> dict:
        platforms = {self.devices[rank].platform for rank in self.local_devices}
        if len(platforms) != 1 or not platforms <= {"cpu", "gpu"}:
            raise ValueError("SkyRL publication requires one local CPU or CUDA device platform")
        return {
            "version": version,
            "payload_order": tuple((source.rank, source.key) for stage in self.stages for source in stage.local_sources),
            "inventories": tuple(self.inventories[rank] for rank in sorted(self.local_devices)),
            "devices": dict(self.local_devices),
            "device_type": "cuda" if "gpu" in platforms else "cpu",
        }

    def begin(self, model: SnowballLMHeadModel, version: int) -> dict:
        if self.session is not None:
            raise RuntimeError("A Snowball publication is already in progress")
        self.version = version
        self.exhausted = False
        self.session = self._payloads(model, version)
        return {"version": version}

    def next(self) -> dict | None:
        """Return the next local source slice, or None at exhaustion."""
        if self.session is None:
            raise RuntimeError("No Snowball publication is in progress")
        payload = next(self.session, None)
        self.exhausted = payload is None
        return payload

    def end(self) -> dict:
        """Close the stream and report whether the caller observed exhaustion."""
        if self.session is None:
            raise RuntimeError("No Snowball publication is in progress")
        self.session.close()
        self.session = None
        return {"version": self.version, "complete": self.exhausted}

    def _payloads(self, model: SnowballLMHeadModel, version: int) -> Generator[dict, None, None]:
        for stage in self.stages:
            names = stage.names
            module = model.transformer
            if stage.layer is not None:
                names = tuple(f"model.layers.0.{name.split('.', 3)[3]}" for name in names)
                module = module.blocks[stage.layer]
            with jax.set_mesh(self.mesh):
                array = self._stage_transform(names, stage.shape, stage.sharding, stage.dtype)(
                    module
                ).block_until_ready()
            shards = {shard.device: shard.data for shard in array.addressable_shards}
            for source in stage.local_sources:
                shard = shards[self.devices[source.rank]]
                if source.local_expert is None:
                    value = np.asarray(shard).reshape(-1)
                else:
                    # Slice the single-device shard outside the learner's global mesh.
                    with jax.set_mesh(None):
                        value = np.asarray(shard[source.local_expert])
                yield self._payload(version, source.rank, source.key, stage.dtype, value)
                del shard, value
            del shards, array

    @staticmethod
    def _payload(version: int, rank: int, key: str, dtype: str, value: np.ndarray) -> dict:
        data = value.tobytes(order="C")
        if len(data) > MAX_PUBLICATION_BYTES:
            raise ValueError("Publication produced an oversized subprocess payload")
        return {"version": version, "root": rank, "key": key, "dtype": dtype, "shape": value.shape, "data": data}
