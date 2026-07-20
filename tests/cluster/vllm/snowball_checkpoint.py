# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint loading for the vendored Snowball training model."""

import hashlib
import json
from typing import Any

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike
from levanter.checkpoint import load_checkpoint as load_levanter_checkpoint
from rigging.filesystem import StoragePath

# The vendored experiment path is the immutable training source for Snowball.
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig as VendoredGrugModelConfig
from experiments.june_tpu_67b_a2b.moe.model import Transformer as VendoredTransformer
from tests.cluster.vllm.snowball import SNOWBALL_GPU, ModelLocation

LOGICAL_ARRAY_DIGEST_SCHEMA = "snowball-logical-array-sha256-v1"


def read_executor_info(location: ModelLocation = SNOWBALL_GPU) -> dict[str, Any]:
    return json.loads(StoragePath(location.executor_info_path).read_text())


def decode_vendored_config(executor_info: dict[str, Any]) -> VendoredGrugModelConfig:
    return draccus.decode(VendoredGrugModelConfig, executor_info["config"]["model"])


def load_checkpoint(
    config: VendoredGrugModelConfig,
    mesh: jax.sharding.Mesh,
    *,
    location: ModelLocation = SNOWBALL_GPU,
    parameter_dtype: DTypeLike | None = None,
) -> tuple[VendoredTransformer, jax.Array]:
    template = eqx.filter_eval_shape(VendoredTransformer.init, config, key=jax.random.PRNGKey(0))
    if parameter_dtype is not None:

        def cast_dtype(value):
            dtype = getattr(value, "dtype", None)
            if dtype is None or not jnp.issubdtype(dtype, jnp.inexact):
                return value
            return jax.ShapeDtypeStruct(
                value.shape,
                parameter_dtype,
                sharding=getattr(value, "sharding", None),
                weak_type=getattr(value, "weak_type", False),
            )

        template = jax.tree.map(cast_dtype, template)
    checkpoint_state = load_levanter_checkpoint(
        {
            "params": template,
            "pending_qb_betas": jax.ShapeDtypeStruct((config.num_layers, config.num_experts), jnp.float32),
        },
        location.checkpoint_path,
        mesh=mesh,
    )
    jax.block_until_ready(checkpoint_state)
    return checkpoint_state["params"], checkpoint_state["pending_qb_betas"]


def apply_pending_qb_betas(model: VendoredTransformer, pending_qb_betas: jax.Array) -> VendoredTransformer:
    assert model.stacked_blocks is not None
    # Mirrors train._apply_qb_betas without importing the training entrypoint.
    router_bias = -pending_qb_betas
    router_bias -= jnp.mean(router_bias, axis=-1, keepdims=True)
    return eqx.tree_at(lambda tree: tree.stacked_blocks.stacked.mlp.router_bias, model, router_bias)


def prepare_bf16_parameters(model: VendoredTransformer, pending_qb_betas: jax.Array) -> VendoredTransformer:
    """Apply the FP32 pending QB update, then match the canonical BF16 compute cast."""
    assert pending_qb_betas.dtype == jnp.float32
    model = apply_pending_qb_betas(model, pending_qb_betas)
    model = jax.tree.map(
        lambda value: value.astype(jnp.bfloat16) if eqx.is_inexact_array(value) else value,
        model,
    )
    jax.block_until_ready(model)
    return model


def _logical_chunk_indices(array: jax.Array | np.ndarray, chunk_bytes: int):
    """Yield bounded C-order logical slices without reading the array."""
    shape = tuple(int(dim) for dim in array.shape)
    if not shape or int(array.size) == 0:
        yield (slice(None),) * len(shape)
        return

    suffix_bytes = int(np.dtype(array.dtype).itemsize)
    for axis in range(len(shape) - 1, -1, -1):
        axis_bytes = suffix_bytes * shape[axis]
        if axis_bytes > chunk_bytes:
            width = max(1, chunk_bytes // suffix_bytes)
            for prefix in np.ndindex(shape[:axis]):
                for start in range(0, shape[axis], width):
                    index = (*prefix, slice(start, min(start + width, shape[axis])))
                    index += (slice(None),) * (len(shape) - axis - 1)
                    yield index
            return
        suffix_bytes = axis_bytes
    yield (slice(None),) * len(shape)


def _slice_bounds(index: slice, dimension: int) -> tuple[int, int]:
    start, stop, step = index.indices(dimension)
    if step != 1:
        raise ValueError(f"Logical digest only supports unit-stride slices, got {index}")
    return start, stop


def _host_logical_chunk(array: jax.Array | np.ndarray, index: tuple[int | slice, ...]) -> np.ndarray:
    """Assemble one logical chunk from physical shards with bounded host memory."""
    if isinstance(array, np.ndarray):
        return np.asarray(array[index])
    if not array.is_fully_addressable:
        raise ValueError("Logical digest requires every array shard to be addressable by this process")

    shape = tuple(int(dim) for dim in array.shape)
    output_shape = tuple(
        stop - start
        for dimension, requested in zip(shape, index, strict=True)
        if isinstance(requested, slice)
        for start, stop in (_slice_bounds(requested, dimension),)
    )
    output = np.empty(output_shape, dtype=array.dtype)
    covered = np.zeros(output_shape, dtype=np.bool_)
    for shard in array.addressable_shards:
        local_index: list[int | slice] = []
        output_index: list[int | slice] = []
        intersects = True
        for dimension, requested, shard_index in zip(shape, index, shard.index, strict=True):
            if isinstance(shard_index, int):
                shard_start, shard_stop = shard_index, shard_index + 1
                shard_dimension_is_collapsed = True
            else:
                shard_start, shard_stop = _slice_bounds(shard_index, dimension)
                shard_dimension_is_collapsed = False

            if isinstance(requested, int):
                requested_value = requested if requested >= 0 else dimension + requested
                if requested_value < shard_start or requested_value >= shard_stop:
                    intersects = False
                    break
                if not shard_dimension_is_collapsed:
                    local_index.append(requested_value - shard_start)
                continue

            requested_start, requested_stop = _slice_bounds(requested, dimension)
            overlap_start = max(requested_start, shard_start)
            overlap_stop = min(requested_stop, shard_stop)
            if overlap_start >= overlap_stop:
                intersects = False
                break
            if not shard_dimension_is_collapsed:
                local_index.append(slice(overlap_start - shard_start, overlap_stop - shard_start))
            output_index.append(slice(overlap_start - requested_start, overlap_stop - requested_start))

        if not intersects:
            continue
        destination = tuple(output_index)
        # ``logical_array_digest`` is called while the model mesh is active, but
        # each addressable shard has single-device sharding. Temporarily clear
        # the mesh so JAX does not try to compile this local slice for all devices.
        with jax.set_mesh(None):
            shard_value = shard.data[tuple(local_index)]
        output[destination] = np.asarray(jax.device_get(shard_value))
        covered[destination] = True

    if not bool(np.all(covered)):
        raise ValueError(f"Addressable shards did not cover logical chunk {index} of shape {shape}")
    return output


def _little_endian_bytes(value: jax.Array | np.ndarray) -> bytes:
    value = np.ascontiguousarray(np.asarray(jax.device_get(value)))
    if value.dtype == np.dtype(jnp.bfloat16):
        words = value.view(np.uint16).astype(np.dtype("<u2"), copy=False)
        return words.tobytes(order="C")
    return value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes(order="C")


def _update_digest_field(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="little", signed=False))
    digest.update(value)


def logical_array_digest(pytree: Any, *, chunk_bytes: int = 64 * 1024**2) -> str:
    """Hash canonical names, metadata, and logical bytes for every array leaf.

    Reading fixed logical slices makes the result independent of device sharding and
    bounds host memory even for the stacked Snowball expert tensors.
    """
    if chunk_bytes <= 0:
        raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")

    path_leaves, _ = jax.tree_util.tree_flatten_with_path(pytree)
    arrays = sorted(
        (jax.tree_util.keystr(path), leaf)
        for path, leaf in path_leaves
        if eqx.is_array(leaf) or isinstance(leaf, np.ndarray)
    )
    digest = hashlib.sha256()
    _update_digest_field(digest, LOGICAL_ARRAY_DIGEST_SCHEMA.encode())
    for name, array in arrays:
        _update_digest_field(digest, name.encode())
        _update_digest_field(digest, str(np.dtype(array.dtype)).encode())
        _update_digest_field(digest, json.dumps(tuple(int(dim) for dim in array.shape)).encode())
        for index in _logical_chunk_indices(array, chunk_bytes):
            digest.update(_little_endian_bytes(_host_logical_chunk(array, index)))
    return digest.hexdigest()
