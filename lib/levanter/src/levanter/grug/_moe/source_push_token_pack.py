# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Package-private token packing kernels for source-push MoE forward."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import AbstractMesh, Mesh, PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.source_push_plan import SOURCE_PUSH_MESH_AXIS, SourcePushPlan
from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed


DEFAULT_SOURCE_PUSH_TOKEN_PACK_HIDDEN_BLOCK = 128


@dataclass(frozen=True, slots=True)
class SourcePushTokenPackPallasBlockSizes:
    """Tile sizes for the local source-push token pack copy kernel."""

    hidden_block: int = DEFAULT_SOURCE_PUSH_TOKEN_PACK_HIDDEN_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushTokenPackPallasBlockSizes":
        return cls()


def source_push_pack_tokens_pallas_mgpu(
    x: Float[Array, "S T D"],
    plan: SourcePushPlan,
    *,
    block_sizes: SourcePushTokenPackPallasBlockSizes | None = None,
    output_dtype: jnp.dtype = jnp.bfloat16,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> Float[Array, "S Dst Q M D"]:
    """Pack source tokens into queue order using a lane-lowered local copy kernel."""

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU source-push token pack requires a GPU backend")
    block_sizes = SourcePushTokenPackPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_source_push_token_pack_pallas_request(x, plan, block_sizes)
    return _source_push_pack_tokens_pallas_call(
        x,
        plan.token_ids,
        plan.valid_mask,
        hidden_block=block_sizes.hidden_block,
        output_dtype=output_dtype,
        interpret=interpret,
        mesh=mesh,
    )


def _source_push_pack_tokens_pallas_call(
    x: Float[Array, "S T D"],
    token_ids: Int[Array, "S Dst Q M"],
    valid_mask: Bool[Array, "S Dst Q M"],
    *,
    hidden_block: int,
    output_dtype: jnp.dtype,
    interpret: bool,
    mesh: Mesh | AbstractMesh | None = None,
    shard_if_needed: bool = True,
) -> Float[Array, "S Dst Q M D"]:
    abstract_mesh = jax.sharding.get_abstract_mesh()
    if shard_if_needed and not interpret and mesh is not None:
        return _source_push_pack_tokens_sharded_pallas_call(
            mesh,
            x,
            token_ids,
            valid_mask,
            hidden_block=hidden_block,
            output_dtype=output_dtype,
        )
    if shard_if_needed and not interpret and not abstract_mesh.empty:
        return _source_push_pack_tokens_sharded_pallas_call(
            abstract_mesh,
            x,
            token_ids,
            valid_mask,
            hidden_block=hidden_block,
            output_dtype=output_dtype,
        )

    source_count, dst_count, entries_per_dst, block_m = token_ids.shape
    hidden_dim = x.shape[-1]
    metadata_entry_block = 1 if interpret else int(math.ceil(128 / block_m))
    if entries_per_dst % metadata_entry_block:
        raise ValueError(
            "Pallas token pack metadata overfetch requires entries_per_dst to be divisible by "
            f"{metadata_entry_block}, got {entries_per_dst}"
        )
    output_dtype = jnp.dtype(output_dtype)
    output_shape = jax.ShapeDtypeStruct((*token_ids.shape, hidden_dim), output_dtype)
    valid_i = valid_mask.astype(jnp.int32)
    kernel = _make_source_push_pack_tokens_kernel(
        block_m=block_m,
        hidden_block=hidden_block,
        metadata_entry_block=metadata_entry_block,
        output_dtype=output_dtype,
    )
    in_specs, out_specs = _source_push_pack_tokens_block_specs(hidden_block=hidden_block)
    cost_estimate = _source_push_pack_tokens_cost_estimate(x, token_ids, valid_mask, output_shape)
    return pl.pallas_call(
        kernel,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=output_shape,
        grid=(source_count, dst_count, entries_per_dst // metadata_entry_block, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_token_pack_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=cost_estimate,
    )(x, token_ids, valid_i)


def _source_push_pack_tokens_sharded_pallas_call(
    mesh: Mesh | AbstractMesh,
    x: Float[Array, "S T D"],
    token_ids: Int[Array, "S Dst Q M"],
    valid_mask: Bool[Array, "S Dst Q M"],
    *,
    hidden_block: int,
    output_dtype: jnp.dtype,
) -> Float[Array, "S Dst Q M D"]:
    """Run the pack kernel source-locally under the expert mesh."""

    x = jax.sharding.reshard(x, P(SOURCE_PUSH_MESH_AXIS, None, None))
    token_ids = jax.sharding.reshard(token_ids, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    valid_mask = jax.sharding.reshard(valid_mask, P(SOURCE_PUSH_MESH_AXIS, None, None, None))

    def local_fn(
        x_local: Float[Array, "1 T D"],
        token_ids_local: Int[Array, "1 Dst Q M"],
        valid_mask_local: Bool[Array, "1 Dst Q M"],
    ) -> Float[Array, "1 Dst Q M D"]:
        return _source_push_pack_tokens_pallas_call(
            x_local,
            token_ids_local,
            valid_mask_local,
            hidden_block=hidden_block,
            output_dtype=output_dtype,
            interpret=False,
            shard_if_needed=False,
        )

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
        check_vma=False,
    )(x, token_ids, valid_mask)


def _source_push_pack_tokens_block_specs(*, hidden_block: int) -> tuple[tuple[pl.BlockSpec, ...], pl.BlockSpec]:
    del hidden_block
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return (gmem_spec, gmem_spec, gmem_spec), gmem_spec


def _make_source_push_pack_tokens_kernel(
    *,
    block_m: int,
    hidden_block: int,
    metadata_entry_block: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        x_ref: Float[pl.Ref, "S T D"],
        token_ids_ref: Int[pl.Ref, "S Dst Q M"],
        valid_mask_ref: Int[pl.Ref, "S Dst Q M"],
        packed_ref: Float[pl.Ref, "S Dst Q M D"],
    ) -> None:
        src = pl.program_id(0)
        dst = pl.program_id(1)
        entry_group = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        hidden_start = hidden_tile * hidden_block
        entry_group_start = entry_group * metadata_entry_block

        valid_tile = valid_mask_ref[
            pl.ds(src, 1),
            pl.ds(dst, 1),
            pl.ds(entry_group_start, metadata_entry_block),
            pl.ds(0, block_m),
        ]
        token_tile = token_ids_ref[
            pl.ds(src, 1),
            pl.ds(dst, 1),
            pl.ds(entry_group_start, metadata_entry_block),
            pl.ds(0, block_m),
        ]
        zero_tile = jnp.zeros((hidden_block,), dtype=output_dtype)
        for metadata_entry in range(metadata_entry_block):
            entry = entry_group_start + metadata_entry
            for row in range(block_m):
                valid = valid_tile[0, 0, metadata_entry, row] != 0
                token = token_tile[0, 0, metadata_entry, row]
                safe_token = jnp.maximum(token, 0)
                x_tile = x_ref[pl.ds(src, 1), pl.ds(safe_token, 1), pl.ds(hidden_start, hidden_block)][0, 0, :]
                packed_ref[
                    pl.ds(src, 1),
                    pl.ds(dst, 1),
                    pl.ds(entry, 1),
                    pl.ds(row, 1),
                    pl.ds(hidden_start, hidden_block),
                ] = jnp.where(valid, x_tile.astype(output_dtype), zero_tile)[None, None, None, None, :]

    return kernel


def _source_push_pack_tokens_reference(
    x: Array,
    token_ids: Array,
    valid_mask: Array,
    *,
    output_dtype: jnp.dtype,
) -> Array:
    source_indices = jnp.arange(token_ids.shape[0], dtype=jnp.int32)[:, None, None, None]
    safe_tokens = jnp.maximum(token_ids, 0)
    packed = x.at[source_indices, safe_tokens].get().astype(output_dtype)
    return jnp.where(valid_mask[..., None], packed, jnp.zeros((), dtype=output_dtype))


def _source_push_pack_tokens_cost_estimate(
    x: Array,
    token_ids: Array,
    valid_mask: Array,
    output_shape: jax.ShapeDtypeStruct,
) -> pl.CostEstimate:
    input_specs = (
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        jax.ShapeDtypeStruct(token_ids.shape, token_ids.dtype),
        jax.ShapeDtypeStruct(valid_mask.shape, valid_mask.dtype),
    )

    def reference(x_spec, token_ids_spec, valid_mask_spec):
        return _source_push_pack_tokens_reference(
            x_spec,
            token_ids_spec,
            valid_mask_spec,
            output_dtype=output_shape.dtype,
        )

    body_cost = pl.estimate_cost(reference, *input_specs)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=input_specs,
        kernel_outputs_specs=output_shape,
    )


def _validate_source_push_token_pack_pallas_request(
    x: Array,
    plan: SourcePushPlan,
    block_sizes: SourcePushTokenPackPallasBlockSizes,
) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must have shape [source, token, hidden], got {x.shape}")
    if plan.token_ids.shape != plan.valid_mask.shape:
        raise ValueError(f"token_ids shape {plan.token_ids.shape} must match valid_mask {plan.valid_mask.shape}")
    if plan.token_ids.ndim != 4:
        raise ValueError(f"token_ids must have shape [source, destination, entry, row], got {plan.token_ids.shape}")
    if plan.token_ids.shape[0] != x.shape[0]:
        raise ValueError(f"x source dim {x.shape[0]} must match plan source dim {plan.token_ids.shape[0]}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if x.shape[-1] % block_sizes.hidden_block:
        raise ValueError(f"x hidden dim {x.shape[-1]} must be divisible by hidden_block={block_sizes.hidden_block}")
