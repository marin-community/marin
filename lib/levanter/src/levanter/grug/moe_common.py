# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared types, routing helpers, and layout utilities for Grug MoE."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast, get_args

import jax
import jax.numpy as jnp
from haliax.jax_utils import named_call
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, get_abstract_mesh, get_mesh, reshard
from jaxtyping import Array, Float, Int

from levanter.utils.activation import ActivationFunctionEnum

_DEFAULT_EP_CAPACITY_FACTOR = 1.25
# #2710 used 1.25 as the practical EP ring default to avoid over/under-packing.

PspecAxis: TypeAlias = str | tuple[str, ...] | None
MoeActivation: TypeAlias = ActivationFunctionEnum | Callable[[jax.Array], jax.Array]
MoeImplementation: TypeAlias = Literal[
    "ring",
    "ragged_all_to_all",
    "scatter",
    "sonic_xla",
    "sonic_xla_interleaved_w13",
    "sonic_xla_interleaved_w13_custom_vjp_down",
]
_VALID_MOE_IMPLEMENTATIONS = get_args(MoeImplementation)
_EP_MOE_IMPLEMENTATIONS = ("ring", "ragged_all_to_all")
_LOCAL_MOE_IMPLEMENTATIONS = (
    "scatter",
    "sonic_xla",
    "sonic_xla_interleaved_w13",
    "sonic_xla_interleaved_w13_custom_vjp_down",
)
_INTERLEAVED_W13_MOE_IMPLEMENTATIONS = (
    "sonic_xla_interleaved_w13",
    "sonic_xla_interleaved_w13_custom_vjp_down",
)
_CUSTOM_VJP_DOWN_MOE_IMPLEMENTATIONS = ("sonic_xla_interleaved_w13_custom_vjp_down",)


@dataclass(frozen=True)
class MoEExpertMlpPspecs:
    """Logical sharding axes for local MoE expert MLP weights."""

    expert: PspecAxis = "expert"
    hidden: PspecAxis = "data"
    intermediate: PspecAxis = "model"

    @property
    def w_gate_up(self) -> P:
        return P(self.expert, self.hidden, self.intermediate)

    @property
    def w_down(self) -> P:
        return P(self.expert, self.intermediate, self.hidden)


def _current_mesh() -> Mesh | jax.sharding.AbstractMesh:
    try:
        mesh = get_mesh()
    except ValueError:
        mesh = None
    if mesh is not None and not mesh.empty:
        return mesh
    return get_abstract_mesh()


def _mesh_has_axis(mesh: Mesh | jax.sharding.AbstractMesh | None, axis_name: str) -> bool:
    if mesh is None or mesh.empty:
        return False
    return axis_name in mesh.shape


def _mesh_axis_size(mesh: Mesh | jax.sharding.AbstractMesh | None, axis_name: str) -> int:
    if mesh is None or mesh.empty:
        return 1
    return int(mesh.shape.get(axis_name, 1))


def _batch_spec(mesh: Mesh | jax.sharding.AbstractMesh | None) -> P:
    if _mesh_has_axis(mesh, "expert"):
        return P(("data", "expert"))
    return P(("data",))


def _batch_spec_from_x(x: jax.Array, mesh: Mesh | jax.sharding.AbstractMesh | None) -> P:
    sharding = getattr(x, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None and len(spec) > 0 and spec[0] is not None:
        return P(spec[0])
    return _batch_spec(mesh)


def _is_replicated_spec(spec: P) -> bool:
    return all(axis is None for axis in spec)


def _value_spec_or_default(x: jax.Array, default: P, *, replace_replicated: bool = False) -> P:
    sharding = getattr(x, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None and not (replace_replicated and _is_replicated_spec(spec)):
        return spec
    return default


def _reshard_for_init(x: jax.Array, spec: P) -> jax.Array:
    mesh = _current_mesh()
    if mesh is None or mesh.empty:
        return x
    return reshard(x, NamedSharding(mesh, spec))


def _reshard_for_shard_map(x: jax.Array, mesh: Mesh | jax.sharding.AbstractMesh | None, spec: P) -> jax.Array:
    if mesh is not None and not mesh.empty:
        return reshard(x, NamedSharding(mesh, spec))
    return x


def resolve_moe_implementation(implementation: MoeImplementation | str | None) -> MoeImplementation:
    if implementation is None:
        return "ring"
    if implementation not in _VALID_MOE_IMPLEMENTATIONS:
        valid = ", ".join(repr(choice) for choice in _VALID_MOE_IMPLEMENTATIONS)
        raise ValueError(f"implementation must be one of {valid} or None, got {implementation!r}")
    return cast(MoeImplementation, implementation)


def moe_implementation_uses_interleaved_w13(
    implementation: MoeImplementation | str | None,
) -> bool:
    return resolve_moe_implementation(implementation) in _INTERLEAVED_W13_MOE_IMPLEMENTATIONS


def interleave_moe_w13(w_gate: jax.Array, w_up: jax.Array) -> jax.Array:
    """Pack concat-style gate/up expert weights into Sonic's interleaved GLU layout."""
    if w_gate.shape != w_up.shape:
        raise ValueError(f"w_gate and w_up must have the same shape, got {w_gate.shape} vs {w_up.shape}")
    return jnp.stack((w_gate, w_up), axis=-1).reshape(*w_gate.shape[:-1], 2 * w_gate.shape[-1])


def split_moe_w13_output(
    w13_out: jax.Array, *, intermediate_dim: int, interleaved: bool
) -> tuple[jax.Array, jax.Array]:
    expected = 2 * intermediate_dim
    if w13_out.shape[-1] != expected:
        raise ValueError(f"w13 output last dimension must be {expected}, got shape={w13_out.shape}")
    if interleaved:
        return w13_out[..., 0::2], w13_out[..., 1::2]
    gate, up = jnp.split(w13_out, [intermediate_dim], axis=-1)
    return gate, up


def _init_weight(key: jax.Array, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * jax.random.truncated_normal(key, -3, 3, shape)


@named_call
def _prepare_moe_dispatch(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    *,
    num_experts: int,
) -> tuple[
    Float[Array, "TK D"],
    Float[Array, "TK"],
    Int[Array, "TK"],
    Int[Array, "E"],
]:
    """Flatten + argsort by expert into grouped layout for GMM."""
    # #2704: keep argsort-grouped dispatch as the canonical compact routing
    # strategy, matching the behavior carried forward from 89318a910.
    tokens, topk = selected_experts.shape
    expert_ids = selected_experts.reshape(tokens * topk)
    dispatch_weights = combine_weights.reshape(tokens * topk)

    sort_idx = jnp.argsort(expert_ids, axis=0)
    token_ids = jnp.arange(tokens * topk, dtype=jnp.int32) // topk
    token_ids_sort = token_ids[sort_idx]
    x_sort = x[token_ids_sort]
    w_sort = dispatch_weights[sort_idx].astype(x.dtype)
    group_sizes = jnp.bincount(expert_ids, length=num_experts).astype(jnp.int32)
    return x_sort, w_sort, token_ids_sort, group_sizes


@named_call
def _prepare_moe_dispatch_indices_with_assignment_ids(
    selected_experts: Int[Array, "T K"],
    *,
    num_experts: int,
) -> tuple[
    Int[Array, "TK"],
    Int[Array, "T K"],
    Int[Array, "E"],
    Int[Array, "TK"],
]:
    """Prepare expert-sorted token ids plus reverse positions without gathering x."""
    tokens, topk = selected_experts.shape
    assignments = tokens * topk
    expert_ids = selected_experts.reshape(assignments)

    sort_idx = jnp.argsort(expert_ids, axis=0)
    assignment_ids = jnp.arange(assignments, dtype=jnp.int32)
    sorted_assignment_ids = assignment_ids[sort_idx]
    token_ids_sort = sorted_assignment_ids // topk

    sorted_positions = jnp.arange(assignments, dtype=jnp.int32)
    dispatch_positions = jnp.zeros((assignments,), dtype=jnp.int32).at[sort_idx].set(sorted_positions)
    dispatch_positions = dispatch_positions.reshape(tokens, topk)

    group_sizes = jnp.bincount(expert_ids, length=num_experts).astype(jnp.int32)
    return token_ids_sort, dispatch_positions, group_sizes, sorted_assignment_ids


def _gather_sum_reference(
    dispatch_output: Float[Array, "TK D"],
    dispatch_positions: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
) -> Float[Array, "T D"]:
    acc = jnp.zeros((dispatch_positions.shape[0], dispatch_output.shape[1]), dtype=dispatch_output.dtype)
    weights = combine_weights.astype(dispatch_output.dtype)
    for topk_index in range(dispatch_positions.shape[1]):
        gathered = jnp.take(dispatch_output, dispatch_positions[:, topk_index], axis=0)
        acc = (acc + (gathered * weights[:, topk_index, None]).astype(dispatch_output.dtype)).astype(
            dispatch_output.dtype
        )
    return acc


def _zero_dropped_assignments() -> Int[Array, ""]:
    return jnp.array(0, dtype=jnp.int32)
