# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Public Grug MoE interface and implementation dispatcher.

Implementation overview:
- Routing keeps the argsort-grouped dispatch path that emerged as the stable
  default from https://github.com/marin-community/marin/issues/2704 and commit
  89318a910 (and its parent).
- Expert parallelism keeps the ring-style strategy from
  https://github.com/marin-community/marin/issues/2710: token-sharded
  `all_gather` for dispatch, then `psum_scatter` for collection.
- Backend bodies live in the private `levanter.grug._moe` package; this module
  keeps the stable public API used by Grug model code and benchmarks.
"""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from haliax.jax_utils import named_call
from jax import shard_map
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from levanter.grug._moe.common import (
    _DEFAULT_EP_CAPACITY_FACTOR,
    _EP_MOE_IMPLEMENTATIONS,
    _init_weight,
    MOE_REMAT_SAVE_NAMES as MOE_REMAT_SAVE_NAMES,
    MoEExpertMlpPspecs,
    MoeActivation,
    MoeImplementation,
    PspecAxis,
    resolve_moe_implementation,
    split_moe_w13_output,
)
from levanter.grug._moe.ep_common import (
    _clip_receiver_group_sizes as _clip_receiver_group_sizes,
    _compact_by_keep_mask as _compact_by_keep_mask,
    _expand_from_keep_mask as _expand_from_keep_mask,
    _shard_a2a_params as _shard_a2a_params,
)
from levanter.grug._moe.ep_deepep import _moe_mlp_ep_deepep_local
from levanter.grug._moe.ep_ragged_all_to_all import _moe_mlp_ep_ragged_a2a_local
from levanter.grug._moe.ep_ring import _moe_mlp_ep_ring_local
from levanter.grug._moe.local import _moe_mlp_local
from levanter.grug.sharding import (
    _batch_spec_from_x,
    _current_mesh,
    _mesh_axis_size,
    _mesh_has_axis,
    _reshard_for_init,
    _reshard_for_shard_map,
    _value_spec_or_default,
)
from levanter.utils.activation import ActivationFunctionEnum


class MoEExpertMlp(eqx.Module):
    """Expert MLP weights for routed MoE calls."""

    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array
    implementation: MoeImplementation = eqx.field(static=True)
    activation: MoeActivation = eqx.field(static=True)
    capacity_factor: float = eqx.field(static=True)

    @staticmethod
    def init(
        *,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        initializer_std: float,
        key: jax.Array,
        implementation: MoeImplementation | str | None = None,
        activation: MoeActivation = ActivationFunctionEnum.silu,
        capacity_factor: float = _DEFAULT_EP_CAPACITY_FACTOR,
        pspecs: MoEExpertMlpPspecs = MoEExpertMlpPspecs(),
    ) -> "MoEExpertMlp":
        resolved_implementation = resolve_moe_implementation(implementation)
        k_gate, k_up, k_down = jax.random.split(key, 3)
        w_gate = _init_weight(k_gate, (num_experts, hidden_dim, intermediate_dim), initializer_std)
        w_up = _init_weight(k_up, (num_experts, hidden_dim, intermediate_dim), initializer_std)
        w_down = _reshard_for_init(
            _init_weight(k_down, (num_experts, intermediate_dim, hidden_dim), initializer_std),
            pspecs.w_down,
        )
        return MoEExpertMlp(
            w_gate=_reshard_for_init(w_gate, pspecs.w_gate_up),
            w_up=_reshard_for_init(w_up, pspecs.w_gate_up),
            w_down=w_down,
            implementation=resolved_implementation,
            activation=activation,
            capacity_factor=capacity_factor,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "T D"],
        selected_experts: Int[Array, "T K"],
        combine_weights: Float[Array, "T K"],
        *,
        mesh: jax.sharding.AbstractMesh | None = None,
        report_capacity_overflow: bool = False,
    ) -> Float[Array, "T D"] | tuple[Float[Array, "T D"], Int[Array, ""]]:
        w_gate_up = jnp.concatenate([self.w_gate, self.w_up], axis=-1)
        return moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_gate_up,
            self.w_down,
            activation=self.activation,
            implementation=self.implementation,
            mesh=mesh,
            capacity_factor=self.capacity_factor,
            report_capacity_overflow=report_capacity_overflow,
        )


@named_call
def moe_mlp(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    w_up_gate: Float[Array, "E D I2"],
    w_down: Float[Array, "E I D"],
    *,
    activation: MoeActivation = ActivationFunctionEnum.silu,
    implementation: MoeImplementation | str | None = None,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = _DEFAULT_EP_CAPACITY_FACTOR,
    report_capacity_overflow: bool = False,
) -> Float[Array, "T D"] | tuple[Float[Array, "T D"], Int[Array, ""]]:
    """Functional routed MoE MLP core used by Grug modules and benchmarks.

    This helper handles dispatch/permute/unpermute (+EP collectives) from
    precomputed token-to-expert assignments. Routing logits/top-k selection
    stays in the caller (e.g. model MLP block).

    Set `report_capacity_overflow=True` to also return a scalar count of
    dropped expert assignments from EP capacity clipping.
    """
    resolved_implementation = resolve_moe_implementation(implementation)

    if mesh is None:
        mesh = _current_mesh()

    if isinstance(activation, ActivationFunctionEnum):
        activation_fn: Callable[[jax.Array], jax.Array] = activation.to_jax_fn()
    else:
        activation_fn = activation

    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [T, D], got shape={x.shape}")
    if selected_experts.ndim != 2:
        raise ValueError(f"selected_experts must be rank-2 [T, K], got shape={selected_experts.shape}")
    if selected_experts.shape != combine_weights.shape:
        raise ValueError(
            "selected_experts and combine_weights must have identical [T, K] shapes; "
            f"got {selected_experts.shape} vs {combine_weights.shape}"
        )
    if selected_experts.shape[0] != x.shape[0]:
        raise ValueError(
            f"selected_experts/combine_weights token dim ({selected_experts.shape[0]}) must match x token "
            f"dim ({x.shape[0]})"
        )

    num_experts = int(w_up_gate.shape[0])
    if w_down.shape[0] != num_experts:
        raise ValueError(
            f"w_down expert dimension ({w_down.shape[0]}) must match w_up_gate expert dimension ({num_experts})"
        )

    has_expert_axis = _mesh_has_axis(mesh, "expert")
    expert_axis_size = _mesh_axis_size(mesh, "expert")

    if mesh is None or mesh.empty:
        out, dropped = _moe_mlp_local(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation_fn=activation_fn,
            num_experts=num_experts,
            implementation=resolved_implementation,
        )
        if report_capacity_overflow:
            return out, dropped
        return out

    batch_spec = _batch_spec_from_x(x, mesh)

    if has_expert_axis and expert_axis_size > 1:
        if resolved_implementation not in _EP_MOE_IMPLEMENTATIONS:
            raise ValueError(
                "Local MoE implementations do not yet support expert-parallel collectives; adding EP support "
                "requires a dispatch/combine schedule inside each expert shard plus cross-shard routing. "
                f"got implementation={resolved_implementation!r} with expert axis size={expert_axis_size}"
            )
        if num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={num_experts} must be divisible by expert axis size={expert_axis_size}")

        if resolved_implementation == "ring":
            shard_local_fn = _moe_mlp_ep_ring_local
        elif resolved_implementation == "ragged_all_to_all":
            shard_local_fn = _moe_mlp_ep_ragged_a2a_local
        elif resolved_implementation == "deepep":
            shard_local_fn = _moe_mlp_ep_deepep_local
        else:
            raise AssertionError(f"Unhandled MoE implementation {resolved_implementation!r}")

        w_up_gate_spec = P("expert", None, None)
        w_down_spec = P("expert", None, None)

        x = _reshard_for_shard_map(x, mesh, batch_spec)
        selected_experts = _reshard_for_shard_map(selected_experts, mesh, batch_spec)
        combine_weights = _reshard_for_shard_map(combine_weights, mesh, batch_spec)
        w_up_gate = _reshard_for_shard_map(w_up_gate, mesh, w_up_gate_spec)
        w_down = _reshard_for_shard_map(w_down, mesh, w_down_spec)

        shard_fn = shard_map(
            partial(
                shard_local_fn,
                activation_fn=activation_fn,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
            ),
            mesh=mesh,
            in_specs=(
                batch_spec,
                batch_spec,
                batch_spec,
                w_up_gate_spec,
                w_down_spec,
            ),
            out_specs=(batch_spec, P()),
            check_vma=False,
        )
        out, dropped = shard_fn(x, selected_experts, combine_weights, w_up_gate, w_down)
        if report_capacity_overflow:
            return out, dropped
        return out

    # Fallback path for no expert axis (or expert axis size 1) keeps routing
    # semantics without EP collectives. JAX 0.9 requires shard_map in_specs to
    # match the actual input sharding, so reshard ordinary inputs to the mesh
    # specs that preserve data-axis parallelism.
    x_spec = _value_spec_or_default(x, batch_spec, replace_replicated=True)
    selected_experts_spec = _value_spec_or_default(selected_experts, batch_spec, replace_replicated=True)
    combine_weights_spec = _value_spec_or_default(combine_weights, batch_spec, replace_replicated=True)
    w_up_gate_spec = _value_spec_or_default(w_up_gate, P(*(None for _ in range(w_up_gate.ndim))))
    w_down_spec = _value_spec_or_default(w_down, P(*(None for _ in range(w_down.ndim))))

    x = _reshard_for_shard_map(x, mesh, x_spec)
    selected_experts = _reshard_for_shard_map(selected_experts, mesh, selected_experts_spec)
    combine_weights = _reshard_for_shard_map(combine_weights, mesh, combine_weights_spec)
    w_up_gate = _reshard_for_shard_map(w_up_gate, mesh, w_up_gate_spec)
    w_down = _reshard_for_shard_map(w_down, mesh, w_down_spec)

    shard_fn = shard_map(
        partial(
            _moe_mlp_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            implementation=resolved_implementation,
        ),
        mesh=mesh,
        in_specs=(
            x_spec,
            selected_experts_spec,
            combine_weights_spec,
            w_up_gate_spec,
            w_down_spec,
        ),
        out_specs=(x_spec, P()),
        check_vma=False,
    )
    out, dropped = shard_fn(x, selected_experts, combine_weights, w_up_gate, w_down)
    if report_capacity_overflow:
        return out, dropped
    return out


__all__ = [
    "MoeActivation",
    "MoEExpertMlp",
    "MoEExpertMlpPspecs",
    "MoeImplementation",
    "PspecAxis",
    "moe_mlp",
    "resolve_moe_implementation",
    "split_moe_w13_output",
]
