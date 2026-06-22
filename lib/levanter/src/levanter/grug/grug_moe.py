# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Public Grug MoE interface and implementation dispatcher.

Implementation overview:
- Routing keeps the argsort-grouped dispatch path that emerged as the stable
  default from https://github.com/marin-community/marin/issues/2704 and commit
  89318a910 (and its parent).
- Expert parallelism keeps the ring-style strategy from
  https://github.com/marin-community/marin/issues/2710: token-sharded
  `all_gather` for dispatch, then `psum_scatter` for collection. The assigned
  token and DeepEP paths provide alternate EP dispatch/combine schedules.
- Backend bodies live in the private `levanter.grug._moe` package; this module
  keeps the stable public API used by Grug model code and benchmarks.
"""

from collections.abc import Callable, Sequence
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
    DEEPEP_REMAT_SAVE_NAMES as DEEPEP_REMAT_SAVE_NAMES,
    MOE_REMAT_HIDDEN_OFFLOAD_NAMES as MOE_REMAT_HIDDEN_OFFLOAD_NAMES,
    MOE_REMAT_HIDDEN_SAVE_NAMES as MOE_REMAT_HIDDEN_SAVE_NAMES,
    MOE_REMAT_OFFLOAD_NAMES as MOE_REMAT_OFFLOAD_NAMES,
    MOE_REMAT_SAVE_NAMES as MOE_REMAT_SAVE_NAMES,
    MoEExpertMlpPspecs,
    MoeActivation,
    MoeImplementation,
    MoERematMode,
    PspecAxis,
    resolve_moe_implementation,
    resolve_moe_remat_mode,
    split_moe_w13_output,
)
from levanter.grug._moe.ep_common import (
    _clip_receiver_group_sizes as _clip_receiver_group_sizes,
    _compact_by_keep_mask as _compact_by_keep_mask,
    _expand_from_keep_mask as _expand_from_keep_mask,
    _shard_a2a_params as _shard_a2a_params,
)
from levanter.grug._moe.ep_assigned_token import _moe_mlp_ep_assigned_token_local
from levanter.grug._moe.ep_deepep import (
    _moe_mlp_ep_deepep_composed_local,
    _moe_mlp_ep_deepep_internode_local,
    _moe_mlp_ep_deepep_local,
)
from levanter.grug._moe.ep_grouped_assigned_token import _moe_mlp_ep_grouped_assigned_token_local
from levanter.grug._moe.ep_padded_all_to_all import _moe_mlp_ep_padded_a2a_local
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
    """Expert MLP weights plus backend-specific W13 layout for routed MoE calls."""

    w_gate_up: jax.Array
    w_down: jax.Array
    implementation: MoeImplementation = eqx.field(static=True)
    activation: MoeActivation = eqx.field(static=True)
    capacity_factor: float = eqx.field(static=True)
    remat_mode: MoERematMode = eqx.field(static=True)

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
        remat_mode: MoERematMode | str = "none",
        pspecs: MoEExpertMlpPspecs = MoEExpertMlpPspecs(),
    ) -> "MoEExpertMlp":
        resolved_implementation = resolve_moe_implementation(implementation)
        resolved_remat_mode = resolve_moe_remat_mode(remat_mode)
        k_gate, k_up, k_down = jax.random.split(key, 3)
        w_gate = _init_weight(k_gate, (num_experts, hidden_dim, intermediate_dim), initializer_std)
        w_up = _init_weight(k_up, (num_experts, hidden_dim, intermediate_dim), initializer_std)
        w_gate_up = jnp.concatenate([w_gate, w_up], axis=-1)

        return MoEExpertMlp(
            w_gate_up=_reshard_for_init(w_gate_up, pspecs.w_gate_up),
            w_down=_reshard_for_init(
                _init_weight(k_down, (num_experts, intermediate_dim, hidden_dim), initializer_std),
                pspecs.w_down,
            ),
            implementation=resolved_implementation,
            activation=activation,
            capacity_factor=capacity_factor,
            remat_mode=resolved_remat_mode,
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
        return moe_mlp(
            x,
            selected_experts,
            combine_weights,
            self.w_gate_up,
            self.w_down,
            activation=self.activation,
            implementation=self.implementation,
            mesh=mesh,
            capacity_factor=self.capacity_factor,
            remat_mode=self.remat_mode,
            report_capacity_overflow=report_capacity_overflow,
        )


class GroupedMoEExpertMlp(eqx.Module):
    """Grouped expert MLP weight bank for several adjacent MoE layers.

    `layer()` exists for correctness checks and incremental migration. The hot
    path should call the grouped module directly so the group axis can remain
    visible to the compiler and optimizer.
    """

    w_gate_up: jax.Array
    w_down: jax.Array
    implementation: MoeImplementation = eqx.field(static=True)
    activation: MoeActivation = eqx.field(static=True)
    capacity_factor: float = eqx.field(static=True)
    remat_mode: MoERematMode = eqx.field(static=True)
    valid_group_size: int = eqx.field(static=True)

    @staticmethod
    def from_layers(layers: Sequence[MoEExpertMlp]) -> "GroupedMoEExpertMlp":
        """Stack ordinary per-layer expert modules into one grouped expert bank."""
        if not layers:
            raise ValueError("layers must contain at least one MoEExpertMlp")

        first = layers[0]
        for i, layer in enumerate(layers[1:], start=1):
            if layer.implementation != first.implementation:
                raise ValueError(
                    "GroupedMoEExpertMlp layers must use the same implementation; "
                    f"layer 0 has {first.implementation!r}, layer {i} has {layer.implementation!r}"
                )
            if layer.activation != first.activation:
                raise ValueError(
                    "GroupedMoEExpertMlp layers must use the same activation; "
                    f"layer 0 has {first.activation!r}, layer {i} has {layer.activation!r}"
                )
            if layer.capacity_factor != first.capacity_factor:
                raise ValueError(
                    "GroupedMoEExpertMlp layers must use the same capacity_factor; "
                    f"layer 0 has {first.capacity_factor!r}, layer {i} has {layer.capacity_factor!r}"
                )
            if layer.remat_mode != first.remat_mode:
                raise ValueError(
                    "GroupedMoEExpertMlp layers must use the same remat_mode; "
                    f"layer 0 has {first.remat_mode!r}, layer {i} has {layer.remat_mode!r}"
                )

        return GroupedMoEExpertMlp(
            w_gate_up=jnp.stack([layer.w_gate_up for layer in layers], axis=0),
            w_down=jnp.stack([layer.w_down for layer in layers], axis=0),
            implementation=first.implementation,
            activation=first.activation,
            capacity_factor=first.capacity_factor,
            remat_mode=first.remat_mode,
            valid_group_size=len(layers),
        )

    def layer(self, local_layer_index: int) -> MoEExpertMlp:
        if local_layer_index < 0 or local_layer_index >= self.valid_group_size:
            raise IndexError(f"local_layer_index={local_layer_index} must be in [0, {self.valid_group_size})")
        return MoEExpertMlp(
            w_gate_up=self.w_gate_up[local_layer_index],
            w_down=self.w_down[local_layer_index],
            implementation=self.implementation,
            activation=self.activation,
            capacity_factor=self.capacity_factor,
            remat_mode=self.remat_mode,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "G T D"],
        selected_experts: Int[Array, "G T K"],
        combine_weights: Float[Array, "G T K"],
        *,
        mesh: jax.sharding.AbstractMesh | None = None,
        report_capacity_overflow: bool = False,
    ) -> Float[Array, "G T D"] | tuple[Float[Array, "G T D"], Int[Array, "G"]]:
        return grouped_moe_mlp(
            x,
            selected_experts,
            combine_weights,
            self.w_gate_up,
            self.w_down,
            valid_group_size=self.valid_group_size,
            activation=self.activation,
            implementation=self.implementation,
            mesh=mesh,
            capacity_factor=self.capacity_factor,
            remat_mode=self.remat_mode,
            report_capacity_overflow=report_capacity_overflow,
        )


@named_call
def grouped_moe_mlp(
    x: Float[Array, "G T D"],
    selected_experts: Int[Array, "G T K"],
    combine_weights: Float[Array, "G T K"],
    w_up_gate: Float[Array, "G E D I2"],
    w_down: Float[Array, "G E I D"],
    *,
    valid_group_size: int | None = None,
    activation: MoeActivation = ActivationFunctionEnum.silu,
    implementation: MoeImplementation | str | None = None,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = _DEFAULT_EP_CAPACITY_FACTOR,
    remat_mode: MoERematMode | str = "none",
    report_capacity_overflow: bool = False,
) -> Float[Array, "G T D"] | tuple[Float[Array, "G T D"], Int[Array, "G"]]:
    """Run routed MoE over a grouped bank of per-layer expert weights."""
    if x.ndim != 3:
        raise ValueError(f"x must be rank-3 [G, T, D], got shape={x.shape}")
    if selected_experts.ndim != 3:
        raise ValueError(f"selected_experts must be rank-3 [G, T, K], got shape={selected_experts.shape}")
    if combine_weights.shape != selected_experts.shape:
        raise ValueError(
            "selected_experts and combine_weights must have identical [G, T, K] shapes; "
            f"got {selected_experts.shape} vs {combine_weights.shape}"
        )
    if w_up_gate.ndim != 4:
        raise ValueError(f"w_up_gate must be rank-4 [G, E, D, I2], got shape={w_up_gate.shape}")
    if w_down.ndim != 4:
        raise ValueError(f"w_down must be rank-4 [G, E, I, D], got shape={w_down.shape}")

    group_size = x.shape[0]
    if selected_experts.shape[0] != group_size or w_up_gate.shape[0] != group_size or w_down.shape[0] != group_size:
        raise ValueError(
            "x, selected_experts, w_up_gate, and w_down must share the same group dimension; "
            f"got {x.shape[0]}, {selected_experts.shape[0]}, {w_up_gate.shape[0]}, {w_down.shape[0]}"
        )
    if valid_group_size is None:
        valid_group_size = group_size
    if valid_group_size < 1 or valid_group_size > group_size:
        raise ValueError(f"valid_group_size must be in [1, {group_size}], got {valid_group_size}")

    resolved_implementation = resolve_moe_implementation(implementation)
    resolved_remat_mode = resolve_moe_remat_mode(remat_mode)
    if mesh is None:
        mesh = _current_mesh()
    if resolved_implementation == "grouped_assigned_token" and (
        mesh is None or mesh.empty or not _mesh_has_axis(mesh, "expert") or _mesh_axis_size(mesh, "expert") <= 1
    ):
        raise ValueError("implementation='grouped_assigned_token' requires grouped_moe_mlp with an expert mesh")

    if isinstance(activation, ActivationFunctionEnum):
        activation_fn: Callable[[jax.Array], jax.Array] = activation.to_jax_fn()
    else:
        activation_fn = activation

    has_expert_axis = _mesh_has_axis(mesh, "expert")
    expert_axis_size = _mesh_axis_size(mesh, "expert")
    if mesh is not None and not mesh.empty and has_expert_axis and expert_axis_size > 1:
        if resolved_implementation not in _EP_MOE_IMPLEMENTATIONS:
            raise ValueError(
                "Grouped MoE with an expert-parallel mesh requires an expert-parallel implementation; "
                f"got implementation={resolved_implementation!r} with expert axis size={expert_axis_size}"
            )
        num_experts = int(w_up_gate.shape[1])
        if num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={num_experts} must be divisible by expert axis size={expert_axis_size}")

        grouped_shard_local_fn = None
        if resolved_implementation == "grouped_assigned_token":
            grouped_shard_local_fn = _moe_mlp_ep_grouped_assigned_token_local
        elif resolved_implementation == "ring":
            shard_local_fn = _moe_mlp_ep_ring_local
        elif resolved_implementation == "assigned_token":
            shard_local_fn = _moe_mlp_ep_assigned_token_local
        elif resolved_implementation == "ragged_all_to_all":
            shard_local_fn = _moe_mlp_ep_ragged_a2a_local
        elif resolved_implementation == "padded_all_to_all":
            shard_local_fn = _moe_mlp_ep_padded_a2a_local
        elif resolved_implementation == "deepep":
            shard_local_fn = _moe_mlp_ep_deepep_local
        elif resolved_implementation == "deepep_composed":
            shard_local_fn = _moe_mlp_ep_deepep_composed_local
        elif resolved_implementation == "deepep_internode":
            shard_local_fn = _moe_mlp_ep_deepep_internode_local
        else:
            raise AssertionError(f"Unhandled MoE implementation {resolved_implementation!r}")

        x_spec = _value_spec_or_default(x, P(None, None, None))
        selected_experts_spec = _value_spec_or_default(selected_experts, x_spec)
        combine_weights_spec = _value_spec_or_default(combine_weights, x_spec)
        w_up_gate_spec = _value_spec_or_default(w_up_gate, P(None, "expert", None, None))
        w_down_spec = _value_spec_or_default(w_down, P(None, "expert", None, None))

        x = _reshard_for_shard_map(x, mesh, x_spec)
        selected_experts = _reshard_for_shard_map(selected_experts, mesh, selected_experts_spec)
        combine_weights = _reshard_for_shard_map(combine_weights, mesh, combine_weights_spec)
        w_up_gate = _reshard_for_shard_map(w_up_gate, mesh, w_up_gate_spec)
        w_down = _reshard_for_shard_map(w_down, mesh, w_down_spec)

        if grouped_shard_local_fn is not None:

            def run_grouped_shard(x_local, selected_local, combine_local, w_up_gate_local, w_down_local):
                out, dropped = grouped_shard_local_fn(
                    x_local,
                    selected_local,
                    combine_local,
                    w_up_gate_local,
                    w_down_local,
                    activation_fn=activation_fn,
                    num_experts=num_experts,
                    capacity_factor=capacity_factor,
                    remat_mode=resolved_remat_mode,
                    valid_group_size=valid_group_size,
                )
                if report_capacity_overflow:
                    return out, dropped
                return out

        else:

            def run_grouped_shard(x_local, selected_local, combine_local, w_up_gate_local, w_down_local):
                def run_one(x_one, selected_one, combine_one, w_up_gate_one, w_down_one):
                    return shard_local_fn(
                        x_one,
                        selected_one,
                        combine_one,
                        w_up_gate_one,
                        w_down_one,
                        activation_fn=activation_fn,
                        num_experts=num_experts,
                        capacity_factor=capacity_factor,
                        remat_mode=resolved_remat_mode,
                    )

                out, dropped = jax.vmap(run_one)(x_local, selected_local, combine_local, w_up_gate_local, w_down_local)
                if valid_group_size != group_size:
                    valid = jnp.arange(group_size) < valid_group_size
                    out = jnp.where(valid[:, None, None], out, jnp.zeros_like(out))
                    dropped = jnp.where(valid, dropped, jnp.zeros_like(dropped))
                if report_capacity_overflow:
                    return out, dropped
                return out

        grouped_shard_fn = shard_map(
            run_grouped_shard,
            mesh=mesh,
            in_specs=(x_spec, selected_experts_spec, combine_weights_spec, w_up_gate_spec, w_down_spec),
            out_specs=(x_spec, P(x_spec[0])) if report_capacity_overflow else x_spec,
            check_vma=False,
        )
        return grouped_shard_fn(x, selected_experts, combine_weights, w_up_gate, w_down)

    def run_one(x_one, selected_one, combine_one, w_up_gate_one, w_down_one):
        return moe_mlp(
            x_one,
            selected_one,
            combine_one,
            w_up_gate_one,
            w_down_one,
            activation=activation,
            implementation=implementation,
            mesh=mesh,
            capacity_factor=capacity_factor,
            remat_mode=remat_mode,
            report_capacity_overflow=report_capacity_overflow,
        )

    result = jax.vmap(run_one)(x, selected_experts, combine_weights, w_up_gate, w_down)
    if valid_group_size == group_size:
        return result

    if report_capacity_overflow:
        out, dropped = result
        out = jnp.where(jnp.arange(group_size)[:, None, None] < valid_group_size, out, jnp.zeros_like(out))
        dropped = jnp.where(jnp.arange(group_size) < valid_group_size, dropped, jnp.zeros_like(dropped))
        return out, dropped
    return jnp.where(jnp.arange(group_size)[:, None, None] < valid_group_size, result, jnp.zeros_like(result))


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
    remat_mode: MoERematMode | str = "none",
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
    resolved_remat_mode = resolve_moe_remat_mode(remat_mode)
    if resolved_implementation == "grouped_assigned_token":
        resolved_implementation = "assigned_token"

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
        elif resolved_implementation == "assigned_token":
            shard_local_fn = _moe_mlp_ep_assigned_token_local
        elif resolved_implementation == "ragged_all_to_all":
            shard_local_fn = _moe_mlp_ep_ragged_a2a_local
        elif resolved_implementation == "padded_all_to_all":
            shard_local_fn = _moe_mlp_ep_padded_a2a_local
        elif resolved_implementation == "deepep":
            shard_local_fn = _moe_mlp_ep_deepep_local
        elif resolved_implementation == "deepep_composed":
            shard_local_fn = _moe_mlp_ep_deepep_composed_local
        elif resolved_implementation == "deepep_internode":
            shard_local_fn = _moe_mlp_ep_deepep_internode_local
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
                remat_mode=resolved_remat_mode,
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
    "GroupedMoEExpertMlp",
    "MoeActivation",
    "MoEExpertMlp",
    "MoEExpertMlpPspecs",
    "MoeImplementation",
    "PspecAxis",
    "grouped_moe_mlp",
    "moe_mlp",
    "resolve_moe_implementation",
    "split_moe_w13_output",
]
