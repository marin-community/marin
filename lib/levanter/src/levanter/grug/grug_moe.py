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

import os
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
    default_moe_expert_pspecs,
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
    _drop_absent_mesh_axes,
    _mesh_axis_size,
    _mesh_has_axis,
    _reshard_for_init,
    _reshard_for_shard_map,
    _value_spec_or_default,
)
from levanter.utils.activation import ActivationFunctionEnum


class MoEExpertMlp(eqx.Module):
    """Expert MLP weights for routed MoE calls."""

    # Either (w_gate, w_up) are populated and w_gate_up is None (default), or — under
    # SCALE_FUSED_GATE_UP_PARAM — w_gate_up holds the pre-concatenated [E, D, I2] tensor and
    # w_gate/w_up are None, so the FSDP gather feeds straight from one stored parameter.
    w_gate: jax.Array | None
    w_up: jax.Array | None
    w_down: jax.Array
    w_gate_up: jax.Array | None
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
        pspecs: MoEExpertMlpPspecs | None = None,
    ) -> "MoEExpertMlp":
        if pspecs is None:
            pspecs = default_moe_expert_pspecs()
        resolved_implementation = resolve_moe_implementation(implementation)
        k_gate, k_up, k_down = jax.random.split(key, 3)
        w_gate = _init_weight(k_gate, (num_experts, hidden_dim, intermediate_dim), initializer_std)
        w_up = _init_weight(k_up, (num_experts, hidden_dim, intermediate_dim), initializer_std)
        w_down = _reshard_for_init(
            _init_weight(k_down, (num_experts, intermediate_dim, hidden_dim), initializer_std),
            pspecs.w_down,
        )
        if os.environ.get("SCALE_FUSED_GATE_UP_PARAM") == "1":
            w_gate_up = _reshard_for_init(jnp.concatenate([w_gate, w_up], axis=-1), pspecs.w_gate_up)
            return MoEExpertMlp(
                w_gate=None,
                w_up=None,
                w_down=w_down,
                w_gate_up=w_gate_up,
                implementation=resolved_implementation,
                activation=activation,
                capacity_factor=capacity_factor,
            )
        return MoEExpertMlp(
            w_gate=_reshard_for_init(w_gate, pspecs.w_gate_up),
            w_up=_reshard_for_init(w_up, pspecs.w_gate_up),
            w_down=w_down,
            w_gate_up=None,
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
        dispatch_slots: Int[Array, "T K"] | None = None,
        w13_pre0: Float[Array, "per D I2"] | None = None,
        w2_pre0: Float[Array, "per I D"] | None = None,
    ) -> Float[Array, "T D"] | tuple[Float[Array, "T D"], Int[Array, ""]]:
        if self.w_gate_up is not None:
            w_gate_up = self.w_gate_up
        elif os.environ.get("SCALE_SPLIT_GATE_UP_GATHER") == "1":
            # Gather w_gate and w_up as two independent all-gathers (then concat the gathered
            # results locally), instead of concatenating first and gathering one monolithic
            # w_gate_up. Two smaller collectives are easier for XLA to schedule/overlap; the
            # concat below is a cheap local op on already-replicated tensors, so moe_mlp's own
            # reshard becomes a no-op.
            m = mesh if mesh is not None else _current_mesh()
            replicated = P(*(None for _ in range(self.w_gate.ndim)))
            w_gate = _reshard_for_shard_map(self.w_gate, m, replicated)
            w_up = _reshard_for_shard_map(self.w_up, m, replicated)
            w_gate_up = jnp.concatenate([w_gate, w_up], axis=-1)
        else:
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
            dispatch_slots=dispatch_slots,
            w13_pre0=w13_pre0,
            w2_pre0=w2_pre0,
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
    dispatch_slots: Int[Array, "T K"] | None = None,
    w13_pre0: Float[Array, "per D I2"] | None = None,
    w2_pre0: Float[Array, "per I D"] | None = None,
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
    if dispatch_slots is not None and dispatch_slots.shape != selected_experts.shape:
        raise ValueError(
            f"dispatch_slots must match selected_experts shape {selected_experts.shape}, got {dispatch_slots.shape}"
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

        if dispatch_slots is not None:
            if resolved_implementation != "ragged_all_to_all":
                raise ValueError("precomputed dispatch slots are only supported by ragged_all_to_all")
            dispatch_slots = _reshard_for_shard_map(dispatch_slots, mesh, batch_spec)
            shard_args = (x, selected_experts, combine_weights, w_up_gate, w_down, dispatch_slots)
            shard_in_specs = (
                batch_spec,
                batch_spec,
                batch_spec,
                w_up_gate_spec,
                w_down_spec,
                batch_spec,
            )
        else:
            shard_args = (x, selected_experts, combine_weights, w_up_gate, w_down)
            shard_in_specs = (
                batch_spec,
                batch_spec,
                batch_spec,
                w_up_gate_spec,
                w_down_spec,
            )

        shard_fn = shard_map(
            partial(
                shard_local_fn,
                activation_fn=activation_fn,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
            ),
            mesh=mesh,
            in_specs=shard_in_specs,
            out_specs=(batch_spec, P()),
            check_vma=False,
        )
        out, dropped = shard_fn(*shard_args)
        if report_capacity_overflow:
            return out, dropped
        return out

    # SCALE_MOE_EXPERT_ESHARD shards experts over the data axis (whole experts per chip). The chunked
    # path gathers the hidden dim per expert-chunk, which does not apply when the expert dim itself is
    # the sharded axis -- a single all-gather over data reconstructs all experts. Fall through to the
    # non-chunked path (reshard experts to replicated), which is that single gather.
    eshard = os.environ.get("SCALE_MOE_EXPERT_ESHARD") == "1"
    chunk_sizes = None if eshard else _resolve_expert_chunk_sizes(num_experts)
    if chunk_sizes is not None and resolved_implementation == "sonic_cute":
        out, dropped = _moe_mlp_chunked_no_ep(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation_fn=activation_fn,
            num_experts=num_experts,
            mesh=mesh,
            chunk_sizes=chunk_sizes,
            w13_pre0=w13_pre0,
            w2_pre0=w2_pre0,
        )
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
    if eshard:
        # Experts are sharded E-over-data; reshard to fully replicated -- one all-gather over the
        # data axis reconstructs all experts on every chip for the local grouped GEMM. (Without this,
        # _value_spec_or_default would keep the E-sharded spec and feed partial experts into the GEMM.)
        w_up_gate_spec = P(*(None for _ in range(w_up_gate.ndim)))
        w_down_spec = P(*(None for _ in range(w_down.ndim)))
    else:
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


def _resolve_expert_chunk_sizes(num_experts: int) -> tuple[int, ...] | None:
    """Static expert-chunk sizes for the no-EP chunked path, or ``None`` when unchunked.

    ``SCALE_MOE_CHUNK_SIZES`` overrides with an explicit comma-separated list (e.g. ``"16,16,96"``
    for a ramp of two small gathers then one large); it must sum to ``num_experts``. Otherwise
    ``SCALE_MOE_EXPERT_CHUNKS`` gives that many equal chunks (``None`` for 1, i.e. unchunked).
    """
    sizes_env = os.environ.get("SCALE_MOE_CHUNK_SIZES")
    if sizes_env:
        sizes = tuple(int(s) for s in sizes_env.split(","))
        if sum(sizes) != num_experts:
            raise ValueError(f"SCALE_MOE_CHUNK_SIZES={sizes} must sum to num_experts={num_experts}")
        return sizes if len(sizes) > 1 else None
    chunks = int(os.environ.get("SCALE_MOE_EXPERT_CHUNKS", "1"))
    if chunks <= 1:
        return None
    if num_experts % chunks != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by SCALE_MOE_EXPERT_CHUNKS={chunks}")
    return (num_experts // chunks,) * chunks


def _moe_mlp_chunked_no_ep(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    w_up_gate: Float[Array, "E D I2"],
    w_down: Float[Array, "E I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    chunk_sizes: tuple[int, ...],
    w13_pre0: Float[Array, "per D I2"] | None = None,
    w2_pre0: Float[Array, "per I D"] | None = None,
) -> tuple[Float[Array, "T D"], Int[Array, ""]]:
    """No-EP chunked sonic_cute path (gate: ``SCALE_MOE_EXPERT_CHUNKS`` > 1).

    Instead of resharding the full expert weights to replicated before the GEMM (the exposed FSDP
    all-gather), pass the H-sharded weights into the shard_map and let the local fn all-gather one
    ``1/chunks`` slice at a time over the ``data`` axis, so each gather fits the scheduler's
    overlap-memory budget and chunk k+1's gather can hide under chunk k's GEMM.

    ``SCALE_MOE_CHUNK_DIM`` selects what the ``1/chunks`` slice partitions: ``expert`` (default)
    gathers a group of experts per chunk and caps each group at a static capacity (drops overflow);
    ``intermediate`` gathers a slice of every expert's intermediate dim per chunk and accumulates the
    partial down-projections (dropless).
    """
    from levanter.grug._moe.sonic_cute import (  # noqa: PLC0415
        _moe_mlp_local_sonic_cute_chunked,
        _moe_mlp_local_sonic_cute_intermediate_chunked,
    )

    chunk_dim = os.environ.get("SCALE_MOE_CHUNK_DIM", "expert")
    if chunk_dim == "expert":
        local_fn = partial(_moe_mlp_local_sonic_cute_chunked, chunk_sizes=chunk_sizes)
    elif chunk_dim == "intermediate":
        if len(set(chunk_sizes)) != 1:
            raise ValueError(
                "SCALE_MOE_CHUNK_DIM=intermediate partitions the intermediate dim uniformly; it does "
                "not support non-uniform SCALE_MOE_CHUNK_SIZES. Use SCALE_MOE_EXPERT_CHUNKS."
            )
        local_fn = partial(_moe_mlp_local_sonic_cute_intermediate_chunked, chunks=len(chunk_sizes))
    else:
        raise ValueError(f"SCALE_MOE_CHUNK_DIM must be 'expert' or 'intermediate', got {chunk_dim!r}")

    batch_spec = _batch_spec_from_x(x, mesh)
    # FSDP layout from MoEExpertMlpPspecs: w_up_gate [E, H/data, 2I/model], w_down [E, I/model, H/data].
    w13_spec = _drop_absent_mesh_axes(mesh, P("expert", "data", "model"))
    w2_spec = _drop_absent_mesh_axes(mesh, P("expert", "model", "data"))

    x = _reshard_for_shard_map(x, mesh, batch_spec)
    selected_experts = _reshard_for_shard_map(selected_experts, mesh, batch_spec)
    combine_weights = _reshard_for_shard_map(combine_weights, mesh, batch_spec)
    w_up_gate = _reshard_for_shard_map(w_up_gate, mesh, w13_spec)
    w_down = _reshard_for_shard_map(w_down, mesh, w2_spec)

    shard_local = partial(
        local_fn,
        activation_fn=activation_fn,
        num_experts=num_experts,
        data_axis_name="data",
    )

    # SCALE_MOE_HOIST_CHUNK0: chunk-0's expert weights are resharded to replicated in the model Block
    # BEFORE attention (see model.Block) and threaded in here, so that all-gather is emitted ahead of
    # attention (operand-gated on the step-start-ready weights) and XLA can overlap it -- vs the
    # in-region gather, pinned to the shard_map's x_flat input, which can't start until attention ends.
    # Chunks 1+ still gather inside the region. Costs holding chunk-0 replicated across attention.
    if chunk_dim == "expert" and w13_pre0 is not None:
        replicated3 = P(None, None, None)
        w13_pre0 = _reshard_for_shard_map(w13_pre0, mesh, replicated3)
        w2_pre0 = _reshard_for_shard_map(w2_pre0, mesh, replicated3)
        shard_fn = shard_map(
            shard_local,
            mesh=mesh,
            in_specs=(batch_spec, batch_spec, batch_spec, w13_spec, w2_spec, replicated3, replicated3),
            out_specs=(batch_spec, P()),
            check_vma=False,
        )
        return shard_fn(x, selected_experts, combine_weights, w_up_gate, w_down, w13_pre0, w2_pre0)

    shard_fn = shard_map(
        shard_local,
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec, w13_spec, w2_spec),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )
    return shard_fn(x, selected_experts, combine_weights, w_up_gate, w_down)


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
