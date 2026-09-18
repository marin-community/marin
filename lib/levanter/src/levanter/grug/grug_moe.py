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
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from haliax.jax_utils import named_call
from jax import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.common import (
    _DEFAULT_EP_CAPACITY_FACTOR,
    _EP_MOE_IMPLEMENTATIONS,
    _init_weight,
    MOE_REMAT_SAVE_NAMES as MOE_REMAT_SAVE_NAMES,
    CapacityDrops,
    MoeDispatchCounts,
    MoEExpertMlpPspecs,
    padding_skipped_assignments,
    MoeActivation,
    MoeImplementation,
    PspecAxis,
    resolve_moe_implementation,
    split_moe_w13_output,
)
from levanter.grug._moe.ep_common import (
    _clip_receiver_group_sizes as _clip_receiver_group_sizes,
    _expert_granular_a2a_params as _expert_granular_a2a_params,
)
from levanter.grug._moe.ep_deepep import _moe_mlp_ep_deepep_local
from levanter.grug._moe.ep_fixed_all_to_all import _moe_mlp_ep_fixed_a2a_local
from levanter.grug._moe.ep_fixed_pooled_wave_all_to_all import _moe_mlp_ep_fixed_pooled_wave_a2a_local
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

MOE_DROPPED_ASSIGNMENTS_METRIC = "moe/dropped_assignments"
MOE_SENDER_DROPPED_ASSIGNMENTS_METRIC = "moe/sender_dropped_assignments"
MOE_RECEIVER_DROPPED_ASSIGNMENTS_METRIC = "moe/receiver_dropped_assignments"
MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC = "moe/skipped_padding_assignments"
MOE_VALID_ASSIGNMENTS_METRIC = "moe/valid_assignments"


def moe_routing_stats(
    selected_experts: Int[Array, "T K"],
    router_probs: Float[Array, "T E"],
    router_logits: Float[Array, "T E"],
    token_valid: Bool[Array, "T"],
    *,
    num_experts: int,
    num_experts_per_token: int,
) -> dict[str, jax.Array]:
    """Compute padding-aware routing distributions and auxiliary losses."""
    router_probs_f = router_probs.astype(jnp.float32)
    router_logits_f = router_logits.astype(jnp.float32)
    valid_f = token_valid.astype(jnp.float32)
    expert_counts = jnp.sum(
        jax.nn.one_hot(selected_experts, num_experts, dtype=jnp.float32) * valid_f[:, None, None],
        axis=(0, 1),
    )
    total_assignments = jnp.maximum(jnp.sum(expert_counts), 1.0)
    assignment_fraction = expert_counts / total_assignments
    routing_entropy = -jnp.sum(assignment_fraction * jnp.log(assignment_fraction + 1e-6))
    token_fraction = assignment_fraction * num_experts_per_token
    valid_tokens = jnp.maximum(jnp.sum(valid_f), 1.0)
    mean_router_probability = jnp.sum(router_probs_f * valid_f[:, None], axis=0) / valid_tokens
    load_balancing_loss = num_experts * jnp.sum(token_fraction * mean_router_probability)
    log_partition = jsp.special.logsumexp(router_logits_f, axis=-1)
    router_z_loss = jnp.sum(log_partition**2 * valid_f) / valid_tokens
    return {
        "routing_counts": expert_counts,
        "routing_entropy": routing_entropy,
        "load_balancing_loss": load_balancing_loss,
        "router_z_loss": router_z_loss,
    }


def moe_routing_stats_local(
    selected_experts: Int[Array, "T K"],
    router_probs: Float[Array, "T E"],
    router_logits: Float[Array, "T E"],
    token_valid: Bool[Array, "T"],
    mesh: jax.sharding.AbstractMesh,
    *,
    batch_axes: tuple[str, ...],
    num_experts: int,
) -> dict[str, jax.Array]:
    """Return shard-local routing metric sufficient statistics.

    Call :func:`reduce_moe_routing_stats` after stacking these outputs over
    layers to combine their cross-device reductions into one collective.
    """

    def _local(sel: jax.Array, probs: jax.Array, logits: jax.Array, valid: jax.Array) -> dict[str, jax.Array]:
        valid_f = valid.astype(jnp.float32)
        counts = jnp.sum(jax.nn.one_hot(sel, num_experts, dtype=jnp.float32) * valid_f[:, None, None], axis=(0, 1))
        log_partition = jsp.special.logsumexp(logits.astype(jnp.float32), axis=-1)
        return {
            "routing_counts_local": counts[None, :],
            "router_prob_sum_local": jnp.sum(probs.astype(jnp.float32) * valid_f[:, None], axis=0)[None, :],
            "router_z_sq_sum_local": jnp.sum(log_partition**2 * valid_f)[None],
            "valid_tokens_local": jnp.sum(valid_f)[None],
        }

    return shard_map(
        _local,
        mesh=mesh,
        in_specs=(P(batch_axes, None), P(batch_axes, None), P(batch_axes, None), P(batch_axes)),
        out_specs={
            "routing_counts_local": P(batch_axes, None),
            "router_prob_sum_local": P(batch_axes, None),
            "router_z_sq_sum_local": P(batch_axes),
            "valid_tokens_local": P(batch_axes),
        },
    )(selected_experts, router_probs, router_logits, token_valid)


def reduce_moe_routing_stats(
    stacked: dict[str, jax.Array],
    *,
    num_experts: int,
    num_experts_per_token: int,
) -> dict[str, jax.Array]:
    """Reduce scan-stacked routing metric partials after the layer scan.

    ``qb_beta_local`` and ``qb_beta_weight_local`` are optionally reduced as
    valid-token-weighted QB thresholds; histogram QB callers instead provide
    an already-reduced ``qb_beta``.
    """
    counts = jnp.sum(stacked["routing_counts_local"], axis=1)
    prob_sum = jnp.sum(stacked["router_prob_sum_local"], axis=1)
    z_sq_sum = jnp.sum(stacked["router_z_sq_sum_local"], axis=1)
    valid_tokens = jnp.maximum(jnp.sum(stacked["valid_tokens_local"], axis=1), 1.0)
    total_assignments = jnp.maximum(jnp.sum(counts, axis=-1, keepdims=True), 1.0)
    assignment_fraction = counts / total_assignments
    routing_entropy = -jnp.sum(assignment_fraction * jnp.log(assignment_fraction + 1e-6), axis=-1)
    load_balancing_loss = num_experts * jnp.sum(
        assignment_fraction * num_experts_per_token * (prob_sum / valid_tokens[:, None]), axis=-1
    )
    out = {
        "routing_counts": counts,
        "routing_entropy": routing_entropy,
        "load_balancing_loss": load_balancing_loss,
        "router_z_loss": z_sq_sum / valid_tokens,
    }
    if "qb_beta_local" in stacked:
        qb_weights = stacked["qb_beta_weight_local"].astype(jnp.float32)
        out["qb_beta"] = jnp.sum(stacked["qb_beta_local"] * qb_weights[:, :, None], axis=1) / jnp.maximum(
            jnp.sum(qb_weights, axis=1)[:, None], 1
        )
    else:
        out["qb_beta"] = stacked["qb_beta"]
    return out


def qb_topk_physical_count(local_tokens: int, *, num_experts_per_token: int, num_experts: int) -> int:
    """Rows `top_k` keeps per expert on one shard, sized for the all-valid case."""
    return max(1, local_tokens * num_experts_per_token // num_experts)


def qb_beta_topk_shard(
    s_local: Float[Array, "t E"],
    valid_local: Bool[Array, "t"],
    *,
    physical_count: int,
    num_experts_per_token: int,
    num_experts: int,
) -> tuple[Float[Array, "E"], Int[Array, ""]]:
    """Per-shard QB threshold over valid tokens plus the valid count that weights it."""
    valid_count = jnp.sum(valid_local, dtype=jnp.int32)
    topk_values, _ = jax.lax.top_k(jnp.where(valid_local[None, :], s_local.T, -jnp.inf), physical_count)
    logical_count = jnp.clip(valid_count * num_experts_per_token // num_experts, 1, physical_count)
    beta = jnp.take(topk_values, logical_count - 1, axis=1)
    return jnp.where(valid_count > 0, beta, 0), valid_count


def estimate_qb_beta_topk(
    s_minus_alpha: Float[Array, "T E"],
    token_valid: Bool[Array, "T"],
    mesh: jax.sharding.AbstractMesh,
    *,
    batch_axes: tuple[str, ...],
    num_experts_per_token: int,
    num_experts: int,
) -> Float[Array, "E"]:
    """Estimate QB thresholds from valid tokens on each batch shard."""
    num_devices = 1
    for axis in batch_axes:
        num_devices *= mesh.shape[axis]
    local_tokens = s_minus_alpha.shape[0] // num_devices
    physical_count = qb_topk_physical_count(
        local_tokens, num_experts_per_token=num_experts_per_token, num_experts=num_experts
    )

    def _local(s_local: jax.Array, valid_local: jax.Array) -> jax.Array:
        beta, valid_count = qb_beta_topk_shard(
            s_local,
            valid_local,
            physical_count=physical_count,
            num_experts_per_token=num_experts_per_token,
            num_experts=num_experts,
        )
        weighted_beta = jax.lax.psum(beta * valid_count, axis_name=batch_axes)
        global_valid_count = jax.lax.psum(valid_count, axis_name=batch_axes)
        return weighted_beta / jnp.maximum(global_valid_count, 1)

    return shard_map(
        _local,
        mesh=mesh,
        in_specs=(P(batch_axes, None), P(batch_axes)),
        out_specs=P(),
    )(s_minus_alpha, token_valid)


class QBRoutedMoE(eqx.Module):
    """QB router and expert MLP composition for flat token batches.

    Model variants retain parameter ownership so checkpoint paths remain
    stable. This component owns the common top-k selection, QB update estimate,
    routing statistics, and expert dispatch.
    """

    num_experts_per_token: int = eqx.field(static=True)
    # The batch-sharded axes used for QB's local statistic and its deferred reduction.
    batch_axes: tuple[str, ...] = eqx.field(static=True)
    # This is part of the QB routing recipe, rather than an independently tuned
    # model hyperparameter: selected sigmoid weights are normalized to this sum.
    routing_renorm_sum: float = eqx.field(static=True, default=2.5)

    @named_call
    def __call__(
        self,
        x: Float[Array, "T D"],
        token_valid: Bool[Array, "T"],
        *,
        router: Float[Array, "D E"],
        router_bias: Float[Array, "E"],
        expert_mlp: "MoEExpertMlp",
        mesh: jax.sharding.AbstractMesh,
        report_capacity_overflow: bool = True,
    ) -> tuple[Float[Array, "T D"], dict[str, jax.Array]]:
        """Route tokens and return their expert output plus router metrics.

        Invalid positions still have static routing tensor shapes, but are
        excluded from QB threshold estimation, routing statistics, logical
        capacity, expert gradients, and the returned dropped-assignment count.
        ``skipped_assignments`` records those omitted assignments even when
        capacity-overflow reporting is disabled.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be rank-2 [T, D], got shape={x.shape}")
        if token_valid.ndim != 1 or token_valid.shape[0] != x.shape[0]:
            raise ValueError(f"token_valid must have shape [{x.shape[0]}], got shape={token_valid.shape}")

        with jax.named_scope("moe_route"):
            router_logits = jnp.einsum("td,de->te", x, reshard(router, P(None, None))).astype(jnp.float32)
            biased_logits = router_logits + jax.lax.stop_gradient(router_bias)
            router_probs = jax.nn.softmax(router_logits, axis=-1)
            topk_logits, selected_experts = jax.lax.top_k(biased_logits, self.num_experts_per_token + 1)
            qb_alpha = topk_logits[:, -1:]
            selected_experts = selected_experts[:, :-1]
            selected_logits = jnp.take_along_axis(router_logits, selected_experts, axis=-1)
            combine_weights = jax.nn.sigmoid(selected_logits)
            combine_weights *= self.routing_renorm_sum / (jnp.sum(combine_weights, axis=-1, keepdims=True) + 1e-9)
            combine_weights = combine_weights.astype(x.dtype)

        with jax.named_scope("moe_router_stats"):
            router_stats = moe_routing_stats(
                selected_experts,
                router_probs,
                router_logits,
                token_valid,
                num_experts=router.shape[1],
                num_experts_per_token=self.num_experts_per_token,
            )

        with jax.named_scope("moe_qb_beta"):
            s_minus_alpha = reshard(router_logits - qb_alpha, P(self.batch_axes, None))
            router_stats["qb_beta"] = estimate_qb_beta_topk(
                s_minus_alpha,
                reshard(token_valid, P(self.batch_axes)),
                mesh,
                batch_axes=self.batch_axes,
                num_experts_per_token=self.num_experts_per_token,
                num_experts=router.shape[1],
            )

        with jax.named_scope("moe_experts"):
            moe_out = expert_mlp(
                x,
                selected_experts.astype(jnp.int32),
                combine_weights,
                token_valid=token_valid,
                mesh=mesh,
                report_capacity_overflow=report_capacity_overflow,
            )
        if report_capacity_overflow:
            routed, dispatch_counts = cast(tuple[Float[Array, "T D"], MoeDispatchCounts], moe_out)
            router_stats["capacity_overflow"] = dispatch_counts.dropped.astype(jnp.float32)
            router_stats["skipped_assignments"] = dispatch_counts.padding_skipped.astype(jnp.float32)
        else:
            routed = cast(Float[Array, "T D"], moe_out)
            router_stats["capacity_overflow"] = jnp.zeros((), dtype=jnp.float32)
            router_stats["skipped_assignments"] = padding_skipped_assignments(
                token_valid, topk=self.num_experts_per_token
            ).astype(jnp.float32)
        return routed, router_stats


class MoEExpertMlp(eqx.Module):
    """Expert MLP weights for routed MoE calls."""

    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array
    implementation: MoeImplementation = eqx.field(static=True)
    activation: MoeActivation = eqx.field(static=True)
    capacity_factor: float = eqx.field(static=True)
    pooled_transport_capacity_factor: float | None = eqx.field(static=True, default=None)
    expert_chunks: int = eqx.field(static=True, default=1)
    num_expert_waves: int = eqx.field(static=True, default=1)

    @staticmethod
    def init(
        *,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        initializer_std: float,
        key: jax.Array,
        gate_up_initializer_std: float | None = None,
        implementation: MoeImplementation | str | None = None,
        activation: MoeActivation = ActivationFunctionEnum.silu,
        capacity_factor: float = _DEFAULT_EP_CAPACITY_FACTOR,
        pooled_transport_capacity_factor: float | None = None,
        expert_chunks: int = 1,
        num_expert_waves: int = 1,
        pspecs: MoEExpertMlpPspecs = MoEExpertMlpPspecs(),
    ) -> "MoEExpertMlp":
        resolved_implementation = resolve_moe_implementation(implementation)
        k_gate, k_up, k_down = jax.random.split(key, 3)
        # `w_gate`/`w_up` contract over `hidden_dim`, so their fan-in moves when the experts run
        # in a latent space; `w_down` contracts over `intermediate_dim` and is unaffected.
        gate_up_std = initializer_std if gate_up_initializer_std is None else gate_up_initializer_std
        w_gate = _init_weight(k_gate, (num_experts, hidden_dim, intermediate_dim), gate_up_std)
        w_up = _init_weight(k_up, (num_experts, hidden_dim, intermediate_dim), gate_up_std)
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
            pooled_transport_capacity_factor=pooled_transport_capacity_factor,
            expert_chunks=expert_chunks,
            num_expert_waves=num_expert_waves,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "T D"],
        selected_experts: Int[Array, "T K"],
        combine_weights: Float[Array, "T K"],
        *,
        token_valid: Bool[Array, "T"] | None = None,
        mesh: jax.sharding.AbstractMesh | None = None,
        report_capacity_overflow: bool = False,
    ) -> Float[Array, "T D"] | tuple[Float[Array, "T D"], MoeDispatchCounts]:
        w_gate_up = jnp.concatenate([self.w_gate, self.w_up], axis=-1)
        return moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_gate_up,
            self.w_down,
            token_valid=token_valid,
            activation=self.activation,
            implementation=self.implementation,
            mesh=mesh,
            capacity_factor=self.capacity_factor,
            pooled_transport_capacity_factor=self.pooled_transport_capacity_factor,
            report_capacity_overflow=report_capacity_overflow,
            expert_chunks=self.expert_chunks,
            num_expert_waves=self.num_expert_waves,
        )


@named_call
def moe_mlp(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    w_up_gate: Float[Array, "E D I2"],
    w_down: Float[Array, "E I D"],
    *,
    token_valid: Bool[Array, "T"] | None = None,
    activation: MoeActivation = ActivationFunctionEnum.silu,
    implementation: MoeImplementation | str | None = None,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = _DEFAULT_EP_CAPACITY_FACTOR,
    pooled_transport_capacity_factor: float | None = None,
    report_capacity_overflow: bool = False,
    expert_chunks: int = 1,
    num_expert_waves: int = 1,
) -> Float[Array, "T D"] | tuple[Float[Array, "T D"], MoeDispatchCounts]:
    """Functional routed MoE MLP core used by Grug modules and benchmarks.

    This helper handles dispatch/permute/unpermute (+EP collectives) from
    precomputed token-to-expert assignments. Routing logits/top-k selection
    stays in the caller (e.g. model MLP block).

    `token_valid` excludes invalid positions from dispatch, capacity accounting,
    and expert gradients. Omitted validity treats every token as valid.

    Set `report_capacity_overflow=True` to also return sender and receiver
    capacity drops plus padding-skipped assignment counts.

    `expert_chunks` applies only to the local `sonic_cute` FSDP path. Values
    greater than one split the expert bank into equal, statically sized chunks.

    `pooled_transport_capacity_factor` sets the sender capacity for each
    destination pool. `num_expert_waves` sets the static wave count for the
    fixed pooled-wave implementation.
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
    if token_valid is None:
        token_valid = jnp.ones((x.shape[0],), dtype=jnp.bool_)
    elif token_valid.ndim != 1 or token_valid.shape[0] != x.shape[0]:
        raise ValueError(f"token_valid must have shape [{x.shape[0]}], got shape={token_valid.shape}")
    elif token_valid.dtype != jnp.bool_:
        raise ValueError(f"token_valid must have boolean dtype, got dtype={token_valid.dtype}")
    # Padding is an input property, so count it once here; backends report only capacity drops.
    padding_skipped = padding_skipped_assignments(token_valid, topk=selected_experts.shape[1])

    def dispatch_counts(drops: CapacityDrops) -> MoeDispatchCounts:
        return MoeDispatchCounts(
            sender_dropped=drops.sender_dropped,
            receiver_dropped=drops.receiver_dropped,
            padding_skipped=padding_skipped,
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
            token_valid,
            w_up_gate,
            w_down,
            activation_fn=activation_fn,
            num_experts=num_experts,
            implementation=resolved_implementation,
            expert_chunks=expert_chunks,
        )
        if report_capacity_overflow:
            return out, dispatch_counts(
                CapacityDrops(sender_dropped=dropped, receiver_dropped=jnp.zeros_like(dropped))
            )
        return out

    batch_spec = _batch_spec_from_x(x, mesh)

    if has_expert_axis and expert_axis_size > 1:
        if expert_chunks != 1:
            raise ValueError("expert_chunks must be 1 when expert parallelism is active")
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
        elif resolved_implementation == "fixed_all_to_all":
            shard_local_fn = _moe_mlp_ep_fixed_a2a_local
        elif resolved_implementation == "fixed_pooled_wave_all_to_all":
            if pooled_transport_capacity_factor is None:
                raise ValueError("fixed_pooled_wave_all_to_all requires pooled_transport_capacity_factor")
            shard_local_fn = partial(
                _moe_mlp_ep_fixed_pooled_wave_a2a_local,
                transport_capacity_factor=pooled_transport_capacity_factor,
                num_expert_waves=num_expert_waves,
            )
        elif resolved_implementation == "deepep":
            shard_local_fn = _moe_mlp_ep_deepep_local
        else:
            raise AssertionError(f"Unhandled MoE implementation {resolved_implementation!r}")

        w_up_gate_spec = P("expert", None, None)
        w_down_spec = P("expert", None, None)

        x = _reshard_for_shard_map(x, mesh, batch_spec)
        selected_experts = _reshard_for_shard_map(selected_experts, mesh, batch_spec)
        combine_weights = _reshard_for_shard_map(combine_weights, mesh, batch_spec)
        token_valid = _reshard_for_shard_map(token_valid, mesh, batch_spec)
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
                batch_spec,
                w_up_gate_spec,
                w_down_spec,
            ),
            out_specs=(batch_spec, CapacityDrops(sender_dropped=P(), receiver_dropped=P())),
            check_vma=False,
        )
        out, drops = shard_fn(x, selected_experts, combine_weights, token_valid, w_up_gate, w_down)
        if report_capacity_overflow:
            return out, dispatch_counts(drops)
        return out

    # Fallback path for no expert axis (or expert axis size 1) keeps routing
    # semantics without EP collectives. JAX 0.9 requires shard_map in_specs to
    # match the actual input sharding, so reshard ordinary inputs to the mesh
    # specs that preserve data-axis parallelism.
    x_spec = _value_spec_or_default(x, batch_spec, replace_replicated=True)
    selected_experts_spec = _value_spec_or_default(selected_experts, batch_spec, replace_replicated=True)
    combine_weights_spec = _value_spec_or_default(combine_weights, batch_spec, replace_replicated=True)
    token_valid_spec = _value_spec_or_default(token_valid, batch_spec, replace_replicated=True)
    if expert_chunks > 1 and resolved_implementation == "sonic_cute":
        # The chunked sonic_cute path all-gathers the hidden dim per expert-chunk over ``data``, so it
        # needs a real data axis; without one the local kernel hits an unbound-axis error.
        if not _mesh_has_axis(mesh, "data") or _mesh_axis_size(mesh, "data") <= 1:
            raise ValueError(
                "chunked sonic_cute (expert_chunks > 1) requires a data axis to all-gather the expert "
                "weights; use expert_chunks=1 on a single device or an unsharded mesh."
            )
        # The local weights must arrive H-sharded ([E, H/data, 2I] / [E, I, H/data]). Force that FSDP
        # layout rather than inheriting whatever (possibly replicated) sharding the layer scan left,
        # which would make the tiled all-gather reconstruct H * data_shards.
        w_up_gate_spec = _drop_absent_mesh_axes(mesh, P("expert", "data", "model"))
        w_down_spec = _drop_absent_mesh_axes(mesh, P("expert", "model", "data"))
    else:
        w_up_gate_spec = _value_spec_or_default(w_up_gate, P(*(None for _ in range(w_up_gate.ndim))))
        w_down_spec = _value_spec_or_default(w_down, P(*(None for _ in range(w_down.ndim))))

    x = _reshard_for_shard_map(x, mesh, x_spec)
    selected_experts = _reshard_for_shard_map(selected_experts, mesh, selected_experts_spec)
    combine_weights = _reshard_for_shard_map(combine_weights, mesh, combine_weights_spec)
    token_valid = _reshard_for_shard_map(token_valid, mesh, token_valid_spec)
    w_up_gate = _reshard_for_shard_map(w_up_gate, mesh, w_up_gate_spec)
    w_down = _reshard_for_shard_map(w_down, mesh, w_down_spec)

    def local_moe(x, selected_experts, combine_weights, token_valid, w_up_gate, w_down):
        out, dropped = _moe_mlp_local(
            x,
            selected_experts,
            combine_weights,
            token_valid,
            w_up_gate,
            w_down,
            activation_fn=activation_fn,
            num_experts=num_experts,
            implementation=resolved_implementation,
            expert_chunks=expert_chunks,
        )
        batch_axis_names = x_spec[0]
        if report_capacity_overflow and batch_axis_names is not None:
            dropped = jax.lax.psum(dropped, axis_name=batch_axis_names)
        return out, dropped

    shard_fn = shard_map(
        local_moe,
        mesh=mesh,
        in_specs=(
            x_spec,
            selected_experts_spec,
            combine_weights_spec,
            token_valid_spec,
            w_up_gate_spec,
            w_down_spec,
        ),
        out_specs=(x_spec, P()),
        check_vma=False,
    )
    out, dropped = shard_fn(x, selected_experts, combine_weights, token_valid, w_up_gate, w_down)
    if report_capacity_overflow:
        return out, dispatch_counts(CapacityDrops(sender_dropped=dropped, receiver_dropped=jnp.zeros_like(dropped)))
    return out


__all__ = [
    "MOE_DROPPED_ASSIGNMENTS_METRIC",
    "MOE_SENDER_DROPPED_ASSIGNMENTS_METRIC",
    "MOE_RECEIVER_DROPPED_ASSIGNMENTS_METRIC",
    "MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC",
    "MOE_VALID_ASSIGNMENTS_METRIC",
    "MoeDispatchCounts",
    "MoeActivation",
    "MoEExpertMlp",
    "MoEExpertMlpPspecs",
    "MoeImplementation",
    "PspecAxis",
    "QBRoutedMoE",
    "moe_mlp",
    "moe_routing_stats",
    "moe_routing_stats_local",
    "reduce_moe_routing_stats",
    "estimate_qb_beta_topk",
    "qb_beta_topk_shard",
    "qb_topk_physical_count",
    "resolve_moe_implementation",
    "split_moe_w13_output",
]
