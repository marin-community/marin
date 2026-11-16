# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Router-metrics telemetry for the EP MoE model: collect per-shard routing partials, reduce them
across devices once after the layer scan, and summarize them for logging. All logging-only -- none
of this feeds the training loss."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int
from levanter.tracker.histogram import Histogram, SummaryStats

try:
    from jax.shard_map import shard_map
except ModuleNotFoundError:
    from jax.experimental.shard_map import shard_map


def local_routing_stats(
    selected_experts: Int[Array, "T K"],
    router_probs: Float[Array, "T E"],
    router_logits: Float[Array, "T E"],
    mesh: jax.sharding.AbstractMesh,
    *,
    num_experts: int,
    batch_axes: tuple[str, ...],
) -> dict[str, jax.Array]:
    """Per-shard partial sums for the router metrics, cross-device reduction deferred.

    Every router metric is a linear reduction over tokens, so reducing here would cost one exposed
    all-reduce per MoE layer per step and buy nothing: the metrics are logging-only, and qb_beta
    reaches the router bias only next step (via the trainer's pending_qb_betas). Return the unreduced
    per-shard partials and let ``reduce_router_stats`` do the collective once over the stacked scan
    outputs.
    """

    def _local(sel: jax.Array, probs: jax.Array, logits: jax.Array) -> dict[str, jax.Array]:
        probs_f = probs.astype(jnp.float32)
        logits_f = logits.astype(jnp.float32)
        counts = jnp.sum(jax.nn.one_hot(sel, num_experts, dtype=jnp.float32), axis=(0, 1))
        z = jsp.special.logsumexp(logits_f, axis=-1)
        return {
            "routing_counts_local": counts[None, :],
            "router_prob_sum_local": jnp.sum(probs_f, axis=0)[None, :],
            "router_z_sq_sum_local": jnp.sum(z**2)[None],
        }

    return shard_map(
        _local,
        mesh=mesh,
        in_specs=(P(batch_axes, None), P(batch_axes, None), P(batch_axes, None)),
        out_specs={
            "routing_counts_local": P(batch_axes, None),
            "router_prob_sum_local": P(batch_axes, None),
            "router_z_sq_sum_local": P(batch_axes),
        },
    )(selected_experts, router_probs, router_logits)


def reduce_router_stats(
    stacked: dict[str, jax.Array],
    *,
    num_experts: int,
    num_experts_per_token: int,
    num_tokens: int,
) -> dict[str, jax.Array]:
    """Reduce the stacked per-shard router partials across devices once, after the layer scan.

    ``stacked`` holds the scan's ``ys``: a leading ``[num_layers]`` axis over a shard axis that is
    still sharded over the batch axes. Summing that shard axis is one all-reduce for the whole stack
    rather than one per layer; XLA's all-reduce combiner then merges the four into a single tupled
    collective, so the layer scan emits none at all. ``num_tokens`` is the global (batch x seq) token
    count, the denominator the per-layer ``jnp.mean(..., axis=0)`` used to carry.
    """
    counts = jnp.sum(stacked["routing_counts_local"], axis=1)
    prob_sum = jnp.sum(stacked["router_prob_sum_local"], axis=1)
    z_sq_sum = jnp.sum(stacked["router_z_sq_sum_local"], axis=1)

    total_assignments = jnp.maximum(jnp.sum(counts, axis=-1, keepdims=True), 1.0)
    assignment_fraction = counts / total_assignments
    routing_entropy = -jnp.sum(assignment_fraction * jnp.log(assignment_fraction + 1e-6), axis=-1)
    token_fraction = assignment_fraction * num_experts_per_token
    p = prob_sum / num_tokens
    load_balancing_loss = num_experts * jnp.sum(token_fraction * p, axis=-1)

    return {
        "routing_counts": counts,
        "routing_entropy": routing_entropy,
        "load_balancing_loss": load_balancing_loss,
        "router_z_loss": z_sq_sum / num_tokens,
        "qb_beta": stacked["qb_beta"],
    }


def summarize_router_metrics(router_metrics: dict[str, jax.Array]) -> dict[str, jax.Array | SummaryStats]:
    routing_entropy = router_metrics["routing_entropy_per_layer"]
    routing_counts = router_metrics["routing_counts_per_layer"]
    load_balancing_loss = router_metrics["load_balancing_loss_per_layer"]
    router_z_loss = router_metrics["router_z_loss_per_layer"]
    capacity_overflow = router_metrics["capacity_overflow_per_layer"]
    sender_capacity_overflow = router_metrics["sender_capacity_overflow_per_layer"]
    receiver_capacity_overflow = router_metrics["receiver_capacity_overflow_per_layer"]
    margin_min = router_metrics["margin_min_per_layer"]  # QB histogram grid lo per layer
    margin_max = router_metrics["margin_max_per_layer"]  # QB histogram grid hi per layer
    qb_beta = router_metrics.get("qb_beta_per_layer")  # per-layer per-expert beta; router_bias = -qb_beta
    num_layers = int(routing_entropy.shape[0])

    # Per-layer total assignments = sum of routing_counts over experts (= tokens * k).
    assignments_per_layer = jnp.sum(routing_counts.astype(jnp.float32), axis=-1)
    capacity_overflow_rate = capacity_overflow.astype(jnp.float32) / jnp.maximum(assignments_per_layer, 1.0)
    sender_overflow_rate = sender_capacity_overflow.astype(jnp.float32) / jnp.maximum(assignments_per_layer, 1.0)
    receiver_overflow_rate = receiver_capacity_overflow.astype(jnp.float32) / jnp.maximum(assignments_per_layer, 1.0)

    out: dict[str, jax.Array | SummaryStats] = {
        "train/router/routing_entropy_mean": jnp.mean(routing_entropy),
        "train/router/load_balancing_loss": jnp.mean(load_balancing_loss),
        "train/router/router_z_loss": jnp.mean(router_z_loss),
        "train/router/routing_counts_per_layer": routing_counts,
        "train/router/capacity_overflow_rate_mean": jnp.mean(capacity_overflow_rate),
        "train/router/sender_overflow_rate_mean": jnp.mean(sender_overflow_rate),
        "train/router/receiver_overflow_rate_mean": jnp.mean(receiver_overflow_rate),
        # QB HIST margin range: min over layers and max over layers, plus per-layer below.
        "train/router/margin_min": jnp.min(margin_min),
        "train/router/margin_max": jnp.max(margin_max),
        "qb_beta_per_layer": qb_beta,
    }
    if qb_beta is not None:
        # Router bias applied to the logits is -qb_beta; log the extent of the per-expert bias.
        out["train/router/bias_min"] = -jnp.max(qb_beta)
        out["train/router/bias_max"] = -jnp.min(qb_beta)
    for i in range(num_layers):
        out[f"train/router/layer_{i}/routing_entropy"] = routing_entropy[i]
        out[f"train/router/layer_{i}/load_balancing_loss"] = load_balancing_loss[i]
        out[f"train/router/layer_{i}/router_z_loss"] = router_z_loss[i]
        out[f"train/router/layer_{i}/routing_hist"] = _histogram_from_expert_counts(routing_counts[i])
        out[f"train/router/layer_{i}/capacity_overflow_rate"] = capacity_overflow_rate[i]
        out[f"train/router/layer_{i}/margin_min"] = margin_min[i]
        out[f"train/router/layer_{i}/margin_max"] = margin_max[i]
    return out


def _histogram_from_expert_counts(expert_counts: jax.Array) -> SummaryStats:
    counts = jnp.asarray(expert_counts, dtype=jnp.float32)
    num_experts = counts.shape[0]
    expert_ids = jnp.arange(num_experts, dtype=jnp.float32)
    num = jnp.sum(counts)
    sum_values = jnp.sum(counts * expert_ids)
    sum_squares = jnp.sum(counts * expert_ids * expert_ids)
    nonzero = counts > 0
    min_value = jnp.where(nonzero, expert_ids, jnp.inf).min()
    max_value = jnp.where(nonzero, expert_ids, -jnp.inf).max()
    min_value = jnp.where(num > 0, min_value, 0.0)
    max_value = jnp.where(num > 0, max_value, 0.0)
    bucket_limits = jnp.arange(num_experts + 1, dtype=jnp.float32)
    histogram = Histogram(bucket_limits=bucket_limits, bucket_counts=counts)
    return SummaryStats.from_reduced_values(
        min=min_value,
        max=max_value,
        num=num,
        nonzero_count=jnp.sum(nonzero),
        sum=sum_values,
        sum_squares=sum_squares,
        histogram=histogram,
    )
