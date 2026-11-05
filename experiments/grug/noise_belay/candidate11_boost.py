# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independently measured Boost readouts on candidate 11's residual writes.

This provides measured hub credit and selected parameter reconstruction. Cut and
several parameter groups remain incomplete; this is not a full training estimator.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import AttentionMask

from experiments.grug.fast_track import model as native
from experiments.grug.noise_belay.candidate11_forward import ForwardCapture, SublayerCapture
from experiments.grug.noise_belay.forward_credit import (
    ReadoutParameterCredit,
    Relu2ParameterCredit,
    attnres_readout_credit,
    causal_convolution_credit,
    causal_convolution_parameter_credit,
    direction_frame,
    readout_parameter_credit,
    relu2_parameter_credit,
)
from experiments.grug.noise_belay.routed_expert_credit import (
    pooled_relu2_parameter_credit,
    pooled_router_parameter_credit,
)

BATCH_AXES = ("replica_dcn", "data", "expert")


def measure_boost_hubs(
    stages: tuple[SublayerCapture, ...],
    tokens: jax.Array,
    loss_weight: jax.Array,
    key: jax.Array,
    *,
    head: jax.Array,
    norm_gain: jax.Array,
    query: jax.Array,
    epsilon: float,
    soft_cap: float,
    relative_step: float,
    token_chunk: int,
    direction_chunk: int,
    z_loss_weight: float,
) -> tuple[jax.Array, ...]:
    """Read every sublayer write through the shared final depth gate and head.

    Returned credits are unnormalized weighted sums, sharded like activations.
    Every stage uses its own forward-loss measurements. No earlier credit is
    passed through a subsequent nonlinear operation.
    """
    width = head.shape[0]
    labels = jnp.pad(tokens[:, 1:], ((0, 0), (0, 1)))
    labels = jnp.where(loss_weight > 0, labels, -1)

    def local_readout(partial, sources, labels, weights, head, gain, query, directions):
        shape = partial.shape
        point = partial.reshape(-1, width)
        fixed = sources.reshape(sources.shape[0], -1, width)
        target = labels.reshape(-1)
        weight = weights.reshape(-1)
        count = point.shape[0]
        padding = (-count) % token_chunk
        point = jnp.pad(point, ((0, padding), (0, 0))).reshape(-1, token_chunk, width)
        fixed = jnp.pad(fixed, ((0, 0), (0, padding), (0, 0))).reshape(fixed.shape[0], -1, token_chunk, width)
        fixed = jnp.swapaxes(fixed, 0, 1)
        target = jnp.pad(target, (0, padding), constant_values=-1).reshape(-1, token_chunk)
        weight = jnp.pad(weight, (0, padding)).reshape(-1, token_chunk)

        def tile(_, args):
            point, fixed, target, weight = args
            credit = attnres_readout_credit(
                point,
                fixed,
                query,
                head,
                gain,
                target,
                directions,
                relative_step=relative_step,
                chunk_size=direction_chunk,
                epsilon=epsilon,
                soft_cap=soft_cap,
                z_loss_weight=z_loss_weight,
            )
            return None, credit * weight[:, None]

        _, credit = jax.lax.scan(tile, None, (point, fixed, target, weight))
        return credit.reshape(-1, width)[:count].reshape(shape)

    specs = (
        P(BATCH_AXES, None, None),
        P(None, BATCH_AXES, None, None),
        P(BATCH_AXES, None),
        P(BATCH_AXES, None),
        P(None, None),
        P(None),
        P(None),
        P(None, None),
    )
    mapped_readout = jax.shard_map(
        local_readout,
        mesh=jax.sharding.get_abstract_mesh(),
        in_specs=specs,
        out_specs=P(BATCH_AXES, None, None),
        check_vma=False,
    )
    hub_credits = []
    for index, stage in enumerate(stages):
        directions = direction_frame(width, jax.random.fold_in(key, index))
        arguments = (
            stage.partial,
            jnp.stack(stage.completed_sources),
            labels,
            loss_weight,
            head,
            norm_gain,
            query,
            directions,
        )
        hub_credits.append(mapped_readout(*(jax.sharding.reshard(a, s) for a, s in zip(arguments, specs, strict=True))))
    return tuple(hub_credits)


def measure_readout_parameters(
    stages: tuple[SublayerCapture, ...],
    tokens: jax.Array,
    loss_weight: jax.Array,
    key: jax.Array,
    *,
    head: jax.Array,
    norm_gain: jax.Array,
    query: jax.Array,
    epsilon: float,
    soft_cap: float,
    head_step: float,
    query_step: float,
    token_chunk: int,
    direction_chunk: int,
    z_loss_weight: float,
) -> ReadoutParameterCredit:
    """Sum shared-head, final-gain and depth-query credits over chosen stages.

    The caller chooses stages: Boost uses intermediate readouts and the final
    one; Cut uses the final readout. Results are global weighted sums, reduced
    across the data/expert batch mesh. Normalize once by the global token weight
    in the training estimator, not by the number of intermediate readouts.
    """
    width = head.shape[0]
    labels = jnp.pad(tokens[:, 1:], ((0, 0), (0, 1)))
    labels = jnp.where(loss_weight > 0, labels, -1)

    def local_readout(sources, labels, weights, head, gain, query, directions):
        sources = sources.reshape(sources.shape[0], -1, width)
        target, weight = labels.reshape(-1), weights.reshape(-1)
        padding = (-target.size) % token_chunk
        sources = jnp.pad(sources, ((0, 0), (0, padding), (0, 0))).reshape(sources.shape[0], -1, token_chunk, width)
        sources = jnp.swapaxes(sources, 0, 1)
        target = jnp.pad(target, (0, padding), constant_values=-1).reshape(-1, token_chunk)
        weight = jnp.pad(weight, (0, padding)).reshape(-1, token_chunk)

        def tile(total, args):
            sources, target, weight = args
            credit = readout_parameter_credit(
                sources,
                query,
                head,
                gain,
                target,
                weight,
                directions,
                head_step=head_step,
                query_step=query_step,
                direction_chunk=direction_chunk,
                epsilon=epsilon,
                soft_cap=soft_cap,
                z_loss_weight=z_loss_weight,
            )
            return jax.tree.map(jnp.add, total, credit), None

        initial = ReadoutParameterCredit(*(jnp.zeros_like(a, dtype=jnp.float32) for a in (head, gain, query)))
        total, _ = jax.lax.scan(tile, initial, (sources, target, weight))
        return jax.tree.map(lambda a: jax.lax.psum(a, BATCH_AXES), total)

    specs = (
        P(None, BATCH_AXES, None, None),
        P(BATCH_AXES, None),
        P(BATCH_AXES, None),
        P(None, None),
        P(None),
        P(None),
        P(None, None),
    )
    mapped_readout = jax.shard_map(
        local_readout,
        mesh=jax.sharding.get_abstract_mesh(),
        in_specs=specs,
        out_specs=ReadoutParameterCredit(P(None, None), P(None), P(None)),
        check_vma=False,
    )
    total = ReadoutParameterCredit(*(jnp.zeros_like(a, dtype=jnp.float32) for a in (head, norm_gain, query)))
    for index, stage in enumerate(stages):
        arguments = (
            jnp.stack((*stage.completed_sources, stage.partial)),
            labels,
            loss_weight,
            head,
            norm_gain,
            query,
            direction_frame(width, jax.random.fold_in(key, index)),
        )
        credit = mapped_readout(*(jax.sharding.reshard(a, s) for a, s in zip(arguments, specs, strict=True)))
        total = jax.tree.map(jnp.add, total, credit)
    return total


def reconstruct_shared_experts(
    model: native.Transformer,
    capture: ForwardCapture,
    hub_credits: tuple[jax.Array, ...],
    *,
    relative_step: float,
) -> tuple[Relu2ParameterCredit, ...]:
    """Reconstruct candidate 11's shared-expert matrices from measured MLP hubs.

    Credits already include token weights. Sum over all batch/expert shards,
    without division; the estimator applies its single global normalization.
    The input gate/norm require their own independent probes and are not
    updated by this function.
    """
    cfg = model.config
    if (
        cfg.num_shared_experts != 1
        or not cfg.shared_ungated_relu2
        or cfg.shared_ungated_activation != native.UngatedExpertActivation.RELU2
        or cfg.expert_leaky_slope != 0
        or cfg.shared_expert_gate
        or cfg.laurel_rank
        or cfg.mlp_in_center
        or cfg.mlp_in_whiten_power
    ):
        raise ValueError("Shared credit requires the candidate 11 expert configuration")

    def local(inputs, preactivation, weight, credit):
        result = relu2_parameter_credit(
            inputs.reshape(-1, inputs.shape[-1]),
            preactivation.reshape(-1, preactivation.shape[-1]),
            weight,
            credit.reshape(-1, credit.shape[-1]),
            relative_step=relative_step,
        )
        return jax.tree.map(lambda x: jax.lax.psum(x, BATCH_AXES), result)

    specs = (
        P(BATCH_AXES, None, None),
        P(BATCH_AXES, None, None),
        P(None, None),
        P(BATCH_AXES, None, None),
    )
    mapped = jax.shard_map(
        local,
        mesh=jax.sharding.get_abstract_mesh(),
        in_specs=specs,
        out_specs=Relu2ParameterCredit(P(None, None), P(None, None)),
        check_vma=False,
    )
    results = []
    for index, layer in enumerate(model.layers()):
        if any(
            x is not None
            for x in (
                layer.sconv_mlp_in,
                layer.out_norm_mlp,
                layer.mlp_out_scale,
                layer.moe_out_gate_w,
            )
        ):
            raise ValueError("Shared credit does not cover extra MLP transforms")
        stage = capture.stages[2 * index + 1]
        inputs = layer.mlp_gated_norm(layer.rms_mlp(stage.input))
        preactivation = capture.shared_preactivations[index].reshape(*inputs.shape[:-1], -1)
        credit = hub_credits[2 * index + 1]
        if layer.sconv_mlp is not None:
            credit = causal_convolution_credit(
                credit,
                jax.sharding.reshard(layer.sconv_mlp.weight, P(None, None)),
                capture.segment_ids,
            )
        arguments = (inputs, preactivation, layer.shared[0].w_down, credit)
        results.append(mapped(*(jax.sharding.reshard(a, s) for a, s in zip(arguments, specs, strict=True))))
    return tuple(results)


def _routed_hub_credit(layer: native.Block, credit: jax.Array, segment_ids: jax.Array | None) -> jax.Array:
    mlp = layer.mlp
    if layer.sconv_mlp is not None:
        credit = causal_convolution_credit(
            credit,
            jax.sharding.reshard(layer.sconv_mlp.weight, P(None, None)),
            segment_ids,
        )
    credit = credit.reshape(-1, credit.shape[-1])
    if mlp.w_latent_up is not None:
        credit = jnp.einsum(
            "td,ld->tl",
            credit,
            jax.sharding.reshard(mlp.w_latent_up, P(None, None)).astype(jnp.float32),
            out_sharding=P(BATCH_AXES, None),
        )
    return credit


def _routed_capacity(mlp: native.MoEMLP) -> tuple[float, float]:
    capacity = mlp.expert_mlp.capacity_factor
    transport_capacity = mlp.expert_mlp.pooled_transport_capacity_factor
    if jax.sharding.get_abstract_mesh().shape["expert"] == 1:
        # Native moe_mlp uses dropless local scatter without expert parallelism.
        capacity = float(mlp.cfg.num_experts)
        transport_capacity = 1.0
    return capacity, transport_capacity


def reconstruct_routed_experts(
    model: native.Transformer,
    capture: ForwardCapture,
    hub_credits: tuple[jax.Array, ...],
    *,
    relative_step: float,
) -> tuple[Relu2ParameterCredit, ...]:
    """Reconstruct routed expert matrices from independently measured MLP hubs.

    Transport credit through the fixed output convolution and latent expansion,
    then probe the expert ReLU² once. Routing, normalization, latent projection
    and depth-gate updates require independent probes and are excluded here.
    """
    results = []
    for index, layer in enumerate(model.layers()):
        mlp = layer.mlp
        if (
            mlp.cfg.moe_implementation != "fixed_pooled_wave_all_to_all"
            or mlp.cfg.moe_expert_waves != 1
            or mlp.cfg.moe_fp8_dispatch
            or not mlp.cfg.moe_drop_renorm
            or mlp.latent_out_norm is not None
            or mlp.expert_mlp.w_gate is not None
            or mlp.expert_mlp_b is not None
            or mlp.cfg.expert_leaky_slope != 0
        ):
            raise ValueError("Routed credit requires candidate 11's expert configuration")
        credit = _routed_hub_credit(layer, hub_credits[2 * index + 1], capture.segment_ids)
        capacity, transport_capacity = _routed_capacity(mlp)
        results.append(
            pooled_relu2_parameter_credit(
                capture.routed_inputs[index],
                capture.routing_selected[index],
                capture.routing_weights[index],
                mlp.expert_mlp.w_up,
                mlp.expert_mlp.w_down,
                credit,
                capacity_factor=capacity,
                transport_capacity_factor=transport_capacity,
                relative_step=relative_step,
            )
        )
    return tuple(results)


def reconstruct_router_matrices(
    model: native.Transformer,
    capture: ForwardCapture,
    hub_credits: tuple[jax.Array, ...],
    *,
    sigma: float,
) -> tuple[jax.Array, ...]:
    """Measure selected-logit responses and reconstruct each router matrix.

    This is conditional credit through candidate 11's sqrt-softplus combine.
    It neither explores alternative selected experts nor updates the separate
    quantile-balancing bias. Those biases use the native forward statistics.
    """
    results = []
    for index, layer in enumerate(model.layers()):
        mlp = layer.mlp
        if (
            mlp.cfg.router_combine != native.RouterCombine.SQRT_SOFTPLUS_RENORM
            or mlp.router is None
            or mlp.router_up is not None
            or mlp.router_mlp_a is not None
            or mlp.router_logit_scale is not None
            or mlp.cfg.router_logit_soft_cap is not None
            or mlp.expert_output_gain is not None
            or mlp.latent_out_norm is not None
            or mlp.expert_mlp.w_gate is not None
            or mlp.cfg.moe_ungated_activation != native.UngatedExpertActivation.RELU2
            or mlp.cfg.moe_implementation != "fixed_pooled_wave_all_to_all"
            or mlp.cfg.moe_expert_waves != 1
            or mlp.cfg.moe_fp8_dispatch
            or not mlp.cfg.moe_drop_renorm
        ):
            raise ValueError("Router credit requires candidate 11's combine and expert configuration")
        router_input = layer.mlp_gated_norm(layer.rms_mlp(capture.stages[2 * index + 1].input))
        credit = _routed_hub_credit(layer, hub_credits[2 * index + 1], capture.segment_ids)
        capacity, transport_capacity = _routed_capacity(mlp)
        results.append(
            pooled_router_parameter_credit(
                capture.routed_inputs[index],
                capture.routing_selected[index],
                capture.routing_logits[index],
                mlp.expert_mlp.w_up,
                mlp.expert_mlp.w_down,
                credit,
                router_input.reshape(-1, router_input.shape[-1]),
                capacity_factor=capacity,
                transport_capacity_factor=transport_capacity,
                sigma=sigma,
                renorm_sum=mlp.cfg.routing_renorm_sum,
            )
        )
    return tuple(results)


def reconstruct_latent_expansions(
    model: native.Transformer,
    capture: ForwardCapture,
    hub_credits: tuple[jax.Array, ...],
) -> tuple[jax.Array | None, ...]:
    """Reconstruct the linear expansion after each routed-expert combination.

    Replay native dispatch and drop renormalization with frozen routing. The
    expansion reads those clean activations and independently measured MLP hub
    credit, transported through the fixed output convolution. No nonlinear
    derivative or additional nonlinear credit transport is used. Candidate 11's
    final two layers write at full width directly and return None in their slots.
    """

    def local_matrix(inputs, credit):
        return jax.lax.psum(inputs.astype(jnp.float32).T @ credit.astype(jnp.float32), BATCH_AXES)

    spec = P(BATCH_AXES, None)
    matrix_credit = jax.shard_map(
        local_matrix,
        mesh=jax.sharding.get_abstract_mesh(),
        in_specs=(spec, spec),
        out_specs=P(None, None),
        check_vma=False,
    )
    results = []
    for index, layer in enumerate(model.layers()):
        mlp = layer.mlp
        if mlp.w_latent_up is None:
            results.append(None)
            continue
        if mlp.latent_out_norm is not None or mlp.expert_mlp_b is not None:
            raise ValueError("Latent expansion credit requires candidate 11's single expert bank")
        inputs = capture.routed_inputs[index]
        weights = capture.routing_weights[index]
        combined, overflow = native._run_expert_bank(
            mlp.expert_mlp,
            native._bank_config(mlp.cfg, 1),
            inputs,
            capture.routing_selected[index],
            weights.astype(inputs.dtype),
        )
        if overflow.assignment_keep is not None:
            factor, _ = native._drop_renorm_factor(weights, overflow.assignment_keep)
            combined = combined * factor[:, None].astype(combined.dtype)
        credit = hub_credits[2 * index + 1]
        if layer.sconv_mlp is not None:
            credit = causal_convolution_credit(
                credit,
                jax.sharding.reshard(layer.sconv_mlp.weight, P(None, None)),
                capture.segment_ids,
            )
        results.append(
            matrix_credit(
                jax.sharding.reshard(combined, spec),
                jax.sharding.reshard(credit.reshape(-1, credit.shape[-1]), spec),
            )
        )
    return tuple(results)


def reconstruct_mlp_convolutions(
    model: native.Transformer,
    capture: ForwardCapture,
    hub_credits: tuple[jax.Array, ...],
) -> tuple[jax.Array | None, ...]:
    """Reconstruct MLP output-convolution weights from independent hub credit.

    Replay each native MLP branch up to its convolution. Contract those inputs
    with the original measured output credit, restricting every lag to its
    packed document. This is linear parameter reconstruction without backprop.
    """
    segments = capture.segment_ids
    mask = AttentionMask(
        is_causal=True,
        segment_ids=None if segments is None else (segments, segments),
    )
    spec = P(BATCH_AXES, None, None)
    results = []
    for index, layer in enumerate(model.layers()):
        if layer.sconv_mlp is None:
            results.append(None)
            continue
        if layer.out_norm_mlp is not None or layer.mlp_out_scale is not None:
            raise ValueError("Convolution credit requires an untransformed branch output")
        kernel_size = layer.sconv_mlp.kernel_size
        before_convolution = eqx.tree_at(lambda block: block.sconv_mlp, layer, None)
        inputs, _ = before_convolution.mlp_branch(capture.stages[2 * index + 1].input, mask)

        def local(inputs, credit, segments, kernel_size=kernel_size):
            weights = causal_convolution_parameter_credit(inputs, credit, segments, kernel_size=kernel_size)
            return jax.lax.psum(weights, BATCH_AXES)

        mapped = jax.shard_map(
            local,
            mesh=jax.sharding.get_abstract_mesh(),
            in_specs=(spec, spec, None if segments is None else P(BATCH_AXES, None)),
            out_specs=P(None, None),
            check_vma=False,
        )
        result = mapped(
            jax.sharding.reshard(inputs, spec),
            jax.sharding.reshard(hub_credits[2 * index + 1], spec),
            segments,
        )
        results.append(jax.sharding.reshard(result, native._partition_spec_of(layer.sconv_mlp.weight)))
    return tuple(results)
