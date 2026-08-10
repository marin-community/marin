#!/usr/bin/env python3
# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Functional probe for the aggregate-scheduled MoK expert MLP."""

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from iris.runtime.jax_init import initialize_jax
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P

from levanter.grug._moe.mok_aggregate_mlp import aggregate_expert_mlp_forward, default_gemm_config
from levanter.grug._moe.mok_megakernel import backward_epilogue, dispatch_gate_up, dispatch_mlp_swiglu_combine_backward
from levanter.grug._moe.mok_schedule import EXPERT_PADDING, build_dispatch_schedule

AXIS = "ep"


def log(message: str) -> None:
    print(f"[mok-aggregate p{jax.process_index()}] {message}", flush=True)


def router_weight_windows(router_weights):
    flat = router_weights.reshape(-1)
    padded = jnp.pad(flat, (0, 3))
    return jnp.stack([padded[offset : offset + flat.shape[0]] for offset in range(4)], axis=1)


def dense_reference(x_all, routing, router_weights, w_gate, w_up, w_down, shared):
    world, tokens, hidden = x_all.shape
    flat_x = x_all.reshape(-1, hidden).astype(jnp.float32)
    flat_routing = routing.reshape(-1, routing.shape[-1])
    flat_weights = router_weights.reshape(-1, router_weights.shape[-1])
    out = jnp.zeros_like(flat_x)
    for expert in range(w_gate.shape[0]):
        gate = flat_x @ w_gate[expert].astype(jnp.float32)
        up = flat_x @ w_up[expert].astype(jnp.float32)
        value = (jax.nn.silu(gate) * up) @ w_down[expert].astype(jnp.float32)
        scale = jnp.sum(jnp.where(flat_routing == expert, flat_weights, 0.0), axis=-1)
        out += scale[:, None] * value
    shared_gate, shared_up, shared_down = shared
    shared_y = (jax.nn.silu(flat_x @ shared_gate) * (flat_x @ shared_up)) @ shared_down
    return (out + shared_y).reshape(world, tokens, hidden)


def staged_reference_loss(x, routing, router_weights, w_gate, w_up, w_down, shared, d_out):
    world, tokens, hidden = x.shape
    flat_x = x.reshape(-1, hidden)
    flat_routing = routing.reshape(-1, routing.shape[-1])
    flat_gate = w_gate.reshape(-1, hidden, w_gate.shape[-1])
    flat_up = w_up.reshape(-1, hidden, w_up.shape[-1])
    flat_down = w_down.reshape(-1, w_down.shape[-2], hidden)
    contributions = jnp.zeros((*flat_routing.shape, hidden), x.dtype)
    for expert in range(flat_gate.shape[0]):
        gate = (flat_x.astype(jnp.float32) @ flat_gate[expert].astype(jnp.float32)).astype(x.dtype)
        up = (flat_x.astype(jnp.float32) @ flat_up[expert].astype(jnp.float32)).astype(x.dtype)
        hidden_value = (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(x.dtype)
        value = (hidden_value.astype(jnp.float32) @ flat_down[expert].astype(jnp.float32)).astype(x.dtype)
        contributions = jnp.where(flat_routing[:, :, None] == expert, value[:, None, :], contributions)
    routed = (contributions * router_weights.reshape(-1, routing.shape[-1], 1).astype(x.dtype)).sum(axis=1)

    shared_gate, shared_up, shared_down = shared
    gate = jnp.einsum("wth,whi->wti", x.astype(jnp.float32), shared_gate.astype(jnp.float32)).astype(x.dtype)
    up = jnp.einsum("wth,whi->wti", x.astype(jnp.float32), shared_up.astype(jnp.float32)).astype(x.dtype)
    hidden_value = (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(x.dtype)
    shared_y = jnp.einsum("wti,wih->wth", hidden_value.astype(jnp.float32), shared_down.astype(jnp.float32)).astype(
        x.dtype
    )
    output = routed.reshape(world, tokens, hidden) + shared_y
    return jnp.sum(output.astype(jnp.float32) * d_out.astype(jnp.float32))


def benchmark(args, world) -> int:
    tokens = args.tokens_per_rank
    hidden = args.hidden
    intermediate = args.intermediate
    topk = args.topk
    experts_per_rank = args.experts_per_rank
    experts = world * experts_per_rank
    capacity = experts_per_rank * EXPERT_PADDING * args.capacity_multiple
    local_ranks = [device.id for device in jax.local_devices()]
    local_devices = len(local_ranks)
    local_rank_indices = {rank: index for index, rank in enumerate(local_ranks)}
    routing = np.empty((world, tokens, topk), dtype=np.int32)
    x_local = np.empty((local_devices * tokens, hidden), dtype=jnp.bfloat16)
    d_out_local = np.empty_like(x_local)
    router_weights_local = np.empty((local_devices * tokens, topk), dtype=np.float32)
    gate_local = np.empty((local_devices, experts_per_rank, hidden, intermediate), dtype=jnp.bfloat16)
    up_local = np.empty_like(gate_local)
    down_local = np.empty((local_devices, experts_per_rank, intermediate, hidden), dtype=jnp.bfloat16)
    shared_gate_local = np.empty((local_devices, hidden, intermediate), dtype=jnp.bfloat16)
    shared_up_local = np.empty_like(shared_gate_local)
    shared_down_local = np.empty((local_devices, intermediate, hidden), dtype=jnp.bfloat16)
    for rank in range(world):
        rng = np.random.default_rng(1234 + args.seed + rank)
        router_logits = rng.standard_normal((tokens, experts), dtype=np.float32)
        topk_experts = np.argpartition(router_logits, -topk, axis=1)[:, -topk:]
        routing[rank] = topk_experts
        if rank not in local_rank_indices:
            continue
        local_rank = local_rank_indices[rank]
        local_token_slice = slice(local_rank * tokens, (local_rank + 1) * tokens)
        topk_values = np.take_along_axis(router_logits, topk_experts, axis=1)
        topk_values -= topk_values.max(axis=1, keepdims=True)
        router_weights = np.exp(topk_values)
        router_weights_local[local_token_slice] = router_weights / router_weights.sum(axis=1, keepdims=True)
        x_local[local_token_slice] = rng.standard_normal((tokens, hidden), dtype=np.float32)
        shared_gate_local[local_rank] = rng.standard_normal((hidden, intermediate), dtype=np.float32) * hidden**-0.5
        shared_up_local[local_rank] = rng.standard_normal((hidden, intermediate), dtype=np.float32) * hidden**-0.5
        shared_down_local[local_rank] = (
            rng.standard_normal((intermediate, hidden), dtype=np.float32) * intermediate**-0.5
        )
        gate_local[local_rank] = (
            rng.standard_normal((experts_per_rank, hidden, intermediate), dtype=np.float32) * hidden**-0.5
        )
        up_local[local_rank] = (
            rng.standard_normal((experts_per_rank, hidden, intermediate), dtype=np.float32) * hidden**-0.5
        )
        down_local[local_rank] = (
            rng.standard_normal((experts_per_rank, intermediate, hidden), dtype=np.float32) * intermediate**-0.5
        )
        d_out_local[local_token_slice] = rng.standard_normal((tokens, hidden), dtype=np.float32) * hidden**-0.5
    routing_array = jnp.asarray(routing)
    schedules = {
        rank: build_dispatch_schedule(
            routing_array,
            num_local_experts=experts_per_rank,
            rank=rank,
            schedule_capacity=capacity,
        )
        for rank in local_ranks
    }
    peer_rank_local = np.stack([np.asarray(schedules[rank][0]) for rank in local_ranks])
    if any(int(schedules[rank][2]) > capacity for rank in local_ranks):
        raise ValueError(f"{capacity=} is smaller than the padded route count")
    peer_token_local = np.stack([np.asarray(schedules[rank][1]) for rank in local_ranks])
    num_routed_local = np.stack([np.asarray(schedules[rank][2]) for rank in local_ranks])
    group_sizes_local = np.stack([np.asarray(schedules[rank][3]) for rank in local_ranks])
    active_rows = np.asarray(multihost_utils.process_allgather(num_routed_local)).reshape(-1)
    log(f"capacity={capacity} active_rows={active_rows.tolist()}")

    mesh = jax.make_mesh((world,), (AXIS,), axis_types=(jax.sharding.AxisType.Explicit,))
    jax.set_mesh(mesh)

    def from_local(array, spec, global_shape):
        return jax.make_array_from_process_local_data(
            jax.sharding.NamedSharding(mesh, spec),
            array,
            global_shape=global_shape,
        )

    x = from_local(x_local, P(AXIS, None), (world * tokens, hidden))
    d_out = from_local(d_out_local, P(AXIS, None), (world * tokens, hidden))
    router_weights = from_local(router_weights_local, P(AXIS, None), (world * tokens, topk))
    peer_rank = from_local(peer_rank_local, P(AXIS, None), (world, capacity))
    peer_token = from_local(peer_token_local, P(AXIS, None), (world, capacity))
    num_routed = from_local(num_routed_local, P(AXIS), (world,))
    group_sizes = from_local(group_sizes_local, P(AXIS, None), (world, experts_per_rank))
    gate = from_local(
        gate_local,
        P(AXIS, None, None, None),
        (world, experts_per_rank, hidden, intermediate),
    )
    up = from_local(
        up_local,
        P(AXIS, None, None, None),
        (world, experts_per_rank, hidden, intermediate),
    )
    down = from_local(
        down_local,
        P(AXIS, None, None, None),
        (world, experts_per_rank, intermediate, hidden),
    )
    shared_gate = from_local(shared_gate_local, P(AXIS, None, None), (world, hidden, intermediate))
    shared_up = from_local(shared_up_local, P(AXIS, None, None), (world, hidden, intermediate))
    shared_down = from_local(shared_down_local, P(AXIS, None, None), (world, intermediate, hidden))

    def forward_body(x_s, _d_out_s, weights_s, rank_s, token_s, num_s, groups_s, gate_s, up_s, down_s, sg, su, sd):
        return aggregate_expert_mlp_forward(
            x_s,
            weights_s,
            gate_s[0],
            up_s[0],
            down_s[0],
            (sg[0], su[0], sd[0]),
            rank_s[0],
            token_s[0],
            num_s[0],
            groups_s[0],
            axis_name=AXIS,
            block_rows=args.block_rows,
            combine_block_rows=args.combine_block_rows,
            num_comm_sms=args.num_comm_sms,
            minibatch_size=args.minibatch_size,
        )

    def forward_context_body(
        x_s, _d_out_s, weights_s, rank_s, token_s, num_s, groups_s, gate_s, up_s, down_s, sg, su, sd
    ):
        shared_weights = (sg[0], su[0], sd[0])
        forward = dispatch_gate_up(
            x_s,
            gate_s[0],
            up_s[0],
            rank_s[0],
            token_s[0],
            num_s[0],
            groups_s[0],
            axis_name=AXIS,
            topk=topk,
            num_comm_sms=args.forward_num_comm_sms,
            minibatch_size=args.minibatch_size,
            gemm_config=default_gemm_config(),
            w_down=down_s[0],
            shared=shared_weights,
        )
        return forward[0], forward[1], forward[2], forward[3], forward[6], forward[7], forward[8]

    def backward_body(
        x_s,
        d_out_s,
        weights_s,
        rank_s,
        token_s,
        num_s,
        groups_s,
        gate_s,
        up_s,
        down_s,
        sg,
        su,
        sd,
        x_routed,
        gate_routed,
        up_routed,
        hidden_routed,
        gate_shared,
        up_shared,
        hidden_shared,
    ):
        shared_weights = (sg[0], su[0], sd[0])
        backward = dispatch_mlp_swiglu_combine_backward(
            router_weight_windows(weights_s),
            d_out_s,
            x_s,
            x_routed,
            gate_routed,
            up_routed,
            hidden_routed,
            gate_shared,
            up_shared,
            hidden_shared,
            gate_s[0],
            up_s[0],
            down_s[0],
            shared_weights,
            rank_s[0],
            token_s[0],
            num_s[0],
            groups_s[0],
            axis_name=AXIS,
            topk=topk,
            num_comm_sms=args.num_comm_sms,
            minibatch_size=args.minibatch_size,
            gemm_config=default_gemm_config(),
        )
        d_x = backward_epilogue(backward[9], backward[5][:-1], topk=topk, axis_name=AXIS)
        d_router = backward[18][:-1].reshape(tokens, topk)
        return d_x, d_router, *backward[10:16]

    out_specs = (
        (
            P(AXIS, None),
            P(AXIS, None),
            P(AXIS, None, None),
            P(AXIS, None, None),
            P(AXIS, None, None),
            P(AXIS, None),
            P(AXIS, None),
            P(AXIS, None),
        )
        if args.backward
        else P(AXIS, None)
    )
    input_specs = (
        P(AXIS, None),
        P(AXIS, None),
        P(AXIS, None),
        P(AXIS, None),
        P(AXIS, None),
        P(AXIS),
        P(AXIS, None),
        P(AXIS, None, None, None),
        P(AXIS, None, None, None),
        P(AXIS, None, None, None),
        P(AXIS, None, None),
        P(AXIS, None, None),
        P(AXIS, None, None),
    )
    inputs = (
        x,
        d_out,
        router_weights,
        peer_rank,
        peer_token,
        num_routed,
        group_sizes,
        gate,
        up,
        down,
        shared_gate,
        shared_up,
        shared_down,
    )
    if args.backward:
        run_forward = jax.jit(
            jax.shard_map(
                forward_context_body,
                in_specs=input_specs,
                out_specs=(P(AXIS, None),) * 7,
                check_vma=False,
            )
        )
        run_backward = jax.jit(
            jax.shard_map(
                backward_body,
                in_specs=input_specs + (P(AXIS, None),) * 7,
                out_specs=out_specs,
                check_vma=False,
            )
        )
        context = run_forward(*inputs)
        jax.block_until_ready(context)
        jax.block_until_ready(run_backward(*inputs, *context))
        log("aggregate forward_backward complete")
        for _ in range(args.warmup_iterations):
            context = run_forward(*inputs)
            jax.block_until_ready(context)
            jax.block_until_ready(run_backward(*inputs, *context))
        stage = "backward"
    else:
        run_forward = jax.jit(
            jax.shard_map(
                forward_body,
                in_specs=input_specs,
                out_specs=out_specs,
                check_vma=False,
            )
        )
        jax.block_until_ready(run_forward(*inputs))
        log("aggregate forward complete")
        for _ in range(args.warmup_iterations):
            jax.block_until_ready(run_forward(*inputs))
        stage = "forward"
    multihost_utils.sync_global_devices(f"mok-aggregate-{stage}-benchmark")
    samples = []
    for _ in range(args.timed_iterations):
        if args.backward:
            context = run_forward(*inputs)
            jax.block_until_ready(context)
        start = time.perf_counter()
        if args.backward:
            jax.block_until_ready(run_backward(*inputs, *context))
        else:
            jax.block_until_ready(run_forward(*inputs))
        samples.append((time.perf_counter() - start) * 1e3)
    if samples:
        all_samples = np.asarray(multihost_utils.process_allgather(np.asarray(samples)))
        if jax.process_index() == 0:
            milliseconds = float(np.median(all_samples.max(axis=0)))
            log(f"benchmark aggregate_{stage}={milliseconds:.3f} ms")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-rank", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--capacity-multiple", type=int, default=8)
    parser.add_argument("--block-rows", type=int, default=128)
    parser.add_argument("--combine-block-rows", type=int, default=16)
    parser.add_argument("--num-comm-sms", type=int)
    parser.add_argument("--forward-num-comm-sms", type=int)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--dispatch-only", action="store_true")
    parser.add_argument("--zero-routed", action="store_true")
    parser.add_argument("--skip-dispatch-gather", action="store_true")
    parser.add_argument("--skip-dispatch-store", action="store_true")
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--gate-up-only", action="store_true")
    parser.add_argument("--shared-only", action="store_true")
    parser.add_argument("--routed-only", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--timed-iterations", type=int, default=0)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.forward_num_comm_sms is None:
        args.forward_num_comm_sms = args.num_comm_sms

    initialize_jax()
    world = jax.device_count()
    log(f"processes={jax.process_count()} devices={world} kind={jax.devices()[0].device_kind}")
    if args.skip_correctness:
        return benchmark(args, world)

    tokens = args.tokens_per_rank
    hidden = args.hidden
    intermediate = args.intermediate
    topk = args.topk
    experts = world * args.experts_per_rank
    capacity = args.experts_per_rank * EXPERT_PADDING * args.capacity_multiple
    rng = np.random.default_rng(args.seed)

    routing = rng.integers(0, experts, size=(world, tokens, topk), dtype=np.int32)
    router_weights = rng.random((world, tokens, topk), dtype=np.float32)
    x_all = (rng.standard_normal((world, tokens, hidden)) * 0.1).astype(np.float32)
    w_gate = (rng.standard_normal((experts, hidden, intermediate)) * 0.05).astype(np.float32)
    w_up = (rng.standard_normal((experts, hidden, intermediate)) * 0.05).astype(np.float32)
    w_down = (rng.standard_normal((experts, intermediate, hidden)) * 0.05).astype(np.float32)
    shared = (
        (rng.standard_normal((hidden, intermediate)) * 0.05).astype(np.float32),
        (rng.standard_normal((hidden, intermediate)) * 0.05).astype(np.float32),
        (rng.standard_normal((intermediate, hidden)) * 0.05).astype(np.float32),
    )

    schedules = [
        build_dispatch_schedule(
            jnp.asarray(routing),
            num_local_experts=args.experts_per_rank,
            rank=rank,
            schedule_capacity=capacity,
        )
        for rank in range(world)
    ]
    if any(int(schedule[2]) > capacity for schedule in schedules):
        raise ValueError(f"{capacity=} is smaller than the padded route count")
    log(f"capacity={capacity} active_rows={[int(schedule[2]) for schedule in schedules]}")

    mesh = jax.make_mesh((world,), (AXIS,), axis_types=(jax.sharding.AxisType.Explicit,))
    jax.set_mesh(mesh)
    reshard = jax.sharding.reshard

    x = reshard(jnp.asarray(x_all.reshape(world * tokens, hidden), jnp.bfloat16), P(AXIS, None))
    weights = reshard(jnp.asarray(router_weights.reshape(world * tokens, topk)), P(AXIS, None))
    peer_rank = reshard(jnp.stack([schedule[0] for schedule in schedules]), P(AXIS, None))
    peer_token = reshard(jnp.stack([schedule[1] for schedule in schedules]), P(AXIS, None))
    num_routed = reshard(jnp.stack([schedule[2] for schedule in schedules]), P(AXIS))
    group_sizes = reshard(jnp.stack([schedule[3] for schedule in schedules]), P(AXIS, None))
    gate = reshard(
        jnp.asarray(w_gate.reshape(world, args.experts_per_rank, hidden, intermediate), jnp.bfloat16),
        P(AXIS, None, None, None),
    )
    up = reshard(
        jnp.asarray(w_up.reshape(world, args.experts_per_rank, hidden, intermediate), jnp.bfloat16),
        P(AXIS, None, None, None),
    )
    down = reshard(
        jnp.asarray(w_down.reshape(world, args.experts_per_rank, intermediate, hidden), jnp.bfloat16),
        P(AXIS, None, None, None),
    )
    shared_sharded = tuple(
        reshard(
            jnp.asarray(np.broadcast_to(weight, (world,) + weight.shape).copy(), jnp.bfloat16), P(AXIS, None, None)
        )
        for weight in shared
    )

    if args.backward:
        if args.num_comm_sms is None:
            raise ValueError("--backward requires --num-comm-sms")

        def backward_body(x_s, weights_s, gate_s, up_s, down_s, sg, su, sd, rank_s, token_s, num_s, groups_s):
            shared_weights = (sg[0], su[0], sd[0])
            forward = dispatch_gate_up(
                x_s,
                gate_s[0],
                up_s[0],
                rank_s[0],
                token_s[0],
                num_s[0],
                groups_s[0],
                axis_name=AXIS,
                topk=topk,
                num_comm_sms=args.num_comm_sms,
                minibatch_size=args.minibatch_size,
                gemm_config=default_gemm_config(),
                w_down=down_s[0],
                shared=shared_weights,
            )
            d_out = x_s
            backward = dispatch_mlp_swiglu_combine_backward(
                router_weight_windows(weights_s),
                d_out,
                x_s,
                forward[0],
                forward[1],
                forward[2],
                forward[3],
                forward[6],
                forward[7],
                forward[8],
                gate_s[0],
                up_s[0],
                down_s[0],
                shared_weights,
                rank_s[0],
                token_s[0],
                num_s[0],
                groups_s[0],
                axis_name=AXIS,
                topk=topk,
                num_comm_sms=args.num_comm_sms,
                minibatch_size=args.minibatch_size,
                gemm_config=default_gemm_config(),
            )
            d_x = backward_epilogue(backward[9], backward[5][:-1], topk=topk, axis_name=AXIS)
            d_router = backward[18][:-1].reshape(tokens, topk)
            nonempty = groups_s[0][:, None, None] > 0
            return (
                d_x,
                d_router,
                jnp.where(nonempty, backward[10], 0),
                jnp.where(nonempty, backward[11], 0),
                jnp.where(nonempty, backward[12], 0),
                backward[13],
                backward[14],
                backward[15],
            )

        backward_outputs = jax.jit(
            jax.shard_map(
                backward_body,
                in_specs=(
                    P(AXIS, None),
                    P(AXIS, None),
                    P(AXIS, None, None, None),
                    P(AXIS, None, None, None),
                    P(AXIS, None, None, None),
                    P(AXIS, None, None),
                    P(AXIS, None, None),
                    P(AXIS, None, None),
                    P(AXIS, None),
                    P(AXIS, None),
                    P(AXIS),
                    P(AXIS, None),
                ),
                out_specs=(
                    P(AXIS, None),
                    P(AXIS, None),
                    P(AXIS, None, None),
                    P(AXIS, None, None),
                    P(AXIS, None, None),
                    P(AXIS, None),
                    P(AXIS, None),
                    P(AXIS, None),
                ),
                check_vma=False,
            )
        )(x, weights, gate, up, down, *shared_sharded, peer_rank, peer_token, num_routed, group_sizes)
        jax.block_until_ready(backward_outputs)

        reference_inputs = (
            jnp.asarray(x_all, jnp.bfloat16),
            jnp.asarray(router_weights),
            jnp.asarray(w_gate.reshape(world, args.experts_per_rank, hidden, intermediate), jnp.bfloat16),
            jnp.asarray(w_up.reshape(world, args.experts_per_rank, hidden, intermediate), jnp.bfloat16),
            jnp.asarray(w_down.reshape(world, args.experts_per_rank, intermediate, hidden), jnp.bfloat16),
            *(jnp.asarray(np.broadcast_to(weight, (world,) + weight.shape).copy(), jnp.bfloat16) for weight in shared),
        )
        reference_grad = jax.jit(
            jax.grad(
                lambda x_ref, router_ref, gate_ref, up_ref, down_ref, sg_ref, su_ref, sd_ref: staged_reference_loss(
                    x_ref,
                    jnp.asarray(routing),
                    router_ref,
                    gate_ref,
                    up_ref,
                    down_ref,
                    (sg_ref, su_ref, sd_ref),
                    jnp.asarray(x_all, jnp.bfloat16),
                ),
                argnums=tuple(range(8)),
            )
        )(*reference_inputs)
        expected = (
            reference_grad[0].reshape(world * tokens, hidden),
            reference_grad[1].reshape(world * tokens, topk),
            reference_grad[2].reshape(world * args.experts_per_rank, hidden, intermediate),
            reference_grad[3].reshape(world * args.experts_per_rank, hidden, intermediate),
            reference_grad[4].reshape(world * args.experts_per_rank, intermediate, hidden),
            reference_grad[5].reshape(world * hidden, intermediate),
            reference_grad[6].reshape(world * hidden, intermediate),
            reference_grad[7].reshape(world * intermediate, hidden),
        )
        names = (
            "d_x",
            "d_router",
            "d_w_gate",
            "d_w_up",
            "d_w_down",
            "d_w_shared_gate",
            "d_w_shared_up",
            "d_w_shared_down",
        )
        failed = False
        for name, actual, wanted in zip(names, backward_outputs, expected, strict=True):
            actual_value = np.asarray(actual, np.float32)
            wanted_value = np.asarray(wanted, np.float32)
            relative_error = np.linalg.norm(actual_value - wanted_value) / max(np.linalg.norm(wanted_value), 1e-30)
            log(f"backward_{name}_relative_norm_err={relative_error:.5g}")
            failed |= not np.isfinite(relative_error) or relative_error > 2e-2
        if failed:
            log("FAIL: persistent MoK backward does not match the staged reference")
            return 1
        log("PASS: persistent MoK forward/backward matches the staged reference")
        return 0

    if args.shared_only or args.routed_only:
        if args.num_comm_sms is None:
            raise ValueError("--shared-only and --routed-only require --num-comm-sms")

        def shared_body(x_s, weights_s, gate_s, up_s, down_s, sg, su, sd, rank_s, token_s, num_s, groups_s):
            outputs = dispatch_gate_up(
                x_s,
                gate_s[0],
                up_s[0],
                rank_s[0],
                token_s[0],
                num_s[0],
                groups_s[0],
                axis_name=AXIS,
                topk=topk,
                num_comm_sms=args.num_comm_sms,
                minibatch_size=args.minibatch_size,
                gemm_config=default_gemm_config(),
                w_down=down_s[0],
                shared=(sg[0], su[0], sd[0]) if args.shared_only else None,
            )
            routed = outputs[5][:-1].reshape(tokens, topk, hidden)
            routed = (routed * weights_s[:, :, None].astype(routed.dtype)).sum(axis=1)
            stages = outputs[6:] if args.shared_only else outputs[1:5]
            return *stages, routed, outputs[5][:-1]

        shared_outputs = jax.jit(
            jax.shard_map(
                shared_body,
                in_specs=(
                    P(AXIS, None),
                    P(AXIS, None),
                    P(AXIS, None, None, None),
                    P(AXIS, None, None, None),
                    P(AXIS, None, None, None),
                    P(AXIS, None, None),
                    P(AXIS, None, None),
                    P(AXIS, None, None),
                    P(AXIS, None),
                    P(AXIS, None),
                    P(AXIS),
                    P(AXIS, None),
                ),
                out_specs=(P(AXIS, None),) * 6,
                check_vma=False,
            )
        )(x, weights, gate, up, down, *shared_sharded, peer_rank, peer_token, num_routed, group_sizes)
        x_bf16 = jnp.asarray(x_all.reshape(world * tokens, hidden), jnp.bfloat16)
        gate_w = jnp.asarray(shared[0], jnp.bfloat16)
        up_w = jnp.asarray(shared[1], jnp.bfloat16)
        down_w = jnp.asarray(shared[2], jnp.bfloat16)
        want_gate = (x_bf16.astype(jnp.float32) @ gate_w.astype(jnp.float32)).astype(jnp.bfloat16)
        want_up = (x_bf16.astype(jnp.float32) @ up_w.astype(jnp.float32)).astype(jnp.bfloat16)
        want_hidden = (jax.nn.silu(want_gate.astype(jnp.float32)) * want_up.astype(jnp.float32)).astype(jnp.bfloat16)
        want_shared = (want_hidden.astype(jnp.float32) @ down_w.astype(jnp.float32)).astype(jnp.bfloat16)
        if args.shared_only:
            for name, output, expected in zip(
                ("gate", "up", "hidden", "down"),
                shared_outputs[:4],
                (want_gate, want_up, want_hidden, want_shared),
                strict=True,
            ):
                got_value = np.asarray(output, np.float32)
                want_value = np.asarray(expected, np.float32)
                relative_error = np.linalg.norm(got_value - want_value) / max(np.linalg.norm(want_value), 1e-30)
                log(f"shared_{name}_relative_norm_err={relative_error:.5g}")
        else:
            expected_stages = [[] for _ in range(4)]
            x_bf16_all = np.asarray(jnp.asarray(x_all, jnp.bfloat16))
            gate_bf16 = np.asarray(jnp.asarray(w_gate, jnp.bfloat16))
            up_bf16 = np.asarray(jnp.asarray(w_up, jnp.bfloat16))
            down_bf16 = np.asarray(jnp.asarray(w_down, jnp.bfloat16))
            for rank, schedule in enumerate(schedules):
                owners = np.asarray(schedule[0])
                peer_tokens = np.asarray(schedule[1])
                group_lengths = np.asarray(schedule[3])
                valid = owners >= 0
                received = np.zeros((capacity, hidden), dtype=jnp.bfloat16)
                received[valid] = x_bf16_all[owners[valid], peer_tokens[valid] // topk]
                rank_gate = np.zeros((capacity, intermediate), dtype=jnp.bfloat16)
                rank_up = np.zeros_like(rank_gate)
                rank_hidden = np.zeros_like(rank_gate)
                rank_down = np.zeros((capacity, hidden), dtype=jnp.bfloat16)
                offset = 0
                for local_expert, length in enumerate(group_lengths):
                    row_slice = slice(offset, offset + int(length))
                    expert = rank * args.experts_per_rank + local_expert
                    rank_gate[row_slice] = np.asarray(
                        jnp.asarray(received[row_slice]).astype(jnp.float32)
                        @ jnp.asarray(gate_bf16[expert]).astype(jnp.float32),
                        jnp.bfloat16,
                    )
                    rank_up[row_slice] = np.asarray(
                        jnp.asarray(received[row_slice]).astype(jnp.float32)
                        @ jnp.asarray(up_bf16[expert]).astype(jnp.float32),
                        jnp.bfloat16,
                    )
                    rank_hidden[row_slice] = np.asarray(
                        jax.nn.silu(jnp.asarray(rank_gate[row_slice]).astype(jnp.float32))
                        * jnp.asarray(rank_up[row_slice]).astype(jnp.float32),
                        jnp.bfloat16,
                    )
                    rank_down[row_slice] = np.asarray(
                        jnp.asarray(rank_hidden[row_slice]).astype(jnp.float32)
                        @ jnp.asarray(down_bf16[expert]).astype(jnp.float32),
                        jnp.bfloat16,
                    )
                    offset += int(length)
                for values, expected in zip(
                    (rank_gate, rank_up, rank_hidden, rank_down),
                    expected_stages,
                    strict=True,
                ):
                    expected.append(values)
            for name, output, expected in zip(
                ("gate", "up", "hidden", "down"),
                shared_outputs[:4],
                expected_stages,
                strict=True,
            ):
                got_value = np.asarray(output, np.float32).reshape(world, capacity, -1)
                want_value = np.asarray(expected, np.float32)
                active = np.stack([np.arange(capacity) < int(schedule[2]) for schedule in schedules])
                got_value = got_value[active]
                want_value = want_value[active]
                relative_error = np.linalg.norm(got_value - want_value) / max(np.linalg.norm(want_value), 1e-30)
                log(f"routed_{name}_relative_norm_err={relative_error:.5g}")
            expected_contributions = np.zeros((world, tokens * topk, hidden), dtype=jnp.bfloat16)
            for source_rank, schedule in enumerate(schedules):
                owners = np.asarray(schedule[0])
                peer_tokens = np.asarray(schedule[1])
                valid = owners >= 0
                expected_contributions[owners[valid], peer_tokens[valid]] = expected_stages[3][source_rank][valid]
            got_contributions = np.asarray(shared_outputs[5], np.float32).reshape(world, tokens * topk, hidden)
            want_contributions = np.asarray(expected_contributions, np.float32)
            relative_error = np.linalg.norm(got_contributions - want_contributions) / max(
                np.linalg.norm(want_contributions), 1e-30
            )
            bad_rows = np.any(np.abs(got_contributions - want_contributions) > 1e-4, axis=-1)
            log(f"combine_relative_norm_err={relative_error:.5g} bad_rows={bad_rows.sum(axis=1).tolist()}")
        zero_shared = tuple(jnp.zeros_like(jnp.asarray(weight, jnp.bfloat16)) for weight in shared)
        want_routed = dense_reference(
            jnp.asarray(x_all, jnp.bfloat16),
            jnp.asarray(routing),
            jnp.asarray(router_weights),
            jnp.asarray(w_gate, jnp.bfloat16),
            jnp.asarray(w_up, jnp.bfloat16),
            jnp.asarray(w_down, jnp.bfloat16),
            zero_shared,
        ).reshape(world * tokens, hidden)
        got_routed = np.asarray(shared_outputs[4], np.float32)
        want_routed = np.asarray(want_routed, np.float32)
        relative_error = np.linalg.norm(got_routed - want_routed) / max(np.linalg.norm(want_routed), 1e-30)
        log(f"routed_relative_norm_err={relative_error:.5g}")
        return 0

    if args.dispatch_only or args.gate_only or args.gate_up_only:
        if args.num_comm_sms is None:
            raise ValueError("--dispatch-only requires --num-comm-sms")

        def dispatch_body(x_s, gate_s, up_s, rank_s, token_s, num_s, groups_s):
            outputs = dispatch_gate_up(
                x_s,
                gate_s[0],
                up_s[0],
                rank_s[0],
                token_s[0],
                jnp.int32(0) if args.zero_routed else num_s[0],
                groups_s[0],
                axis_name=AXIS,
                topk=topk,
                num_comm_sms=args.num_comm_sms,
                minibatch_size=args.minibatch_size,
                gemm_config=default_gemm_config(),
                run_compute=args.gate_only or args.gate_up_only,
                run_up=args.gate_up_only,
                run_dispatch_gather=not args.skip_dispatch_gather,
                run_dispatch_store=not args.skip_dispatch_store,
            )
            if args.gate_up_only:
                return outputs[1], outputs[2]
            return outputs[1 if args.gate_only else 0]

        dispatched = jax.jit(
            jax.shard_map(
                dispatch_body,
                in_specs=(
                    P(AXIS, None),
                    P(AXIS, None, None, None),
                    P(AXIS, None, None, None),
                    P(AXIS, None),
                    P(AXIS, None),
                    P(AXIS),
                    P(AXIS, None),
                ),
                out_specs=(P(AXIS, None), P(AXIS, None)) if args.gate_up_only else P(AXIS, None),
                check_vma=False,
            )
        )(x, gate, up, peer_rank, peer_token, num_routed, group_sizes)
        jax.block_until_ready(dispatched)
        if args.dispatch_only:
            x_bf16 = x_all.astype(jnp.bfloat16)
            for shard in dispatched.addressable_shards:
                row_slice = shard.index[0]
                rank = row_slice.start // capacity
                owners = np.asarray(schedules[rank][0])
                tokens = np.asarray(schedules[rank][1])
                valid = owners >= 0
                expected = np.zeros((capacity, hidden), dtype=jnp.bfloat16)
                expected[valid] = x_bf16[owners[valid], tokens[valid] // topk]
                np.testing.assert_array_equal(np.asarray(shard.data), expected)
        stage = "dispatch/gate/up" if args.gate_up_only else "dispatch/gate" if args.gate_only else "dispatch"
        log(f"PASS: persistent {stage} stage completed")
        return 0

    def shard_body(x_s, weights_s, rank_s, token_s, num_s, groups_s, gate_s, up_s, down_s, sg, su, sd):
        return aggregate_expert_mlp_forward(
            x_s,
            weights_s,
            gate_s[0],
            up_s[0],
            down_s[0],
            (sg[0], su[0], sd[0]),
            rank_s[0],
            token_s[0],
            num_s[0],
            groups_s[0],
            axis_name=AXIS,
            block_rows=args.block_rows,
            combine_block_rows=args.combine_block_rows,
            num_comm_sms=args.num_comm_sms,
            minibatch_size=args.minibatch_size,
        )

    got = jax.jit(
        jax.shard_map(
            shard_body,
            in_specs=(
                P(AXIS, None),
                P(AXIS, None),
                P(AXIS, None),
                P(AXIS, None),
                P(AXIS),
                P(AXIS, None),
                P(AXIS, None, None, None),
                P(AXIS, None, None, None),
                P(AXIS, None, None, None),
                P(AXIS, None, None),
                P(AXIS, None, None),
                P(AXIS, None, None),
            ),
            out_specs=P(AXIS, None),
            check_vma=False,
        )
    )(x, weights, peer_rank, peer_token, num_routed, group_sizes, gate, up, down, *shared_sharded)

    want = np.asarray(
        dense_reference(
            jnp.asarray(x_all, jnp.bfloat16),
            jnp.asarray(routing),
            jnp.asarray(router_weights),
            jnp.asarray(w_gate, jnp.bfloat16),
            jnp.asarray(w_up, jnp.bfloat16),
            jnp.asarray(w_down, jnp.bfloat16),
            tuple(jnp.asarray(weight, jnp.bfloat16).astype(jnp.float32) for weight in shared),
        ),
        np.float32,
    )
    squared_error = 0.0
    squared_norm = 0.0
    for shard in got.addressable_shards:
        mine = np.asarray(shard.data, np.float32)
        expected = want.reshape(got.shape)[shard.index]
        squared_error += float(np.sum((mine - expected) ** 2))
        squared_norm += float(np.sum(expected**2))

    local = np.asarray([squared_error, squared_norm], np.float64)
    totals = np.asarray(multihost_utils.process_allgather(local)).sum(axis=0)
    relative_error = totals[0] ** 0.5 / max(totals[1] ** 0.5, 1e-30)
    log(f"relative_norm_err={relative_error:.5g}")
    if not np.isfinite(relative_error) or relative_error > 2e-2:
        log("FAIL: aggregate expert MLP does not match the dense reference")
        return 1

    log(f"PASS: aggregate expert MLP matches dense reference across {world} peers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
