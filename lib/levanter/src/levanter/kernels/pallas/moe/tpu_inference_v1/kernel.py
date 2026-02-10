# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU-Friendly Fused Mixture of Experts (MoE) kernel."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .tuned_block_sizes import get_tuned_block_sizes

P = jax.sharding.PartitionSpec

cdiv = pl.cdiv


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_packing(dtype):
    bits = dtypes.bit_width(dtype) if hasattr(dtypes, "bit_width") else dtypes.itemsize_bits(dtype)
    return 32 // bits


def broadcast_minor(src, shape):
    if src.shape == shape:
        return src
    assert src.shape[:-1] == shape[:-1]
    assert src.shape[-1] % 128 == 0
    target_minor = align_to(shape[-1], src.shape[-1])
    # no-op concatenation.
    return jnp.concatenate([src for _ in range(target_minor // src.shape[-1])], axis=-1)[..., : shape[-1]]


def apply_scoring_fn(scoring_fn: str, x):
    match scoring_fn:
        case "softmax":
            return jax.nn.softmax(x, axis=-1)
        case "sigmoid":
            # TODO(catswe): use jax.nn.sigmoid once mosaic lowering bug with bf16 input is fixed
            return 1 / (1 + jnp.exp(-x))
        case _:
            raise NotImplementedError(f"Unsupported scoring function: {scoring_fn}")


def swigluoai(gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7.0) -> jax.Array:
    """Activation used in some models such as GPT-OSS."""
    gate = jnp.clip(gate, a_max=limit)
    up = jnp.clip(up, a_min=-limit, a_max=limit)
    glu = gate * jax.nn.sigmoid(alpha * gate)
    return (up + 1.0) * glu


def apply_act_fn(acc1, acc3, act_fn):
    if act_fn == "silu":
        return jax.nn.silu(acc1) * acc3
    elif act_fn == "gelu":
        return jax.nn.gelu(acc1) * acc3
    elif act_fn == "swigluoai":
        return swigluoai(acc1, acc3)
    else:
        raise NotImplementedError(f"Unsupported activation function: {act_fn}")


def ref_moe(
    tokens: jax.Array,  # (num_tokens, hidden_size)
    w1: jax.Array,  # (num_experts, 2, hidden_size, intermediate_size)
    w2: jax.Array,  # (num_experts, intermediate_size, hidden_size)
    gating_output: jax.Array,  # (num_tokens, num_experts)
    top_k: int,
    *,
    renormalize_topk_logits: bool = False,
    act_fn: str = "silu",
    scoring_fn: str = "softmax",
    subc_quant_w1_sz: int | None = None,
    subc_quant_w2_sz: int | None = None,
    w1_scale: jax.Array | None = None,  # F32(num_experts, 2, hidden_size //subc_quant_w1_sz, 1, intermediate_size)
    w2_scale: jax.Array | None = None,  # F32(num_experts, intermediate_size // subc_quant_w2_sz, 1, hidden_size)
    b1: jax.Array | None = None,  # F32(num_experts, 2, 1, intermediate_size)
    b2: jax.Array | None = None,  # F32(num_experts, 1, hidden_size)
):
    n_tokens = tokens.shape[0]  # num_tokens

    # Compute gating scores for all experts
    gating_logits = apply_scoring_fn(scoring_fn, gating_output)  # [num_tokens, n_experts]

    # Select top-k experts per token
    top_k_logits, top_k_indices = lax.top_k(gating_logits, top_k)  # [num_tokens, top_k], [num_tokens, top_k]

    if renormalize_topk_logits:
        top_k_logits = top_k_logits / jnp.sum(top_k_logits, axis=-1, keepdims=True)

    t_outputs = []
    hidden_size, intermediate_size = w1.shape[-2:]

    # Process each token individually
    for i in range(n_tokens):
        curr_token = jnp.expand_dims(tokens[i], axis=0)  # [1, hidden_size]
        assigned_expert_ids = top_k_indices[i]  # [top_k] - indices of selected experts for token i
        tok_expert_act = []

        # Process each selected expert for the current token
        for expert_id in assigned_expert_ids:
            # Get expert weights
            expert_w1 = w1[expert_id, 0].astype(jnp.float32)
            expert_w3 = w1[expert_id, 1].astype(jnp.float32)
            if w1_scale is not None:
                assert subc_quant_w1_sz is not None
                expert_w1 *= jnp.repeat(w1_scale[expert_id, 0, :, 0], subc_quant_w1_sz, axis=0)[:hidden_size]
                expert_w3 *= jnp.repeat(w1_scale[expert_id, 1, :, 0], subc_quant_w1_sz, axis=0)[:hidden_size]
            expert_weight_1 = jnp.concat([expert_w1, expert_w3], axis=-1)  # [hidden_size, 2 * intermediate_size]
            expert_weight_2 = w2[expert_id].astype(jnp.float32)  # [intermediate_size, hidden_size]
            if w2_scale is not None:
                assert subc_quant_w2_sz is not None
                expert_weight_2 *= jnp.repeat(w2_scale[expert_id, :, 0], subc_quant_w2_sz, axis=0)[:intermediate_size]

            # First linear layer with SwiGLU activation
            gmm_1_out = curr_token @ expert_weight_1  # [1, 2 * intermediate_size]

            # Split into gate and up projections for SwiGLU
            gmm1_w1_proj, gmm1_w3_proj = jnp.split(
                gmm_1_out, 2, axis=-1
            )  # [1, intermediate_size], [1, intermediate_size]
            if b1 is not None:
                gmm1_w1_proj += b1[expert_id : expert_id + 1, 0, 0]
                gmm1_w3_proj += b1[expert_id : expert_id + 1, 1, 0]

            # Apply gated activation: activation(gate) * up
            act = apply_act_fn(gmm1_w1_proj, gmm1_w3_proj, act_fn)

            # Second linear layer (down projection)
            gmm_2_out = act @ expert_weight_2  # [1, hidden_size]
            if b2 is not None:
                gmm_2_out += b2[expert_id : expert_id + 1, 0]
            tok_expert_act.append(gmm_2_out)

        # Combine outputs from all selected experts
        experts_act = jnp.concatenate(tok_expert_act, axis=0)  # [top_k, hidden_size]

        # Weighted sum using top-k gating weights
        top_k_weights = top_k_logits[i]  # [top_k]
        top_k_weights = jnp.expand_dims(top_k_weights, axis=1)  # [top_k, 1]
        weighted_output = jnp.sum(experts_act * top_k_weights, axis=0, keepdims=True)  # [1, hidden_size]

        t_outputs.append(weighted_output.astype(tokens.dtype))

    return jnp.concatenate(t_outputs, axis=0)  # [actual_num_tokens, hidden_size]


def _fused_ep_moe_kernel(
    # Input
    tokens_hbm,  # (local_num_tokens, t_packing, hidden_size // t_packing)
    w1_hbm,  # (local_num_experts, 2, hidden_size, intermediate_size)
    w2_hbm,  # (local_num_experts, intermediate_size, hidden_size)
    # TODO(jevinjiang): We choose F32 scale for easier slicing. The extra
    # latency should be hidden in the pipeline overlaping. But is there a better
    # way to do this?
    w1_scale_hbm,  # None | F32(local_num_experts, 2, cdiv(hidden_size, subc_quant_w1_sz), 1, intermediate_size)
    w2_scale_hbm,  # None | F32(local_num_experts, cdiv(intermediate_size, subc_quant_w2_sz), 1, hidden_size)
    b1_hbm,  # None | F32(local_num_experts, 2, 1, intermediate_size)
    b2_hbm,  # None | F32(local_num_experts, 1, hidden_size)
    gating_hbm,  # (local_num_tokens, padded_num_experts)
    a2a_g_hbm,  # (num_experts, bt, t_packing, hidden_size // t_packing)
    # Output
    output_hbm,  # (local_num_tokens, hidden_size)
    # Scratch
    t2e_routing_x2_smem,  # <bt_sem_id> (2, bt, padded_top_k)
    d2e_count_x2_smem,  # <bt_sem_id> (2, num_devices, 1, padded_num_experts)
    expert_offsets_x2_smem,  # <bt_sem_id> (2, 2, padded_num_experts): for a2a_s and a2a_g
    expert_starts_x2_smem,  # <bt_sem_id> (2, 1, padded_num_experts)
    expert_sizes_x2_smem,  # <bt_sem_id> (2, 1, padded_num_experts)
    a2a_s_sends_x2_smem,  # <e_sem_id> (2,)
    a2a_s_x2_vmem,  # <e_sem_id> (2, bt * num_devices, t_packing, hidden_size // t_packing)
    a2a_s_acc_x2_vmem,  # <e_sem_id> (2, bt * num_devices, t_packing, hidden_size // t_packing)
    ### Accumulation for gathered tokens:
    a2a_g_acc_vmem,  # (top_k, bt, t_packing, hidden_size // t_packing)
    ### Expert weight double buffering:
    b_gating_x2_vmem,  # <bt_sem_id> (2, bt, padded_num_experts)
    b_output_x2_vmem,  # <bt_sem_id> (2, bt, hidden_size)
    b_w1_x2_vmem,  # <bw_sem_id> (2, t_packing, bd1 // t_packing, bf)
    b_w3_x2_vmem,  # <bw_sem_id> (2, t_packing, bd1 // t_packing, bf)
    b_w2_x2_vmem,  # <bw_sem_id> (2, t_packing, bf, bd2 // t_packing)
    b_w1_scale_x2_vmem,  # None | <bw_sem_id> (2, t_packing, bd1 // t_packing // subc_quant_w1_sz, 1, bf)
    b_w3_scale_x2_vmem,  # None | <bw_sem_id> (2, t_packing, bd1 // t_packing // subc_quant_w1_sz, 1, bf)
    b_w2_scale_x2_vmem,  # None | <bw_sem_id> (2, t_packing, bf // subc_quant_w2_sz, 1, bd2 // t_packing)
    b_b1_x2_vmem,  # None | <bw_sem_id> (2, 1, bf)
    b_b3_x2_vmem,  # None | <bw_sem_id> (2, 1, bf)
    b_b2_x2_vmem,  # None | <bw_sem_id> (2, t_packing, 1, bd2 // t_packing)
    b_acc_vmem,  # F32(2, bt * num_devices, 1, bf)
    ### Semaphores:
    local_sems,  # (2, 5): 2 x [b_gating_sem, b_w1_sem, b_w2_sem, b_w3_sem, b_output_sem]
    send_sems,  # <e_sem_id> (2,)
    recv_sems,  # <e_sem_id> (2,)
    a2a_gather_sem,
    a2a_acc_sem,
    *,
    top_k: int,
    renormalize_topk_logits: bool,
    ep_axis_name: str,
    act_fn: str,
    scoring_fn: str,
    subc_quant_w1_sz: int | None = None,
    subc_quant_w2_sz: int | None = None,
    # Kernel tuning params.
    bt: int,  # Block size of local_num_tokens.
    bf: int,  # Block size of intermediate_size.
    bd1: int,  # Block size of hidden_size in w1.
    bd2: int,  # Block size of hidden_size in w2.
    btc: int,  # Compute size of block tokens for active expert.
    bfc: int,  # Compute size of block intermediate_size.
    bd1c: int,  # Compute size of block hidden_size.
    bd2c: int,  # Compute size of block hidden_size.
):
    my_id = lax.axis_index(ep_axis_name)
    num_devices = lax.axis_size(ep_axis_name)
    local_num_tokens = tokens_hbm.shape[0]
    local_num_experts, intermediate_size, hidden_size = w2_hbm.shape
    right_id = (my_id + 1) % num_devices
    num_experts = a2a_g_hbm.shape[0]
    padded_num_experts = d2e_count_x2_smem.shape[-1]
    padded_top_k = t2e_routing_x2_smem.shape[-1]
    assert padded_num_experts == align_to(num_experts, 128)
    assert padded_top_k == align_to(top_k, 128)

    t_dtype = tokens_hbm.dtype
    t_packing = get_dtype_packing(t_dtype)
    t_bitwidth = 32 // t_packing
    assert a2a_g_hbm.dtype == t_dtype
    assert w1_hbm.dtype == w2_hbm.dtype

    assert bd1 % bd1c == 0
    assert bd2 % bd2c == 0
    assert bf % bfc == 0
    assert hidden_size % t_packing == 0
    assert bd1 % t_packing == 0
    assert bd2 % t_packing == 0
    assert bd1c % t_packing == 0
    assert bd2c % t_packing == 0

    h_per_t_packing = hidden_size // t_packing
    assert tokens_hbm.shape[-1] == h_per_t_packing
    bd1_per_t_packing = bd1 // t_packing
    bd2_per_t_packing = bd2 // t_packing
    bd1c_per_t_packing = bd1c // t_packing
    bd2c_per_t_packing = bd2c // t_packing

    if subc_quant_w1_sz is not None:
        if subc_quant_w1_sz < hidden_size:
            assert subc_quant_w1_sz % 256 == 0
            assert bd1c_per_t_packing == subc_quant_w1_sz
            assert bd1 % subc_quant_w1_sz == 0
            assert bd1_per_t_packing % subc_quant_w1_sz == 0
            assert h_per_t_packing % subc_quant_w1_sz == 0
        else:
            assert subc_quant_w1_sz == hidden_size

    if subc_quant_w2_sz is not None:
        if subc_quant_w2_sz < intermediate_size:
            assert subc_quant_w2_sz % 256 == 0
            assert bfc == subc_quant_w2_sz
            assert bf % subc_quant_w2_sz == 0
        else:
            assert subc_quant_w2_sz == intermediate_size

    num_bt = cdiv(local_num_tokens, bt)
    num_bf = cdiv(intermediate_size, bf)
    num_bd1 = cdiv(hidden_size, bd1)
    num_bd2 = cdiv(hidden_size, bd2)
    use_a2a = num_devices > 1

    def get_mesh_device_id(ep_rank):
        dp_rank = jax.lax.axis_index("data")
        return (dp_rank, ep_rank)

    def sync_barrier():
        if not use_a2a:
            return
        barrier_sem = pltpu.get_barrier_semaphore()
        for i in range(num_devices):
            pltpu.semaphore_signal(
                barrier_sem,
                device_id=get_mesh_device_id(i),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pltpu.semaphore_wait(barrier_sem, num_devices)

    def start_fetch_b_gating(bt_id, priority=0):
        is_valid = jnp.logical_and(0 <= bt_id, bt_id < num_bt)
        sz = pl.multiple_of(lax.select(is_valid, bt, 0), bt)
        bt_sem_id = (bt_id + 2) % 2
        b_gating_sem = local_sems.at[bt_sem_id, 0]
        pltpu.make_async_copy(
            src_ref=gating_hbm.at[pl.ds(bt_id * bt, sz)],
            dst_ref=b_gating_x2_vmem.at[bt_sem_id, pl.ds(0, sz)],
            sem=b_gating_sem,
        ).start(priority=priority)

    def wait_fetch_b_gating(bt_id):
        bt_sem_id = bt_id % 2
        b_gating_sem = local_sems.at[bt_sem_id, 0]
        pltpu.make_async_copy(
            src_ref=b_gating_x2_vmem.at[bt_sem_id],
            dst_ref=b_gating_x2_vmem.at[bt_sem_id],
            sem=b_gating_sem,
        ).wait()

    def get_top_k(input, top_k, renormalize_topk_logits):
        assert len(input.shape) == 2, input.shape
        input = input.astype(jnp.float32)
        padded_k_shape = (input.shape[0], padded_top_k)
        top_k_logits_lst = []
        top_k_indices_lst = []
        t2e = jnp.zeros(input.shape, dtype=jnp.int32)
        t2e_routing = jnp.zeros(padded_k_shape, dtype=jnp.int32)
        iota = jax.lax.broadcasted_iota(jnp.int32, input.shape, 1)
        padded_k_iota = jax.lax.broadcasted_iota(jnp.int32, padded_k_shape, 1)
        top_k_logits_sum = jnp.zeros(padded_k_shape, jnp.float32)

        for k_id in range(top_k):
            # TODO(jevinjiang): return both top_k values and indices in Mosaic
            top_k_logits = jnp.broadcast_to(
                jnp.max(input[:, :num_experts], axis=1, keepdims=True),
                padded_k_shape,
            ).astype(input.dtype)
            top_k_logits_lst.append(top_k_logits)
            if renormalize_topk_logits:
                top_k_logits_sum += top_k_logits
            # TODO(jevinjiang): support bf16 argmax in Mosaic
            top_k_indices = jnp.broadcast_to(
                jnp.argmax(input[:, :num_experts], axis=1, keepdims=True),
                padded_k_shape,
            )
            top_k_indices_lst.append(top_k_indices)
            t2e_routing = jnp.where(padded_k_iota == k_id, top_k_indices, t2e_routing)
            mask = iota == broadcast_minor(top_k_indices, input.shape)
            t2e += mask.astype(jnp.int32)
            if k_id != top_k - 1:
                input = jnp.where(mask, -jnp.inf, input)

        if renormalize_topk_logits:
            for k_id in range(top_k):
                top_k_logits_lst[k_id] /= top_k_logits_sum

        expert_sizes = jnp.sum(t2e, axis=0, keepdims=True)
        expert_starts = jnp.zeros_like(expert_sizes)
        return top_k_logits_lst, t2e_routing, expert_sizes, expert_starts

    def all_reduce_metadata(bt_sem_id, t2e_routing, starts, sizes):
        send_sem = send_sems.at[0]
        recv_sem = recv_sems.at[0]

        # All-reduce to accumulate starts and sizes and transfer to SMEM.
        def _all_reduce_metadata(
            t2e_routing_vmem,
            d2e_count_vmem,
            offsets_vmem,
            starts_vmem,
            sizes_vmem,
        ):
            offsets_vmem[...] = jnp.zeros_like(offsets_vmem)
            # TODO(jevinjiang): check how slow is VMEM -> SMEM.
            offsets_copy = pltpu.async_copy(
                src_ref=offsets_vmem,
                dst_ref=expert_offsets_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )
            t2e_routing_vmem[...] = t2e_routing
            t2e_routing_copy = pltpu.async_copy(
                src_ref=t2e_routing_vmem,
                dst_ref=t2e_routing_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )
            reduced_sizes = sizes
            reduced_starts = starts
            row_id = my_id
            d2e_count_vmem[...] = jnp.zeros_like(d2e_count_vmem)
            d2e_count_vmem[row_id] = sizes
            if use_a2a:
                for i in range(num_devices - 1):
                    sync_barrier()
                    # TODO(jevinjiang): we can use double buffering to improve AR if needed.
                    pltpu.async_remote_copy(
                        src_ref=d2e_count_vmem.at[row_id],
                        dst_ref=d2e_count_vmem.at[row_id],
                        send_sem=send_sem,
                        recv_sem=recv_sem,
                        device_id=get_mesh_device_id(right_id),
                        device_id_type=pltpu.DeviceIdType.MESH,
                    ).wait()
                    row_id = (row_id + num_devices - 1) % num_devices
                    new_sizes = d2e_count_vmem[row_id]
                    reduced_sizes += new_sizes
                    reduced_starts += lax.select(my_id > i, new_sizes, jnp.zeros_like(new_sizes))
            starts_vmem[...] = reduced_starts
            sizes_vmem[...] = reduced_sizes

            starts_copy = pltpu.async_copy(
                src_ref=starts_vmem,
                dst_ref=expert_starts_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )
            sizes_copy = pltpu.async_copy(
                src_ref=sizes_vmem,
                dst_ref=expert_sizes_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )

            # TODO(jevinjiang): if d2e_count is too big, we can store in HBM and fetch
            # to SMEM partially.
            d2e_count_copy = pltpu.async_copy(
                src_ref=d2e_count_vmem,
                dst_ref=d2e_count_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )

            t2e_routing_copy.wait()
            d2e_count_copy.wait()
            offsets_copy.wait()
            starts_copy.wait()
            sizes_copy.wait()

        pl.run_scoped(
            _all_reduce_metadata,
            pltpu.VMEM(t2e_routing_x2_smem.shape[1:], t2e_routing_x2_smem.dtype),
            pltpu.VMEM(d2e_count_x2_smem.shape[1:], d2e_count_x2_smem.dtype),
            pltpu.VMEM(expert_offsets_x2_smem.shape[1:], expert_offsets_x2_smem.dtype),
            pltpu.VMEM(expert_starts_x2_smem.shape[1:], expert_starts_x2_smem.dtype),
            pltpu.VMEM(expert_sizes_x2_smem.shape[1:], expert_sizes_x2_smem.dtype),
        )

    def start_a2a_scatter(bt_id, e_sem_id, local_e_id):
        bt_sem_id = bt_id % 2

        # Counting the number of remote sends from the current device.
        send_sz = 0
        for bt_t_id in range(bt):
            for k_id in range(top_k):
                e_id = t2e_routing_x2_smem[bt_sem_id, bt_t_id, k_id]
                is_active_expert = e_id % local_num_experts == local_e_id
                recv_id = e_id // local_num_experts
                offset = expert_offsets_x2_smem[bt_sem_id, 0, e_id]
                sz = lax.select(is_active_expert, 1, 0)
                is_local = recv_id == my_id
                local_sz = lax.select(is_local, sz, 0)
                remote_sz = lax.select(is_local, 0, sz)
                send_sz += remote_sz
                expert_offsets_x2_smem[bt_sem_id, 0, e_id] = offset + local_sz + remote_sz
                start = expert_starts_x2_smem[bt_sem_id, 0, e_id] + offset
                t_id = bt * bt_id + bt_t_id
                # TODO(jevinjiang): compare the perf when using branches.
                pltpu.make_async_copy(
                    src_ref=tokens_hbm.at[pl.ds(t_id, local_sz)],
                    dst_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(start, local_sz)],
                    sem=recv_sems.at[e_sem_id],
                ).start()
                if use_a2a:
                    pltpu.make_async_remote_copy(
                        src_ref=tokens_hbm.at[pl.ds(t_id, remote_sz)],
                        dst_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(start, remote_sz)],
                        send_sem=send_sems.at[e_sem_id],
                        recv_sem=recv_sems.at[e_sem_id],
                        device_id=get_mesh_device_id(recv_id),
                        device_id_type=pltpu.DeviceIdType.MESH,
                    ).start()
        if use_a2a:
            a2a_s_sends_x2_smem[e_sem_id] = send_sz
        else:
            a2a_s_sends_x2_smem[e_sem_id] = 0

    def wait_a2a_scatter_recv(bt_id, e_sem_id, local_e_id):
        bt_sem_id = bt_id % 2
        e_id = my_id * local_num_experts + local_e_id
        sz = expert_sizes_x2_smem[bt_sem_id, 0, e_id]
        pltpu.make_async_copy(
            src_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(0, sz)],
            dst_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(0, sz)],
            sem=recv_sems.at[e_sem_id],
        ).wait()

    def wait_a2a_scatter_send(bt_id, e_sem_id, local_e_id):
        if not use_a2a:
            return
        del bt_id, local_e_id
        sz = a2a_s_sends_x2_smem[e_sem_id]
        pltpu.make_async_copy(
            src_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(0, sz)],
            dst_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(0, sz)],
            sem=send_sems.at[e_sem_id],
        ).wait()

    def start_a2a_gather(bt_id, e_sem_id, local_e_id):
        my_e_id = my_id * local_num_experts + local_e_id
        bt_sem_id = bt_id % 2
        start = 0
        if not use_a2a:
            sz = d2e_count_x2_smem[bt_sem_id, my_id, 0, my_e_id]
            pltpu.make_async_copy(
                src_ref=a2a_s_acc_x2_vmem.at[e_sem_id, pl.ds(start, sz)],
                dst_ref=a2a_g_hbm.at[my_e_id, pl.ds(0, sz)],
                sem=a2a_gather_sem,
            ).start()
            return
        for recv_id in range(num_devices):
            sz = d2e_count_x2_smem[bt_sem_id, recv_id, 0, my_e_id]
            is_local = recv_id == my_id
            local_sz = lax.select(is_local, sz, 0)
            remote_sz = lax.select(is_local, 0, sz)
            pltpu.make_async_copy(
                src_ref=a2a_s_acc_x2_vmem.at[e_sem_id, pl.ds(start, local_sz)],
                dst_ref=a2a_g_hbm.at[my_e_id, pl.ds(0, local_sz)],
                sem=a2a_gather_sem,
            ).start()
            pltpu.make_async_remote_copy(
                src_ref=a2a_s_acc_x2_vmem.at[e_sem_id, pl.ds(start, remote_sz)],
                dst_ref=a2a_g_hbm.at[my_e_id, pl.ds(0, remote_sz)],
                send_sem=send_sems.at[e_sem_id],
                recv_sem=a2a_gather_sem,
                device_id=get_mesh_device_id(recv_id),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()
            start += sz

    def wait_a2a_gather_send(bt_id, e_sem_id, local_e_id):
        if not use_a2a:
            return
        my_e_id = my_id * local_num_experts + local_e_id
        bt_sem_id = bt_id % 2
        sz = expert_sizes_x2_smem[bt_sem_id, 0, my_e_id]
        local_sz = d2e_count_x2_smem[bt_sem_id, my_id, 0, my_e_id]
        remote_sz = sz - local_sz
        is_valid = jnp.logical_and(0 <= local_e_id, local_e_id < local_num_experts)
        remote_sz = lax.select(is_valid, remote_sz, 0)
        ref = a2a_g_hbm.reshape(num_experts * bt, t_packing, hidden_size // t_packing)
        pltpu.make_async_copy(
            src_ref=ref.at[pl.ds(0, remote_sz)],
            dst_ref=ref.at[pl.ds(0, remote_sz)],
            sem=send_sems.at[e_sem_id],
        ).wait()

    def wait_a2a_gather_recv_all():
        sz = top_k * bt
        ref = a2a_g_hbm.reshape(num_experts * bt, t_packing, hidden_size // t_packing)
        pltpu.make_async_copy(
            src_ref=ref.at[pl.ds(0, sz)],
            dst_ref=ref.at[pl.ds(0, sz)],
            sem=a2a_gather_sem,
        ).wait()

    def start_fetch_bw1(local_e_id, bw1_sem_id, bf_id, bd1_id):
        for p in range(t_packing):
            offset = p * h_per_t_packing + bd1_id * bd1_per_t_packing
            pltpu.make_async_copy(
                src_ref=w1_hbm.at[
                    local_e_id,
                    0,
                    pl.ds(offset, bd1_per_t_packing),
                    pl.ds(bf_id * bf, bf),
                ],
                dst_ref=b_w1_x2_vmem.at[bw1_sem_id, p],
                sem=local_sems.at[bw1_sem_id, 1],
            ).start()
            if w1_scale_hbm is not None:
                assert subc_quant_w1_sz is not None
                pltpu.make_async_copy(
                    src_ref=w1_scale_hbm.at[
                        local_e_id,
                        0,
                        pl.ds(
                            offset // subc_quant_w1_sz,
                            cdiv(bd1_per_t_packing, subc_quant_w1_sz),
                        ),
                        pl.ds(0, 1),
                        pl.ds(bf_id * bf, bf),
                    ],
                    dst_ref=b_w1_scale_x2_vmem.at[bw1_sem_id, p],
                    sem=local_sems.at[bw1_sem_id, 1],
                ).start()
        if b1_hbm is not None and bd1_id == 0:
            pltpu.make_async_copy(
                src_ref=b1_hbm.at[local_e_id, 0, pl.ds(0, 1), pl.ds(bf_id * bf, bf)],
                dst_ref=b_b1_x2_vmem.at[bf_id % 2],
                sem=local_sems.at[bw1_sem_id, 1],
            ).start()

    def start_fetch_bw2(local_e_id, bw2_sem_id, bf_id, bd2_id):
        for p in range(t_packing):
            offset = p * h_per_t_packing + bd2_id * bd2_per_t_packing
            pltpu.make_async_copy(
                src_ref=w2_hbm.at[
                    local_e_id,
                    pl.ds(bf_id * bf, bf),
                    pl.ds(offset, bd2_per_t_packing),
                ],
                dst_ref=b_w2_x2_vmem.at[bw2_sem_id, p],
                sem=local_sems.at[bw2_sem_id, 2],
            ).start()
            if w2_scale_hbm is not None:
                assert subc_quant_w2_sz is not None
                pltpu.make_async_copy(
                    src_ref=w2_scale_hbm.at[
                        local_e_id,
                        pl.ds(
                            bf_id * bf // subc_quant_w2_sz,  # 0 for per-channel quantization
                            cdiv(bf, subc_quant_w2_sz),  # 1 for per-channel quantization
                        ),
                        pl.ds(0, 1),
                        pl.ds(offset, bd2_per_t_packing),
                    ],
                    dst_ref=b_w2_scale_x2_vmem.at[bw2_sem_id, p],
                    sem=local_sems.at[bw2_sem_id, 2],
                ).start()
            if b2_hbm is not None and bf_id == 0:
                pltpu.make_async_copy(
                    src_ref=b2_hbm.at[local_e_id, pl.ds(0, 1), pl.ds(offset, bd2_per_t_packing)],
                    dst_ref=b_b2_x2_vmem.at[bd2_id % 2, p],
                    sem=local_sems.at[bw2_sem_id, 2],
                ).start()

    def start_fetch_bw3(local_e_id, bw3_sem_id, bf_id, bd3_id):
        for p in range(t_packing):
            offset = p * h_per_t_packing + bd3_id * bd1_per_t_packing
            pltpu.make_async_copy(
                src_ref=w1_hbm.at[
                    local_e_id,
                    1,
                    pl.ds(offset, bd1_per_t_packing),
                    pl.ds(bf_id * bf, bf),
                ],
                dst_ref=b_w3_x2_vmem.at[bw3_sem_id, p],
                sem=local_sems.at[bw3_sem_id, 3],
            ).start()
            if w1_scale_hbm is not None:
                assert subc_quant_w1_sz is not None
                pltpu.make_async_copy(
                    src_ref=w1_scale_hbm.at[
                        local_e_id,
                        1,
                        pl.ds(
                            offset // subc_quant_w1_sz,  # 0 for per-channel quantization
                            cdiv(bd1_per_t_packing, subc_quant_w1_sz),  # 1 for per-channel quantization
                        ),
                        pl.ds(0, 1),
                        pl.ds(bf_id * bf, bf),
                    ],
                    dst_ref=b_w3_scale_x2_vmem.at[bw3_sem_id, p],
                    sem=local_sems.at[bw3_sem_id, 3],
                ).start()
        if b1_hbm is not None and bd3_id == 0:
            pltpu.make_async_copy(
                src_ref=b1_hbm.at[local_e_id, 1, pl.ds(0, 1), pl.ds(bf_id * bf, bf)],
                dst_ref=b_b3_x2_vmem.at[bf_id % 2],
                sem=local_sems.at[bw3_sem_id, 3],
            ).start()

    def wait_fetch_bw1(local_e_id, bw1_sem_id, bf_id, bd1_id):
        del local_e_id
        pltpu.make_async_copy(
            src_ref=b_w1_x2_vmem.at[bw1_sem_id],
            dst_ref=b_w1_x2_vmem.at[bw1_sem_id],
            sem=local_sems.at[bw1_sem_id, 1],
        ).wait()
        if w1_scale_hbm is not None:
            pltpu.make_async_copy(
                src_ref=b_w1_scale_x2_vmem.at[bw1_sem_id],
                dst_ref=b_w1_scale_x2_vmem.at[bw1_sem_id],
                sem=local_sems.at[bw1_sem_id, 1],
            ).wait()
        if b1_hbm is not None and bd1_id == 0:
            pltpu.make_async_copy(
                src_ref=b_b1_x2_vmem.at[bf_id % 2],
                dst_ref=b_b1_x2_vmem.at[bf_id % 2],
                sem=local_sems.at[bw1_sem_id, 1],
            ).wait()

    def wait_fetch_bw2(local_e_id, bw2_sem_id, bf_id, bd2_id):
        del local_e_id
        pltpu.make_async_copy(
            src_ref=b_w2_x2_vmem.at[bw2_sem_id],
            dst_ref=b_w2_x2_vmem.at[bw2_sem_id],
            sem=local_sems.at[bw2_sem_id, 2],
        ).wait()
        if w2_scale_hbm is not None:
            pltpu.make_async_copy(
                src_ref=b_w2_scale_x2_vmem.at[bw2_sem_id],
                dst_ref=b_w2_scale_x2_vmem.at[bw2_sem_id],
                sem=local_sems.at[bw2_sem_id, 2],
            ).wait()
        if b2_hbm is not None and bf_id == 0:
            pltpu.make_async_copy(
                src_ref=b_b2_x2_vmem.at[bd2_id % 2],
                dst_ref=b_b2_x2_vmem.at[bd2_id % 2],
                sem=local_sems.at[bw2_sem_id, 2],
            ).wait()

    def wait_fetch_bw3(local_e_id, bw3_sem_id, bf_id, bd3_id):
        del local_e_id
        pltpu.make_async_copy(
            src_ref=b_w3_x2_vmem.at[bw3_sem_id],
            dst_ref=b_w3_x2_vmem.at[bw3_sem_id],
            sem=local_sems.at[bw3_sem_id, 3],
        ).wait()
        if w1_scale_hbm is not None:
            pltpu.make_async_copy(
                src_ref=b_w3_scale_x2_vmem.at[bw3_sem_id],
                dst_ref=b_w3_scale_x2_vmem.at[bw3_sem_id],
                sem=local_sems.at[bw3_sem_id, 3],
            ).wait()
        if b1_hbm is not None and bd3_id == 0:
            pltpu.make_async_copy(
                src_ref=b_b3_x2_vmem.at[bf_id % 2],
                dst_ref=b_b3_x2_vmem.at[bf_id % 2],
                sem=local_sems.at[bw3_sem_id, 3],
            ).wait()

    def start_fetch_next_bw(local_e_id, bw_sem_id, bf_id, bd1_id, bd2_id):
        next_bd1_id = bd1_id + 1
        next_bd2_id = bd2_id + 1
        next_sem_id = (bw_sem_id + 1) % 2

        if bf_id >= num_bf:
            return
        if next_bd1_id < num_bd1:
            start_fetch_bw1(local_e_id, next_sem_id, bf_id, next_bd1_id)
            start_fetch_bw3(local_e_id, next_sem_id, bf_id, next_bd1_id)
        elif next_bd1_id == num_bd1:
            start_fetch_bw2(local_e_id, next_sem_id, bf_id, 0)
        elif next_bd2_id < num_bd2:
            start_fetch_bw2(local_e_id, next_sem_id, bf_id, next_bd2_id)
        elif next_bd2_id == num_bd2:
            start_fetch_next_bw(local_e_id, bw_sem_id, bf_id + 1, -1, -1)
        else:
            raise RuntimeError("Unreachable")

    def dynamic_ffn1(
        t_b32_vmem,
        w1_vmem,
        w1_scale_vmem,
        b1_vmem,
        w3_vmem,
        w3_scale_vmem,
        b3_vmem,
        acc1_vmem,
        acc3_vmem,
        dyn_sz,
        should_init,
    ):
        assert t_b32_vmem.shape == (bt * num_devices, bd1 // t_packing)
        assert w1_vmem.shape == w3_vmem.shape == (t_packing, bd1_per_t_packing, bf)
        assert acc1_vmem.shape == acc3_vmem.shape == (bt * num_devices, bf)
        assert bd1 % (t_packing * 128) == 0, (bd1, t_packing)
        assert bd1c % (t_packing * 128) == 0, (bd1c, t_packing)
        assert bd1_per_t_packing % bd1c_per_t_packing == 0
        if w1_scale_vmem is not None:
            assert w1_scale_vmem.shape == (
                t_packing,
                cdiv(bd1_per_t_packing, subc_quant_w1_sz),
                1,
                bf,
            )
            if subc_quant_w1_sz < hidden_size:
                assert bd1c_per_t_packing == subc_quant_w1_sz
        if w3_scale_vmem is not None:
            assert w3_scale_vmem.shape == (
                t_packing,
                cdiv(bd1_per_t_packing, subc_quant_w1_sz),
                1,
                bf,
            )
            if subc_quant_w1_sz < hidden_size:
                assert bd1c_per_t_packing == subc_quant_w1_sz

        num_loops = cdiv(dyn_sz, btc)
        repack_ty = jnp.dtype(f"int{t_bitwidth}")

        def body(btc_id, _):
            for bd1c_id in range(cdiv(bd1, bd1c)):
                t_b32 = t_b32_vmem[
                    pl.ds(btc_id * btc, btc),
                    pl.ds(bd1c_id * bd1c_per_t_packing, bd1c_per_t_packing),
                ]
                for p_id in range(t_packing):
                    t = pltpu.bitcast(t_b32.astype(repack_ty), t_dtype)
                    t_b32 = t_b32 >> t_bitwidth
                    for bfc_id in range(cdiv(bf, bfc)):
                        w_slices = (
                            p_id,
                            pl.ds(bd1c_id * bd1c_per_t_packing, bd1c_per_t_packing),
                            pl.ds(bfc_id * bfc, bfc),
                        )
                        w1 = w1_vmem[*w_slices]
                        acc1 = jnp.dot(t, w1, preferred_element_type=jnp.float32)

                        if w1_scale_vmem is not None:
                            w1_scale_slices = (
                                p_id,
                                (bd1c_id * bd1c_per_t_packing) // subc_quant_w1_sz,
                                pl.ds(0, 1),
                                pl.ds(bfc_id * bfc, bfc),
                            )
                            # TODO(jevinjiang): can use mosaic to load with stride 0.
                            w1_scale = jnp.broadcast_to(w1_scale_vmem[*w1_scale_slices], acc1.shape)
                            acc1 *= w1_scale

                        w3 = w3_vmem[*w_slices]

                        acc3 = jnp.dot(t, w3, preferred_element_type=jnp.float32)

                        if w3_scale_vmem is not None:
                            w3_scale_slices = (
                                p_id,
                                (bd1c_id * bd1c_per_t_packing) // subc_quant_w1_sz,
                                pl.ds(0, 1),
                                pl.ds(bfc_id * bfc, bfc),
                            )
                            w3_scale = jnp.broadcast_to(w3_scale_vmem[*w3_scale_slices], acc3.shape)
                            acc3 *= w3_scale

                        acc_slices = (pl.ds(btc_id * btc, btc), pl.ds(bfc_id * bfc, bfc))
                        if should_init and p_id == bd1c_id == 0:
                            if b1_vmem is not None:
                                b1_scale_slices = (
                                    pl.ds(0, 1),
                                    pl.ds(bfc_id * bfc, bfc),
                                )
                                b1 = jnp.broadcast_to(b1_vmem[*b1_scale_slices], acc1.shape)
                                acc1 += b1
                            if b3_vmem is not None:
                                b3_scale_slices = (
                                    pl.ds(0, 1),
                                    pl.ds(bfc_id * bfc, bfc),
                                )
                                b3 = jnp.broadcast_to(b3_vmem[*b3_scale_slices], acc1.shape)
                                acc3 += b3
                            acc1_vmem[*acc_slices] = acc1
                            acc3_vmem[*acc_slices] = acc3
                        else:
                            acc1_vmem[*acc_slices] += acc1
                            acc3_vmem[*acc_slices] += acc3

        lax.fori_loop(0, num_loops, body, None)

    def dynamic_ffn2(
        acc1_vmem,
        acc3_vmem,
        w2_vmem,
        w2_scale_vmem,
        b2_vmem,
        res_b32_vmem,
        dyn_sz,
        should_init,
    ):
        assert res_b32_vmem.shape == (bt * num_devices, bd2_per_t_packing)
        assert w2_vmem.shape == (t_packing, bf, bd2_per_t_packing)
        assert acc1_vmem.shape == acc3_vmem.shape == (bt * num_devices, bf)
        assert bd2 % (t_packing * 128) == 0, (bd2, t_packing)
        assert bd2c % (t_packing * 128) == 0, (bd2c, t_packing)
        assert t_dtype in (jnp.float32, jnp.bfloat16)

        if w2_scale_vmem is not None:
            assert w2_scale_vmem.shape == (
                t_packing,
                cdiv(bf, subc_quant_w2_sz),
                1,
                bd2_per_t_packing,
            )
            if subc_quant_w2_sz < intermediate_size:
                assert bfc == subc_quant_w2_sz

        num_loops = cdiv(dyn_sz, btc)
        assert bd2c % (t_packing * 128) == 0, (bd2c, t_packing)

        def body(btc_id, _):
            for bd2c_id in range(cdiv(bd2, bd2c)):
                res_lst = []
                for p_id in range(t_packing):
                    res = jnp.zeros((btc, bd2c_per_t_packing), dtype=jnp.float32)

                    if b2_vmem is not None and should_init:
                        b2_scale_slices = (
                            p_id,
                            pl.ds(0, 1),
                            pl.ds(bd2c_id * bd2c_per_t_packing, bd2c_per_t_packing),
                        )
                        b2 = jnp.broadcast_to(b2_vmem[*b2_scale_slices], res.shape)
                        res += b2

                    for bfc_id in range(cdiv(bf, bfc)):
                        acc_slices = (pl.ds(btc_id * btc, btc), pl.ds(bfc_id * bfc, bfc))
                        acc1 = acc1_vmem[*acc_slices]
                        acc3 = acc3_vmem[*acc_slices]
                        act = apply_act_fn(acc1, acc3, act_fn)
                        w2 = w2_vmem[
                            p_id,
                            pl.ds(bfc_id * bfc, bfc),
                            pl.ds(bd2c_id * bd2c_per_t_packing, bd2c_per_t_packing),
                        ]
                        acc = jnp.dot(act, w2, preferred_element_type=jnp.float32)
                        if w2_scale_vmem is not None:
                            assert subc_quant_w2_sz is not None
                            w2_scale_slices = (
                                p_id,
                                (bfc_id * bfc) // subc_quant_w2_sz,
                                pl.ds(0, 1),
                                pl.ds(bd2c_id * bd2c_per_t_packing, bd2c_per_t_packing),
                            )
                            w2_scale = jnp.broadcast_to(w2_scale_vmem[*w2_scale_slices], acc.shape)
                            acc *= w2_scale
                        res += acc
                    res = pltpu.bitcast(res, jnp.uint32)
                    if t_packing == 2:
                        res = res >> 16 << (16 * p_id)
                    else:
                        assert t_packing == 1
                    res_lst.append(res)
                res = res_lst[0]
                # TODO(jevinjiang): use interleaved packing when it is exposed to Pallas
                for i in range(1, t_packing):
                    res |= res_lst[i]
                sliced_res_vmem = res_b32_vmem.at[
                    pl.ds(btc_id * btc, btc),
                    pl.ds(bd2c_id * bd2c_per_t_packing, bd2c_per_t_packing),
                ]
                if should_init:
                    sliced_res_vmem[...] = res
                else:
                    sliced_res_vmem[...] = pltpu.bitcast(
                        sliced_res_vmem.bitcast(t_dtype)[...] + pltpu.bitcast(res, t_dtype),
                        sliced_res_vmem.dtype,
                    )

        lax.fori_loop(0, num_loops, body, None)

    def expert_ffn(bt_id, e_sem_id, local_e_id):
        bt_sem_id = bt_id % 2
        bw_sem_id = 0
        # start_fetch_bw1(local_e_id, bw_sem_id, 0, 0)
        # start_fetch_bw3(local_e_id, bw_sem_id, 0, 0)
        a2a_s_b32_vmem = (
            a2a_s_x2_vmem.bitcast(jnp.uint32).reshape(2, bt * num_devices, hidden_size // t_packing).at[e_sem_id]
        )
        a2a_s_acc_b32_vmem = (
            a2a_s_acc_x2_vmem.bitcast(jnp.uint32).reshape(2, bt * num_devices, hidden_size // t_packing).at[e_sem_id]
        )
        b_acc_vmem_2d = b_acc_vmem.reshape(2, bt * num_devices, bf)
        b_acc1_vmem = b_acc_vmem_2d.at[0]
        b_acc3_vmem = b_acc_vmem_2d.at[1]

        e_id = my_id * local_num_experts + local_e_id
        dyn_sz = expert_sizes_x2_smem[bt_sem_id, 0, e_id]

        bd1_per_t_packing = bd1 // t_packing
        bd2_per_t_packing = bd2 // t_packing

        for bf_id in range(num_bf):
            for bd1_id in range(num_bd1):
                start_fetch_next_bw(local_e_id, bw_sem_id, bf_id, bd1_id, 0)
                w1_scale_vmem = None if b_w1_scale_x2_vmem is None else b_w1_scale_x2_vmem.at[bw_sem_id]
                w3_scale_vmem = None if b_w3_scale_x2_vmem is None else b_w3_scale_x2_vmem.at[bw_sem_id]
                b1_vmem = None if b_b1_x2_vmem is None else b_b1_x2_vmem.at[bf_id % 2]
                b3_vmem = None if b_b3_x2_vmem is None else b_b3_x2_vmem.at[bf_id % 2]
                wait_fetch_bw1(local_e_id, bw_sem_id, bf_id, bd1_id)
                wait_fetch_bw3(local_e_id, bw_sem_id, bf_id, bd1_id)

                dynamic_ffn1(
                    t_b32_vmem=a2a_s_b32_vmem.at[..., pl.ds(bd1_id * bd1_per_t_packing, bd1_per_t_packing)],
                    w1_vmem=b_w1_x2_vmem.at[bw_sem_id],
                    w1_scale_vmem=w1_scale_vmem,
                    b1_vmem=b1_vmem,
                    w3_vmem=b_w3_x2_vmem.at[bw_sem_id],
                    w3_scale_vmem=w3_scale_vmem,
                    b3_vmem=b3_vmem,
                    acc1_vmem=b_acc1_vmem,
                    acc3_vmem=b_acc3_vmem,
                    dyn_sz=dyn_sz,
                    should_init=(bd1_id == 0),
                )
                bw_sem_id = (bw_sem_id + 1) % 2

            for bd2_id in range(num_bd2):
                start_fetch_next_bw(local_e_id, bw_sem_id, bf_id, num_bd1, bd2_id)
                wait_fetch_bw2(local_e_id, bw_sem_id, bf_id, bd2_id)
                if bf_id == bd2_id == 0:
                    wait_a2a_gather_send(bt_id, e_sem_id, local_e_id - 2)

                w2_scale_vmem = None if b_w2_scale_x2_vmem is None else b_w2_scale_x2_vmem.at[bw_sem_id]
                b2_vmem = None if b_b2_x2_vmem is None else b_b2_x2_vmem.at[bd2_id % 2]
                dynamic_ffn2(
                    acc1_vmem=b_acc1_vmem,
                    acc3_vmem=b_acc3_vmem,
                    w2_vmem=b_w2_x2_vmem.at[bw_sem_id],
                    w2_scale_vmem=w2_scale_vmem,
                    b2_vmem=b2_vmem,
                    res_b32_vmem=a2a_s_acc_b32_vmem.at[..., pl.ds(bd2_id * bd2_per_t_packing, bd2_per_t_packing)],
                    dyn_sz=dyn_sz,
                    should_init=(bf_id == 0),
                )
                bw_sem_id = (bw_sem_id + 1) % 2

    def bt_acc(bt_id, top_k_logits_lst):
        bt_sem_id = bt_id % 2
        for bt_t_id in range(bt):
            for k_id in range(top_k):
                e_id = t2e_routing_x2_smem[bt_sem_id, bt_t_id, k_id]
                offset = expert_offsets_x2_smem[bt_sem_id, 1, e_id]
                expert_offsets_x2_smem[bt_sem_id, 1, e_id] = offset + 1
                pltpu.make_async_copy(
                    src_ref=a2a_g_hbm.at[e_id, pl.ds(offset, 1)],
                    dst_ref=a2a_g_acc_vmem.at[k_id, pl.ds(bt_t_id, 1)],
                    sem=a2a_acc_sem,
                ).start()
        pltpu.make_async_copy(
            src_ref=a2a_g_acc_vmem,
            dst_ref=a2a_g_acc_vmem,
            sem=a2a_acc_sem,
        ).wait()
        output = None
        for k_id in range(top_k):
            acc = a2a_g_acc_vmem[k_id].reshape(bt, hidden_size)
            logits = broadcast_minor(top_k_logits_lst[k_id], acc.shape)
            acc *= logits
            if output is None:
                output = acc
            else:
                output += acc
        assert output is not None
        return output.astype(output_hbm.dtype)

    def start_send_bo(bt_id, priority=0):
        bt_sem_id = bt_id % 2
        b_output_sem = local_sems.at[bt_sem_id, 4]
        pltpu.make_async_copy(
            src_ref=b_output_x2_vmem.at[bt_sem_id],
            dst_ref=output_hbm.at[pl.ds(bt_id * bt, bt)],
            sem=b_output_sem,
        ).start(priority=priority)

    def wait_send_bo(bt_id):
        is_valid = jnp.logical_and(0 <= bt_id, bt_id < num_bt)
        sz = pl.multiple_of(lax.select(is_valid, bt, 0), bt)
        bt_sem_id = (bt_id + 2) % 2
        b_output_sem = local_sems.at[bt_sem_id, 4]
        pltpu.make_async_copy(
            src_ref=output_hbm.at[pl.ds(0, sz)],
            dst_ref=output_hbm.at[pl.ds(0, sz)],
            sem=b_output_sem,
        ).wait()

    ### ------- Kernel start ------- ###
    sync_barrier()
    start_fetch_b_gating(bt_id=0)

    def run_per_bt(bt_id, e_sem_id):
        bt_sem_id = bt_id % 2
        next_bt_id = bt_id + 1
        start_fetch_b_gating(next_bt_id)
        wait_fetch_b_gating(bt_id)

        b_gating = b_gating_x2_vmem[bt_sem_id]
        b_gating_score = apply_scoring_fn(scoring_fn, b_gating)
        top_k_logits_lst, t2e_routing, expert_sizes, expert_starts = get_top_k(
            b_gating_score, top_k, renormalize_topk_logits
        )

        all_reduce_metadata(bt_sem_id, t2e_routing, expert_starts, expert_sizes)
        sync_barrier()

        # Start a2a scatter for first active expert.
        start_a2a_scatter(bt_id=bt_id, e_sem_id=e_sem_id, local_e_id=0)

        def run_per_expert(local_e_id, e_sem_id):
            # Prefetch weights for CURRENT active expert.
            # TODO(jevinjiang): It is hard to prefetch weights in previous iteration
            # because the expert_ffn keeps overwriting the buffers. Triple buffering
            # could resolve this but it takes more VMEM scratch. Need further
            # experiment on this.
            start_fetch_bw1(local_e_id, bw1_sem_id=0, bf_id=0, bd1_id=0)
            start_fetch_bw3(local_e_id, bw3_sem_id=0, bf_id=0, bd3_id=0)

            # Next ids.
            next_e_sem_id = lax.select(e_sem_id == 0, 1, 0)
            next_local_e_id = local_e_id + 1

            # Start a2a scatter for NEXT active expert.
            @pl.when(next_local_e_id < local_num_experts)
            def _():
                start_a2a_scatter(bt_id, next_e_sem_id, next_local_e_id)

            # Wait a2a scatter for CURRENT active expert.
            wait_a2a_scatter_recv(bt_id, e_sem_id, local_e_id)

            # Perform FFN for CURRENT active expert.
            expert_ffn(bt_id, e_sem_id, local_e_id)

            # Start a2a gather to send back tokens for CURRENT active expert.
            start_a2a_gather(bt_id, e_sem_id, local_e_id)

            # A must-wait before next sync_barrier.
            wait_a2a_scatter_send(bt_id, e_sem_id, local_e_id)
            sync_barrier()
            return next_e_sem_id

        e_sem_id = lax.fori_loop(0, local_num_experts, run_per_expert, e_sem_id, unroll=False)

        # Wait to receive a2a gather for ALL experts.
        wait_a2a_gather_recv_all()
        sync_barrier()

        # Accumulate results for current batch.
        output = bt_acc(bt_id, top_k_logits_lst)

        # Make sure it is safe to overwrite output buffer.
        wait_send_bo(bt_id=bt_id - 2)
        b_output_x2_vmem[bt_sem_id] = output

        start_send_bo(bt_id)

        wait_a2a_gather_send(
            bt_id,
            e_sem_id=e_sem_id,
            local_e_id=local_num_experts - 2,
        )
        wait_a2a_gather_send(
            bt_id,
            e_sem_id=lax.select(e_sem_id == 0, 1, 0),
            local_e_id=local_num_experts - 1,
        )
        sync_barrier()
        return e_sem_id

    lax.fori_loop(0, num_bt, run_per_bt, 0, unroll=False)
    wait_send_bo(bt_id=num_bt - 2)
    wait_send_bo(bt_id=num_bt - 1)

    ### ------- Kernel end ------- ###


@functools.partial(
    jax.jit,
    static_argnames=[
        "mesh",
        "top_k",
        "renormalize_topk_logits",
        "act_fn",
        "scoring_fn",
        "subc_quant_w1_sz",
        "subc_quant_w2_sz",
        "bt",
        "bf",
        "bd1",
        "bd2",
        "btc",
        "bfc",
        "bd1c",
        "bd2c",
        "ep_axis_name",
    ],
)
def fused_ep_moe(
    mesh: jax.sharding.Mesh,
    tokens: jax.Array,  # (num_tokens, hidden_size)
    w1: jax.Array,  # (num_experts, 2, hidden_size, intermediate_size)
    w2: jax.Array,  # (num_experts, intermediate_size, hidden_size)
    gating_output: jax.Array,  # (num_tokens, num_experts)
    top_k: int,
    *,
    renormalize_topk_logits: bool = False,
    act_fn: str = "silu",
    scoring_fn: str = "softmax",
    subc_quant_w1_sz: int | None = None,
    subc_quant_w2_sz: int | None = None,
    w1_scale: jax.Array | None = None,  # F32(num_experts, 2, hidden_size // subc_quant_w1_sz, 1, intermediate_size)
    w2_scale: jax.Array | None = None,  # F32(num_experts, intermediate_size // subc_quant_w2_sz, 1, hidden_size)
    b1: jax.Array | None = None,  # F32(num_experts, 2, 1, intermediate_size)
    b2: jax.Array | None = None,  # F32(num_experts, 1, hidden_size)
    # Kernel tuning parameters.
    bt: int | None = None,
    bf: int | None = None,
    bd1: int | None = None,
    bd2: int | None = None,
    btc: int | None = None,
    bfc: int | None = None,
    bd1c: int | None = None,
    bd2c: int | None = None,
    ep_axis_name: str = "model",
):
    # TODO(jevinjiang): move all these assertions to validation function.
    if len(mesh.shape) != 2:
        raise NotImplementedError("Only 2D mesh is supported.")

    for axis_name in mesh.axis_names:
        if axis_name == ep_axis_name:
            continue
        if mesh.shape[axis_name] != 1:
            raise NotImplementedError(f"Expected all non-ep axis to have size 1 in {mesh.shape=}")

    ep_size = mesh.shape[ep_axis_name]
    num_devices = ep_size

    num_tokens, hidden_size = tokens.shape
    num_experts, intermediate_size, _ = w2.shape

    if w1.shape != (num_experts, 2, hidden_size, intermediate_size):
        raise ValueError(f"Expected {w1.shape=} to be" f" {(num_experts, 2, hidden_size, intermediate_size)}.")

    if w2.shape != (num_experts, intermediate_size, hidden_size):
        raise ValueError(f"Expected {w2.shape=} to be" f" {(num_experts, intermediate_size, hidden_size)}.")

    if gating_output.shape != (num_tokens, num_experts):
        raise ValueError(f"Expected {gating_output.shape=} to be {(num_tokens, num_experts)}.")

    if not (0 < top_k <= num_experts):
        raise ValueError(f"Expected {top_k=} to be in range (0, {num_experts=}].")

    if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
        raise ValueError(
            f"Expected {hidden_size=} and {intermediate_size=} to be aligned to"
            " 128. Did you pad them with zeros outside the kernel?"
        )
    if num_tokens % ep_size != 0:
        raise ValueError(f"Expected {num_tokens=} to be aligned to {ep_size=}.")
    if num_experts % ep_size != 0:
        raise ValueError(f"Expected {num_experts=} to be aligned to {ep_size=}.")

    local_num_tokens = num_tokens // ep_size
    # local_num_experts = num_experts // ep_size
    padded_num_experts = align_to(num_experts, 128)
    padded_top_k = align_to(top_k, 128)
    t_dtype = tokens.dtype
    t_packing = get_dtype_packing(t_dtype)
    assert w1.dtype == w2.dtype
    w_packing = get_dtype_packing(w1.dtype)

    if None in {bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c}:
        bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c = get_tuned_block_sizes(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            t_packing=t_packing,
            w_packing=w_packing,
            num_tokens=num_tokens,
            ep_size=ep_size,
        )
    if None in {bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c}:
        raise ValueError("Block sizes must be fully specified or resolved from tuned table.")
    bt = int(bt)
    bf = int(bf)
    bd1 = int(bd1)
    bd2 = int(bd2)
    btc = int(btc)
    bfc = int(bfc)
    bd1c = int(bd1c)
    bd2c = int(bd2c)

    # Override bt
    if local_num_tokens <= t_packing * 8:
        bt = local_num_tokens
        btc = bt
    bt = min(local_num_tokens, bt)
    # The worst case is that all devices send bt to one device.
    btc = min(bt, btc)
    assert bt % btc == 0

    if local_num_tokens % t_packing != 0:
        raise ValueError(f"Expected {local_num_tokens=} to be aligned to {t_packing=}.")

    if bt % t_packing != 0:
        raise ValueError(f"Expected {bt=} to be aligned to {t_packing=}.")
    if local_num_tokens % bt != 0:
        raise ValueError(f"Expected {local_num_tokens=} to be aligned to {bt=}.")

    if subc_quant_w1_sz is not None:
        if subc_quant_w1_sz <= 0:
            raise ValueError(f"Expected {subc_quant_w1_sz=} to be non-negative.")
        if subc_quant_w1_sz % 256 != 0:
            raise ValueError("Expected {subc_quant_w1_sz=} to be aligned to 256.")
        if hidden_size % subc_quant_w1_sz != 0:
            raise ValueError(f"Expected {hidden_size=} to be aligned to {subc_quant_w1_sz=}.")
        # We force compute size of contracting dim to be subc_quant_w1_sz. So we can
        # apply same scale after matmul and accumulation. It only makes sense for
        # non per-channel quantization
        if subc_quant_w1_sz < hidden_size:
            bd1c = subc_quant_w1_sz * t_packing

    if subc_quant_w2_sz is not None:
        if subc_quant_w2_sz <= 0:
            raise ValueError(f"Expected {subc_quant_w2_sz=} to be non-negative.")
        if subc_quant_w2_sz % 256 != 0:
            raise ValueError("Expected {subc_quant_w2_sz=} to be aligned to 256.")
        if intermediate_size % subc_quant_w2_sz != 0:
            raise ValueError(f"Expected {intermediate_size=} to be aligned to {subc_quant_w2_sz=}.")
        # We force compute size of contracting dim to be subc_quant_w2_sz. So we can
        # apply same scale after matmul and accumulation. It only makes sense for
        # non per-channel quantization
        if subc_quant_w2_sz < intermediate_size:
            bfc = subc_quant_w2_sz

    if bfc % 128 != 0:
        raise ValueError(f"Expected {bfc=} to be aligned to 128.")
    if bd1c % (t_packing * 128) != 0:
        raise ValueError(f"Expected {bd1c=} to be aligned to {t_packing * 128}.")
    if bd2c % (t_packing * 128) != 0:
        raise ValueError(f"Expected {bd2c=} to be aligned to {t_packing * 128}.")
    if bf % bfc != 0:
        raise ValueError(f"Expected {bf=} to be aligned to {bfc=}.")
    if bd1 % bd1c != 0:
        raise ValueError(f"Expected {bd1=} to be aligned to {bd1c=}.")
    if bd2 % bd2c != 0:
        raise ValueError(f"Expected {bd2=} to be aligned to {bd2c=}.")
    if hidden_size % bd1 != 0 or hidden_size % bd2 != 0:
        raise ValueError(f"Expected {hidden_size=} to be aligned to {bd1=} and {bd2=}.")
    if intermediate_size % bf != 0:
        raise ValueError(f"Expected {intermediate_size=} to be aligned to {bf=}.")

    # Note: we should dump scale as the kernel expected shape in the
    # checkpoint offline or reshape right after weight loading.
    if w1_scale is not None:
        assert subc_quant_w1_sz is not None
        expected_w1_scale_shape = (
            num_experts,
            2,
            hidden_size // subc_quant_w1_sz,
            1,
            intermediate_size,
        )
        if w1_scale.shape != expected_w1_scale_shape:
            raise ValueError(f"Expected {w1_scale.shape=} to be {expected_w1_scale_shape}.")
        if w1_scale.dtype != jnp.float32:
            w1_scale = w1_scale.astype(jnp.float32)

    if w2_scale is not None:
        assert subc_quant_w2_sz is not None
        expected_w2_scale_shape = (
            num_experts,
            intermediate_size // subc_quant_w2_sz,
            1,
            hidden_size,
        )
        if w2_scale.shape != expected_w2_scale_shape:
            raise ValueError(f"Expected {w2_scale.shape=} to be {expected_w2_scale_shape}.")
        if w2_scale.dtype != jnp.float32:
            w2_scale = w2_scale.astype(jnp.float32)

    if b1 is not None:
        expected_b1_shape = (num_experts, 2, 1, intermediate_size)
        if b1.shape != expected_b1_shape:
            raise ValueError(f"Expected {b1.shape=} to be {expected_b1_shape}.")
        if b1.dtype != jnp.float32:
            b1 = b1.astype(jnp.float32)

    if b2 is not None:
        expected_b2_shape = (num_experts, 1, hidden_size)
        if b2.shape != expected_b2_shape:
            raise ValueError(f"Expected {b2.shape=} to be {expected_b2_shape}.")
        if b2.dtype != jnp.float32:
            b2 = b2.astype(jnp.float32)

    # Prepare inputs for the kernel.
    if padded_num_experts != gating_output.shape[-1]:
        gating_output = jnp.pad(
            gating_output,
            ((0, 0), (0, padded_num_experts - gating_output.shape[-1])),
            constant_values=-jnp.inf,
        )

    tokens = tokens.reshape(-1, t_packing, hidden_size // t_packing)

    hbm_block_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    renorm_str = "-renorm_k" if renormalize_topk_logits else ""
    scope_name = f"fused-moe-k_{top_k}{renorm_str}-bt_{bt}_{btc}-bf_{bf}_{bfc}-bd1_{bd1}_{bd1c}-bd2_{bd2}_{bd2c}"
    compiler_params = pltpu.CompilerParams(
        vmem_limit_bytes=100 * 1024 * 1024,
        **({"collective_id": 0} if num_devices > 1 else {}),
    )

    fused_moe = pl.pallas_call(
        functools.partial(
            _fused_ep_moe_kernel,
            top_k=top_k,
            renormalize_topk_logits=renormalize_topk_logits,
            ep_axis_name=ep_axis_name,
            act_fn=act_fn,
            scoring_fn=scoring_fn,
            subc_quant_w1_sz=subc_quant_w1_sz,
            subc_quant_w2_sz=subc_quant_w2_sz,
            bt=bt,
            bf=bf,
            bd1=bd1,
            bd2=bd2,
            btc=btc,
            bfc=bfc,
            bd1c=bd1c,
            bd2c=bd2c,
        ),
        out_shape=jax.ShapeDtypeStruct((local_num_tokens, hidden_size), t_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                hbm_block_spec,  # tokens_hbm
                hbm_block_spec,  # w1_hbm
                hbm_block_spec,  # w2_hbm
                None if w1_scale is None else hbm_block_spec,  # w1_scale_hbm
                None if w2_scale is None else hbm_block_spec,  # w2_scale_hbm
                None if b1 is None else hbm_block_spec,  # b1_hbm
                None if b2 is None else hbm_block_spec,  # b2_hbm
                hbm_block_spec,  # gating_output_hbm
                hbm_block_spec,  # a2a_g_hbm
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(
                [
                    # t2e_routing_x2_smem
                    pltpu.SMEM((2, bt, padded_top_k), jnp.int32),
                    # d2e_count_x2_smem
                    pltpu.SMEM((2, num_devices, 1, padded_num_experts), jnp.int32),
                    # expert_offsets_x2_smem
                    pltpu.SMEM((2, 2, padded_num_experts), jnp.int32),
                    # expert_starts_x2_smem
                    pltpu.SMEM((2, 1, padded_num_experts), jnp.int32),
                    # expert_sizes_x2_smem
                    pltpu.SMEM((2, 1, padded_num_experts), jnp.int32),
                    # a2a_s_sends_x2_smem
                    pltpu.SMEM((2,), jnp.int32),
                    # a2a_s_x2_vmem
                    pltpu.VMEM(
                        (
                            2,
                            bt * num_devices,
                            t_packing,
                            hidden_size // t_packing,
                        ),
                        t_dtype,
                    ),
                    # a2a_s_acc_x2_vmem
                    pltpu.VMEM(
                        (
                            2,
                            bt * num_devices,
                            t_packing,
                            hidden_size // t_packing,
                        ),
                        t_dtype,
                    ),
                    # a2a_g_acc_vmem
                    pltpu.VMEM((top_k, bt, t_packing, hidden_size // t_packing), t_dtype),
                    # b_gating_x2_vmem
                    pltpu.VMEM((2, bt, padded_num_experts), t_dtype),
                    # b_output_x2_vmem
                    pltpu.VMEM((2, bt, hidden_size), t_dtype),
                    # b_w1_x2_vmem
                    pltpu.VMEM((2, t_packing, bd1 // t_packing, bf), w1.dtype),
                    # b_w3_x2_vmem
                    pltpu.VMEM((2, t_packing, bd1 // t_packing, bf), w1.dtype),
                    # b_w2_x2_vmem
                    pltpu.VMEM((2, t_packing, bf, bd2 // t_packing), w2.dtype),
                    # b_w1_scale_x2_vmem
                    (
                        None
                        if w1_scale is None
                        else pltpu.VMEM(
                            (
                                2,
                                t_packing,
                                cdiv(bd1 // t_packing, subc_quant_w1_sz),
                                1,
                                bf,
                            ),
                            jnp.float32,
                        )
                    ),
                    # b_w3_scale_x2_vmem
                    (
                        None
                        if w1_scale is None
                        else pltpu.VMEM(
                            (
                                2,
                                t_packing,
                                cdiv(bd1 // t_packing, subc_quant_w1_sz),
                                1,
                                bf,
                            ),
                            jnp.float32,
                        )
                    ),
                    # b_w2_scale_x2_vmem
                    (
                        None
                        if w2_scale is None
                        else pltpu.VMEM(
                            (
                                2,
                                t_packing,
                                cdiv(bf, subc_quant_w2_sz),
                                1,
                                bd2 // t_packing,
                            ),
                            jnp.float32,
                        )
                    ),
                    # b_b1_x2_vmem
                    (
                        None
                        if b1 is None
                        else pltpu.VMEM(
                            (
                                2,
                                1,
                                bf,
                            ),
                            jnp.float32,
                        )
                    ),
                    # b_b3_x2_vmem
                    (
                        None
                        if b1 is None
                        else pltpu.VMEM(
                            (
                                2,
                                1,
                                bf,
                            ),
                            jnp.float32,
                        )
                    ),
                    # b_b2_x2_vmem
                    (
                        None
                        if b2 is None
                        else pltpu.VMEM(
                            (
                                2,
                                t_packing,
                                1,
                                bd2 // t_packing,
                            ),
                            jnp.float32,
                        )
                    ),
                    # b_acc_vmem
                    pltpu.VMEM((2, bt * num_devices, 1, bf), jnp.float32),
                    # local_sems
                    pltpu.SemaphoreType.DMA((2, 5)),
                    # send_sems
                    pltpu.SemaphoreType.DMA((2,)),
                    # recv_sems
                    pltpu.SemaphoreType.DMA((2,)),
                    # a2a_gather_sem
                    pltpu.SemaphoreType.DMA,
                    # a2a_acc_sem
                    pltpu.SemaphoreType.DMA,
                ]
            ),
        ),
        compiler_params=compiler_params,
        name=scope_name,
    )

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(
            P(ep_axis_name),  # tokens_hbm
            P(ep_axis_name),  # w1_hbm
            P(ep_axis_name),  # w2_hbm
            None if w1_scale is None else P(ep_axis_name),  # w1_scale_hbm
            None if w2_scale is None else P(ep_axis_name),  # w2_scale_hbm
            None if b1 is None else P(ep_axis_name),  # b1_hbm
            None if b2 is None else P(ep_axis_name),  # b2_hbm
            P(ep_axis_name),  # gating_output_hbm
            P(),  # a2a_g_hbm
        ),
        out_specs=P(ep_axis_name),
        check_vma=False,
    )
    def kernel(
        tokens,
        w1,
        w2,
        w1_scale,
        w2_scale,
        b1,
        b2,
        gating_output,
        a2a_g_hbm_scratch,
    ):
        return fused_moe(
            pltpu.with_memory_space_constraint(tokens, pltpu.HBM),  # tokens_hbm
            pltpu.with_memory_space_constraint(w1, pltpu.HBM),  # w1_hbm
            pltpu.with_memory_space_constraint(w2, pltpu.HBM),  # w2_hbm
            (None if w1_scale is None else pltpu.with_memory_space_constraint(w1_scale, pltpu.HBM)),  # w1_scale_hbm
            (None if w2_scale is None else pltpu.with_memory_space_constraint(w2_scale, pltpu.HBM)),  # w2_scale_hbm
            (None if b1 is None else pltpu.with_memory_space_constraint(b1, pltpu.HBM)),  # b1_hbm
            (None if b2 is None else pltpu.with_memory_space_constraint(b2, pltpu.HBM)),  # b2_hbm
            pltpu.with_memory_space_constraint(gating_output, pltpu.HBM),  # gating_output_hbm
            pltpu.with_memory_space_constraint(a2a_g_hbm_scratch, pltpu.HBM),  # a2a_g_hbm
        )

    a2a_g_hbm_scratch = pl.empty((num_experts, bt, t_packing, hidden_size // t_packing), t_dtype)
    return kernel(
        tokens,
        w1,
        w2,
        w1_scale,
        w2_scale,
        b1,
        b2,
        gating_output,
        a2a_g_hbm_scratch,
    )
