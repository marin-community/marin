# Utils ported from Verl
# https://github.com/volcengine/verl/blob/e1603dc97f3c20c58feed1f5be34acd5c72a830c/verl/utils/megatron_utils.py#L4
# https://github.com/volcengine/verl/blob/dfa3933ac44b545fca1f6a8519fd07394a2cde1c/verl/models/mcore/util.py
# The original copyright is reproduced below:

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import gc
import torch
import torch.nn as nn
from loguru import logger
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.transformer.module import Float16Module
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
from megatron.core import parallel_state as mpu
from megatron.core.utils import get_attr_wrapped_model
from megatron.core.packed_seq_params import PackedSeqParams

from skyrl_train.distributed.megatron.grug_muonh import MegatronGrugMuonH

ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def materialize_megatron_params(model_chunks: list[nn.Module]) -> None:
    """Finish deferred parameter gathers before checkpointing or exporting."""
    for module in model_chunks:
        if isinstance(module, DDP) and module.ddp_config.overlap_param_gather:
            module.start_param_sync(force_sync=True)


@torch.no_grad()
def dp_weight_checksum_mismatch(model_chunks: list[nn.Module]) -> float:
    """Return 1.0 when GPU parameter checksums differ across matching DP replicas, otherwise 0.0."""
    materialize_megatron_params(model_chunks)
    mismatch = torch.zeros((), dtype=torch.int64, device=torch.cuda.current_device())
    for expert in (False, True):
        parameters = [
            parameter.detach()
            for chunk in model_chunks
            for parameter in chunk.parameters()
            if bool(getattr(parameter, "allreduce", True)) != expert
        ]
        if not parameters:
            continue
        checksum = torch.stack(
            [
                value
                for parameter in parameters
                for value in (
                    parameter.sum(dtype=torch.float64),
                    torch.linalg.vector_norm(parameter, dtype=torch.float64),
                )
            ]
        )
        group = mpu.get_expert_data_parallel_group() if expert else mpu.get_data_parallel_group()
        replicas = [torch.empty_like(checksum) for _ in range(torch.distributed.get_world_size(group))]
        torch.distributed.all_gather(replicas, checksum, group=group)
        matches = torch.stack([torch.all(replica == checksum) for replica in replicas]).all()
        mismatch = torch.maximum(mismatch, (~matches).to(torch.int64))
    torch.distributed.all_reduce(mismatch, op=torch.distributed.ReduceOp.MAX)
    return float(mismatch.item())


def make_batch_generator(batches, vpp_size):
    """
    Creates a batch generator suitable for Megatron pipeline parallelism,
    handling virtual pipeline parallelism (VPP).

    If VPP is used (vpp_size > 1), it duplicates the batch iterator for each
    virtual pipeline stage. Otherwise, it returns a single iterator.

    Args:
        batches: An iterable (e.g., list) of micro-batches.
        vpp_size (int): The virtual pipeline model parallel size.

    Returns:
        An iterator or a list of iterators over the micro-batches.
    """
    if vpp_size > 1:
        # has vpp
        batch_generator = [batches] * vpp_size  # number of vpp chunks
        batch_generator = [iter(b) for b in batch_generator]
    else:
        # no vpp
        batch_generator = iter(batches)
    return batch_generator


def print_model_size(model: nn.Module, name: str = None):
    n_params, scale = get_model_size(model, scale="auto")
    if name is None:
        name = model.__class__.__name__
    logger.info(f"{name} contains {n_params:.2f}{scale} parameters")


def get_model_size(model: nn.Module, scale="auto"):
    n_params = sum(p.numel() for p in model.parameters())

    if scale == "auto":
        if n_params > 1e9:
            scale = "B"
        elif n_params > 1e6:
            scale = "M"
        elif n_params > 1e3:
            scale = "K"
        else:
            scale = ""

    if scale == "B":
        n_params = n_params / 1e9
    elif scale == "M":
        n_params = n_params / 1e6
    elif scale == "K":
        n_params = n_params / 1e3
    elif scale == "":
        pass
    else:
        raise NotImplementedError(f"Unknown scale {scale}")

    return n_params, scale


def freeze_moe_router(model):
    for layer in model.decoder.layers:
        if hasattr(layer.mlp, "router"):
            if hasattr(layer.mlp.router, "weight"):
                layer.mlp.router.weight.requires_grad = False
            if hasattr(layer.mlp.router, "bias"):
                layer.mlp.router.bias.requires_grad = False
        if hasattr(layer.mlp, "shared_experts"):
            if hasattr(layer.mlp.shared_experts, "gate_weight"):
                layer.mlp.shared_experts.gate_weight.requires_grad = False
            if hasattr(layer.mlp.shared_experts, "gate_bias"):
                layer.mlp.shared_experts.gate_bias.requires_grad = False


@torch.no_grad()
def offload_megatron_grads_to_cpu(models):
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in model_chunk_all_buffers:
                for buffer in buffers:
                    if buffer.grad_data.storage().size() > 0:
                        buffer.grad_data_size = buffer.grad_data.storage().size()
                        buffer.grad_data.storage().resize_(0)
        else:
            # we need this for ref module
            for _, param in model_chunk.named_parameters():
                if param.grad is not None:
                    param.grad = param.grad.to("cpu", non_blocking=True)
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def load_megatron_grads_to_gpu(models):
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in model_chunk_all_buffers:
                for buffer in buffers:
                    buffer.grad_data.storage().resize_(buffer.grad_data_size)
                    buffer.grad_data.zero_()
        else:
            # we need this for ref module
            for _, param in model_chunk.named_parameters():
                if param.grad is not None:
                    param.grad = param.grad.to(torch.cuda.current_device(), non_blocking=True)
    gc.collect()
    torch.cuda.empty_cache()


def _megatron_parameter_buffers(model_chunk: DDP) -> list:
    buffers = [*model_chunk.buffers, *model_chunk.expert_parallel_buffers]
    backed = [buffer for buffer in buffers if buffer.param_data is not None]
    if backed and len(backed) != len(buffers):
        raise RuntimeError("Megatron model has mixed sharded and unsharded parameter buffers")
    return backed


@torch.no_grad()
def offload_megatron_model_to_cpu(models):
    """
    In megatron, the model and optimizer storage are:
    - bf16 parameter data chunked in model parallel group
    - fp32 grad chunked in model parallel group
    - fp32 main_parameter chunked in model and dp group
    - fp32 optimizer state chunked in model and dp group
    """
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            backed = _megatron_parameter_buffers(model_chunk)
            if backed:
                for buffer in backed:
                    if buffer.param_data.storage().size() > 0:
                        buffer.param_data.cpu_data = torch.empty_like(buffer.param_data, device="cpu", pin_memory=True)
                        buffer.param_data.cpu_data.copy_(buffer.param_data)
                        buffer.param_data_size = buffer.param_data.storage().size()
                        buffer.param_data.storage().resize_(0)
                    assert buffer.param_data_size == buffer.param_data.cpu_data.storage().size()
                continue
        # Non-distributed Megatron optimizers and reference modules own parameter storage directly.
        for param in model_chunk.parameters():
            param.data = param.data.to("cpu", non_blocking=True)
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def load_megatron_model_to_gpu(models):
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            backed = _megatron_parameter_buffers(model_chunk)
            if backed:
                for buffer in backed:
                    if buffer.param_data.storage().size() == 0:
                        buffer.param_data.storage().resize_(buffer.param_data_size)
                        buffer.param_data.copy_(buffer.param_data.cpu_data, non_blocking=True)
                        del buffer.param_data.cpu_data
                continue
        device_id = torch.cuda.current_device()
        for param in model_chunk.parameters():
            param.data = param.data.to(device_id, non_blocking=True)
    gc.collect()
    torch.cuda.synchronize()
    torch.accelerator.memory.empty_host_cache()
    torch.cuda.empty_cache()


@torch.no_grad()
def offload_megatron_copy_params(optimizers):
    """
    Offload optimizer parameters to CPU. Supports both Megatron optimizers
    and `ChainedOptimizer`, which wraps a list of underlying optimizers.

    Args:
        optimizers: The optimizer or ChainedOptimizer instance.
    """

    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    def offload_tensor_to_cpu(tensor):
        if tensor is None:
            return
        tensor.data = tensor.data.to("cpu", non_blocking=True)

    def offload_group_to_cpu(group):
        if group is None:
            return

        if isinstance(group, list):
            for param_group in group:
                if isinstance(param_group, list):
                    for param in param_group:
                        offload_tensor_to_cpu(param)
                else:
                    offload_tensor_to_cpu(param_group)
        else:
            offload_tensor_to_cpu(group)

    # Offload all parameter groups to CPU for each underlying optimizer

    for _opt in _iter_opts(optimizers):
        if hasattr(_opt, "shard_fp32_from_float16_groups"):
            offload_group_to_cpu(_opt.shard_fp32_from_float16_groups)
        if hasattr(_opt, "fp32_from_float16_groups"):
            offload_group_to_cpu(_opt.fp32_from_float16_groups)


@torch.no_grad()
def load_megatron_copy_params(optimizers):
    """
    Load optimizer parameters back to GPU. Handles ChainedOptimizer.

    Args:
        optimizers: Optimizer or ChainedOptimizer instance.
    """

    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    def load_tensor_to_gpu(tensor):
        if tensor is None:
            return
        device_id = torch.cuda.current_device()
        tensor.data = tensor.data.to(device_id, non_blocking=True)

    def load_group_to_gpu(group):
        if group is None:
            return

        if isinstance(group, list):
            for param_group in group:
                if isinstance(param_group, list):
                    for param in param_group:
                        load_tensor_to_gpu(param)
                else:
                    load_tensor_to_gpu(param_group)
        else:
            load_tensor_to_gpu(group)

    # Load all parameter groups to GPU for each underlying optimizer

    for _opt in _iter_opts(optimizers):
        if hasattr(_opt, "shard_fp32_from_float16_groups"):
            load_group_to_gpu(_opt.shard_fp32_from_float16_groups)
        if hasattr(_opt, "fp32_from_float16_groups"):
            load_group_to_gpu(_opt.fp32_from_float16_groups)


@torch.no_grad()
def offload_megatron_optimizer(optimizers):
    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    for _opt in _iter_opts(optimizers):
        offload_megatron_copy_params(_opt)
        for state in _opt.optimizer.state.values():
            for name in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
                if name in state:
                    state[name] = state[name].to("cpu", non_blocking=True)
        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def load_megatron_optimizer(optimizers):
    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    for _opt in _iter_opts(optimizers):
        load_megatron_copy_params(_opt)
        # if we are using HybridDeviceOptimizer, we need to only move gpu optimizer state to gpu
        if hasattr(_opt.optimizer, "_move_new_state_to_right_device"):
            _opt.optimizer._move_new_state_to_right_device()
        else:
            keep_cpu_momentum = isinstance(_opt.optimizer, MegatronGrugMuonH) and _opt.optimizer.offload_momentum
            for state in _opt.optimizer.state.values():
                for name in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
                    if name in state and not (keep_cpu_momentum and name == "momentum_buffer"):
                        state[name] = state[name].to(torch.cuda.current_device(), non_blocking=True)
        gc.collect()
        torch.cuda.empty_cache()


def restore_offloaded_optimizer_state(optimizer) -> None:
    """Repair MCore 0.18's CPU-offloaded optimizer state after bucket restore."""
    # Revisit on MCore upgrades; remove once native restore passes two-update
    # continuation parity for mixed CPU/GPU state. Upstream step fix (partial):
    # https://github.com/NVIDIA/Megatron-LM/pull/7629
    optimizers = optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer]
    for distributed_optimizer in optimizers:
        inner = distributed_optimizer.optimizer
        if not isinstance(inner, HybridDeviceOptimizer):
            continue
        # MCore 0.18's bucket load overwrites CPU Adam's saved step with local
        # state. Only the inner GPU groups should retain step, or HDO resets it
        # before every update.
        for group in inner.param_groups:
            for parameter in group["params"]:
                state = inner.state[parameter]
                if "step" in state:
                    state["step"].fill_(group["step"])
            group.pop("step", None)
        # Bucket tensors replace state after HDO's load hooks. Rebind moments,
        # restore FP32 masters and place state on the native optimizers' devices.
        inner._sync_hdo_state_to_sub_optimizers()


def preprocess_packed_seqs(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, pre_process: bool = True
) -> tuple[torch.Tensor, PackedSeqParams]:
    """
    Preprocess packed sequences
    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0 gets first and last chunks, GPU1
    gets second and second last chunks, and so on), this is for load balancing with causal masking.
    See https://github.com/NVIDIA/TransformerEngine/issues/1368
    """
    batch_size = input_ids.shape[0]

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size

    pad_size = (align_size - seqlens_in_batch % align_size) % align_size
    seqlens_in_batch_padded = seqlens_in_batch + pad_size

    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)

    # ----------------------------------------------------------------------------
    # Move the index information needed in the subsequent loop to the CPU at once,
    # to avoid frequent .item() calls in the loop that cause D2H synchronization
    # ----------------------------------------------------------------------------
    seqlens_in_batch_cpu: list[int] = seqlens_in_batch.tolist()  # original valid lengths
    seqlens_in_batch_padded_cpu: list[int] = seqlens_in_batch_padded.tolist()  # lengths after padding
    cu_seqlens_padded_cpu: list[int] = cu_seqlens_padded.tolist()  # start positions (after padding)

    # Pure Python int calculation to avoid further synchronization
    max_seqlen_in_batch = max(seqlens_in_batch_padded_cpu)

    shape = list(input_ids.shape[1:])
    shape[0] = sum(seqlens_in_batch_padded_cpu) // cp_size
    if pre_process:
        input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
        for i in range(batch_size):
            # Use Python int, so no GPU→CPU sync in the loop
            if cp_size <= 1:
                seqlen = seqlens_in_batch_cpu[i]
                start_idx = cu_seqlens_padded_cpu[i]
                input_ids_rmpad[start_idx : start_idx + seqlen] = input_ids[i, attention_mask[i]]
                continue

            seqlen_padded_i = seqlens_in_batch_padded_cpu[i]
            seqlen = seqlen_padded_i // cp_size
            half_seqlen = seqlen // 2
            start_idx = cu_seqlens_padded_cpu[i] // cp_size
            # split to 2 chunks
            d = input_ids[i, attention_mask[i]]
            chunks = (
                (half_seqlen * cp_rank, half_seqlen * (cp_rank + 1), start_idx),
                (
                    seqlen_padded_i - half_seqlen * (cp_rank + 1),
                    seqlen_padded_i - half_seqlen * cp_rank,
                    start_idx + half_seqlen,
                ),
            )
            for source_start, source_end, destination_start in chunks:
                source_end = min(source_end, d.shape[0])
                chunk_len = source_end - source_start
                if chunk_len > 0:
                    input_ids_rmpad[destination_start : destination_start + chunk_len] = d[source_start:source_end]

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )
    if pre_process:
        return input_ids_rmpad.unsqueeze(0), packed_seq_params
    else:
        return input_ids, packed_seq_params


def pack_padded_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    packed_seq_params: PackedSeqParams,
) -> torch.Tensor:
    """Pack unsharded token IDs using the offsets consumed by packed logits."""
    cu_padded = packed_seq_params.cu_seqlens_q_padded.tolist()
    packed = torch.zeros((1, cu_padded[-1]), dtype=input_ids.dtype, device=input_ids.device)
    for i, (start, end) in enumerate(zip(cu_padded[:-1], cu_padded[1:], strict=True)):
        tokens = input_ids[i, attention_mask[i]]
        if tokens.numel() > end - start:
            raise ValueError("Packed token offsets are shorter than the unpadded sequence.")
        packed[0, start : start + tokens.numel()] = tokens
    return packed


def scatter_token_values(
    values: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    drop_last: bool,
) -> torch.Tensor:
    """Restore compact per-token values to their original padded positions."""
    output_len = attention_mask.shape[1] - int(drop_last)
    output = torch.zeros((attention_mask.shape[0], output_len), dtype=values.dtype, device=values.device)
    for i in range(attention_mask.shape[0]):
        positions = attention_mask[i].nonzero(as_tuple=False).flatten()
        if drop_last:
            positions = positions[:-1]
        if positions.numel() > 0:
            output[i, positions] = values[i, : positions.numel()]
    return output


def unpack_packed_token_values(
    output: torch.Tensor,
    packed_seq_params: PackedSeqParams,
    attention_mask: torch.Tensor,
    post_process: bool = True,
) -> torch.Tensor:
    """Restore packed scalar values without materializing padded vocab logits."""
    if not post_process:
        return output
    if output.ndim != 2 or output.shape[0] != 1:
        raise ValueError(f"Expected packed scalar values with shape [1, T], got {tuple(output.shape)}.")

    # -------------------------------------------------------------------------
    # Move the lengths and offsets needed for subsequent Python-level indexing to the CPU in advance,
    # to avoid a large number of .item() calls in the loop
    # -------------------------------------------------------------------------
    cu_padded_cpu: list[int] = packed_seq_params.cu_seqlens_q_padded.tolist()
    seq_lens_cpu: list[int] = attention_mask.sum(dim=1, dtype=torch.int32).cpu().tolist()

    batch_size, seq_len = attention_mask.shape
    shape = [batch_size, seq_len]
    output_new = torch.zeros(shape, dtype=output.dtype, device=output.device)

    cp_size = mpu.get_context_parallel_world_size()
    # All-gather scalar values across context-parallel ranks.
    if cp_size > 1:
        output_list = [torch.empty_like(output) for _ in range(cp_size)]
        torch.distributed.all_gather(output_list, output.detach(), group=mpu.get_context_parallel_group())
        output_list[mpu.get_context_parallel_rank()] = output
    else:
        output_list = [output]
    for i in range(batch_size):
        if cp_size <= 1:
            s = seq_lens_cpu[i]
            start_idx = cu_padded_cpu[i]
            output_new[i, attention_mask[i]] = output[0][start_idx : start_idx + s]
            continue
        s_len_padded_chunk = (cu_padded_cpu[i + 1] - cu_padded_cpu[i]) // cp_size
        half_seqlen = s_len_padded_chunk // 2
        s_len = seq_lens_cpu[i]
        s_len_padded = s_len_padded_chunk * cp_size
        tmp = torch.empty(s_len_padded, device=output.device, dtype=output.dtype)
        for j in range(cp_size):
            o = output_list[j][0]
            # split to 2 chunks
            packed_start_idx = cu_padded_cpu[i] // cp_size
            o0, o1 = (
                o[packed_start_idx : packed_start_idx + half_seqlen],
                o[packed_start_idx + half_seqlen : packed_start_idx + s_len_padded_chunk],
            )
            tmp[j * half_seqlen : (j + 1) * half_seqlen] = o0
            tmp[s_len_padded - (j + 1) * half_seqlen : s_len_padded - j * half_seqlen] = o1
        output_new[i, attention_mask[i]] = tmp[:s_len]

    return output_new


def compact_left_padded_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Remove left padding from token IDs for a vocab-reduced logprob computation."""
    seq_lens = attention_mask.sum(dim=1, dtype=torch.int32)
    compact_len = int(seq_lens.max().item())
    tp_size = mpu.get_tensor_model_parallel_world_size()
    compact_len += (tp_size - compact_len % tp_size) % tp_size
    compact = torch.zeros((input_ids.shape[0], compact_len), dtype=input_ids.dtype, device=input_ids.device)
    for i, seq_len in enumerate(seq_lens.tolist()):
        compact[i, :seq_len] = input_ids[i, attention_mask[i]]
    return compact


def remove_left_padding(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    pre_process: bool = True,
):
    """
    Remove left padding from input_ids, attention_mask and position_ids
    return new_input_ids, new_attention_mask, new_position_ids
    """
    assert attention_mask.ndim == 2
    assert position_ids.ndim == 2
    cp_size = mpu.get_context_parallel_world_size()
    assert cp_size == 1, "Context parallel size without seq_pack is not supported"
    batch_size = input_ids.shape[0]
    shape = list(input_ids.shape)  # batch_size, seq_len,...
    seq_lens = attention_mask.sum(dim=1)
    seq_len = seq_lens.max().item()
    if mpu.get_tensor_model_parallel_world_size() > 1:
        sp_world_size = mpu.get_tensor_model_parallel_world_size()
        pad_size = (sp_world_size - seq_len % sp_world_size) % sp_world_size
        seq_len = seq_len + pad_size
    shape[1] = seq_len
    if pre_process:
        new_input_ids = torch.zeros(dtype=input_ids.dtype, device=input_ids.device, size=shape)
    new_attention_mask = torch.zeros(
        dtype=attention_mask.dtype, device=attention_mask.device, size=(batch_size, seq_len)
    )
    new_position_ids = torch.zeros(dtype=position_ids.dtype, device=position_ids.device, size=(batch_size, seq_len))
    for i in range(batch_size):
        if pre_process:
            new_input_ids[i, : seq_lens[i]] = input_ids[i, attention_mask[i]]
        new_attention_mask[i, : seq_lens[i]] = attention_mask[i, attention_mask[i]]
        new_position_ids[i, : seq_lens[i]] = position_ids[i, attention_mask[i]]
    if pre_process:
        return new_input_ids, new_attention_mask, new_position_ids
    else:
        return input_ids, new_attention_mask, new_position_ids


def get_model_config(model):
    return get_attr_wrapped_model(model, "config", allow_none=False)
