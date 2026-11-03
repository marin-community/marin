# Utils ported from NeMo-Aligner by way of NeMo-RL
# https://github.com/NVIDIA-NeMo/RL/blob/9301d36cbf847212430b84a27cfe6990f773b7cf/nemo_rl/distributed/model_utils.py#L4
# The original copyright is reproduced below:

#  Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Optional

import torch
import torch.distributed as dist
import megatron.core.parallel_state as mpu


@torch.no_grad()
def _compute_distributed_log_softmax(
    vocab_parallel_logits: torch.Tensor, group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Compute a stable distributed log softmax across tensor parallel workers.

    Taken from: https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L265

    Args:
        vocab_parallel_logits (torch.Tensor): Logits tensor with shape [batch_size, seq_length, vocab_size//TP]
            where TP is the tensor parallel size.
        group (torch.distributed.ProcessGroup): Process group for the all-reduce operations.

    Returns:
        torch.Tensor: Log softmax output with the same shape as input, but values represent
            log probabilities normalized across the full vocabulary dimension.
    """
    logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(
        logits_max,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )

    # Subtract the maximum value.
    vocab_parallel_logits = vocab_parallel_logits - logits_max

    sum_exp_logits = vocab_parallel_logits.exp().sum(-1, keepdim=True).float()

    torch.distributed.all_reduce(
        sum_exp_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=group,
    )

    return vocab_parallel_logits - sum_exp_logits.log_().to(vocab_parallel_logits.dtype)


class DistributedLogprob(torch.autograd.Function):
    """Custom autograd function for computing log probabilities in a distributed setting.

    Taken from https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L286
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]  Always ignore torch.autograd.Function.forward's type since it's always more specific than the base class
        ctx: Any,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        group: torch.distributed.ProcessGroup,
        inference_only: bool = False,
    ) -> torch.Tensor:
        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target - vocab_start_index
        masked_target[target_mask] = 0

        vocab_parallel_logits = vocab_parallel_logits.to(dtype=torch.float32)

        log_probs = _compute_distributed_log_softmax(vocab_parallel_logits, group=group)
        softmax_output = log_probs.exp()

        log_probs = torch.gather(log_probs, -1, masked_target.unsqueeze(-1)).squeeze(-1)
        log_probs[target_mask] = 0.0

        torch.distributed.all_reduce(
            log_probs,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )

        if not inference_only:
            # only save for backward when we have inference only=False
            ctx.save_for_backward(softmax_output, target_mask, masked_target)

        return log_probs

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        grad_output = grad_outputs[0]
        softmax, target_mask, masked_target = ctx.saved_tensors

        if softmax.ndim == 3:
            B, S, V = softmax.shape

            # skip `torch.nn.functional.one_hot`
            row = torch.arange(B, device=softmax.device).view(-1, 1).expand(-1, S).reshape(-1)
            col = torch.arange(S, device=softmax.device).expand(B, -1).reshape(-1)
            flat_idx = (row * S + col) * V

            flat_chosen = flat_idx.masked_select(~target_mask.reshape(-1)) + masked_target.masked_select(~target_mask)

            # `neg` is zero-copy
            grad_input = softmax.neg()
            grad_input = grad_input.mul_(grad_output.unsqueeze(-1))

            grad_output_selected = grad_output.masked_select(~target_mask)
            grad_input.view(-1).scatter_add_(0, flat_chosen, grad_output_selected)
        else:
            V = softmax.size(-1)
            is_chosen = (~target_mask).unsqueeze(-1) * torch.nn.functional.one_hot(masked_target, num_classes=V)
            grad_input = is_chosen.float().sub_(softmax)
            grad_input.mul_(grad_output.unsqueeze(-1))

        # if you add an argument to the forward method, then you must add a corresponding None here
        return grad_input, None, None, None, None, None, None


class ChunkedDistributedLogprob(torch.autograd.Function):
    """Custom autograd function for computing log probabilities in a distributed setting.

    The log probabilities computation is chunked in the sequence dimension
    to mitigate GPU OOM (especially during backward pass).
    In addition, logits casting from float16 or bfloat16 -> float32 is performed
    inside the chunk loop to avoid materializing a whole float32 logits tensor.

    Adapted from https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L286
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]  Always ignore torch.autograd.Function.forward's type since it's always more specific than the base class
        ctx: Any,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        chunk_size: int,
        tp_group: torch.distributed.ProcessGroup,
        inference_only: bool = False,
    ) -> torch.Tensor:
        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target - vocab_start_index
        masked_target[target_mask] = 0

        seq_size = int(vocab_parallel_logits.shape[1])
        num_chunks = (seq_size + chunk_size - 1) // chunk_size
        all_log_probs = []

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(seq_size, (chunk_idx + 1) * chunk_size)

            logits = vocab_parallel_logits[:, chunk_start:chunk_end, :]
            logits = logits.to(dtype=torch.float32)

            log_probs = _compute_distributed_log_softmax(
                logits,
                group=tp_group,
            )

            log_probs = torch.gather(log_probs, -1, masked_target[:, chunk_start:chunk_end].unsqueeze(-1)).squeeze(-1)
            log_probs[target_mask[:, chunk_start:chunk_end]] = 0.0

            torch.distributed.all_reduce(
                log_probs,
                op=torch.distributed.ReduceOp.SUM,
                group=tp_group,
            )

            all_log_probs.append(log_probs)

        log_probs = torch.cat(all_log_probs, dim=1)

        if not inference_only:
            # only save for backward when we have inference only=False
            ctx.save_for_backward(vocab_parallel_logits, target_mask, masked_target)
            ctx.chunk_size = chunk_size
            ctx.tp_group = tp_group

        return log_probs

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        grad_output = grad_outputs[0]
        vocab_parallel_logits, target_mask, masked_target = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        tp_group = ctx.tp_group

        partition_vocab_size = int(vocab_parallel_logits.shape[-1])
        seq_size = int(vocab_parallel_logits.shape[1])
        num_chunks = (seq_size + chunk_size - 1) // chunk_size

        # Autograd must receive one [B, S, V_local] gradient, but it need not be
        # fp32: the model activation is bf16/fp16 and PyTorch accumulates its
        # gradient in that dtype.  Every sequence chunk below overwrites its
        # destination exactly once, so ``empty_like`` is safe and avoids the
        # extra full fp32 buffer (12.9 GiB in the failing 30B RL batch).
        #
        # The temporary log-softmax and chosen-token mask remain fp32 *inside*
        # the bounded chunk loop; only the returned activation gradient uses
        # the model's native dtype.
        grad_input = torch.empty_like(vocab_parallel_logits)

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(seq_size, (chunk_idx + 1) * chunk_size)

            logits = vocab_parallel_logits[:, chunk_start:chunk_end, :]
            logits = logits.to(dtype=torch.float32)

            softmax_output = _compute_distributed_log_softmax(
                logits,
                group=tp_group,
            )
            softmax_output = softmax_output.exp()

            # 1 if it's the chosen log prob, 0 otherwise
            is_chosen = (~(target_mask[:, chunk_start:chunk_end])).unsqueeze(-1) * torch.nn.functional.one_hot(
                masked_target[:, chunk_start:chunk_end],
                num_classes=partition_vocab_size,
            )

            grad_input_chunk = grad_input[:, chunk_start:chunk_end, :]
            grad_input_chunk.copy_(is_chosen.float().sub_(softmax_output))
            grad_input_chunk.mul_(grad_output[:, chunk_start:chunk_end].unsqueeze(dim=-1))

            del softmax_output, is_chosen, logits

        # TODO: Investigate PrimeRL's streamed token-and-vocab LM-head backward
        # (`prime_rl/trainer/models/layers/lm_head.py:_SequenceChunkedLogProbEntropyFn`)
        # for a future design that also avoids a full vocab-parallel gradient buffer.

        # if you add an argument to the forward method, then you must add a corresponding None here
        return grad_input, None, None, None, None, None, None


def _parallel_logprobs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    group: dist.ProcessGroup,
    inference_only: bool,
    chunk_size: int | None,
) -> torch.Tensor:
    if logits.is_cuda and logits.dtype in (torch.bfloat16, torch.float32) and dist.get_world_size(group) == 1:
        # CPU test profiles import this module without the optional TE package.
        from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

        # TE reuses its model-dtype training workspace as the gradient.
        # Bound workspace copies during scoring, when no gradient is needed.
        with torch.set_grad_enabled(torch.is_grad_enabled() and not inference_only):
            if not torch.is_grad_enabled() and chunk_size is not None:
                return torch.cat(
                    [
                        -parallel_cross_entropy(x, y, dist_process_group=group, overwrite_input=False)
                        for x, y in zip(logits.split(chunk_size, dim=1), targets.split(chunk_size, dim=1), strict=True)
                    ],
                    dim=1,
                )
            return -parallel_cross_entropy(logits, targets, dist_process_group=group, overwrite_input=False)
    if chunk_size is not None:
        return ChunkedDistributedLogprob.apply(
            logits, targets, vocab_start_index, vocab_end_index, chunk_size, group, inference_only
        ).contiguous()
    return DistributedLogprob.apply(
        logits, targets, vocab_start_index, vocab_end_index, group, inference_only
    ).contiguous()


def from_parallel_logits_to_logprobs(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup,
    inference_only: bool = False,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Get log probabilities from TP+CP sharded vocab logits.

    Args:
        vocab_parallel_logits (torch.Tensor): Logits tensor with shape [batch_size, seq_len // CP, vocab_size // TP]
            where TP is the tensor parallel size.
        target (torch.Tensor): Target token indices with shape [batch_size, seq_len].
            NOTE: Must be the unmodified targets as this function will shift them internally.
        vocab_start_index (int): Starting vocabulary index for this worker's partition.
        vocab_end_index (int): Ending vocabulary index for this worker's partition.
        tp_group (torch.distributed.ProcessGroup): Process group for distributed communication.
        inference_only (bool, optional): If True, tensors won't be saved for backward pass. Defaults to False.
        cp_group (torch.distributed.ProcessGroup, optional): Context parallelism process group. Defaults to None.
        chunk_size (int, optional): Sequence dimension chunk size for computing the log probabilities.

    Returns:
        torch.Tensor: Log probabilities tensor with shape [batch_size, seq_len-1].
            The sequence dimension is reduced by 1 due to the target shifting.

    Taken from: https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L354
    """
    target = target.roll(shifts=-1, dims=-1)
    cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)
    # Pad the targets to local size * cp_size
    pad_len = vocab_parallel_logits.shape[1] * cp_size - target.shape[1]
    if pad_len > 0:
        target = torch.nn.functional.pad(target, (0, pad_len), value=0)

    # Shard the targets by context parallelism
    cp_rank = torch.distributed.get_rank(cp_group)
    target = _get_tokens_on_this_cp_rank(target, cp_rank, cp_size, seq_dim=1)

    logprobs = _parallel_logprobs(
        vocab_parallel_logits, target, vocab_start_index, vocab_end_index, tp_group, inference_only, chunk_size
    )

    if cp_size > 1:
        # we need to gather the logits by context parallelism
        logprobs = allgather_cp_sharded_tensor(logprobs, cp_group, seq_dim=1)  # , unpadded_seqlen=target.shape[1])

    if pad_len > 0:
        logprobs = logprobs[:, :-pad_len]

    return logprobs[:, :-1]


def from_parallel_logits_to_logprobs_packed_sequences(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    attention_mask: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    group: torch.distributed.ProcessGroup,
    inference_only: bool = False,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Get log probabilities from TP sharded vocab logits for packed sequences.

    Args:
        vocab_parallel_logits (torch.Tensor): Packed logits tensor with shape [1, T // CP, vocab_size//TP]
            where T is the total number of tokens across all packed sequences.
        target (torch.Tensor): Packed target token indices with shape [1, T].
            NOTE: Must be the unmodified targets as this function will shift them internally.
        cu_seqlens (torch.Tensor): Cumulative sequence lengths tensor with shape [batch_size + 1].
            cu_seqlens[i] indicates the start position of sequence i in the packed format.
        attention_mask (torch.Tensor): Original padded token positions. Values are
            scattered back to these positions after vocab reduction.
        vocab_start_index (int): Starting vocabulary index for this worker's partition.
        vocab_end_index (int): Ending vocabulary index for this worker's partition.
        group (torch.distributed.ProcessGroup): Process group for distributed communication.
        inference_only (bool, optional): If True, tensors won't be saved for backward pass. Defaults to False.
        cp_group (torch.distributed.ProcessGroup, optional): Context parallelism process group. Defaults to None.
        chunk_size (int, optional): Sequence dimension chunk size for computing the log probabilities.

    Returns:
        torch.Tensor: Log probabilities in original padded positions with shape
            [batch_size, sequence_length - 1].
    """
    # Remove batch dimension to work with [T, vocab_size] and [T]
    vocab_parallel_logits = vocab_parallel_logits.squeeze(0)
    target = target.squeeze(0)

    batch_size = cu_seqlens_padded.shape[0] - 1
    cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)
    cp_rank = 0 if cp_group is None else torch.distributed.get_rank(cp_group)

    # Roll each sequence individually
    rolled_targets = torch.zeros(target.shape[0] // cp_size, dtype=target.dtype, device=target.device)
    for i in range(batch_size):
        start_idx = cu_seqlens_padded[i].item()
        end_idx = cu_seqlens_padded[i + 1].item()

        # Get the sequence targets and roll by -1
        seq_targets = target[start_idx:end_idx]
        rolled_seq_targets = seq_targets.roll(shifts=-1, dims=0)
        rolled_targets[start_idx // cp_size : end_idx // cp_size] = _get_tokens_on_this_cp_rank(
            rolled_seq_targets, cp_rank, cp_size, seq_dim=0
        )

    # Logprob kernels consume [batch, sequence, vocabulary] tensors.
    rolled_targets = rolled_targets.unsqueeze(0)
    vocab_parallel_logits = vocab_parallel_logits.unsqueeze(0)

    # Apply distributed log probability computation
    probs = _parallel_logprobs(
        vocab_parallel_logits, rolled_targets, vocab_start_index, vocab_end_index, group, inference_only, chunk_size
    )

    # Remove batch dimension for filtering
    probs = probs.squeeze(0)

    # Ensure probs is 1D after squeezing
    if probs.dim() != 1:
        raise ValueError(
            f"Expected probs to be 1D after squeezing, but got shape {probs.shape}. "
            f"Original shape before squeeze: {probs.unsqueeze(0).shape}"
        )

    if cp_size > 1:
        # per-sequence cp_allgather
        final_probs = torch.zeros(probs.shape[0] * cp_size, device=probs.device)
        for i in range(batch_size):
            start_idx = cu_seqlens_padded[i].item()
            end_idx = cu_seqlens_padded[i + 1].item()
            final_probs[start_idx:end_idx] = allgather_cp_sharded_tensor(
                probs[start_idx // cp_size : end_idx // cp_size], cp_group, seq_dim=0
            )
        probs = final_probs

    out_logprobs = torch.zeros((batch_size, attention_mask.shape[1] - 1), dtype=probs.dtype, device=probs.device)
    # Filter out the last token of each sequence
    for i in range(batch_size):
        start_idx = cu_seqlens_padded[i].item()
        end_idx = cu_seqlens_padded[i + 1].item()

        # Exclude the last position (which has the rolled target from position 0)
        if end_idx - start_idx > 0:
            seq_probs = probs[start_idx : end_idx - 1]
            # Ensure seq_probs is 1D
            if seq_probs.dim() > 1:
                seq_probs = seq_probs.squeeze()

            positions = attention_mask[i].nonzero(as_tuple=False).flatten()[:-1]
            if positions.numel() > 0:
                out_logprobs[i, positions] = seq_probs[: positions.numel()]

    return out_logprobs


def _get_tokens_on_this_cp_rank(
    input_ids: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Get tokens on this context parallelism rank.

    Assumes that input_ids are already padded to a multiple of cp_size * 2 or cp_size == 1.

    Args:
        input_ids: Input token IDs [seq_length, ]
        cp_rank: Context parallelism rank
        cp_size: Context parallelism size

    Returns:
        Tokens on this context parallelism rank [1, seq_length // cp_size]
    """
    if cp_size == 1:
        return input_ids

    # load balance for causal attention
    shard_size = input_ids.shape[seq_dim] // (cp_size * 2)
    shard_inds = (cp_rank, (cp_size * 2) - cp_rank - 1)

    # Create slices for each dimension
    slices = [slice(None)] * input_ids.dim()
    ids_chunks = []

    for ind in shard_inds:
        slices[seq_dim] = slice(ind * shard_size, (ind + 1) * shard_size)
        ids_chunks.append(input_ids[slices])

    ids = torch.cat(ids_chunks, dim=seq_dim)
    return ids


def allgather_cp_sharded_tensor(tensor, cp_group, seq_dim=1):  # , unpadded_seqlen=None):
    return AllGatherCPTensor.apply(tensor, cp_group, seq_dim)  # , unpadded_seqlen)


class AllGatherCPTensor(torch.autograd.Function):
    def forward(
        ctx, tensor, cp_group: torch.distributed.ProcessGroup, seq_dim=1
    ):  # , unpadded_seqlen: Optional[int] = None):
        cp_size = torch.distributed.get_world_size(cp_group)
        cp_rank_chunks = []
        for _ in range(cp_size):
            cp_rank_chunks.append(torch.empty_like(tensor))

        torch.distributed.all_gather(tensor_list=cp_rank_chunks, tensor=tensor, group=cp_group)

        # undo the CP load balancing chunking
        tensor_chunks = []
        for logit_chunk in cp_rank_chunks:
            tensor_chunks.extend(torch.chunk(logit_chunk, chunks=2, dim=seq_dim))

        chunk_indices = []
        for cp_rank in range(cp_size):
            chunk_indices.append(cp_rank)
            chunk_indices.append(2 * cp_size - cp_rank - 1)

        chunks_and_indices = list(zip(tensor_chunks, chunk_indices))
        chunks_and_indices = sorted(chunks_and_indices, key=lambda tup: tup[1])
        ret_tensor = [chunk for chunk, _ in chunks_and_indices]
        ret_tensor = torch.cat(ret_tensor, dim=seq_dim)

        ctx.seq_dim = seq_dim
        ctx.cp_group = cp_group
        # ctx.unpadded_seqlen = unpadded_seqlen

        return ret_tensor

    def backward(ctx, grad_output):
        cp_size = torch.distributed.get_world_size(ctx.cp_group)
        cp_rank = torch.distributed.get_rank(ctx.cp_group)
        # chunk the seqdim in 2*cp chunks, and select with a CP load balanced indexing
        seq_dim = ctx.seq_dim
        # if ctx.unpadded_seqlen is not None:
        # # Zero out grad_output along the seq_dim after unpadded_seqlen
        # slicer = [slice(None)] * grad_output.dim()
        # slicer[seq_dim] = slice(ctx.unpadded_seqlen, None)
        #     grad_output[tuple(slicer)] = 0

        grad_output = grad_output.view(
            *grad_output.shape[0:seq_dim],
            2 * cp_size,
            grad_output.shape[seq_dim] // (2 * cp_size),
            *grad_output.shape[(seq_dim + 1) :],
        )

        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True).cuda(
            non_blocking=True
        )

        grad_input = grad_output.index_select(seq_dim, index)
        grad_input = grad_input.view(*grad_input.shape[0:seq_dim], -1, *grad_input.shape[(seq_dim + 2) :])

        return grad_input, None, None  # , None


# Below ported from https://github.com/volcengine/verl/blob/main/verl/utils/megatron/tensor_parallel.py#L109
class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        grad_input = vocab_parallel_logits.sub(sum_softmax_times_logits)
        grad_input.mul_(softmax_logits)
        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        grad_input.neg_()
        return grad_input


def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor, chunk_size: Optional[int] = None) -> torch.Tensor:
    """Compute entropy when the logits are sharded across tensor-parallel ranks.

    Args:
        vocab_parallel_logits: Logits ending in ``(sequence, vocab_size // tp_size)``.
        chunk_size: Optional bound on the sequence positions processed together.

    Returns:
        Per-token entropy with the vocabulary dimension removed.

    """
    if chunk_size is None or vocab_parallel_logits.shape[-2] <= chunk_size:
        return _VocabParallelEntropy.apply(vocab_parallel_logits)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    entropy_chunks = [
        _VocabParallelEntropy.apply(logits_chunk) for logits_chunk in vocab_parallel_logits.split(chunk_size, dim=-2)
    ]
    return torch.cat(entropy_chunks, dim=-1)
