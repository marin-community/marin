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


from skyrl_train.distributed.megatron.stable_softmax import token_entropy, token_logprobs


def _parallel_logprobs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    group: dist.ProcessGroup,
    inference_only: bool,
    chunk_size: int | None,
) -> torch.Tensor:
    return token_logprobs(
        logits, targets, vocab_start_index, vocab_end_index, group, inference_only, chunk_size
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


def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor, chunk_size: Optional[int] = None) -> torch.Tensor:
    """Return fp32 per-token entropy over the tensor-parallel vocabulary."""
    return token_entropy(vocab_parallel_logits, mpu.get_tensor_model_parallel_group(), chunk_size)
