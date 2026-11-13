# This code is adapted from VERL
# https://github.com/volcengine/verl/blob/main/verl/utils/torch_functional.py
# The original copyright is reproduced below:
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from contextlib import nullcontext
import torch
import torch.nn.functional as F
from jaxtyping import Float, Integer

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


CHUNK_SIZE = 1024


class _EntropyFromLogits(torch.autograd.Function):
    """Memory-efficient entropy computation via activation checkpointing.

    Standard entropy computation with requires_grad=True retains the full
    (batch, seq, vocab_size) softmax tensor in the computation graph for
    backward. For large-vocab models (e.g. Qwen3 with 152K vocab), this
    can add ~18GB per 32K-token sequence in bf16.

    This custom autograd function saves only the raw logits (which are
    already stored by the model's backward pass) and recomputes softmax
    during backward, trading ~1 extra softmax computation for ~18GB of
    peak VRAM savings.

    Forward:  H = logsumexp(x) - sum(softmax(x) * x)
    Backward: dH/dx_i = p_i * (E[x] - x_i)  where p = softmax(x), E[x] = sum(p * x)
    """

    @staticmethod
    def forward(ctx, logits, attention_mask):
        # logits: (batch, seq_chunk, vocab)
        # Compute entropy using the numerically stable formulation
        p = F.softmax(logits, dim=-1)
        mean_logit = (p * logits).sum(-1)  # (batch, seq_chunk)
        entropy = torch.logsumexp(logits, dim=-1) - mean_logit  # (batch, seq_chunk)

        if attention_mask is not None:
            entropy = entropy * attention_mask

        # Save only logits and mask for backward — NOT the (batch, seq, vocab) softmax
        ctx.save_for_backward(logits, attention_mask)
        return entropy

    @staticmethod
    def backward(ctx, grad_output):
        logits, attention_mask = ctx.saved_tensors

        # Recompute softmax (the key memory saving — avoid storing it from forward)
        p = F.softmax(logits, dim=-1)
        mean_logit = (p * logits).sum(-1, keepdim=True)  # (batch, seq_chunk, 1)

        # dH/dx_i = p_i * (E[x] - x_i)
        grad_logits = p * (mean_logit - logits)  # (batch, seq_chunk, vocab)

        if attention_mask is not None:
            grad_logits = grad_logits * attention_mask.unsqueeze(-1)

        grad_logits = grad_logits * grad_output.unsqueeze(-1)
        return grad_logits, None  # None for attention_mask


def _memory_efficient_entropy_from_logits(logits, attention_mask=None):
    """Apply _EntropyFromLogits in chunks along the sequence dimension."""
    chunk_size = CHUNK_SIZE
    num_chunks = (logits.size(1) + chunk_size - 1) // chunk_size
    entropy_tensor = torch.zeros((logits.shape[0], logits.shape[1]), dtype=logits.dtype, device=logits.device)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, logits.size(1))
        chunk = logits[:, start_idx:end_idx]
        chunk_mask = attention_mask[:, start_idx:end_idx] if attention_mask is not None else None
        entropy_tensor[:, start_idx:end_idx] = _EntropyFromLogits.apply(chunk, chunk_mask)

    return entropy_tensor


def chunked_cross_entropy_from_log_probs(
    logprobs: Float[torch.Tensor, "batch_size seqlen vocab_size"], requires_grad: bool = False
) -> Float[torch.Tensor, "batch_size seqlen"]:
    cm = nullcontext() if requires_grad else torch.no_grad()
    with cm:
        # Calculate entropy in chunks to avoid OOM
        chunk_size = CHUNK_SIZE
        num_chunks = (logprobs.size(1) + chunk_size - 1) // chunk_size
        entropy_tensor = torch.zeros(
            (logprobs.shape[0], logprobs.shape[1]), dtype=logprobs.dtype, device=logprobs.device
        )

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, logprobs.size(1))
            # (bsz, seq, vocab_size)
            chunk = logprobs[:, start_idx:end_idx]

            # Calculate entropy for this chunk
            chunk_probs = chunk.exp()
            chunk_entropy = -(chunk_probs * chunk).sum(-1)
            entropy_tensor[:, start_idx:end_idx] = chunk_entropy
    return entropy_tensor


# NOTE: we don't actually use jaxtype for runtime type checking since it doesn't play well with torch compile
def chunked_entropy_from_logits(
    logits: Float[torch.Tensor, "batch_size seqlen vocab"],
    requires_grad: bool = False,
    attention_mask: Float[torch.Tensor, "batch_size seqlen"] = None,
) -> Float[torch.Tensor, "batch_size seqlen"]:
    """Chunked entropy calculation from logits.

    Avoids allocating a full log probabilities tensor to save memory. For models like Qwen with large vocab sizes, this can reduce gpu memory significantly (~O(10GB))

    Args:
        logits: Input logits of shape (batch_size, seqlen, vocab_size)
        requires_grad: Whether to enable gradient computation
        attention_mask: Optional attention mask of shape (batch_size, seqlen). When provided,
                       entropy values for padded positions (mask=0) will be zeroed out.

    Returns:
        Entropy tensor of shape (batch_size, seqlen). If attention_mask is provided,
        positions with mask=0 will have entropy=0.
    """
    # Validate attention mask shape if provided
    if attention_mask is not None:
        if attention_mask.shape != (logits.shape[0], logits.shape[1]):
            raise ValueError(
                f"attention_mask shape {attention_mask.shape} does not match logits shape "
                f"(batch_size={logits.shape[0]}, seqlen={logits.shape[1]}). "
                f"Expected attention_mask shape: ({logits.shape[0]}, {logits.shape[1]})"
            )

    if requires_grad:
        # Use memory-efficient custom autograd: recomputes softmax in backward
        # instead of storing the full (batch, seq, vocab_size) tensor.
        # Saves ~18GB for 32K-token Qwen3 (152K vocab) sequences.
        return _memory_efficient_entropy_from_logits(logits, attention_mask)

    with torch.no_grad():
        # Calculate entropy in chunks to avoid OOM
        chunk_size = CHUNK_SIZE
        num_chunks = (logits.size(1) + chunk_size - 1) // chunk_size
        entropy_tensor = torch.zeros((logits.shape[0], logits.shape[1]), dtype=logits.dtype, device=logits.device)

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, logits.size(1))
            # (bsz, seq, vocab_size)
            chunk = logits[:, start_idx:end_idx]
            chunk_logprob = F.log_softmax(chunk, dim=-1)

            # Calculate entropy for this chunk
            chunk_probs = chunk_logprob.exp()
            chunk_entropy = -(chunk_probs * chunk_logprob).sum(-1)

            # Apply attention mask if provided
            if attention_mask is not None:
                chunk_mask = attention_mask[:, start_idx:end_idx]
                chunk_entropy = chunk_entropy * chunk_mask

            entropy_tensor[:, start_idx:end_idx] = chunk_entropy
    return entropy_tensor


# Adapt from VERL
def logprobs_from_logits(
    logits: Float[torch.Tensor, "batch_size seqlen vocab_size"],
    labels: Integer[torch.Tensor, "batch_size seqlen"],
    inplace_backward=True,
) -> Float[torch.Tensor, "batch_size seqlen"]:
    """
    Compute per-token log-probabilities for the given labels.

    Uses a Flash-Attention–based cross-entropy (if available) for efficient backward,
    otherwise falls back to a standard log-softmax+gather approach.

    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591

    Args:
        logits (Tensor): Model outputs of shape (..., vocab_size).
        labels (LongTensor): True class indices of shape matching logits[..., :-1].
        inplace_backward (bool): If True and Flash-Attn is available, perform backward in-place.

    Returns:
        Tensor: Log-probabilities of the target labels, shape logits.shape[:-1].
    """
    if FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits = logits.reshape(-1, last_dim)
        labels = labels.reshape(-1)
        output = logprobs_from_logits_flash_attn(logits, labels, inplace_backward=inplace_backward)
        output = output.view(*batch_dim)
    else:
        output = logprobs_from_logits_v2(logits, labels)
    return output


def logprobs_from_logits_flash_attn(logits, labels, inplace_backward=True):
    output = cross_entropy_loss(logits, labels, inplace_backward=inplace_backward)
    assert isinstance(output, tuple), (
        "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    )
    return -output[0]


# Credits: https://www.tylerromero.com/posts/2025-02-selective-log-softmax/
# https://github.com/volcengine/verl/pull/220
def logprobs_from_logits_v2(
    logits: Float[torch.Tensor, "batch_size seqlen vocab_size"], labels: Integer[torch.Tensor, "batch_size seqlen"]
) -> Float[torch.Tensor, "batch_size seqlen"]:
    """
    A memory efficient implementation of logprobs_from_logits
    """
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(logit, dim=-1) for logit in logits])
        logprobs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        logprobs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_logprobs = F.log_softmax(row_logits, dim=-1)
            row_logprobs_labels = row_logprobs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            logprobs_labels.append(row_logprobs_labels)
        logprobs_labels = torch.stack(logprobs_labels)
    return logprobs_labels
