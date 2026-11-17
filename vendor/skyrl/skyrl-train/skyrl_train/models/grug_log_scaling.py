"""Grug's position-dependent log scaling of global-layer queries.

A query that sees ``n`` keys of its own document is multiplied by
``(1 + alpha * ln(max(n / n_floor, 1))) ** power``. Like the JAX hero model,
``n`` counts from the document start, so it restarts at one in each packed sample.
"""

import torch


def grug_log_scaling_factor(visible_keys: torch.Tensor, n_floor: int, alpha: float, power: float) -> torch.Tensor:
    """Return the FP32 query multiplier for each query's visible-key count."""
    factor = 1.0 + alpha * torch.log(torch.clamp(visible_keys.float() / n_floor, min=1.0))
    return factor if power == 1.0 else factor**power


def thd_visible_keys(
    cu_seqlens_padded: torch.Tensor, num_tokens: int, cp_size: int, cp_rank: int
) -> torch.Tensor:
    """Return how many keys of its packed document each CP-local THD token sees, itself included.

    Megatron's THD context parallelism splits every padded document into
    ``2 * cp_size`` equal chunks and gives rank ``r`` chunks ``r`` and
    ``2 * cp_size - 1 - r``, in that order (``preprocess_packed_seqs``).
    ``num_tokens`` is the CP-local token count; passing it keeps this free of host syncs.
    """
    lengths = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    local_lengths = lengths // cp_size
    document = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=lengths.device), local_lengths, output_size=num_tokens
    )
    offset = torch.arange(num_tokens, device=lengths.device) - (cu_seqlens_padded[:-1] // cp_size)[document]
    if cp_size == 1:
        return offset + 1
    half = (local_lengths // 2)[document]
    position = torch.where(
        offset < half,
        cp_rank * half + offset,
        lengths[document] - (cp_rank + 2) * half + offset,
    )
    return position + 1
