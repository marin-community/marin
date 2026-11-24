"""Causal ShortConv with document boundaries and Megatron's zigzag CP layout."""

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as distributed_nn
import torch.nn.functional as F


def causal_short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    cp_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Convolve ``[sequence, batch, channel]`` without crossing document boundaries, all documents at once.

    Args:
        x: CP-local sequence, before any TP sequence sharding.
        weight: ``[tap, channel]`` weights, current-token tap first.
        cu_seqlens: Cumulative global padded lengths of the packed documents, ``[documents + 1]``,
            starting at zero and covering ``x.shape[0] * cp_size`` positions. Document lengths are
            unchecked: each must be positive and divisible by twice the CP size (by one without CP).
            None means one global document per batch column, partitioned across CP ranks.
        cp_group: Context group. Each document uses the two-chunk zigzag layout.

    Each chunk uses the preceding ``tap - 1`` tokens in its document, with zeros before the
    document's start. CP ranks exchange halos once per call when the kernel has multiple taps.
    """
    cp_size = 1 if cp_group is None else dist.get_world_size(cp_group)
    cp_rank = 0 if cp_group is None else dist.get_rank(cp_group)
    if weight.ndim != 2 or weight.shape[1] != x.shape[-1]:
        raise ValueError("ShortConv weight must have shape [tap, channel]")
    halo = weight.shape[0] - 1
    if halo == 0:
        return (x.float() * weight[0].float()).to(x.dtype)
    device = x.device
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, x.shape[0] * cp_size], device=device)
    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)
    chunks_per_document = 1 if cp_size == 1 else 2
    global_chunks = chunks_per_document * cp_size
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    chunk_size = (lengths // global_chunks).repeat_interleave(chunks_per_document)
    num_chunks = chunk_size.numel()
    chunk_start = torch.cumsum(chunk_size, 0) - chunk_size
    # This rank's chunks in local order, and each one's index among its document's global chunks.
    local_slot = torch.arange(num_chunks, device=device) % chunks_per_document
    global_index = torch.where(local_slot == 0, cp_rank, global_chunks - 1 - cp_rank)

    # The last ``halo`` tokens of every local chunk, right-aligned and zero-filled when the chunk is shorter.
    back = torch.arange(halo, 0, -1, device=device)
    tail_source = (chunk_start + chunk_size)[:, None] - back
    tail_valid = back <= chunk_size[:, None]
    tails = x[tail_source.clamp(min=0)] * tail_valid[..., None, None].to(x.dtype)
    if cp_size > 1:
        gathered = torch.stack(distributed_nn.all_gather(tails.contiguous(), group=cp_group))
    else:
        gathered = tails.unsqueeze(0)

    # History slot j of a chunk is the token ``halo - j`` positions before the chunk's first token.
    steps_back = back[None, :].expand(num_chunks, halo)
    source_global = global_index[:, None] - (steps_back + chunk_size[:, None] - 1) // chunk_size[:, None]
    from_end = (steps_back - 1) % chunk_size[:, None]
    in_document = source_global >= 0
    source_global = source_global.clamp(min=0)
    if cp_size > 1:
        first_half = source_global < cp_size
        source_rank = torch.where(first_half, source_global, global_chunks - 1 - source_global)
        source_slot = torch.where(first_half, 0, 1)
    else:
        source_rank = torch.zeros_like(source_global)
        source_slot = torch.zeros_like(source_global)
    document = torch.arange(num_chunks, device=device)[:, None] // chunks_per_document
    history = gathered[source_rank, document * chunks_per_document + source_slot, halo - 1 - from_end]
    history = history * in_document[..., None, None].to(x.dtype)

    # One sequence of [history, chunk] per chunk, so a single valid convolution never mixes chunks.
    chunk_of_token = torch.repeat_interleave(torch.arange(num_chunks, device=device), chunk_size)
    token_position = torch.arange(x.shape[0], device=device) + (chunk_of_token + 1) * halo
    history_position = (chunk_start + torch.arange(num_chunks, device=device) * halo)[:, None] + torch.arange(
        halo, device=device
    )
    joined = x.new_zeros((x.shape[0] + num_chunks * halo, *x.shape[1:]))
    joined = joined.index_copy(0, token_position, x)
    joined = joined.index_copy(0, history_position.reshape(-1), history.reshape(-1, *x.shape[1:]))
    kernel = weight.flip(0).T.unsqueeze(1).float()
    convolved = F.conv1d(joined.permute(1, 2, 0).float(), kernel, groups=x.shape[-1]).permute(2, 0, 1)
    return convolved[token_position - halo].to(x.dtype)
