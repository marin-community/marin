"""Causal ShortConv with document boundaries and Megatron's zigzag CP layout."""

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as distributed_nn
import torch.nn.functional as F


def _convolve(x: torch.Tensor, weight: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    # Accumulate taps in FP32 and round once at the output.
    joined = torch.cat((history, x), dim=0).permute(1, 2, 0).float()
    kernel = weight.flip(0).T.unsqueeze(1).float()
    return F.conv1d(joined, kernel, groups=x.shape[-1]).permute(2, 0, 1).to(x.dtype)


def causal_short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    sequence_lengths: tuple[int, ...] | None = None,
    cp_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Convolve ``[sequence, batch, channel]`` without crossing document boundaries.

    Args:
        x: CP-local sequence, before any TP sequence sharding.
        weight: ``[tap, channel]`` weights, current-token tap first.
        sequence_lengths: Global padded lengths of packed documents. None means
            each batch column is one independent document.
        cp_group: Context group. Each document uses the two-chunk zigzag layout.

    Only the last kernel-width-minus-one tokens of each chunk are exchanged.
    The differentiable gather returns halo gradients to their source rank.
    """
    cp_size = 1 if cp_group is None else dist.get_world_size(cp_group)
    cp_rank = 0 if cp_group is None else dist.get_rank(cp_group)
    lengths = sequence_lengths or (x.shape[0] * cp_size,)
    if sum(lengths) != x.shape[0] * cp_size:
        raise ValueError("ShortConv document lengths do not cover the local sequence")
    if weight.ndim != 2 or weight.shape[1] != x.shape[-1]:
        raise ValueError("ShortConv weight must have shape [tap, channel]")
    halo = weight.shape[0] - 1
    if halo == 0:
        return (x.float() * weight[0].float()).to(x.dtype)
    results = []
    offset = 0
    for length in lengths:
        if length <= 0 or length % cp_size or (cp_size > 1 and length % (2 * cp_size)):
            raise ValueError("Packed ShortConv lengths must be positive and divisible by twice CP size")
        local = x[offset : offset + length // cp_size]
        offset += length // cp_size
        if cp_size == 1:
            results.append(_convolve(local, weight, x.new_zeros((halo, *x.shape[1:]))))
            continue
        first, second = local.chunk(2, dim=0)
        chunk_size = first.shape[0]
        tails = torch.stack((first[-halo:], second[-halo:])).contiguous()
        gathered = distributed_nn.all_gather(tails, group=cp_group)
        chunks = [gathered[r][0] for r in range(cp_size)] + [gathered[r][1] for r in reversed(range(cp_size))]
        for chunk, index in ((first, cp_rank), (second, 2 * cp_size - cp_rank - 1)):
            count = (halo + chunk_size - 1) // chunk_size
            preceding = chunks[max(0, index - count) : index]
            history = torch.cat([x.new_zeros((halo, *x.shape[1:])), *preceding], dim=0)[-halo:]
            results.append(_convolve(chunk, weight, history))
    return torch.cat(results, dim=0)
