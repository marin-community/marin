"""Dependency-light tensor primitives shared by training objectives."""

from typing import Optional

import torch


LOG_PROB_DELTA_CLIP = 20.0


def safe_exp_delta(delta: torch.Tensor, clip: float = LOG_PROB_DELTA_CLIP, out_dtype=None) -> torch.Tensor:
    """Exponentiate a bounded log-probability delta without low-precision overflow."""
    result = torch.exp(delta.float().clamp(min=-clip, max=clip))
    return result.to(out_dtype or delta.dtype)


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: Optional[int] = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim).clamp(min=1.0)
