"""Offline per-token KL between two models' frozen-sequence scores.

The reference pass stores, per position, the top-K token ids, their logprobs, and the
log of the remaining tail mass. The candidate pass scores the reference's top-K ids at
each position. This module reduces those rows to per-token KL over the reference's
top-K support plus one tail bin — the same two-bin structure as the training-side
teacher objectives in ``objective/teacher.py`` — and bounds the estimator with exact
full-vocabulary KL on a seeded subsample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# A partial-vocabulary tail has probability at least this floor, bounding its negative log;
# identical to the training-side teacher tail bound in objective/teacher.py.
TAIL_MASS_FLOOR = 1e-12


@dataclass(frozen=True)
class SequenceKLRow:
    """One scored position, keyed for columnar storage and tagged for probe analysis."""

    sequence_id: str
    position: int
    token_id: int
    kl: float
    reference_logprob: float
    candidate_logprob: float
    token_type_tag: int | None = None
    position_tag: int | None = None
    exact_kl: float | None = None


def _relative_entropy(probability: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """KL contribution per bin with the 0·log(0) = 0 convention."""
    positive = probability > 0
    log_probability = torch.where(positive, probability, torch.ones_like(probability)).log()
    log_reference = torch.where(positive, reference, torch.ones_like(reference)).log()
    return torch.where(positive, probability * (log_probability - log_reference), torch.zeros_like(probability))


def _tail_log_mass(retained_log_mass: torch.Tensor) -> torch.Tensor:
    """The complement's log probability, floored like the training-side teacher tail."""
    log_mass = retained_log_mass.clamp(max=math.log1p(-TAIL_MASS_FLOOR))
    return torch.where(
        log_mass < -math.log(2),
        torch.log1p(-log_mass.exp()),
        torch.log(-torch.expm1(log_mass)),
    )


def binned_kl(
    reference_topk_logprobs: torch.Tensor,
    candidate_support_logprobs: torch.Tensor,
    *,
    direction: str,
    retained_log_mass: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-position KL over the reference's top-K support plus one tail bin.

    ``reference_topk_logprobs`` and ``candidate_support_logprobs`` are ``[..., K]``
    log probabilities over the same top-K token ids; ``retained_log_mass`` is the
    log of the reference's remaining probability mass with the K dim removed
    (``[...]``, required unless the support spans the vocabulary). ``direction``
    selects the divergence form: ``forward`` scores KL(reference ‖ candidate),
    ``reverse`` scores KL(candidate ‖ reference).
    """
    if reference_topk_logprobs.shape != candidate_support_logprobs.shape:
        raise ValueError("reference and candidate support logprobs must share a shape")
    if direction not in ("forward", "reverse"):
        raise ValueError(f"direction must be forward or reverse, got {direction!r}")
    reference_support = reference_topk_logprobs.double().exp()
    candidate_support = candidate_support_logprobs.double().exp()
    if retained_log_mass is None:
        reference_bins = reference_support
        candidate_bins = candidate_support
    else:
        if retained_log_mass.shape != reference_support.shape[:-1]:
            raise ValueError("retained_log_mass must drop the top-K dimension")
        reference_tail = _tail_log_mass(retained_log_mass.double()).exp().unsqueeze(-1)
        # The candidate's tail can round to zero, which would make the forward form
        # infinite; floor it like the reference tail, whose mass is unmeasurable
        # below the same bound.
        candidate_tail = (1.0 - candidate_support.sum(-1, keepdim=True)).clamp(min=TAIL_MASS_FLOOR)
        reference_bins = torch.cat((reference_support, reference_tail), dim=-1)
        candidate_bins = torch.cat((candidate_support, candidate_tail), dim=-1)
    if direction == "forward":
        return _relative_entropy(reference_bins, candidate_bins).sum(-1)
    return _relative_entropy(candidate_bins, reference_bins).sum(-1)


def exact_kl(
    reference_logprobs: torch.Tensor,
    candidate_logprobs: torch.Tensor,
    *,
    direction: str,
) -> torch.Tensor:
    """Full-vocabulary per-position KL, the subsample bound for the binned estimator."""
    if reference_logprobs.shape != candidate_logprobs.shape:
        raise ValueError("reference and candidate logprobs must share a shape")
    return binned_kl(
        reference_logprobs,
        candidate_logprobs,
        direction=direction,
        retained_log_mass=None,
    )


def position_subsample_mask(num_positions: int, *, rate: float, seed: int) -> torch.Tensor:
    """The seeded exact-scoring subsample: roughly ``rate`` of positions, at least one."""
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"subsample rate must be in (0, 1], got {rate}")
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand(num_positions, generator=generator) < rate
    if not mask.any():
        mask[torch.randint(num_positions, (1,), generator=generator)] = True
    return mask
