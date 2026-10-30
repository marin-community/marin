# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline per-token KL between two models' frozen-sequence scores.

One scoring pass stores, per position, the top-K token ids and their log
probabilities; the other pass scores those same ids. This module reduces such
rows to per-token KL over one model's top-K support plus a single tail bin —
the same two-bin structure as the training-side teacher objectives in
``objective/teacher.py`` — and bounds the estimator with exact full-vocabulary
KL on a seeded subsample of scored positions.

The two divergences differ only in which model supplies the support. Forward
KL(reference ‖ candidate) scores the reference's top-K plus its tail; the
reverse form is a role swap — the candidate's top-K and tail are the support
and the reference is gathered on those ids. Neither tail is floored: a floor
replaces the tail's value with a constant, so a candidate tail of 0.01 against
a floored support tail adds about 0.23 nats the models did not produce.

Each tail is the scoring pass's own log mass outside the support, a logsumexp
over the complement of the full log-softmax. Deriving it from the stored top-K
as ``1 - sum(exp)`` cannot resolve a tail below the fp32 rounding of that sum
(about 1e-6): on the dev pair it rounded 28% of candidate tails to exactly zero
and their KL to infinity. ``binned_kl`` checks that each side's support and tail
sum to one distribution, so a tail cannot silently disagree with its support.

Rows compared across checkpoints must come from scoring passes that share one
engine, dtype and kernel set for both models, with a B-vs-B rescoring floor
pre-registered beside the subsample's exact-error bound; Phase 1a deltas are
otherwise comparable to engine and batch-layout logprob noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from experiments.post_training.mismatch_probe.protocol import request_seed

# Full-vocabulary logprobs must satisfy logsumexp == 0 within this tolerance.
NORMALIZATION_TOLERANCE = 1e-6
# A support's logprobs and its tail log mass, both from fp32 log-softmax over a
# ~128k vocabulary, must sum to one distribution within this float noise.
BIN_NORMALIZATION_TOLERANCE = 1e-5


class KLForm(StrEnum):
    """Which model supplies the support; forward scores KL(reference ‖ candidate)."""

    FORWARD = "forward"
    REVERSE = "reverse"


class ProbeBucket(StrEnum):
    """Token bucket tags, mirrored from WP8's ``experiments/post_training/probe_sets/buckets.py``.

    WP8 (PR #10511) owns the values; reconcile this mirror when that PR lands.
    Only bucketed tokens are scored — fixed context carries no bucket — so every
    scored position tags with one of these. ``assistant_text`` is earlier assistant
    content outside the structured buckets, and ``delimiter`` the generated
    structural tokens; both are assistant output and are scored.
    """

    REASONING = "reasoning"
    TOOL_NAME = "tool_name"
    TOOL_ARGUMENTS = "tool_arguments"
    FINAL_MESSAGE = "final_message"
    ASSISTANT_TEXT = "assistant_text"
    DELIMITER = "delimiter"


@dataclass(frozen=True)
class SequenceKLRow:
    """One scored position, keyed for columnar storage and tagged for probe analysis.

    ``token_type_tag`` is WP8's bucket for the token and ``position_tag`` its turn
    index; both are required because unbucketed positions are never scored.
    """

    sequence_id: str
    position: int
    token_id: int
    kl: float
    reference_logprob: float
    candidate_logprob: float
    candidate_step: int
    reference_id: str
    probe_hash: str
    top_k: int
    reference_tail_mass: float
    candidate_tail_mass: float
    form: KLForm
    token_type_tag: ProbeBucket
    position_tag: int
    exact_kl: float | None = None


@dataclass(frozen=True)
class SequenceFlipRow:
    """One scored position's greedy token against the reference's top-1.

    ``reference_top1_logprob`` is the reference's own confidence at the position,
    so summaries can report flip rates per reference-margin quartile.
    """

    sequence_id: str
    position: int
    reference_top1: int
    candidate_top1: int
    flipped: bool
    reference_top1_logprob: float
    candidate_top1_logprob: float
    candidate_step: int
    reference_id: str
    candidate_id: str
    probe_hash: str
    token_type_tag: ProbeBucket
    position_tag: int


def _relative_entropy(probability: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """KL contribution per bin with the 0·log(0) = 0 convention (objective/teacher.py)."""
    positive = probability > 0
    log_probability = torch.where(positive, probability, torch.ones_like(probability)).log()
    log_reference = torch.where(positive, reference, torch.ones_like(reference)).log()
    return torch.where(positive, probability * (log_probability - log_reference), torch.zeros_like(probability))


def _bins(support_logprobs: torch.Tensor, tail_logmass: torch.Tensor, name: str) -> torch.Tensor:
    """Probabilities of the K support bins and the tail bin, after checking they sum to one."""
    log_bins = torch.cat((support_logprobs.double(), tail_logmass.double().unsqueeze(-1)), dim=-1)
    total = torch.logsumexp(log_bins, dim=-1)
    if bool((total.abs() > BIN_NORMALIZATION_TOLERANCE).any()):
        worst = float(total.abs().max())
        raise ValueError(f"{name} support and tail do not sum to one distribution (|log total| up to {worst:.2e})")
    return log_bins.exp()


def binned_kl(
    support_logprobs: torch.Tensor,
    support_tail_logmass: torch.Tensor,
    scored_support_logprobs: torch.Tensor,
    scored_tail_logmass: torch.Tensor,
) -> torch.Tensor:
    """Per-position KL over the support model's top-K plus one tail bin.

    ``support_logprobs`` are the support model's stored top-K log probabilities
    (``[..., K]``) and ``support_tail_logmass`` its log mass outside them
    (``[...]``); ``scored_support_logprobs`` are the other model's log
    probabilities gathered on the same ids and ``scored_tail_logmass`` its log
    mass outside those ids. Forward KL(reference ‖ candidate) passes the
    reference's top-K as the support; the reverse form is the role swap — the
    candidate's top-K and tail are the support and the reference is gathered on
    those ids. Neither tail is floored: a scored tail that is truly zero against
    a positive support tail is an infinite divergence.
    """
    if support_logprobs.shape != scored_support_logprobs.shape:
        raise ValueError("support and scored logprobs must share a shape")
    if (
        support_tail_logmass.shape != support_logprobs.shape[:-1]
        or scored_tail_logmass.shape != support_tail_logmass.shape
    ):
        raise ValueError("each tail must hold one log mass per support row")
    support_bins = _bins(support_logprobs, support_tail_logmass, "support")
    scored_bins = _bins(scored_support_logprobs, scored_tail_logmass, "scored")
    return _relative_entropy(support_bins, scored_bins).sum(-1)


def exact_kl(reference_logprobs: torch.Tensor, candidate_logprobs: torch.Tensor) -> torch.Tensor:
    """Full-vocabulary per-position KL, the subsample bound for the binned estimator.

    Asserts both sides span the vocabulary: each side's log probabilities must sum
    to a normalized distribution. Partial support without a tail computes an
    unnormalized KL, which is why the binned form owns the tail bin instead.
    """
    if reference_logprobs.shape != candidate_logprobs.shape:
        raise ValueError("reference and candidate logprobs must share a shape")
    for name, logprobs in (("reference", reference_logprobs), ("candidate", candidate_logprobs)):
        total = torch.logsumexp(logprobs.double(), dim=-1)
        if bool((total.abs() > NORMALIZATION_TOLERANCE).any()):
            raise ValueError(f"{name} logprobs do not sum to a normalized distribution; pass the full vocabulary")
    return _relative_entropy(reference_logprobs.double().exp(), candidate_logprobs.double().exp()).sum(-1)


def exact_subsample_seed(master_seed: int, probe_hash: str) -> int:
    """The scored-position subsample's seed, from the probe's request-seed derivation."""
    return request_seed(master_seed, f"{probe_hash}:exact-subsample", 0)


def position_subsample_mask(num_scored_positions: int, *, rate: float, seed: int) -> torch.Tensor:
    """The seeded exact-scoring subsample over scored positions.

    The mask indexes the positions the binned estimator scored — the stored rows —
    not raw sequence positions. It is exactly the ``torch.rand`` draws, so an empty
    mask is a valid outcome and keeps the exact-KL bound unbiased.
    """
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"subsample rate must be in (0, 1], got {rate}")
    if num_scored_positions < 1:
        raise ValueError(f"num_scored_positions must be positive, got {num_scored_positions}")
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(num_scored_positions, generator=generator) < rate
