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

Both retained log masses derive on device from the stored top-K logprobs, so a
row's tail can never disagree with its support, and a peaked model keeps its
true (near-zero) tail instead of an fp32 ``1 - sum(exp)`` rounding artifact.

Rows compared across checkpoints must come from scoring passes that share one
engine, dtype and kernel set for both models, with a B-vs-B rescoring floor
pre-registered beside the subsample's exact-error bound; Phase 1a deltas are
otherwise comparable to engine and batch-layout logprob noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import torch

from experiments.post_training.mismatch_probe.protocol import request_seed

# Full-vocabulary logprobs must satisfy logsumexp == 0 within this tolerance, and a
# retained log mass may exceed 0 by no more than this float noise before the input
# is rejected as summing to more than the whole distribution.
NORMALIZATION_TOLERANCE = 1e-6


class KLForm(StrEnum):
    """Which model supplies the support; forward scores KL(reference ‖ candidate)."""

    FORWARD = "forward"
    REVERSE = "reverse"


class ProbeBucket(StrEnum):
    """Token bucket tags, mirrored from WP8's ``experiments/post_training/probe_sets/buckets.py``.

    WP8 (PR #10511) owns the values; reconcile this mirror when that PR lands.
    Only bucketed tokens are scored — fixed context carries no bucket — so every
    scored position tags with one of these.
    """

    REASONING = "reasoning"
    TOOL_NAME = "tool_name"
    TOOL_ARGUMENTS = "tool_arguments"
    FINAL_MESSAGE = "final_message"


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


def _relative_entropy(probability: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """KL contribution per bin with the 0·log(0) = 0 convention (objective/teacher.py)."""
    positive = probability > 0
    log_probability = torch.where(positive, probability, torch.ones_like(probability)).log()
    log_reference = torch.where(positive, reference, torch.ones_like(reference)).log()
    return torch.where(positive, probability * (log_probability - log_reference), torch.zeros_like(probability))


def _retained_log_mass(topk_logprobs: torch.Tensor) -> torch.Tensor:
    """The support's log mass, computed on device in float64 from the stored top-K logprobs."""
    retained = torch.logsumexp(topk_logprobs.double(), dim=-1)
    if bool((retained > NORMALIZATION_TOLERANCE).any()):
        raise ValueError("top-K logprobs sum to more than the full distribution (retained log mass > 0)")
    # A subset's retained mass is at most the whole distribution; anything in
    # (0, NORMALIZATION_TOLERANCE] is float noise for a support that carries all of
    # it, and would otherwise propagate a NaN tail through log(-expm1(x)).
    return retained.clamp(max=0.0)


def tail_log_mass(topk_logprobs: torch.Tensor) -> torch.Tensor:
    """The complement's log mass of a stored top-K support, derived from its logprobs.

    Public so row builders record exactly the tail the KL used. Never floored: a
    peaked model's tail is its own near-zero mass, not a constant.
    """
    retained = _retained_log_mass(topk_logprobs)
    return torch.where(
        retained < -math.log(2),
        torch.log1p(-retained.exp()),
        torch.log(-torch.expm1(retained)),
    )


def binned_kl(support_logprobs: torch.Tensor, scored_support_logprobs: torch.Tensor) -> torch.Tensor:
    """Per-position KL over the support model's top-K plus one tail bin.

    ``support_logprobs`` are the support model's stored top-K log probabilities
    (``[..., K]``); ``scored_support_logprobs`` are the other model's log
    probabilities gathered on the same ids (``[..., K]``). Both tails derive from
    these tensors on device and neither is floored. Forward KL(reference ‖
    candidate) passes the reference's top-K as the support; the reverse form is
    the role swap — the candidate's top-K and tail are the support and the
    reference is gathered on those ids.
    """
    if support_logprobs.shape != scored_support_logprobs.shape:
        raise ValueError("support and scored logprobs must share a shape")
    support_bins = torch.cat(
        (support_logprobs.double().exp(), tail_log_mass(support_logprobs).exp().unsqueeze(-1)), dim=-1
    )
    scored_bins = torch.cat(
        (scored_support_logprobs.double().exp(), tail_log_mass(scored_support_logprobs).exp().unsqueeze(-1)), dim=-1
    )
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
