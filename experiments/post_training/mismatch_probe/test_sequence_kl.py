# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The binned sequence-KL estimator against exact full-vocabulary KL."""

import hashlib
import math

import pytest
import torch

from experiments.post_training.mismatch_probe.protocol import request_seed
from experiments.post_training.mismatch_probe.sequence_kl import (
    KLForm,
    ProbeBucket,
    SequenceKLRow,
    binned_kl,
    exact_kl,
    exact_subsample_seed,
    position_subsample_mask,
    tail_log_mass,
)


def _normalized(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.double(), dim=-1)


def test_identical_models_score_zero_over_the_support_and_tail():
    torch.manual_seed(0)
    full = _normalized(torch.randn(5, 16, 32))
    support, _ids = full.topk(8, dim=-1)
    assert torch.all(binned_kl(support, support).abs() < 1e-9)


def test_forward_and_reverse_are_role_swaps_of_one_math():
    torch.manual_seed(1)
    reference = _normalized(torch.randn(4, 24))
    candidate = _normalized(torch.randn(4, 24))
    reference_topk, reference_ids = reference.topk(12, dim=-1)
    candidate_topk, candidate_ids = candidate.topk(12, dim=-1)
    forward = binned_kl(reference_topk, candidate.gather(-1, reference_ids))
    reverse = binned_kl(candidate_topk, reference.gather(-1, candidate_ids))
    # The forms score different supports, so the same pair of models gives two
    # different numbers, and each is a lower bound on its own exact divergence.
    assert not torch.allclose(forward, reverse)
    assert bool((forward <= exact_kl(reference, candidate) + 1e-9).all())
    assert bool((reverse <= exact_kl(candidate, reference) + 1e-9).all())


def test_binned_matches_exact_when_the_support_spans_the_vocabulary():
    # Powers of two so the support sums to exactly 1 in float64: zero retained mass
    # must be tolerated as an empty tail bin, and the match is nontrivial because
    # the exact KL is strictly positive.
    reference = torch.tensor(
        [[math.log(0.5), math.log(0.25), math.log(0.125), math.log(0.0625), math.log(0.03125), math.log(0.03125)]],
        dtype=torch.float64,
    )
    candidate = torch.tensor(
        [[math.log(0.25), math.log(0.25), math.log(0.25), math.log(0.125), math.log(0.0625), math.log(0.0625)]],
        dtype=torch.float64,
    )
    exact = exact_kl(reference, candidate)
    assert float(exact) > 0
    support, ids = reference.topk(reference.shape[-1], dim=-1)
    assert torch.allclose(binned_kl(support, candidate.gather(-1, ids)), exact, atol=1e-12)


def test_forward_gap_shrinks_monotonically_as_k_grows():
    torch.manual_seed(3)
    reference = torch.log_softmax(torch.randn(3, 96).double() * 5.0, dim=-1)
    candidate = torch.log_softmax(torch.randn(3, 96).double() * 5.0, dim=-1)
    exact = exact_kl(reference, candidate)
    gaps = []
    for top_k in (4, 16, 64):
        reference_topk, reference_ids = reference.topk(top_k, dim=-1)
        binned = binned_kl(reference_topk, candidate.gather(-1, reference_ids))
        gaps.append((binned - exact).abs().max().item())
    assert gaps[0] > gaps[1] > gaps[2] > 0.0
    # At K = V the gap closes exactly (powers-of-two test); below V the residual
    # tail bin's mass sets it, here around 1e-4.
    assert gaps[2] < 1e-3


def test_reverse_matches_the_exact_reverse_kl_as_k_grows():
    torch.manual_seed(4)
    reference = torch.log_softmax(torch.randn(3, 96).double() * 5.0, dim=-1)
    candidate = torch.log_softmax(torch.randn(3, 96).double() * 5.0, dim=-1)
    exact_reverse = exact_kl(candidate, reference)
    gaps = []
    for top_k in (4, 16, 64):
        candidate_topk, candidate_ids = candidate.topk(top_k, dim=-1)
        reverse = binned_kl(candidate_topk, reference.gather(-1, candidate_ids))
        gaps.append((reverse - exact_reverse).abs().max().item())
    assert gaps[0] > gaps[1] > gaps[2] > 0.0
    assert gaps[2] < 1e-3


def test_no_tail_is_floored():
    # A support tail below the removed 1e-12 floor must keep its own mass: flooring
    # would set the bin by the constant rather than by the model.
    tail = 1e-13
    reference = torch.tensor(
        [[math.log(0.25), math.log(0.25), math.log(0.5 - tail), math.log(tail)]], dtype=torch.float64
    )
    candidate = torch.tensor([[math.log(0.25), math.log(0.25), math.log(0.49), math.log(0.01)]], dtype=torch.float64)
    support, ids = reference.topk(3, dim=-1)
    assert float(tail_log_mass(support).exp()) == pytest.approx(tail, rel=1e-2, abs=0)
    assert torch.allclose(
        binned_kl(support, candidate.gather(-1, ids)), exact_kl(reference, candidate), rtol=1e-10, atol=0
    )


def test_tail_log_mass_is_derived_from_the_stored_logprobs():
    support = torch.tensor([[math.log(0.4), math.log(0.3), math.log(0.3 - 1e-13)]], dtype=torch.float64)
    assert torch.allclose(tail_log_mass(support).exp(), torch.tensor([1e-13], dtype=torch.float64), rtol=1e-2, atol=0)
    # A peaked support keeps its true near-zero tail; no floor substitutes a constant.
    peaked = torch.tensor([[0.0, -40.0, -40.0]], dtype=torch.float64)
    assert float(tail_log_mass(peaked).exp()) < 1e-15
    with pytest.raises(ValueError, match="more than the full distribution"):
        tail_log_mass(torch.zeros(1, 3))


def test_a_peaked_scored_model_gets_an_honest_divergence():
    torch.manual_seed(5)
    reference = _normalized(torch.randn(2, 8))
    reference_topk, reference_ids = reference.topk(6, dim=-1)
    # All candidate mass on the reference's top id: the candidate's tail is zero, so
    # forward KL is genuinely infinite wherever the reference keeps tail mass...
    peaked = torch.full_like(reference, -40.0).scatter(-1, reference_ids[..., :1], 0.0)
    assert bool(torch.isinf(binned_kl(reference_topk, peaked.gather(-1, reference_ids))).all())
    # ...and the role-swapped form is finite, because the support's own tail is zero.
    candidate_topk, candidate_ids = peaked.topk(6, dim=-1)
    reverse = binned_kl(candidate_topk, reference.gather(-1, candidate_ids))
    assert bool(torch.isfinite(reverse).all())
    assert bool((reverse >= 0).all())


def test_logprobs_summing_past_one_are_rejected():
    support = torch.zeros(2, 4)
    scored = torch.full((2, 4), -1.0)
    with pytest.raises(ValueError, match="more than the full distribution"):
        binned_kl(support, scored)
    with pytest.raises(ValueError, match="normalized distribution"):
        exact_kl(support, scored)


def test_exact_kl_matches_a_manual_computation_and_rejects_partial_support():
    torch.manual_seed(6)
    reference = _normalized(torch.randn(3, 16))
    candidate = _normalized(torch.randn(3, 16))
    manual = (reference.exp() * (reference - candidate)).sum(-1)
    assert torch.allclose(exact_kl(reference, candidate), manual, atol=1e-12)
    with pytest.raises(ValueError, match="normalized distribution"):
        exact_kl(reference[..., :8], candidate[..., :8])
    with pytest.raises(ValueError, match="share a shape"):
        exact_kl(reference, candidate[..., :8])
    with pytest.raises(ValueError, match="share a shape"):
        binned_kl(torch.zeros(2, 4), torch.zeros(2, 5))


def test_subsample_mask_is_seeded_over_scored_positions_and_stays_unbiased():
    mask = position_subsample_mask(1000, rate=0.01, seed=11)
    again = position_subsample_mask(1000, rate=0.01, seed=11)
    other = position_subsample_mask(1000, rate=0.01, seed=12)
    assert mask.shape == (1000,)
    assert torch.equal(mask, again)
    assert not torch.equal(mask, other)
    assert 1 <= int(mask.sum()) <= 40
    # No at-least-one backfill: the mask is exactly the raw draws, so a draw that
    # misses every scored position stays empty and the exact bound stays unbiased.
    generator = torch.Generator().manual_seed(0)
    raw_draw = torch.rand(5, generator=generator) < 0.01
    assert int(raw_draw.sum()) == 0
    assert torch.equal(position_subsample_mask(5, rate=0.01, seed=0), raw_draw)
    with pytest.raises(ValueError, match="rate must be in"):
        position_subsample_mask(10, rate=0.0, seed=1)
    with pytest.raises(ValueError, match="must be positive"):
        position_subsample_mask(0, rate=0.5, seed=1)


def test_exact_subsample_seed_matches_the_request_seed_derivation():
    probe_hash = "sha256:abc"
    prompt_id = f"{probe_hash}:exact-subsample"
    payload = f"42:{len(prompt_id)}:{prompt_id}:0".encode()
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    assert exact_subsample_seed(42, probe_hash) == expected
    assert exact_subsample_seed(42, probe_hash) == request_seed(42, prompt_id, 0)
    assert exact_subsample_seed(42, "sha256:abd") != expected


def test_rows_carry_the_full_scoring_identity_and_required_tags():
    row = SequenceKLRow(
        sequence_id="P_think:0007",
        position=42,
        token_id=311,
        kl=0.125,
        reference_logprob=-1.5,
        candidate_logprob=-1.75,
        candidate_step=120,
        reference_id="B@checkpoint-100",
        probe_hash="sha256:deadbeef",
        top_k=256,
        reference_tail_mass=1e-9,
        candidate_tail_mass=0.01,
        form=KLForm.FORWARD,
        token_type_tag=ProbeBucket.REASONING,
        position_tag=2,
    )
    assert (row.sequence_id, row.position, row.token_id) == ("P_think:0007", 42, 311)
    assert row.exact_kl is None
    # Tags are required columns, not optional: unbucketed positions are never scored.
    with pytest.raises(TypeError):
        SequenceKLRow(
            sequence_id="P_think:0007",
            position=42,
            token_id=311,
            kl=0.125,
            reference_logprob=-1.5,
            candidate_logprob=-1.75,
            candidate_step=120,
            reference_id="B@checkpoint-100",
            probe_hash="sha256:deadbeef",
            top_k=256,
            reference_tail_mass=1e-9,
            candidate_tail_mass=0.01,
            form=KLForm.FORWARD,
            position_tag=2,
        )
