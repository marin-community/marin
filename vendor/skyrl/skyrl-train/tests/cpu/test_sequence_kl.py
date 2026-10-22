"""The binned sequence-KL estimator against exact full-vocabulary KL."""

import math

import pytest
import torch

from skyrl_train.sequence_kl import (
    SequenceKLRow,
    binned_kl,
    exact_kl,
    position_subsample_mask,
)


def _log_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.double(), dim=-1)


def _reference_rows(full_logprobs: torch.Tensor, topk: int):
    """Top-K ids, their logprobs, and the retained mass's log, as the reference pass writes them."""
    values, ids = full_logprobs.topk(topk, dim=-1)
    retained = values.logsumexp(-1)
    return ids, values, retained


def test_identical_models_score_zero_kl_in_both_directions():
    torch.manual_seed(0)
    logits = torch.randn(5, 16, 32)
    full = _log_softmax(logits)
    _ids, topk_logprobs, retained = _reference_rows(full, topk=8)
    for direction in ("forward", "reverse"):
        kl = binned_kl(topk_logprobs, topk_logprobs, direction=direction, retained_log_mass=retained)
        assert torch.all(kl.abs() < 1e-9)


def test_binned_kl_matches_exact_kl_when_support_spans_the_vocabulary():
    torch.manual_seed(1)
    reference = _log_softmax(torch.randn(4, 24))
    candidate = _log_softmax(torch.randn(4, 24))
    for direction in ("forward", "reverse"):
        exact = exact_kl(reference, candidate, direction=direction)
        binned = binned_kl(reference, candidate, direction=direction, retained_log_mass=None)
        assert torch.allclose(binned, exact, atol=1e-9)


def test_binned_kl_with_a_tail_bin_approaches_exact_kl_as_k_grows():
    torch.manual_seed(2)
    reference = _log_softmax(torch.randn(3, 64))
    candidate = _log_softmax(torch.randn(3, 64))
    exact = exact_kl(reference, candidate, direction="forward")
    gaps = []
    for topk in (4, 16, 64):
        ids, topk_logprobs, retained = _reference_rows(reference, topk=topk)
        candidate_support = torch.gather(candidate, -1, ids)
        binned = binned_kl(topk_logprobs, candidate_support, direction="forward", retained_log_mass=retained)
        gaps.append((binned - exact).abs().max().item())
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 1e-6


def test_tail_mass_floor_bounds_the_reference_tail():
    torch.manual_seed(3)
    reference = _log_softmax(torch.randn(2, 8))
    candidate = _log_softmax(torch.randn(2, 8))
    ids, topk_logprobs, retained = _reference_rows(reference, topk=7)
    candidate_support = torch.gather(candidate, -1, ids)
    # The true tail is tiny; the floor keeps the bin's log finite and the KL finite.
    kl = binned_kl(topk_logprobs, candidate_support, direction="forward", retained_log_mass=retained)
    assert torch.isfinite(kl).all()
    exact = exact_kl(reference, candidate, direction="forward")
    assert torch.allclose(kl, exact, atol=1e-4)


def test_estimator_rejects_mismatched_shapes_and_unknown_directions():
    support = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="share a shape"):
        binned_kl(support, torch.zeros(2, 5), direction="forward")
    with pytest.raises(ValueError, match="forward or reverse"):
        binned_kl(support, support, direction="symmetric")


def test_position_subsample_is_seeded_and_never_empty():
    mask = position_subsample_mask(1000, rate=0.01, seed=11)
    again = position_subsample_mask(1000, rate=0.01, seed=11)
    other = position_subsample_mask(1000, rate=0.01, seed=12)
    assert torch.equal(mask, again)
    assert not torch.equal(mask, other)
    assert 1 <= mask.sum() <= 40
    assert position_subsample_mask(3, rate=0.01, seed=5).sum() >= 1
    with pytest.raises(ValueError, match="rate must be in"):
        position_subsample_mask(10, rate=0.0, seed=1)


def test_score_rows_carry_the_columnar_key_and_optional_tags():
    row = SequenceKLRow(
        sequence_id="P_chat-0007",
        position=42,
        token_id=311,
        kl=0.125,
        reference_logprob=-1.5,
        candidate_logprob=-1.75,
        token_type_tag=2,
        position_tag=4,
        exact_kl=None,
    )
    assert (row.sequence_id, row.position, row.token_id) == ("P_chat-0007", 42, 311)
    assert row.exact_kl is None and math.isfinite(row.kl)
