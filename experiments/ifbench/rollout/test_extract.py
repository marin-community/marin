# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `extract.py` — synthetic rollouts, real verifier registry."""

from __future__ import annotations

from experiments.ifbench.data.prepare import PreparedRow, _stable_prompt_id
from experiments.ifbench.rollout.backend import Rollout
from experiments.ifbench.rollout.extract import (
    extract_pairs_and_sft,
    verify_rollouts,
)


def _row(prompt: str, instruction_id: str, kwargs: dict | None) -> PreparedRow:
    msgs = [{"role": "user", "content": prompt}]
    gt = repr([{"instruction_id": [instruction_id], "kwargs": [kwargs]}])
    return PreparedRow(
        prompt_id=_stable_prompt_id(msgs),
        messages=msgs,
        constraint=instruction_id,
        constraint_type="multi",
        ground_truth=gt,
        num_constraints=1,
    )


def _rollout(prompt_id: str, model: str, response: str) -> Rollout:
    return Rollout(
        prompt_id=prompt_id,
        model_id=model,
        backend="stub",
        response_text=response,
        finish_reason="stop",
        input_tokens=10,
        output_tokens=20,
        thinking_tokens=None,
        seed=0,
        sampling_config_hash="abc",
    )


def test_verify_rollouts_classifies_correctly() -> None:
    """Use a known constraint to confirm pass/fail classification works end-to-end."""
    prepared = _row("write something all lowercase", "change_case:english_lowercase", None)
    rollouts = [
        _rollout(prepared.prompt_id, "good_model", "the quick brown fox jumps"),
        _rollout(prepared.prompt_id, "bad_model", "Mixed Case Response Here"),
    ]
    verified = verify_rollouts(prepared, rollouts)
    assert len(verified) == 2
    by_model = {v.rollout.model_id: v for v in verified}
    assert by_model["good_model"].passes_all
    assert not by_model["bad_model"].passes_all


def test_extract_yields_pair_when_both_buckets_nonempty() -> None:
    prepared = _row("write all lowercase", "change_case:english_lowercase", None)
    rollouts_by_prompt = {
        prepared.prompt_id: [
            _rollout(prepared.prompt_id, "strong", "all good lowercase response here"),
            _rollout(prepared.prompt_id, "weak", "All Mixed Case Response Here"),
        ]
    }
    pairs, sft, stats = extract_pairs_and_sft([prepared], rollouts_by_prompt, seed=0)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.chosen_response == "all good lowercase response here"
    assert pair.chosen_model == "strong"
    assert pair.rejected_model == "weak"
    assert pair.rejected_failed_constraints == ["change_case:english_lowercase"]
    assert len(sft) == 1
    assert sft[0].response == "all good lowercase response here"
    assert stats.n_prompts_yielding_pair == 1
    assert stats.pair_yield == 1.0


def test_extract_skips_when_no_passers() -> None:
    prepared = _row("be lowercase", "change_case:english_lowercase", None)
    rollouts_by_prompt = {
        prepared.prompt_id: [
            _rollout(prepared.prompt_id, "weak1", "ALL UPPERCASE"),
            _rollout(prepared.prompt_id, "weak2", "MORE Uppercase Stuff"),
        ]
    }
    pairs, sft, stats = extract_pairs_and_sft([prepared], rollouts_by_prompt, seed=0)
    assert pairs == []
    assert sft == []
    assert stats.n_prompts_no_passers == 1
    assert stats.pair_yield == 0.0


def test_extract_skips_when_no_failers_but_keeps_sft() -> None:
    """If everything passes, no DPO pair, but every passing rollout is a valid SFT example."""
    prepared = _row("be lowercase", "change_case:english_lowercase", None)
    rollouts_by_prompt = {
        prepared.prompt_id: [
            _rollout(prepared.prompt_id, "good1", "all lowercase one"),
            _rollout(prepared.prompt_id, "good2", "all lowercase two"),
            _rollout(prepared.prompt_id, "good3", "all lowercase three"),
        ]
    }
    pairs, sft, stats = extract_pairs_and_sft([prepared], rollouts_by_prompt, seed=0)
    assert pairs == []
    assert len(sft) == 3
    assert stats.n_prompts_no_failers == 1
    assert stats.n_sft_examples == 3


def test_extract_handles_missing_rollouts() -> None:
    """If a prompt has no rollouts at all, skip cleanly."""
    prepared = _row("anything", "change_case:english_lowercase", None)
    pairs, sft, stats = extract_pairs_and_sft([prepared], {}, seed=0)
    assert pairs == []
    assert sft == []
    assert stats.n_prompts_no_passers == 1


def test_extract_records_skip_strata() -> None:
    """Skip counts are bucketed by num_constraints for diagnosis."""
    prepared = _row("be lowercase", "change_case:english_lowercase", None)
    rollouts_by_prompt = {
        prepared.prompt_id: [
            _rollout(prepared.prompt_id, "weak", "ALL UPPER"),
        ]
    }
    _, _, stats = extract_pairs_and_sft([prepared], rollouts_by_prompt, seed=0)
    assert stats.skip_by_num_constraints == {1: 1}
    assert stats.yield_by_num_constraints == {}


def test_extract_pair_yield_aggregate() -> None:
    """Yield is computed across multiple prompts."""
    rows = [_row(f"prompt {i} be lowercase", "change_case:english_lowercase", None) for i in range(10)]
    rollouts_by_prompt = {}
    for i, r in enumerate(rows):
        if i < 6:
            rollouts_by_prompt[r.prompt_id] = [
                _rollout(r.prompt_id, "strong", "all lowercase here please"),
                _rollout(r.prompt_id, "weak", "MIXED CASE"),
            ]
        else:
            # No passers
            rollouts_by_prompt[r.prompt_id] = [
                _rollout(r.prompt_id, "weak", "MIXED CASE"),
            ]
    pairs, _, stats = extract_pairs_and_sft(rows, rollouts_by_prompt, seed=0)
    assert len(pairs) == 6
    assert stats.pair_yield == 0.6
    assert stats.n_prompts_no_passers == 4
