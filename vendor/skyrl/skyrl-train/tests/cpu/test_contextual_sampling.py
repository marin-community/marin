# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest

from skyrl_train.contextual_sampling import PromptAcquisition, PromptFeedback, PromptThompsonOrder, PublicPrompt


@pytest.fixture
def prompts():
    return [
        PublicPrompt(str(index), "math", f"{text} with inputs {index + 100} and 141.")
        for index, text in enumerate(
            ["Sort the numbers and put the smallest first"] * 40 + ["Compute the remainder after division"] * 40
        )
    ]


def _order(prompts, acquisition, held_out=()):
    return PromptThompsonOrder(
        prompts,
        set(held_out),
        {},
        acquisition=acquisition,
        seed=22,
        window_size=12,
        decay=0.95,
        epsilon=0.2,
        reference_tokens=100,
    )


def _feedback(index, step, *, useful, tokens):
    return PromptFeedback(f"{step}/{index}", str(index), step, 2, (0.0, float(useful)), tokens, int(useful))


def test_contextual_posterior_transfers_within_source_to_unvisited_prompts(prompts):
    contextual = _order(prompts, PromptAcquisition.CONTEXTUAL)
    source = _order(prompts, PromptAcquisition.SOURCE)
    for step in range(1, 5):
        examples = [_feedback(index, step, useful=index < 40, tokens=100) for index in (*range(20), *range(40, 60))]
        contextual.finish_step(step, examples)
        source.finish_step(step, examples)

    assert source.prediction("30")[0] == source.prediction("70")[0]
    assert contextual.prediction("30")[0] > contextual.prediction("70")[0] + 0.5
    selected = [contextual.next_index() for _ in range(12)]
    assert len(set(selected)) == len(selected)
    assert sum(index < 40 for index in selected) >= 9
    assert any(20 <= index < 40 for index in selected)


def test_cost_aware_acquisition_prefers_cheaper_unvisited_prompts(prompts):
    contextual = _order(prompts, PromptAcquisition.CONTEXTUAL)
    cost_aware = _order(prompts, PromptAcquisition.COST_AWARE)
    for step in range(1, 5):
        examples = [
            _feedback(index, step, useful=index % 2 == 0, tokens=40 if index < 40 else 240)
            for index in (*range(20), *range(40, 60))
        ]
        contextual.finish_step(step, examples)
        cost_aware.finish_step(step, examples)

    assert cost_aware.prediction("30")[0] == pytest.approx(contextual.prediction("30")[0])
    assert cost_aware.prediction("30")[1] < 65
    assert cost_aware.prediction("70")[1] > 190
    contextual_expensive = sum(contextual.next_index() >= 40 for _ in range(120))
    cost_aware_expensive = sum(cost_aware.next_index() >= 40 for _ in range(120))
    assert cost_aware_expensive < contextual_expensive - 25


def test_resume_restores_pending_window_and_feedback_bookkeeping(prompts):
    original = _order(prompts, PromptAcquisition.COST_AWARE)
    for _ in range(5):
        index = original.next_index()
        original.assign(prompts[index].uid)
    good = _feedback(0, 1, useful=True, tokens=100)
    untrained = replace(good, lease_id="untrained", generated_tokens=35, nonzero_advantage_tokens=None)
    missing_grade = replace(good, lease_id="missing", generated_tokens=17, verified_scores=(1.0, None))
    zero_advantage = replace(good, lease_id="zero", generated_tokens=10, nonzero_advantage_tokens=0)
    metrics = original.finish_step(1, [good, untrained, missing_grade, zero_advantage])
    assert metrics["sampling/groups"] == 4
    assert metrics["sampling/fully_verified_groups"] == 3
    assert metrics["sampling/useful_groups"] == 1
    assert metrics["sampling/optimizer_groups"] == 3
    assert metrics["sampling/generated_tokens"] == 162

    restored = _order(prompts, PromptAcquisition.COST_AWARE)
    restored.load_state_dict(json.loads(json.dumps(original.state_dict())))
    with pytest.raises(ValueError, match="only once"):
        restored.finish_step(2, [good])

    following = [_feedback(1, 1, useful=False, tokens=25), _feedback(41, 2, useful=True, tokens=65)]
    assert restored.finish_step(2, following) == original.finish_step(2, following)
    assert [restored.next_index() for _ in range(36)] == [original.next_index() for _ in range(36)]


def test_held_out_content_cannot_join_the_training_catalog(prompts):
    with pytest.raises(ValueError, match="held-out public prompt"):
        _order(prompts, PromptAcquisition.CONTEXTUAL, held_out=[replace(prompts[0], uid="heldout").digest])
