# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
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


def _order(prompts, acquisition, held_out=(), *, reference=None, epsilon=0.2):
    return PromptThompsonOrder(
        prompts,
        set(held_out),
        reference or {},
        acquisition=acquisition,
        seed=22,
        window_size=12,
        decay=0.95,
        epsilon=epsilon,
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


def test_within_source_uses_the_same_cost_acquisition_on_a_single_source(prompts):
    original = _order(prompts, PromptAcquisition.COST_AWARE, epsilon=0.05)
    stratified = _order(prompts, PromptAcquisition.WITHIN_SOURCE_COST_AWARE, epsilon=0.05)
    for step in range(1, 5):
        observations = [
            _feedback(i, step, useful=i % 2 == 0, tokens=40 if i < 40 else 240)
            for i in (*range(20), *range(40, 60))
        ]
        assert original.finish_step(step, observations) == stratified.finish_step(step, observations)
        assert [original.next_index() for _ in range(12)] == [stratified.next_index() for _ in range(12)]
    with pytest.raises(ValueError, match="another training catalog or policy"):
        stratified.load_state_dict(original.state_dict())


@pytest.mark.parametrize("reference", [{}, {"tiny": 1000, "rare": 1, "common": 17}])
def test_within_source_preserves_reference_shares_and_pairs_source_windows(prompts, reference):
    prompts = [
        replace(prompt, source="tiny" if i < 2 else "rare" if i < 12 else "common")
        for i, prompt in enumerate(prompts)
    ]
    acquisition = PromptAcquisition.WITHIN_SOURCE_COST_AWARE
    ranked = _order(prompts, acquisition, reference=reference, epsilon=0.05)
    uniform = _order(prompts, acquisition, reference=reference, epsilon=1.0)
    ranked.finish_step(1, [_feedback(i, 1, useful=i < 12, tokens=20 if i < 12 else 200) for i in range(80)])

    # Reference counts are 2/10/68. With the override, the tiny source exhausts its two
    # rows first and the remaining ten slots divide in the ratio 1:17.
    expected = {"tiny": 2, "rare": 10/18, "common": 170/18} if reference else {
        "tiny": 12*2/80, "rare": 12*10/80, "common": 12*68/80,
    }
    totals = Counter()
    for _ in range(120):
        rows = [ranked.next_index() for _ in range(12)]
        assert len(set(rows)) == len(rows)
        sources = [prompts[i].source for i in rows]
        assert sources == [prompts[uniform.next_index()].source for _ in range(12)]
        counts = Counter(sources)
        for source, target in expected.items():
            assert abs(counts[source] - target) < 1
        totals.update(counts)
    for source, target in expected.items():
        assert totals[source] / 120 == pytest.approx(target, abs=0.15)
    for _ in range(5):
        ranked.next_index()
    restored = _order(prompts, acquisition, reference=reference, epsilon=0.05)
    restored.load_state_dict(json.loads(json.dumps(ranked.state_dict())))
    assert [ranked.next_index() for _ in range(48)] == [restored.next_index() for _ in range(48)]


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
