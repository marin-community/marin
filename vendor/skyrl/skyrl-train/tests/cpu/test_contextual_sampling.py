# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
from dataclasses import replace

import pytest

from skyrlrecipe.declared import PromptAcquisition
from skyrl_train.contextual_sampling import PromptFeedback, PromptThompsonOrder, PublicPrompt


@pytest.fixture
def prompts():
    return [
        PublicPrompt(str(index), "math", f"{text} with inputs {index + 100} and 141.")
        for index, text in enumerate(
            ["Sort the numbers and put the smallest first"] * 40 + ["Compute the remainder after division"] * 40
        )
    ]


def _order(prompts, acquisition, held_out=(), *, reference=None, epsilon=0.2, groups_per_family=None):
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
        groups_per_family=groups_per_family,
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


def test_source_passes_cover_skewed_pools_and_replay_at_rollover(prompts):
    prompts = [replace(p, source="rare" if i < 5 else "common") for i, p in enumerate(prompts)]
    kind = PromptAcquisition.WITHIN_SOURCE_PASS_COST_AWARE
    weights = {"rare": 1, "common": 3}
    ranked = _order(prompts, kind, reference=weights, epsilon=0.05)
    uniform = _order(prompts, kind, reference=weights, epsilon=1.0)
    baseline = _order(prompts, PromptAcquisition.WITHIN_SOURCE_COST_AWARE, reference=weights)
    observations = [_feedback(i, 1, useful=i < 40, tokens=20 if i < 40 else 200) for i in range(80)]
    ranked.finish_step(1, observations)
    visits = [[], []]
    for _ in range(24):
        a, b, c = ([policy.next_index() for _ in range(12)] for policy in (ranked, uniform, baseline))
        assert len(set(a)) == len(set(b)) == 12
        assert [prompts[i].source for i in a] == [prompts[i].source for i in b] == [prompts[i].source for i in c]
        assert sum(i < 5 for i in a) == 3
        visits[0].extend(a)
        visits[1].extend(b)
    assert visits[0] != visits[1]
    for sequence in visits:
        for rows in (set(range(5)), set(range(5, 80))):
            within_source = [i for i in sequence if i in rows]
            for start in range(0, len(within_source), len(rows)):
                one_pass = within_source[start:start + len(rows)]
                assert len(set(one_pass)) == len(one_pass)
    for _ in range(5):
        ranked.next_index()
    restored = _order(prompts, kind, reference=weights, epsilon=0.05)
    restored.load_state_dict(json.loads(json.dumps(ranked.state_dict())))
    assert [ranked.next_index() for _ in range(96)] == [restored.next_index() for _ in range(96)]
    with pytest.raises(ValueError, match="another training catalog or policy"):
        baseline.load_state_dict(ranked.state_dict())


@pytest.mark.parametrize("reference", [{}, {"rare": 3, "common": 1}])
def test_family_windows_condition_the_reference_and_resume(prompts, reference):
    quotas = {"sorting": 8, "division": 12}
    prompts = [
        replace(
            prompt,
            family="sorting" if i < 40 else "division",
            source="rare" if i < 10 or 40 <= i < 70 else "common",
        )
        for i, prompt in enumerate(prompts)
    ]
    pools = {family: [i for i, prompt in enumerate(prompts) if prompt.family == family] for family in quotas}

    def order(epsilon):
        return _order(
            prompts,
            PromptAcquisition.WITHIN_SOURCE_COST_AWARE,
            reference=reference,
            epsilon=epsilon,
            groups_per_family=quotas,
        )

    ranked, random = order(0.05), order(1.0)
    ranked.finish_step(1, [_feedback(i, 1, useful=i < 30, tokens=20 if i < 30 else 200) for i in range(80)])
    expected_rare = {"sorting": 4, "division": 10.8} if reference else {"sorting": 2, "division": 9}

    def draw(sampler, family, live=()):
        return sampler.next_family_index(family, pools[family], lambda row: row not in live)

    for family, count in quotas.items():
        for _ in range(12):
            rows = [draw(ranked, family) for _ in range(count)]
            assert len(set(rows)) == len(rows)
            assert all(prompts[row].family == family for row in rows)
            sources = [prompts[row].source for row in rows]
            assert sources == [prompts[draw(random, family)].source for _ in range(count)]
            assert abs(sources.count("rare") - expected_rare[family]) < 1

    # One shared source has no capacity in this family. Cover the remaining rows without crossing families.
    live = {row for row in pools["sorting"] if prompts[row].source == "rare"}
    live.update(draw(ranked, "sorting", live) for _ in range(8))
    assert len(live) == 18
    draw(ranked, "division")
    restored = order(0.05)
    restored.load_state_dict(json.loads(json.dumps(ranked.state_dict())))
    assert draw(restored, "division", pools["division"]) is None
    delayed = [
        replace(_feedback(5, 1, useful=True, tokens=25), lease_id="prefetched"),
        _feedback(55, 2, useful=False, tokens=220),
    ]
    assert restored.finish_step(2, delayed) == ranked.finish_step(2, delayed)
    for family in ["sorting", "division"] * 30:
        assert draw(restored, family) == draw(ranked, family)


def test_held_out_content_cannot_join_the_training_catalog(prompts):
    with pytest.raises(ValueError, match="held-out public prompt"):
        _order(prompts, PromptAcquisition.CONTEXTUAL, held_out=[replace(prompts[0], uid="heldout").digest])
