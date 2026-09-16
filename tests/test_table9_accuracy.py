# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.evaluation.olmo_base_eval.accuracy import choice_metrics, coverage_report, validate_task_samples
from marin.evaluation.olmo_base_eval.components import MT_MBPP_SUBTASKS, scored_tasks

from experiments.domain_phase_mix.evaluate_table9_accuracy import digest, evaluate_task, protocol


def test_choice_normalizations_can_disagree():
    scores = choice_metrics([-2.0, -3.0], [1, 3], [" a", " long answer"], 1)
    assert scores == {"acc": 0.0, "acc_per_token": 1.0, "acc_per_char": 1.0}
    with pytest.raises(ValueError, match="gold-only"):
        choice_metrics([-2.0], [1], [" a"], 0)


def test_deferred_languages_never_make_table9_complete():
    scores = {task: 0.5 for task in scored_tasks() if task not in MT_MBPP_SUBTASKS}
    report = coverage_report(scores)
    assert report["covered_components"] == 34
    assert report["requested_scope_complete"]
    assert not report["complete"]
    del scores["mmlu_anatomy"]
    report = coverage_report(scores)
    assert report["covered_components"] == 33
    assert report["missing"] == ["mmlu_other"]


def test_incomplete_samples_cannot_supply_a_task_score():
    samples = [{"doc_id": i, "metrics": {"exact_match": float(i == 0)}} for i in range(2)]
    assert validate_task_samples("minerva_math_algebra", samples, [0, 1], "exact_match") == 0.5
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        validate_task_samples("minerva_math_algebra", [samples[0], samples[0]], [0, 1], "exact_match")
    samples[1]["metrics"]["exact_match"] = float("nan")
    with pytest.raises(ValueError, match="Invalid"):
        validate_task_samples("minerva_math_algebra", samples, [0, 1], "exact_match")


def test_generation_requests_never_alias_the_frozen_plan_spec():
    """lm-eval appends the EOS string to `until` in place; the plan's protocol hash must survive a generation task."""

    class AppendingHarness:
        def generate_until(self, instances):
            for instance in instances:
                instance.args[1]["until"].append("<|end_of_text|>")
            return ["42"] * len(instances)

    spec = {"generation": {"max_gen_toks": 8, "temperature": 0, "seed": 0, "n": 1, "until": ["Problem:"]}}
    plan = {"schema_version": 1, "request_manifest": {"tasks": {"minerva_math_algebra": spec}}, "rows": []}
    before = digest(protocol(plan))
    requests = [{"task": "minerva_math_algebra", "doc_id": i, "context": f"Problem {i}"} for i in range(2)]
    samples = evaluate_task(AppendingHarness(), requests, spec)
    assert [s["generation"] for s in samples] == ["42", "42"]
    assert spec["generation"]["until"] == ["Problem:"]
    assert digest(protocol(plan)) == before
