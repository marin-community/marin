# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.post_training.curriculum_sft.ablation.glm_generation import (
    CURRICULUM_PACKET,
    generation_body,
)
from experiments.post_training.curriculum_sft.ablation.matrix import (
    AblationCell,
    CurriculumCondition,
    GenerationSpec,
    SftDose,
    ablation_matrix,
    build_generation_prompt,
    build_held_out_tasks,
    generated_payloads_to_rows,
)
from experiments.post_training.curriculum_sft.ablation.tasks import synthetic_task, task_payload
from experiments.post_training.curriculum_sft.ablation.verifier import (
    evaluate_responses,
    score_response,
    task_payload_has_schema,
    verify_task_payload,
)


def _response(task, *, result=None, evidence=None):
    return {
        "result": task.expected_result if result is None else result,
        "evidence": list(task.evidence_ids) if evidence is None else evidence,
    }


def test_synthetic_task_has_exact_machine_checkable_oracle() -> None:
    task = synthetic_task(0)

    assert task.expected_result == {"gross_profit": 47047, "margin_bps": 4700}
    assert verify_task_payload(task_payload(task)).accepted


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("answer", {"gross_profit": 0, "margin_bps": 0}, "arithmetic_valid"),
        ("evidence", ["disclosure.revenue"], "evidence_valid"),
        ("question", "", "format_valid"),
    ],
)
def test_task_verifier_separates_quality_failures(field: str, value: object, expected: str) -> None:
    payload = task_payload(synthetic_task(1))
    payload[field] = value

    verification = verify_task_payload(payload)

    assert not getattr(verification, expected)
    assert not verification.accepted


def test_response_scorer_separates_arithmetic_evidence_and_format() -> None:
    task = synthetic_task(2)

    arithmetic = score_response(task, _response(task, result={"gross_profit": 1, "margin_bps": 1}))
    evidence = score_response(task, _response(task, evidence=["disclosure.operating_cost", "disclosure.revenue"]))
    malformed = score_response(task, "not json")

    assert arithmetic.format_valid and not arithmetic.arithmetic_valid and arithmetic.evidence_valid
    assert evidence.format_valid and evidence.arithmetic_valid and not evidence.evidence_valid
    assert not malformed.format_valid


def test_evaluation_summary_uses_deterministic_component_rates() -> None:
    tasks = [synthetic_task(index) for index in range(2)]
    responses = [_response(tasks[0]), _response(tasks[1], evidence=["disclosure.revenue"])]

    summary = evaluate_responses(tasks, responses)

    assert summary.count == 2
    assert summary.format_rate == 1.0
    assert summary.arithmetic_rate == 1.0
    assert summary.evidence_rate == 0.5
    assert summary.exact_rate == 0.5


def test_ablation_matrix_is_matched_across_three_factors() -> None:
    cells = ablation_matrix(accepted_examples=8)

    assert len(cells) == 8
    assert {cell.curriculum for cell in cells} == set(CurriculumCondition)
    assert {cell.generation_spec for cell in cells} == set(GenerationSpec)
    assert {cell.dose for cell in cells} == set(SftDose)
    assert {cell.accepted_examples for cell in cells} == {8}


def test_generation_prompt_exposes_only_requested_curriculum_condition() -> None:
    task_only = AblationCell(CurriculumCondition.TASK_ONLY, GenerationSpec.WEAK, SftDose.LOW)
    conditioned = AblationCell(CurriculumCondition.CURRICULUM_CONDITIONED, GenerationSpec.STRICT, SftDose.LOW)

    task_prompt = build_generation_prompt(
        task_only,
        subject_area="financial reporting",
        task_family="fictional company questions",
        curriculum_packet="PRIVATE CURRICULUM PACKET",
        task_count=3,
    )
    conditioned_prompt = build_generation_prompt(
        conditioned,
        subject_area="financial reporting",
        task_family="fictional company questions",
        curriculum_packet="PRIVATE CURRICULUM PACKET",
        task_count=3,
    )

    assert "PRIVATE CURRICULUM PACKET" not in task_prompt
    assert "PRIVATE CURRICULUM PACKET" in conditioned_prompt
    assert "gross profit" in conditioned_prompt
    assert "benchmark examples" in task_prompt


def test_live_generation_body_uses_pinned_curriculum_and_paired_seed() -> None:
    task_only = AblationCell(CurriculumCondition.TASK_ONLY, GenerationSpec.STRICT, SftDose.LOW)
    conditioned = AblationCell(CurriculumCondition.CURRICULUM_CONDITIONED, GenerationSpec.STRICT, SftDose.LOW)

    task_body = generation_body(task_only, expected_tasks=3, seed=23)
    conditioned_body = generation_body(conditioned, expected_tasks=3, seed=23)

    assert CURRICULUM_PACKET not in task_body["messages"][1]["content"]
    assert CURRICULUM_PACKET in conditioned_body["messages"][1]["content"]
    assert task_body["seed"] == conditioned_body["seed"] == 23
    assert task_body["temperature"] == conditioned_body["temperature"]
    assert task_body["tools"] == conditioned_body["tools"]


def test_payload_conversion_uses_one_canonical_target_for_every_generation_spec() -> None:
    strict_tasks = [synthetic_task(index) for index in range(4)]
    weak_tasks = [synthetic_task(index + 10) for index in range(4)]
    strict_payloads = [task_payload(task) for task in strict_tasks]
    weak_payloads = [task_payload(task) for task in weak_tasks]
    strict = AblationCell(CurriculumCondition.TASK_ONLY, GenerationSpec.STRICT, SftDose.LOW, accepted_examples=4)
    weak = AblationCell(CurriculumCondition.TASK_ONLY, GenerationSpec.WEAK, SftDose.LOW, accepted_examples=4)

    strict_rows = generated_payloads_to_rows(strict, strict_payloads)
    weak_rows = generated_payloads_to_rows(weak, weak_payloads)

    assert json.loads(strict_rows[0]["messages"][-1]["content"]) == {
        "result": strict_tasks[0].expected_result,
        "evidence": list(strict_tasks[0].evidence_ids),
    }
    assert strict_rows[0]["messages"][1] != weak_rows[0]["messages"][1]
    assert strict_rows[0]["metadata"]["deterministic_task_verification"] == "accepted"
    assert weak_rows[0]["metadata"]["deterministic_task_verification"] == "accepted"


def test_weak_conversion_rejects_generator_answer_errors() -> None:
    payload = task_payload(synthetic_task(0))
    payload["answer"] = {"gross_profit": 1, "margin_bps": 1}
    payload["evidence"] = ["A prose calculation", "Another prose calculation"]
    cell = AblationCell(CurriculumCondition.TASK_ONLY, GenerationSpec.WEAK, SftDose.LOW, accepted_examples=1)

    with pytest.raises(ValueError, match="unique oracle-accepted"):
        generated_payloads_to_rows(cell, [payload])


def test_weak_conversion_rejects_schema_valid_unsolvable_task() -> None:
    payload = task_payload(synthetic_task(0))
    payload["question"] = "Compute the requested measures from the disclosure."
    cell = AblationCell(CurriculumCondition.TASK_ONLY, GenerationSpec.WEAK, SftDose.LOW, accepted_examples=1)

    assert task_payload_has_schema(payload)
    assert not verify_task_payload(payload).format_valid
    with pytest.raises(ValueError, match="unique oracle-accepted"):
        generated_payloads_to_rows(cell, [payload])


def test_sft_dose_reuses_identical_accepted_examples() -> None:
    tasks = [synthetic_task(index) for index in range(4)]
    payloads = [task_payload(task) for task in tasks]
    low = AblationCell(CurriculumCondition.TASK_ONLY, GenerationSpec.STRICT, SftDose.LOW, accepted_examples=4)
    high = AblationCell(CurriculumCondition.TASK_ONLY, GenerationSpec.STRICT, SftDose.HIGH, accepted_examples=4)
    low_rows = generated_payloads_to_rows(low, payloads)
    high_rows = generated_payloads_to_rows(high, payloads)

    assert [row["metadata"]["source_task_id"] for row in low_rows] == [
        row["metadata"]["source_task_id"] for row in high_rows
    ]
    assert len(low_rows) == 4
    assert len(high_rows) == 4
    assert [row["messages"] for row in low_rows] == [row["messages"] for row in high_rows]
    for row in low_rows + high_rows:
        response = json.loads(row["messages"][-1]["content"])
        task = next(task for task in tasks if task.task_id == row["metadata"]["source_task_id"])
        assert score_response(task, response).accepted


def test_strict_conversion_requires_exactly_the_accepted_example_budget() -> None:
    valid = task_payload(synthetic_task(0))
    invalid = dict(valid, evidence=["disclosure.revenue"])
    cell = AblationCell(CurriculumCondition.TASK_ONLY, GenerationSpec.STRICT, SftDose.LOW, accepted_examples=2)

    with pytest.raises(ValueError, match="needed 2"):
        generated_payloads_to_rows(cell, [invalid, valid])


def test_task_verifier_accepts_comma_formatted_figures() -> None:
    payload = task_payload(synthetic_task(0))
    revenue = payload["facts"]["revenue"]
    operating_cost = payload["facts"]["operating_cost"]
    payload["question"] = f"Revenue was {revenue:,} and operating cost was {operating_cost:,}."

    assert verify_task_payload(payload).accepted


def test_task_verifier_requires_complete_integer_tokens() -> None:
    payload = task_payload(synthetic_task(0))
    payload["question"] = (
        f"Revenue was 9{payload['facts']['revenue']} and operating cost was " f"{payload['facts']['operating_cost']}0."
    )

    assert not verify_task_payload(payload).format_valid


def test_fenced_json_preserves_component_scores_but_fails_exact_format() -> None:
    task = synthetic_task(0)
    response = f"```json\n{json.dumps(_response(task))}\n```"

    score = score_response(task, response)

    assert not score.format_valid
    assert score.arithmetic_valid
    assert score.evidence_valid
    assert not score.accepted


def test_native_qwen_reasoning_envelope_scores_only_final_answer() -> None:
    task = synthetic_task(0)
    answer = json.dumps(_response(task))

    exact = score_response(task, f"<think>Reasoning may contain {{braces}}.</think>\n\n{answer}")
    fenced = score_response(task, f"<think>Reasoning.</think>\n\n```json\n{answer}\n```")

    assert exact.accepted
    assert not fenced.format_valid
    assert fenced.arithmetic_valid
    assert fenced.evidence_valid


def test_held_out_tasks_are_shared_and_disjoint_from_training() -> None:
    held_out = build_held_out_tasks(task_count=3, seed=17)
    training = [synthetic_task(index, seed=17) for index in range(3)]

    assert [task.task_id for task in held_out] == [task.task_id for task in build_held_out_tasks(task_count=3)]
    assert {task.task_id for task in held_out}.isdisjoint(task.task_id for task in training)
    assert all(verify_task_payload(task_payload(task)).accepted for task in held_out)


def test_held_out_fact_tuples_are_unique() -> None:
    held_out = build_held_out_tasks(task_count=64, seed=17)

    assert len({(task.revenue, task.operating_cost) for task in held_out}) == 64
