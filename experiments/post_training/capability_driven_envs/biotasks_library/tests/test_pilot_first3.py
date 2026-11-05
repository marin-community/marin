# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from taskcompendium.grading import grade_answer
from taskcompendium.models import ConversationTrace, GradingAttempt, TextMessage

from experiments.post_training.capability_driven_envs.biotasks_library.pilot_first3 import ideas, task_spec
from experiments.post_training.capability_driven_envs.biotasks_library.run_first3_zephyr import review_prompt
from experiments.post_training.capability_driven_envs.biotasks_library.taskforge_flow import run_idea


def test_first_three_sources_produce_stable_candidate_and_null_ideas():
    rows = ideas()

    assert len(rows) == 4
    assert len({row.idea_id for row in rows}) == 4
    assert [row.author_assessment.recommended_action for row in rows] == ["accept", "accept", "reject", "revise"]
    assert sum(row.null_reason is not None for row in rows) == 1
    assert len({row.source_document_key for row in rows}) == 3


def test_every_non_null_reference_grades_one_and_empty_json_grades_zero():
    for idea in ideas():
        if idea.null_reason is not None:
            continue
        task = task_spec(idea)
        reference = GradingAttempt(
            conversation=ConversationTrace(
                events=(*task.context.events, TextMessage(role="assistant", content=json.dumps(idea.expected)))
            )
        )
        negative = GradingAttempt(
            conversation=ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="{}")))
        )
        assert grade_answer(task, reference).reward == 1
        assert grade_answer(task, negative).reward == 0


@pytest.mark.asyncio
async def test_ideas_cross_taskforge_with_null_rejected_and_candidates_accepted(tmp_path):
    results = [await run_idea(tmp_path, idea) for idea in ideas()]

    assert [result["terminal"] for result in results] == ["accepted", "accepted", "held", "accepted"]
    assert sum("task_json" in result for result in results) == 3


def test_glm_review_packet_is_blind_to_author_assessment():
    idea = ideas()[0]
    packet = review_prompt(idea, {"terminal": "accepted", "task_json": task_spec(idea).model_dump_json()})

    assert "author_assessment" not in packet
    assert idea.author_assessment.rationale not in packet
    assert idea.idea_id in packet
