# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the bounded dummy feedback loop."""

from experiments.post_training.baby_rsi.dummy import MULTIPLICATION, run_dummy_rsi


def test_feedback_round_targets_failed_capability_and_improves_evaluation():
    result = run_dummy_rsi(rounds=1)

    assert [rollout.reward for rollout in result.initial_evaluation.rollouts] == [1.0, 0.0]
    assert [rollout.reward for rollout in result.final_evaluation.rollouts] == [1.0, 1.0]
    assert len(result.rounds) == 1

    round_result = result.rounds[0]
    assert [prompt.capability_id for prompt in round_result.prompts.prompts] == [MULTIPLICATION]
    evaluation_ids = {rollout.task_id for rollout in result.initial_evaluation.rollouts}
    training_ids = {task.id for task in round_result.training_tasks}
    assert evaluation_ids.isdisjoint(training_ids)
