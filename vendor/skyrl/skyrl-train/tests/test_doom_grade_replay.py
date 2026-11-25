# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
from tests.cpu.task_specs import text_task
from taskcompendium.models import AgentMachine, FileReward, RewardFile, RewardFileFormat, ScriptGrader
from taskcompendium.runtime.grading import file_grade

from experiments.post_training.mochi_doom_ftpo.grades import full_credit_task
from vendor.skyrl.scripts.replay_doom_grades import replay_row


@pytest.mark.parametrize("score,passed", [(0.0, False), (0.5, False), (math.nextafter(1.0, 0.0), False), (1.0, True)])
def test_calendar_file_scores_replay_the_native_full_credit_contract(score, passed):
    reward = FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),))
    grader = ScriptGrader(argv=("true",), placement=AgentMachine(), reward=reward)
    task = text_task().model_copy(update={"grader": grader})
    qualified = full_credit_task(task)
    qualified_reward = qualified.grader.reward
    native = file_grade(str(score).encode(), qualified_reward.files[0], qualified_reward, {})
    original = {"task_id": task.id, "response_token_ids": [1, 2, 3],
                "verification": {"status": "verified", "score": score, "passed": None}}
    replayed = replay_row(original, task)
    assert native.passed is passed
    assert replayed == {**original, "verification": {**original["verification"], "passed": passed}}
    assert task.grader.reward.pass_above is None
