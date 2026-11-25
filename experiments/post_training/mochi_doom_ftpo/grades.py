# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declare the campaign's full-credit correctness contract on native graders."""

import math

from taskcompendium.models import FileReward, RewardFileFormat, ScriptGrader, StdoutReward, TaskSpec, VerifyitGrader

# FileReward uses an exclusive threshold. Its adjacent float below one admits
# exactly full credit for native binary64 scores in the declared [0, 1] range.
FILE_FULL_CREDIT_THRESHOLD = math.nextafter(1.0, -math.inf)


def full_credit_task(task: TaskSpec) -> TaskSpec:
    """Declare full credit on the campaign's native programmatic graders."""
    grader = task.grader
    if isinstance(grader, VerifyitGrader):
        return task
    if not isinstance(grader, ScriptGrader):
        raise ValueError(f"Unexpected campaign grader for {task.id}: {grader.kind}")
    reward = grader.reward
    if isinstance(reward, StdoutReward):
        qualified_reward = StdoutReward(pass_at_least=1.0)
    elif isinstance(reward, FileReward) and all(file.format == RewardFileFormat.NUMBER for file in reward.files):
        qualified_reward = reward.model_copy(update={"pass_above": FILE_FULL_CREDIT_THRESHOLD})
    else:
        raise ValueError(f"Unexpected campaign reward for {task.id}: {reward.kind}")
    return task.model_copy(update={"grader": grader.model_copy(update={"reward": qualified_reward})})
