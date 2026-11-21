# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The forge cohort's declared calibration controls for the Atlas replay.

The cohort carries its authored golden control as private oracle resources
(``forge_release.golden_oracle_resources``): ``solution/solve.sh`` rewrites the
control's workspace files and ``solution/answer.txt`` holds its reply. This
module declares how the replay's oracle delivers that control per task shape.
Bind it with ``task_spec_harbor prepare --controls
experiments.post_training.task_curation.forge_controls:CONTROLS``.
"""

from taskcompendium.models import AnswerType, TaskSpec
from taskcompendium.pipeline.models import Controls, ControlSubmission, OracleCommand
from taskcompendium.runtime.resources import has_oracle_solution

SOLUTION_SCRIPT_COMMAND = "bash /solution/solve.sh"
SOLUTION_ANSWER_FILE = "/solution/answer.txt"
REPLY_ANSWER_TYPES = frozenset({AnswerType.TEXT, AnswerType.NUMBER, AnswerType.JSON})


def golden(task: TaskSpec) -> ControlSubmission | None:
    """The declared oracle delivery for one forge task.

    A task whose grader reads the reply (text, number, json) runs its solve
    script and submits the reply file as the answer; a workspace-graded task
    lets the solve script's files be the submission, so no command binding is
    needed. An actorless task without preserved solution material submits the
    calibration response the calibration-solutions directory supplies, so it
    declares nothing here.
    """
    machine = task.environment_requirements.command_semantics is not None
    if not has_oracle_solution(task):
        return None
    if task.answer_type not in REPLY_ANSWER_TYPES:
        return None
    return OracleCommand(
        command=SOLUTION_SCRIPT_COMMAND if machine else "true",
        answer_file=SOLUTION_ANSWER_FILE,
    )


CONTROLS = Controls(golden=golden)
