# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run an in-process TaskSpec grader inside its enforced trusted worker."""

import argparse
import os
from pathlib import Path

from verifyit.modes.grade_judge import API_KEY_ENV, BASE_URL_ENV, MODEL_ENV, JudgeConnection

from taskcompendium.grading import grade_answer, grade_remote_judge_answer
from taskcompendium.grading_result import verdict_bytes
from taskcompendium.models import ConversationTrace, GradingAttempt, TaskSpec, grades_with_judge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--conversation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-reason")
    args = parser.parse_args()
    task = TaskSpec.model_validate_json(args.task.read_text())
    if task.grader.environment is not None:
        raise ValueError("The TaskSpec declares its own grading environment")
    attempt = GradingAttempt(
        ConversationTrace.model_validate_json(args.conversation.read_text()), stop_reason=args.stop_reason
    )
    if grades_with_judge(task.grader):
        connection = JudgeConnection(
            os.environ[BASE_URL_ENV], os.environ[API_KEY_ENV], default_model=os.environ[MODEL_ENV]
        )
        grade = grade_remote_judge_answer(task, attempt, connection)
    else:
        grade = grade_answer(task, attempt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(verdict_bytes(grade))


if __name__ == "__main__":
    main()
