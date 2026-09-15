# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from tasktrove_verify import grade as grade_module
from tasktrove_verify.grade import InvalidTask, Status
from tasktrove_verify.spec import McqSpec, Mode


def test_invalid_task_becomes_invalid_task_reward(tmp_path, monkeypatch):
    def broken(spec, tests_dir, workspace):
        raise InvalidTask("no reference")

    monkeypatch.setitem(grade_module.GRADERS, Mode.MCQ, broken)
    reward = grade_module.grade(McqSpec("A"), tmp_path, tmp_path)
    assert reward.status == Status.INVALID_TASK and reward.detail == {"error": "no reference"}
