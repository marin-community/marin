# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys

from tasktrove_verify import grade as grade_module
from tasktrove_verify.grade import InvalidTask, Status
from tasktrove_verify.spec import McqSpec, Mode


def test_mode_modules_import_on_first_use_only(monkeypatch):
    monkeypatch.setattr(grade_module, "GRADERS", {})
    for name in [m for m in sys.modules if m.startswith("tasktrove_verify.modes.grade_mcq")]:
        monkeypatch.delitem(sys.modules, name)
    assert "tasktrove_verify.modes.grade_mcq" not in sys.modules
    grader = grade_module.grader_for(Mode.MCQ)
    assert "tasktrove_verify.modes.grade_mcq" in sys.modules
    assert grade_module.grader_for(Mode.MCQ) is grader


def test_invalid_task_becomes_invalid_task_reward(tmp_path, monkeypatch):
    def broken(spec, tests_dir, workspace):
        raise InvalidTask("no reference")

    monkeypatch.setitem(grade_module.GRADERS, Mode.MCQ, broken)
    reward = grade_module.grade(McqSpec("A"), tmp_path, tmp_path)
    assert reward.status == Status.INVALID_TASK and reward.detail == {"error": "no reference"}
