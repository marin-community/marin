# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
from test_judge import fake_judge  # noqa: F401  (fixture)
from verifyit.adapters.openhands_next_action import NextActionJudge, TurnState, grade_next_action
from verifyit.grade import Status
from verifyit.modes.grade_judge import JudgeConnection

ROOT = "/workspace/acme__widgets__1.2"
STATE = TurnState(repo_root=ROOT, cwd=ROOT)


def _bash(command: str) -> dict:
    return {"name": "execute_bash", "arguments": json.dumps({"command": command})}


def _editor(command: str, path: str, **arguments) -> dict:
    return {"name": "str_replace_editor", "arguments": json.dumps({"command": command, "path": path, **arguments})}


def _reply(call: dict) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"function": call}]}


VIEW_CORE = _editor("view", f"{ROOT}/widgets/core.py")
SNIPPET_EXPERT = _bash(f'cd {ROOT} && python -c "import widgets; print(widgets.__version__)"')
SNIPPET_STUDENT = _bash("grep -n __version__ widgets/__init__.py")


@pytest.mark.parametrize(
    "expert,reply,components",
    [
        # Same purpose by another tool or other arguments.
        (
            _editor("view", f"{ROOT}/widgets/core.py", view_range=[1, 40]),
            _bash(f"cd {ROOT} && cat widgets/core.py"),
            (0, 1, 1, 0),
        ),
        (VIEW_CORE, _bash("sed -n '100,140p' widgets/core.py"), (0, 1, 1, 0)),
        (_editor("view", f"{ROOT}/widgets"), _bash(f"cd {ROOT} && ls -la widgets/"), (0, 1, 1, 0)),
        (
            _bash(f"cd {ROOT} && python -m pytest tests/test_core.py::TestGear::test_spin -xvs"),
            _bash("pytest tests/test_core.py -q"),
            (1, 1, 1, 0),
        ),
        (
            _bash(f'cd {ROOT} && grep -rn "def spin_gear" widgets/'),
            _bash("grep -n spin_gear widgets/core.py"),
            (1, 1, 1, 0),
        ),
        (
            _bash(f"cd {ROOT} && python reproduce_issue.py"),
            _bash("timeout 60 python reproduce_issue.py 2>&1 | tail -5"),
            (1, 1, 1, 0),
        ),
        (
            _editor(
                "str_replace", f"{ROOT}/widgets/core.py", old_str="    return gear\n", new_str="    return gear.copy()\n"
            ),
            _editor("str_replace", f"{ROOT}/widgets/core.py", old_str="def spin(gear):\n    return gear\n", new_str="x"),
            (1, 1, 1, 0),
        ),
        (
            _editor("create", f"{ROOT}/widgets/util.py", file_text="A"),
            _editor("create", f"{ROOT}/widgets/util.py", file_text="B"),
            (1, 1, 1, 0),
        ),
        (
            {"name": "finish", "arguments": json.dumps({"message": "done"})},
            {"name": "finish", "arguments": "{}"},
            (1, 1, 1, 0),
        ),
        # A different target, operation, or no action at all.
        (VIEW_CORE, _editor("view", f"{ROOT}/widgets/gears.py"), (1, 1, 0, 0)),
        (_bash('grep -rn "spin_gear" widgets'), _bash('grep -rn "load_config" widgets'), (1, 1, 0, 0)),
        (_bash("pytest tests/test_core.py"), _bash("python reproduce_issue.py"), (1, 0, 0, 0)),
        (
            _editor("create", f"{ROOT}/widgets/util.py", file_text="A"),
            _editor("create", f"{ROOT}/widgets/x.py", file_text="A"),
            (1, 1, 0, 0),
        ),
        (
            _editor("str_replace", f"{ROOT}/a.py", old_str="x = 1", new_str="y"),
            _editor("view", f"{ROOT}/a.py"),
            (1, 0, 0, 0),
        ),
        (VIEW_CORE, {"name": "think", "arguments": json.dumps({"thought": "hmm"})}, (0, 0, 0, 0)),
        (VIEW_CORE, {"name": "finish", "arguments": "{}"}, (0, 0, 0, 0)),
        (VIEW_CORE, None, (0, 0, 0, 0)),
        # Neither rule settles it; with the judge off it scores 0.
        (SNIPPET_EXPERT, SNIPPET_STUDENT, (1, 0, 0, 1)),
    ],
)
def test_grade_next_action_scores_by_purpose(expert, reply, components):
    message = {"role": "assistant", "content": "I will look", "tool_calls": []} if reply is None else _reply(reply)
    verdict = grade_next_action(expert, message, STATE)
    tool_name, same_operation, functional, undecided = components
    assert verdict.reward == functional, verdict.detail
    assert verdict.detail["components"] == {
        "tool_name": tool_name,
        "same_operation": same_operation,
        "functional": functional,
        "undecided": undecided,
    }


def test_relative_paths_resolve_against_the_working_directory():
    inside_package = TurnState(repo_root=ROOT, cwd=f"{ROOT}/widgets")
    assert grade_next_action(VIEW_CORE, _reply(_bash("cat core.py")), inside_package).reward == 1.0
    assert grade_next_action(VIEW_CORE, _reply(_bash("cat core.py")), STATE).reward == 0.0


def _judge() -> NextActionJudge:
    connection = JudgeConnection(base_url=os.environ["VERIFYIT_JUDGE_BASE_URL"], api_key="test-key")
    return NextActionJudge(model="fake/judge-9b", connection=connection)


def test_only_undecided_pairs_go_to_the_judge(fake_judge):  # noqa: F811
    fake_judge.replies = ["The candidate reads the same version string.\nSCORE: 1"]
    settled = grade_next_action(_editor("view", f"{ROOT}/a.py"), _reply(_bash("cat a.py")), STATE, judge=_judge())
    assert (settled.reward, fake_judge.prompts) == (1.0, [])
    verdict = grade_next_action(SNIPPET_EXPERT, _reply(SNIPPET_STUDENT), STATE, observation="1.2.0", judge=_judge())
    assert (verdict.reward, verdict.detail["reason"]) == (1.0, "judged")
    assert "1.2.0" in fake_judge.prompts[0]


def test_judge_outage_is_an_infrastructure_error_not_a_zero(fake_judge):  # noqa: F811
    fake_judge.http_status = 503
    verdict = grade_next_action(SNIPPET_EXPERT, _reply(SNIPPET_STUDENT), STATE, judge=_judge())
    assert verdict.status == Status.INFRA_ERROR
