# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import textwrap
from pathlib import Path

import pytest
from tasktrove_verify.modes import script
from tasktrove_verify.reward import InvalidTask, Status
from tasktrove_verify.spec import ScriptSpec

# The grading script is handed the workspace as its cwd and the three TASKTROVE_* variables.
BASH_REWARD_JSON = """\
#!/bin/bash
set -euo pipefail
answer=$(cat "$TASKTROVE_WORKSPACE/answer.txt")
[[ "$PWD" == "$TASKTROVE_WORKSPACE" ]] || exit 9
[[ -f "$TASKTROVE_TESTS_DIR/verifier.toml" ]] || exit 10
if [[ "$answer" == "$1" ]]; then reward=1.0; else reward=0.0; fi
printf '{"reward": %s, "note": "compared"}\\n' "$reward" > "$TASKTROVE_LOGS_DIR/reward.json"
# Lower-precedence channels disagree on purpose.
printf '0.25\\n' > "$TASKTROVE_LOGS_DIR/reward.txt"
echo 0.5
"""

PYTHON_REWARD_TXT = """\
import os
from pathlib import Path

logs = Path(os.environ["TASKTROVE_LOGS_DIR"])
(logs / "reward.txt").write_text("0.75\\n")
print("grading finished")
"""


def _tests_dir(tmp_path: Path, body: str, name: str) -> Path:
    tests = tmp_path / "tests"
    tests.mkdir(exist_ok=True)
    (tests / "verifier.toml").write_text(f'mode = "script"\npath = "{name}"\n')
    (tests / name).write_text(body)
    return tests


def _workspace(tmp_path: Path, answer: str = "42") -> Path:
    workspace = tmp_path / "app"
    workspace.mkdir(exist_ok=True)
    (workspace / "answer.txt").write_text(answer)
    return workspace


def test_reward_json_wins_over_reward_txt_and_stdout(tmp_path):
    tests = _tests_dir(tmp_path, BASH_REWARD_JSON, "test.sh")
    spec = ScriptSpec(path="test.sh", args=("42",))
    reward = script.grade(spec, tests, _workspace(tmp_path))
    assert (reward.reward, reward.status) == (1.0, Status.SCORED)
    assert reward.detail["channel"] == "reward.json"
    assert reward.detail["exit_code"] == 0


def test_reward_json_reports_a_wrong_answer_as_zero(tmp_path):
    tests = _tests_dir(tmp_path, BASH_REWARD_JSON, "test.sh")
    reward = script.grade(ScriptSpec(path="test.sh", args=("42",)), tests, _workspace(tmp_path, answer="7"))
    assert (reward.reward, reward.status) == (0.0, Status.SCORED)


def test_python_script_reports_through_reward_txt(tmp_path):
    tests = _tests_dir(tmp_path, PYTHON_REWARD_TXT, "grade.py")
    reward = script.grade(ScriptSpec(path="grade.py"), tests, _workspace(tmp_path))
    assert reward.reward == 0.75
    assert reward.detail["channel"] == "reward.txt"


def test_last_stdout_line_is_the_reward_of_last_resort(tmp_path):
    body = "#!/bin/bash\necho 'checking things'\necho 0.5\necho\n"
    tests = _tests_dir(tmp_path, body, "test.sh")
    reward = script.grade(ScriptSpec(path="test.sh"), tests, _workspace(tmp_path))
    assert reward.reward == 0.5
    assert reward.detail["channel"] == "stdout"


def test_reward_outside_the_unit_interval_scores_zero(tmp_path):
    body = '#!/bin/bash\nprintf \'{"reward": 7.0}\' > "$TASKTROVE_LOGS_DIR/reward.json"\n'
    tests = _tests_dir(tmp_path, body, "test.sh")
    reward = script.grade(ScriptSpec(path="test.sh"), tests, _workspace(tmp_path))
    assert (reward.reward, reward.status) == (0.0, Status.SCORED)
    assert reward.detail["reason"] == "reward_out_of_range"
    assert reward.detail["reported"] == 7.0


def test_failing_script_that_still_reported_a_reward_is_scored(tmp_path):
    body = '#!/bin/bash\nprintf \'{"reward": 0.5}\' > "$TASKTROVE_LOGS_DIR/reward.json"\necho boom >&2\nexit 3\n'
    tests = _tests_dir(tmp_path, body, "test.sh")
    reward = script.grade(ScriptSpec(path="test.sh"), tests, _workspace(tmp_path))
    assert (reward.reward, reward.status) == (0.5, Status.SCORED)
    assert reward.detail["exit_code"] == 3
    assert "boom" in reward.detail["stderr"]


def test_failing_script_without_a_reward_is_an_infra_error(tmp_path):
    body = "#!/bin/bash\necho 'no toolchain' >&2\nexit 2\n"
    tests = _tests_dir(tmp_path, body, "test.sh")
    with pytest.raises(RuntimeError, match="no toolchain"):
        script.grade(ScriptSpec(path="test.sh"), tests, _workspace(tmp_path))


def test_missing_script_is_an_invalid_task(tmp_path):
    tests = _tests_dir(tmp_path, "#!/bin/bash\n", "test.sh")
    with pytest.raises(InvalidTask, match=r"absent\.sh"):
        script.grade(ScriptSpec(path="absent.sh"), tests, _workspace(tmp_path))


def test_hanging_script_is_killed_with_its_children_at_the_timeout(tmp_path):
    # The script blocks in a background child that outlives it; only a process-group kill collects
    # the pipe, so a plain wait() here would hang instead of scoring a timeout.
    body = textwrap.dedent(
        """\
        #!/bin/bash
        sleep 120 &
        wait
        """
    )
    tests = _tests_dir(tmp_path, body, "test.sh")
    reward = script.grade(ScriptSpec(path="test.sh", timeout=0.5), tests, _workspace(tmp_path))
    assert (reward.reward, reward.status) == (0.0, Status.SCORED)
    assert reward.detail["reason"] == "timeout"


def test_explicit_spec_workspace_overrides_the_workspace_argument(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    tests = _tests_dir(tmp_path, "#!/bin/bash\ntouch ran-here\necho 1.0\n", "test.sh")
    reward = script.grade(ScriptSpec(path="test.sh", workspace=str(elsewhere)), tests, _workspace(tmp_path))
    assert reward.reward == 1.0
    assert (elsewhere / "ran-here").is_file()
