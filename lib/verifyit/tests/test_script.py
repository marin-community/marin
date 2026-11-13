# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import textwrap
from pathlib import Path

import pytest
from verifyit.grade import InvalidTask, Status, main, run, scored, write_reward
from verifyit.modes import grade_script
from verifyit.spec import ScriptSpec, parse_spec, render_spec

# The grading script is handed the workspace as its cwd and the three VERIFYIT_* variables.
BASH_REWARD_JSON = """\
#!/bin/bash
set -euo pipefail
answer=$(cat "$VERIFYIT_WORKSPACE/answer.txt")
[[ "$PWD" == "$VERIFYIT_WORKSPACE" ]] || exit 9
[[ -f "$VERIFYIT_TESTS_DIR/verifier.toml" ]] || exit 10
if [[ "$answer" == "$1" ]]; then reward=1.0; else reward=0.0; fi
printf '{"reward": %s, "note": "compared"}\\n' "$reward" > "$VERIFYIT_LOGS_DIR/reward.json"
# Lower-precedence channels disagree on purpose.
printf '0.25\\n' > "$VERIFYIT_LOGS_DIR/reward.txt"
echo 0.5
"""

PYTHON_REWARD_TXT = """\
import os
from pathlib import Path

logs = Path(os.environ["VERIFYIT_LOGS_DIR"])
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
    reward = grade_script.grade(spec, tests, _workspace(tmp_path))
    assert (reward.reward, reward.status) == (1.0, Status.SCORED)
    assert reward.detail["channel"] == "reward.json"
    assert reward.detail["exit_code"] == 0


def test_malformed_reward_json_does_not_fall_back_to_another_channel(tmp_path):
    body = """#!/bin/bash
printf '{broken' > "$VERIFYIT_LOGS_DIR/reward.json"
printf '1.0' > "$VERIFYIT_LOGS_DIR/reward.txt"
"""
    tests = _tests_dir(tmp_path, body, "test.sh")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        grade_script.grade(ScriptSpec(path="test.sh"), tests, _workspace(tmp_path))


def test_reward_json_reports_a_wrong_answer_as_zero(tmp_path):
    tests = _tests_dir(tmp_path, BASH_REWARD_JSON, "test.sh")
    reward = grade_script.grade(ScriptSpec(path="test.sh", args=("42",)), tests, _workspace(tmp_path, answer="7"))
    assert (reward.reward, reward.status) == (0.0, Status.SCORED)


def test_python_script_reports_through_reward_txt(tmp_path):
    tests = _tests_dir(tmp_path, PYTHON_REWARD_TXT, "grade.py")
    reward = grade_script.grade(ScriptSpec(path="grade.py"), tests, _workspace(tmp_path))
    assert reward.reward == 0.75
    assert reward.detail["channel"] == "reward.txt"


def test_last_stdout_line_is_the_reward_of_last_resort(tmp_path):
    body = "#!/bin/bash\necho 'checking things'\necho 0.5\necho\n"
    tests = _tests_dir(tmp_path, body, "test.sh")
    reward = grade_script.grade(ScriptSpec(path="test.sh"), tests, _workspace(tmp_path))
    assert reward.reward == 0.5
    assert reward.detail["channel"] == "stdout"


def test_reward_outside_the_unit_interval_scores_zero(tmp_path):
    body = '#!/bin/bash\nprintf \'{"reward": 7.0}\' > "$VERIFYIT_LOGS_DIR/reward.json"\n'
    tests = _tests_dir(tmp_path, body, "test.sh")
    reward = grade_script.grade(ScriptSpec(path="test.sh"), tests, _workspace(tmp_path))
    assert (reward.reward, reward.status) == (0.0, Status.SCORED)
    assert reward.detail["reason"] == "reward_out_of_range"
    assert reward.detail["reported"] == 7.0


def test_failing_script_cannot_report_a_positive_reward(tmp_path):
    body = '#!/bin/bash\nprintf \'{"reward": 0.5}\' > "$VERIFYIT_LOGS_DIR/reward.json"\necho boom >&2\nexit 3\n'
    tests = _tests_dir(tmp_path, body, "test.sh")
    reward = grade_script.grade(ScriptSpec(path="test.sh"), tests, _workspace(tmp_path))
    assert (reward.reward, reward.status) == (0.0, Status.INFRA_ERROR)
    assert reward.detail["exit_code"] == 3
    assert "boom" in reward.detail["stderr"]


def test_failing_script_without_a_reward_is_an_infra_error(tmp_path):
    body = "#!/bin/bash\necho 'no toolchain' >&2\nexit 2\n"
    tests = _tests_dir(tmp_path, body, "test.sh")
    reward = grade_script.grade(ScriptSpec(path="test.sh"), tests, _workspace(tmp_path))
    assert reward.status == Status.INFRA_ERROR
    assert "no toolchain" in reward.detail["stderr"]


def test_missing_script_is_an_invalid_task(tmp_path):
    tests = _tests_dir(tmp_path, "#!/bin/bash\n", "test.sh")
    with pytest.raises(InvalidTask, match=r"absent\.sh"):
        grade_script.grade(ScriptSpec(path="absent.sh"), tests, _workspace(tmp_path))


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
    reward = grade_script.grade(ScriptSpec(path="test.sh", timeout=0.5), tests, _workspace(tmp_path))
    assert (reward.reward, reward.status) == (0.0, Status.SCORED)
    assert reward.detail["reason"] == "timeout"


def test_explicit_spec_workspace_overrides_the_workspace_argument(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    tests = _tests_dir(tmp_path, "#!/bin/bash\ntouch ran-here\necho 1.0\n", "test.sh")
    reward = grade_script.grade(ScriptSpec(path="test.sh", workspace=str(elsewhere)), tests, _workspace(tmp_path))
    assert reward.reward == 1.0
    assert (elsewhere / "ran-here").is_file()


@pytest.mark.parametrize("channel,payload", [("reward.json", '{"reward": "broken"}'), ("reward.txt", "broken")])
def test_invalid_authoritative_reward_cannot_be_replaced_by_stdout(tmp_path, channel, payload):
    body = (
        "import os\nfrom pathlib import Path\n"
        f'(Path(os.environ["VERIFYIT_LOGS_DIR"]) / {channel!r}).write_text({payload!r})\n'
        "print(1.0)\n"
    )
    tests = _tests_dir(tmp_path, body, "grade.py")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "reward.json").write_text('{"reward": 1.0}')
    (logs / "reward.txt").write_text("1.0")
    reward = run(tests / "verifier.toml", _workspace(tmp_path))
    write_reward(logs, reward)
    assert reward.status == Status.INFRA_ERROR
    assert json.loads((logs / "verdict.json").read_text())["status"] == "infra_error"
    assert not (logs / "reward.json").exists()
    assert not (logs / "reward.txt").exists()


def test_named_reward_survives_spec_roundtrip_and_preserves_metrics(tmp_path):
    spec = parse_spec(render_spec(ScriptSpec(path="grade.py", reward_key="accuracy")))
    body = (
        "import os\nfrom pathlib import Path\n"
        '(Path(os.environ["VERIFYIT_LOGS_DIR"]) / "reward.json").write_text('
        '\'{"accuracy": 0.75, "loss": 4.0, "reward": 0.0}\')\n'
    )
    tests = _tests_dir(tmp_path, body, "grade.py")
    reward = grade_script.grade(spec, tests, _workspace(tmp_path))
    assert (reward.reward, reward.status) == (0.75, Status.SCORED)
    assert reward.detail["metrics"] == {"accuracy": 0.75, "loss": 4.0, "reward": 0.0}


def test_named_reward_cannot_be_replaced_by_scalar_channel(tmp_path):
    tests = _tests_dir(tmp_path, '#!/bin/bash\nprintf 1 > "$VERIFYIT_LOGS_DIR/reward.txt"\necho 1\n', "test.sh")
    (tests / "verifier.toml").write_text(render_spec(ScriptSpec(path="test.sh", reward_key="accuracy")))
    reward = run(tests / "verifier.toml", _workspace(tmp_path))
    assert reward.status == Status.INFRA_ERROR


@pytest.mark.parametrize(
    "status,reward_value", [("scored", 0.0), ("scored", 1.0), ("infra_error", 0.0), ("invalid_task", 0.0)]
)
def test_declared_script_verdict_preserves_status_and_native_metadata(tmp_path, status, reward_value):
    payload = {
        "status": status,
        "reward": reward_value,
        "detail": {"native": {"reward_basis": ["state", "nl"], "info": {"runtime": False}}},
    }
    body = (
        "import os\nfrom pathlib import Path\n"
        + f"(Path(os.environ['VERIFYIT_LOGS_DIR'])/'result.json').write_text({json.dumps(payload)!r})\nprint(1)\n"
    )
    tests = _tests_dir(tmp_path, body, "grade.py")
    (tests / "verifier.toml").write_text('mode="script"\npath="grade.py"\nverdict_file="result.json"\n')
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "reward.json").write_text('{"reward":1}')
    (logs / "reward.txt").write_text("1")
    reward = run(tests / "verifier.toml", _workspace(tmp_path))
    write_reward(logs, reward)
    assert reward.status.value == status
    assert reward.reward == reward_value
    assert reward.detail["native"] == payload["detail"]["native"]
    if status == "scored":
        assert json.loads((logs / "reward.json").read_text())["reward"] == reward_value
    else:
        assert not (logs / "reward.json").exists()
        assert not (logs / "reward.txt").exists()


@pytest.mark.parametrize(
    "payload,ending",
    [(None, "print(1)"), ("{broken", "print(1)"), ('{"status":"scored","reward":1,"detail":{}}', "raise SystemExit(3)")],
)
def test_declared_script_failure_cannot_be_replaced_by_success(tmp_path, payload, ending):
    body = "import os\nfrom pathlib import Path\n"
    if payload is not None:
        body += f"(Path(os.environ['VERIFYIT_LOGS_DIR'])/'result.json').write_text({payload!r})\n"
    body += ending + "\n"
    tests = _tests_dir(tmp_path, body, "grade.py")
    (tests / "verifier.toml").write_text('mode="script"\npath="grade.py"\nverdict_file="result.json"\n')
    reward = run(tests / "verifier.toml", _workspace(tmp_path))
    assert reward.status == Status.INFRA_ERROR


@pytest.mark.parametrize("filename", ["../result.json", "/tmp/result.json", "", "."])
def test_declared_verdict_path_cannot_escape_private_logs(tmp_path, filename):
    tests = _tests_dir(tmp_path, "print(1)\n", "grade.py")
    (tests / "verifier.toml").write_text(render_spec(ScriptSpec(path="grade.py", verdict_file=filename)))
    assert run(tests / "verifier.toml", _workspace(tmp_path)).status == Status.INVALID_TASK


def test_declared_producer_timeout_does_not_accept_written_success(tmp_path):
    tests = _tests_dir(
        tmp_path,
        (
            "import os,time\n"
            "from pathlib import Path\n"
            '(Path(os.environ["VERIFYIT_LOGS_DIR"])/"result.json").write_text(\'{"status"'
            ':"scored","reward":1,"detail":{}}\')\n'
            "time.sleep(120)\n"
        ),
        "grade.py",
    )
    (tests / "verifier.toml").write_text(
        render_spec(ScriptSpec(path="grade.py", timeout=0.2, verdict_file="result.json"))
    )
    reward = run(tests / "verifier.toml", _workspace(tmp_path))
    assert reward.status == Status.INFRA_ERROR
    assert reward.detail["script"]["exit_code"] is None


@pytest.mark.parametrize(
    "payload",
    [
        '{"status":"scored","reward":0,"reward":1,"detail":{}}',
        '{"status":"scored","reward":2,"detail":{}}',
        '{"status":"scored","reward":true,"detail":{}}',
        '{"status":"unknown","reward":0,"detail":{}}',
        '{"status":"infra_error","reward":1,"detail":{}}',
    ],
)
def test_declared_verdict_malformed_contract_is_infrastructure_failure(tmp_path, payload):
    tests = _tests_dir(
        tmp_path,
        "import os\nfrom pathlib import Path\n"
        + f'(Path(os.environ["VERIFYIT_LOGS_DIR"])/"result.json").write_text({payload!r})\nprint(1)\n',
        "grade.py",
    )
    (tests / "verifier.toml").write_text(render_spec(ScriptSpec(path="grade.py", verdict_file="result.json")))
    assert run(tests / "verifier.toml", _workspace(tmp_path)).status == Status.INFRA_ERROR


@pytest.mark.parametrize("channel", ["reward.json", "reward.txt", "result.json"])
@pytest.mark.parametrize("producer", ["symlink", "dangling", "oversize"])
def test_invalid_reward_artifact_cannot_fall_back_or_leave_stale_credit(tmp_path, channel, producer):
    payload = {
        "reward.json": '{"reward":1}',
        "reward.txt": "1",
        "result.json": '{"status":"scored","reward":1,"detail":{}}',
    }[channel]
    operation = {
        "symlink": "target.write_text(payload); artifact.symlink_to(target)",
        "dangling": "artifact.symlink_to(target)",
        "oversize": "artifact.write_text(payload + ' ' * 1_000_001)",
    }[producer]
    body = (
        "import os\nfrom pathlib import Path\n"
        "logs = Path(os.environ['VERIFYIT_LOGS_DIR'])\n"
        f"artifact = logs / {channel!r}\npayload = {payload!r}\n"
        "target = logs / 'target'\n"
        f"{operation}\nprint(1)\n"
    )
    tests = _tests_dir(tmp_path, body, "grade.py")
    if channel == "result.json":
        (tests / "verifier.toml").write_text(render_spec(ScriptSpec(path="grade.py", verdict_file=channel)))
    logs = tmp_path / "logs"
    write_reward(logs, scored(1))
    assert main([str(tests / "verifier.toml"), "--workspace", str(_workspace(tmp_path)), "--logs-dir", str(logs)]) == 0
    verdict = json.loads((logs / "verdict.json").read_text())
    assert (verdict["status"], verdict["reward"]) == ("infra_error", 0)
    assert not (logs / "reward.txt").exists()
    assert not (logs / "reward.json").exists()


@pytest.mark.parametrize("payload", ['{"reward":0,"reward":1}', '{"reward":1,"metadata":{"value":1e999}}'])
def test_ambiguous_or_nonfinite_json_metadata_cannot_award_credit(tmp_path, payload):
    body = (
        "import os\nfrom pathlib import Path\n"
        f"(Path(os.environ['VERIFYIT_LOGS_DIR']) / 'reward.json').write_text({payload!r})\nprint(1)\n"
    )
    tests = _tests_dir(tmp_path, body, "grade.py")
    reward = run(tests / "verifier.toml", _workspace(tmp_path))
    assert (reward.status, reward.reward) == (Status.INFRA_ERROR, 0)
