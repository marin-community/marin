# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consequential ScriptSpec behavior for legacy Harbor reward files."""

import json

import pytest
from verifyit.grade import Status, run, write_reward
from verifyit.spec import ScriptSpec, render_spec


@pytest.fixture
def native_task(tmp_path, monkeypatch):
    tests = tmp_path / "tests"
    tests.mkdir()
    workspace = tmp_path / "app"
    workspace.mkdir()
    logs = tmp_path / "legacy-logs"
    logs.mkdir()
    monkeypatch.setenv("VERIFYIT_NATIVE_LOGS_DIR", str(logs))
    (tests / "native_bridge.py").write_text(
        "from verifyit.adapters.harbor_runtime import main\n" "raise SystemExit(main(['test.sh']))\n"
    )
    (tests / "verifier.toml").write_text(
        render_spec(ScriptSpec(path="native_bridge.py", verdict_file="native-verdict.json"))
    )
    return tests, workspace, logs


def test_native_script_zero_replaces_stale_positive_reward(native_task, tmp_path):
    tests, workspace, logs = native_task
    (logs / "reward.txt").write_text("1")
    (tests / "test.sh").write_text("#!/bin/bash\n" 'echo 0 > "$VERIFYIT_NATIVE_LOGS_DIR/reward.txt"\n')
    reward = run(tests / "verifier.toml", workspace)
    assert (reward.status, reward.reward) == (Status.SCORED, 0.0)
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "reward.txt").write_text("1")
    write_reward(outer, reward)
    assert (outer / "reward.txt").read_text().strip() == "0.0"


@pytest.mark.parametrize(
    "script",
    [
        "#!/bin/bash\nexit 1\n",
        '#!/bin/bash\necho 1 > "$VERIFYIT_NATIVE_LOGS_DIR/reward.txt"\nexit 7\n',
        "#!/bin/bash\ntrue\n",
        '#!/bin/bash\necho garbage > "$VERIFYIT_NATIVE_LOGS_DIR/reward.txt"\n',
        '#!/bin/bash\necho 1 > "$VERIFYIT_NATIVE_LOGS_DIR/reward.txt"\n'
        'echo broken > "$VERIFYIT_NATIVE_LOGS_DIR/reward.json"\n',
    ],
)
def test_native_runtime_failure_cannot_reuse_previous_reward(native_task, tmp_path, script):
    tests, workspace, logs = native_task
    (logs / "reward.txt").write_text("1")
    (tests / "test.sh").write_text(script)
    reward = run(tests / "verifier.toml", workspace)
    assert reward.status is Status.INFRA_ERROR and reward.reward == 0.0
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "reward.txt").write_text("1")
    write_reward(outer, reward)
    assert not (outer / "reward.txt").exists()
    assert json.loads((outer / "verdict.json").read_text())["status"] == "infra_error"


def test_native_json_keeps_named_numeric_metrics(native_task):
    tests, workspace, _ = native_task
    (tests / "test.sh").write_text(
        "#!/bin/bash\n" 'echo \'{"reward":0.5,"pass_rate":0.75}\' > "$VERIFYIT_NATIVE_LOGS_DIR/reward.json"\n'
    )
    reward = run(tests / "verifier.toml", workspace)
    assert (reward.status, reward.reward) == (Status.SCORED, 0.5)
    assert reward.detail["native_metrics"] == {"pass_rate": 0.75}


def test_native_json_preserves_gdb_optional_metrics(native_task):
    tests, workspace, _ = native_task
    # Produced by the pinned GDB evaluator's write_reward serialization helper.
    payload = {
        "reward": 0.75,
        "accuracy": 0.75,
        "nima_score": None,
        "format": "png",
        "per_example": [{"score": 0.75, "note": "optional metric unavailable"}],
    }
    (tests / "native_reward.json").write_text(json.dumps(payload))
    (tests / "test.sh").write_text(
        'cp "$VERIFYIT_TESTS_DIR/native_reward.json" "$VERIFYIT_NATIVE_LOGS_DIR/reward.json"\n'
    )
    reward = run(tests / "verifier.toml", workspace)
    assert (reward.status, reward.reward) == (Status.SCORED, 0.75)
    assert reward.detail["native_metrics"] == {key: value for key, value in payload.items() if key != "reward"}


@pytest.mark.parametrize("primary", [True, -0.1, 1.1, "NaN", None])
def test_auxiliary_metadata_does_not_rescue_invalid_primary(native_task, primary):
    tests, workspace, _ = native_task
    (tests / "native_reward.json").write_text(json.dumps({"reward": primary, "accuracy": 1.0, "format": "png"}))
    (tests / "test.sh").write_text(
        'cp "$VERIFYIT_TESTS_DIR/native_reward.json" "$VERIFYIT_NATIVE_LOGS_DIR/reward.json"\n'
    )
    reward = run(tests / "verifier.toml", workspace)
    assert (reward.status, reward.reward) == (Status.INFRA_ERROR, 0.0)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param('{"reward":1,"metrics":{"samples":[1e999]}}', id="nested_nonfinite"),
        pytest.param('{"reward":0,"reward":1}', id="duplicate_primary"),
    ],
)
def test_invalid_native_reward_payload_is_unscored(native_task, payload):
    tests, workspace, _ = native_task
    (tests / "native_reward.json").write_text(payload)
    (tests / "test.sh").write_text(
        'cp "$VERIFYIT_TESTS_DIR/native_reward.json" "$VERIFYIT_NATIVE_LOGS_DIR/reward.json"\n'
    )
    reward = run(tests / "verifier.toml", workspace)
    assert (reward.status, reward.reward) == (Status.INFRA_ERROR, 0.0)
