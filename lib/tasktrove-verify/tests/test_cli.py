# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from tasktrove_verify import grade as grade_module
from tasktrove_verify.cli import main
from tasktrove_verify.reward import Status, scored
from tasktrove_verify.spec import Mode


def _verdict(logs: Path) -> dict:
    return json.loads((logs / "verdict.json").read_text())


def test_malformed_spec_writes_invalid_task_and_exits_zero(tmp_path):
    spec = tmp_path / "tests" / "verifier.toml"
    spec.parent.mkdir()
    spec.write_text('mode = "mcq"\n')
    logs = tmp_path / "logs"
    assert main([str(spec), "--logs-dir", str(logs), "--workspace", str(tmp_path)]) == 0
    assert _verdict(logs)["status"] == Status.INVALID_TASK
    assert not (logs / "reward.json").exists()
    assert not (logs / "reward.txt").exists()


def test_crashing_grader_writes_infra_error(tmp_path, monkeypatch):
    def boom(spec, tests_dir, workspace):
        raise RuntimeError("no toolchain")

    monkeypatch.setitem(grade_module.GRADERS, Mode.MCQ, boom)
    spec = tmp_path / "verifier.toml"
    spec.write_text('mode = "mcq"\nexpected = "C"\n')
    logs = tmp_path / "logs"
    main([str(spec), "--logs-dir", str(logs)])
    assert _verdict(logs) == {"reward": 0.0, "status": "infra_error", "detail": {"error": "RuntimeError: no toolchain"}}
    assert not (logs / "reward.json").exists()


def test_scored_reward_writes_the_verdict_and_harbor_reward_files(tmp_path, monkeypatch):
    monkeypatch.setitem(grade_module.GRADERS, Mode.MCQ, lambda spec, tests_dir, workspace: scored(1.0, extracted="C"))
    spec = tmp_path / "verifier.toml"
    spec.write_text('mode = "mcq"\nexpected = "C"\n')
    logs = tmp_path / "logs"
    main([str(spec), "--logs-dir", str(logs)])
    assert _verdict(logs) == {"reward": 1.0, "status": "scored", "detail": {"extracted": "C"}}
    assert json.loads((logs / "reward.json").read_text()) == {"reward": 1.0}
    assert (logs / "reward.txt").read_text() == "1.0\n"
