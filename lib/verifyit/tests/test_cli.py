# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from verifyit import grade as grade_module
from verifyit.candidate_file import main as candidate_file_main
from verifyit.grade import Status, main, scored
from verifyit.json_comparison import NumericTypePolicy
from verifyit.spec import (
    ExactSpec,
    FunctionCall,
    JsonSchemaSpec,
    Mode,
    NumericSpec,
    PredictedActionSpec,
    StructuredExactSpec,
    render_spec,
)


@pytest.mark.parametrize(
    "spec,candidate,invalid_contract",
    [
        *[
            (
                NumericSpec("0.3", tolerance_abs=0.0, tolerance_rel=0.0),
                "0.3",
                'mode = "numeric"\n' + invalid_fields + "\n",
            )
            for invalid_fields in (
                "expected = 0.30000000000000004\ntolerance_abs = 0.0\ntolerance_rel = 0.0",
                'expected = "0.3"\ntolerance_abs = "0.01"\ntolerance_rel = 0.0',
                'expected = "0.3"\ntolerance_abs = 0.0',
            )
        ],
        *[
            (
                StructuredExactSpec(expected={"answer": 12}),
                '{"answer":12}',
                f'mode = "structured_exact"\nexpected = {malformed_expected}\n',
            )
            for malformed_expected in ("true", "42", "[]")
        ],
    ],
)
def test_invalid_private_contract_clears_previous_rewards(tmp_path, spec, candidate, invalid_contract):
    config = tmp_path / "verifier.toml"
    config.write_text(render_spec(replace(spec, output=str(tmp_path / "answer.txt"))))
    (tmp_path / "answer.txt").write_text(candidate)
    logs = tmp_path / "logs"
    arguments = [str(config), "--logs-dir", str(logs), "--workspace", str(tmp_path)]
    assert main(arguments) == 0
    assert json.loads((logs / "reward.json").read_text()) == {"reward": 1.0}
    config.write_text(invalid_contract)
    assert main(arguments) == 0
    assert _verdict(logs)["status"] == Status.INVALID_TASK
    assert not (logs / "reward.json").exists()
    assert not (logs / "reward.txt").exists()


def _verdict(logs: Path) -> dict:
    return json.loads((logs / "verdict.json").read_text())


@pytest.mark.parametrize("candidate,reward", [("12", 1.0), (r"\boxed{12}", 0.0), (None, 0.0)])
def test_candidate_file_grades_full_text_or_missing_answer(tmp_path, candidate, reward):
    config = tmp_path / "contract.toml"
    config.write_text(render_spec(ExactSpec(expected=("12",))))
    manifest = tmp_path / "resources.json"
    manifest.write_text("[]")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if candidate is not None:
        (workspace / "submission.txt").write_text(candidate)
    logs = tmp_path / "logs"

    assert (
        candidate_file_main(
            [
                "--spec",
                str(config),
                "--answer",
                "/app/submission.txt",
                "--workspace",
                str(workspace),
                "--logs-dir",
                str(logs),
                "--resources-manifest",
                str(manifest),
            ]
        )
        == 0
    )
    assert _verdict(logs)["status"] == Status.SCORED
    assert json.loads((logs / "reward.json").read_text()) == {"reward": reward}
    if candidate is None:
        assert _verdict(logs)["detail"] == {"reason": "no_output"}


def test_candidate_file_module_loads_resources_relative_to_explicit_manifest(tmp_path):
    config = tmp_path / "contract.toml"
    config.write_text(render_spec(JsonSchemaSpec(schema="schemas/order.json")))
    resources = tmp_path / "private"
    (resources / "schemas").mkdir(parents=True)
    (resources / "schemas" / "order.json").write_text(json.dumps({"const": {"quantity": 3}}))
    manifest = resources / "inputs.json"
    manifest.write_text(json.dumps(["schemas/order.json"]))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "order.txt").write_text('```json\n{"quantity": 3}\n```')
    logs = tmp_path / "logs"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "verifyit.candidate_file",
            "--spec",
            str(config),
            "--answer",
            "/app/order.txt",
            "--workspace",
            str(workspace),
            "--logs-dir",
            str(logs),
            "--resources-manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert _verdict(logs)["status"] == Status.SCORED
    assert json.loads((logs / "reward.json").read_text()) == {"reward": 1.0}


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


def test_unscored_rerun_removes_prior_harbor_reward_files(tmp_path, monkeypatch):
    spec = tmp_path / "verifier.toml"
    spec.write_text('mode = "mcq"\nexpected = "C"\n')
    logs = tmp_path / "logs"
    monkeypatch.setitem(grade_module.GRADERS, Mode.MCQ, lambda _spec, _tests_dir, _workspace: scored(1.0))
    main([str(spec), "--logs-dir", str(logs)])

    spec.write_text('mode = "mcq"\n')
    main([str(spec), "--logs-dir", str(logs)])

    assert _verdict(logs)["status"] == Status.INVALID_TASK
    assert not (logs / "reward.json").exists()
    assert not (logs / "reward.txt").exists()


def test_pytest_setup_failure_opt_in_clears_previous_reward(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test_candidate.py").write_text("def test_ok():\n    assert True\n")
    spec = tmp_path / "verifier.toml"
    spec.write_text(
        'mode = "pytest"\npaths = ["test_candidate.py"]\n'
        f'python = "{sys.executable}"\n'
        "setup_failure_is_infra = true\n"
    )
    logs = tmp_path / "logs"
    assert main([str(spec), "--logs-dir", str(logs), "--workspace", str(workspace)]) == 0
    assert _verdict(logs)["status"] == Status.SCORED
    assert (logs / "reward.txt").read_text() == "1.0\n"

    spec.write_text(spec.read_text() + 'setup = "exit 3"\n')
    assert main([str(spec), "--logs-dir", str(logs), "--workspace", str(workspace)]) == 0
    assert _verdict(logs)["status"] == Status.INFRA_ERROR
    assert not (logs / "reward.json").exists()
    assert not (logs / "reward.txt").exists()


def test_predicted_action_overflowing_private_tolerance_clears_stale_reward(tmp_path):
    specification = PredictedActionSpec(
        expected_calls=(FunctionCall("lookup", {"id": 1}),),
        output=str(tmp_path / "answer.json"),
    )
    config = tmp_path / "verifier.toml"
    config.write_text(render_spec(specification))
    (tmp_path / "answer.json").write_text('[{"name":"lookup","arguments":{"id":1}}]')
    logs = tmp_path / "logs"
    arguments = [str(config), "--logs-dir", str(logs), "--workspace", str(tmp_path)]
    assert main(arguments) == 0
    assert json.loads((logs / "reward.json").read_text()) == {"reward": 1.0}
    assert (logs / "reward.txt").read_text() == "1.0\n"

    config.write_text(config.read_text() + f"numeric_tolerance = {10**400}\n")
    assert main(arguments) == 0
    assert _verdict(logs)["status"] == Status.INVALID_TASK
    assert not (logs / "reward.json").exists()
    assert not (logs / "reward.txt").exists()


@pytest.mark.parametrize(
    "spec,candidate,reward",
    [
        pytest.param(
            PredictedActionSpec(
                expected_calls=(FunctionCall("lookup", {"values": [None, True, 1, 1.0, {"text": "value"}]}),)
            ),
            '[{"name":"lookup","arguments":{"values":[null,true,1,1.0,{"text":"value"}]}}]',
            1.0,
            id="action-nested-json-roundtrip",
        ),
        pytest.param(
            PredictedActionSpec(
                expected_calls=(FunctionCall("lookup", {"values": [None, True, 1, 1.0, {"text": "value"}]}),)
            ),
            '[{"name":"lookup","arguments":{"values":[null,true,1,1.0,{"text":"wrong"}]}}]',
            0.0,
            id="action-wrong-nested-value",
        ),
        pytest.param(
            PredictedActionSpec(
                expected_calls=(FunctionCall("lookup", {"values": [None, True, 1, 1.0, {"text": "value"}]}),)
            ),
            "not json",
            0.0,
            id="action-malformed-candidate",
        ),
        pytest.param(
            StructuredExactSpec(expected={"values": [None, True, 1, 1.0]}),
            '{"values":[null,true,1,1.0]}',
            1.0,
            id="nested-json-roundtrip",
        ),
        pytest.param(
            StructuredExactSpec(expected={"values": [None, True, 1, 1.0]}),
            '{"values":[null,1,1,1.0]}',
            0.0,
            id="bool-is-not-number",
        ),
        pytest.param(
            StructuredExactSpec(expected={"values": [None, True, 1, 1.0]}),
            '{"values":[null,true,1.0,1]}',
            1.0,
            id="nested-numeric-value-equality",
        ),
        pytest.param(StructuredExactSpec(expected=None), "null", 1.0, id="null-roundtrip"),
        pytest.param(StructuredExactSpec(expected=None), "not json", 0.0, id="malformed-candidate"),
        pytest.param(
            StructuredExactSpec(expected={"nested": [16]}, numeric_types=NumericTypePolicy.VALUE),
            '{"nested":[16.0]}',
            1.0,
            id="explicit-value-policy",
        ),
        pytest.param(
            StructuredExactSpec(expected={"nested": [16]}, numeric_types=NumericTypePolicy.STRICT),
            '{"nested":[16.0]}',
            0.0,
            id="explicit-strict-policy",
        ),
        pytest.param(
            StructuredExactSpec(expected={"payload": {"id": 1}}),
            '{"payload":{"id":0,"id":1}}',
            0.0,
            id="structured-duplicate-candidate",
        ),
        pytest.param(
            PredictedActionSpec(expected_calls=(FunctionCall("lookup", {"id": 1}),)),
            '[{"name":"lookup","arguments":{"id":0,"id":1}}]',
            0.0,
            id="action-duplicate-candidate",
        ),
    ],
)
def test_json_file_grading_preserves_contract_from_toml(tmp_path, spec, candidate, reward):
    config = tmp_path / "verifier.toml"
    config.write_text(render_spec(replace(spec, output=str(tmp_path / "answer.json"))))
    (tmp_path / "answer.json").write_text(candidate)
    logs = tmp_path / "logs"
    assert main([str(config), "--logs-dir", str(logs), "--workspace", str(tmp_path)]) == 0
    assert _verdict(logs)["status"] == Status.SCORED
    assert json.loads((logs / "reward.json").read_text()) == {"reward": reward}
