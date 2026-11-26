# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from iris.client.client import IrisClient
from marin.execution.lazy import StepContext
from rigging.secrets import SecretResolutionError
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
)
from verifyit.modes.grade_judge import JudgeConnection

from experiments.evaluation.judge_config import MICRO210_JUDGE
from experiments.evaluation.micro210 import micro210_step
from experiments.evaluation.micro210_policy import Micro210Model
from experiments.post_training.long_loop import launch_validation
from experiments.post_training.long_loop.eval_runtime import judge
from experiments.post_training.long_loop.judge_route import judge_route_verdict
from experiments.post_training.long_loop.serving import serving_release
from tests.rl.micro210_fixture import policy_fixture_reference


def _gate_task(model: str):
    """A minimal judge-graded task declaring ``model`` (empty string means the served fallback)."""
    return TaskSpec(
        id="gate-fixture",
        source=Source(dataset="fixture/gate", revision="1", row="0", importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content="Grade me."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=VerifyitGrader(
            mode="judge",
            parameters={
                "references": ["12"],
                "rubric": "labels",
                "prompt_template": "Reference: {reference}\nCandidate: {candidate}",
                "label_scores": {"PASS": 1.0, "FAIL": 0.0},
                "model": model,
            },
        ),
    )


def test_judge_route_gate_refuses_unservable_declared_judges():
    """The CPU gate classifies declared judges and refuses the unservable-no-substitute class.

    A declared judge must be the deployment's served model, mapped in ``substitute_models``
    (a self-map records a served model), or empty for the served fallback; anything else is the
    marin #12374 class and fails the launch before GPU allocation.
    """
    served = JudgeConnection(
        "https://judge.invalid",
        "fixture-key",
        default_model="zai-org/GLM-5.3-Flash",
        substitute_models={
            "MiniMaxAI/MiniMax-M3-MXFP8": "zai-org/GLM-5.3-Flash",
            "glm-5.3": "zai-org/GLM-5.3-Flash",
            "openai/gpt-oss-120b": "openai/gpt-oss-120b",
        },
    )
    worker = served
    assert judge_route_verdict(_gate_task(""), worker) == "default_fallback"
    assert judge_route_verdict(_gate_task("zai-org/GLM-5.3-Flash"), worker) == "served"
    assert judge_route_verdict(_gate_task("glm-5.3"), worker) == "substituted"
    assert judge_route_verdict(_gate_task("MiniMaxAI/MiniMax-M3-MXFP8"), worker) == "substituted"
    assert judge_route_verdict(_gate_task("openai/gpt-oss-120b"), worker) == "substituted"
    with pytest.raises(ValueError, match=r"glm-4\.6"):
        judge_route_verdict(_gate_task("glm-4.6"), worker)
    # Without any deployed connection a declared judge is still refused.
    with pytest.raises(ValueError, match="declares judge model"):
        judge_route_verdict(_gate_task("glm-4.6"), None)


def test_judge_route_gate_passes_tasks_without_judges():
    """Tasks without a judge grader carry no judge-route obligation."""
    task = _gate_task("")
    plain = task.model_copy(update={"grader": NoGrader(reason="no judge")})
    assert judge_route_verdict(plain, SimpleNamespace(judge_connection=None)) == "no_judge"


@pytest.mark.timeout(120)
def test_invalid_launch_fails_before_judge_allocation(tmp_path, monkeypatch):
    allocated = []

    def allocate(*args, **kwargs):
        allocated.append(True)
        raise AssertionError("Invalid config reached judge allocation")

    monkeypatch.setattr(judge, "run_skyrl", allocate)
    credential = tmp_path / "judge-key"
    credential.write_text("configured-key")
    config = SimpleNamespace(
        skyrl=SimpleNamespace(launch_config_yaml="unexpected_root_field: true\n"),
        judge=replace(MICRO210_JUDGE, credential=f"file:{credential}"),
    )
    with pytest.raises(ValueError, match="Strict SkyRL composition failed"):
        judge.run_judged(config)
    assert not allocated


def test_selected_context_binds_exact_future_config_and_overrides(tmp_path, monkeypatch):

    raw = b'{"model_type":"grug", "max_seq_len":32768}'
    (tmp_path / "config.json").write_bytes(raw)
    overrides = {"max_seq_len": 262144, "max_position_embeddings": 262144}
    launch = {
        "inputs": {"model": {"uri": str(tmp_path), "identity": "selected-step-17"}},
        "skyrl": {
            "generator": {"engine_init_kwargs": {"hf_overrides": overrides}},
            "context_budget": {"request_window_tokens": 65536},
        },
    }

    def execute(command, **kwargs):
        root, override_file, requested, output = command[-4:]
        assert Path(root, "config.json").read_bytes() == raw
        assert json.loads(Path(override_file).read_text()) == overrides
        assert "VLLM_ALLOW_LONG_MAX_MODEL_LEN" not in kwargs["env"]
        Path(output).write_text(
            json.dumps(
                {
                    "config_sha256": hashlib.sha256(raw).hexdigest(),
                    "requested_window": int(requested),
                    "overrides": overrides,
                    "vllm_version": serving_release().version,
                    "accepted": True,
                    "max_model_len": 65536,
                }
            )
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(launch_validation.subprocess, "run", execute)
    receipt = launch_validation.validate_selected_context(yaml.safe_dump(launch))
    assert receipt["model_identity"] == "selected-step-17"
    assert receipt["config_sha256"] == hashlib.sha256(raw).hexdigest()


def test_selected_context_subprocess_is_independent_of_marin_installation(tmp_path):
    source = Path(launch_validation.__file__).parent
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    for name in ("selected_context_probe.py", "vllm-release.json"):
        shutil.copyfile(source / name, isolated / name)
    release = json.loads((isolated / "vllm-release.json").read_text())
    expected_version = release["distribution"]["version"]
    dependency = tmp_path / "dependency"
    package = dependency / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f"__version__ = {expected_version!r}\n")
    # Fake the external vLLM library only. The real child must import and execute
    # without the repository or site-packages on its Python path.
    (package / "config.py").write_text(
        "class ModelConfig:\n"
        "    def __init__(self, *, max_model_len, **kwargs):\n"
        "        self.max_model_len = max_model_len\n"
    )
    metadata = dependency / "vllm-0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Name: vllm\nVersion: {expected_version}\n")
    model = tmp_path / "model"
    model.mkdir()
    raw = b'{"model_type":"grug_moe"}'
    (model / "config.json").write_bytes(raw)
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"max_position_embeddings": 262144}))
    receipt = tmp_path / "receipt.json"
    environment = {**os.environ, "PYTHONPATH": str(dependency)}
    environment.pop("VLLM_ALLOW_LONG_MAX_MODEL_LEN", None)
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(isolated / "selected_context_probe.py"),
            str(model),
            str(overrides),
            "65536",
            str(receipt),
        ],
        cwd=isolated,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads(receipt.read_text())
    assert evidence["config_sha256"] == hashlib.sha256(raw).hexdigest()
    assert evidence["vllm_commit"] == release["source"]["fork_commit"]
    assert evidence["vllm_version"] == expected_version
    assert evidence["accepted"] and evidence["max_model_len"] == 65536


def test_regional_preflight_rejects_unreadable_prepared_inputs(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    document = yaml.safe_dump(
        {"inputs": {"validation_data": [{"uri": str(tmp_path / "unreadable"), "relative_path": "rollouts.parquet"}]}}
    )
    with pytest.raises(FileNotFoundError):
        launch_validation.regional_preflight(document, (str(policy),), required_secret_names=())


@pytest.mark.timeout(120)
def test_judged_evaluation_forwards_configured_key_without_ambient_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.evaluation.micro210_policy.POLICY_REFERENCE", policy_fixture_reference())
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    monkeypatch.setenv("DAYTONA_TARGET", "us")
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(tmp_path / "empty-secret-bundle"))
    credential = tmp_path / "judge-key"
    credential.write_text("configured-judge-key")
    step = micro210_step(
        Micro210Model("s3://fixture/model", "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40),
        name="fixture/judged-eval",
        version="2026.11.20",
    )
    context = StepContext.for_run(
        str(tmp_path / "output"), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps
    )
    config = step.build_config(replace(context, is_fingerprint=True))
    client = MagicMock(spec=IrisClient)
    client.submit.side_effect = RuntimeError("CPU scheduler unavailable")
    monkeypatch.setattr(launch_validation, "open_iris_client", lambda **kwargs: nullcontext(client))
    with pytest.raises(RuntimeError, match="CPU scheduler unavailable"):
        judge.run_judged(judge.JudgedEvaluation(config.skyrl, replace(MICRO210_JUDGE, credential=f"file:{credential}")))
    environment = client.submit.call_args.kwargs["environment"]
    assert environment.secrets["JUDGE_API_KEY"] == "configured-judge-key"
    assert environment.secrets["WANDB_API_KEY"] == "offline-fixture"
    assert "WANDB_API_KEY" not in environment.env_vars
    assert environment.env_vars["DAYTONA_TARGET"] == "us"
    assert "DAYTONA_TARGET" not in environment.secrets
    assert "JUDGE_API_KEY" not in environment.env_vars


def test_regional_cpu_gate_requires_every_phase_secret_before_worker_subprocess(tmp_path, monkeypatch):
    launch = {
        "inputs": {"validation_data": []},
        "skyrl": {
            "environment": {
                "task_sessions": {
                    "remote_judge": {"api_key_env": "JUDGE_TEST_KEY"},
                    "daytona": {"api_key_env": "TRAIN_TEST_KEY", "eval_api_key_env": "EVAL_TEST_KEY"},
                }
            },
            "terminal_bench": {
                "harbor": {"daytona_api_key_env": "TRAIN_TEST_KEY", "eval_daytona_api_key_env": "EVAL_TEST_KEY"}
            },
            "trainer": {"callbacks": [{"type": "harbor_evaluation", "daytona_api_key_env": "CALLBACK_TEST_KEY"}]},
        },
    }
    names = {
        "CW_KEY_ID",
        "CW_KEY_SECRET",
        "HF_TOKEN",
        "WANDB_API_KEY",
        "JUDGE_TEST_KEY",
        "TRAIN_TEST_KEY",
        "EVAL_TEST_KEY",
        "CALLBACK_TEST_KEY",
    }
    storage_names = {"CW_KEY_ID", "CW_KEY_SECRET", "HF_TOKEN", "WANDB_API_KEY"}
    assert set(launch_validation.launch_secret_names(launch)) == names - storage_names
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(tmp_path))
    for name in names:
        monkeypatch.delenv(name, raising=False)
        if name != "EVAL_TEST_KEY":
            (tmp_path / name).write_text("fixture-" + name)

    def forbidden_subprocess(*args, **kwargs):
        raise AssertionError("Worker subprocess started without the evaluation-org credential")

    monkeypatch.setattr(launch_validation.subprocess, "run", forbidden_subprocess)
    with pytest.raises(SecretResolutionError, match="EVAL_TEST_KEY"):
        launch_validation.regional_preflight(yaml.safe_dump(launch), (), required_secret_names=tuple(storage_names))
