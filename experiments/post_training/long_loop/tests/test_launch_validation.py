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

from experiments.evaluation.judge_config import MICRO210_JUDGE
from experiments.evaluation.micro210 import micro210_step
from experiments.evaluation.micro210_policy import Micro210Model
from experiments.post_training.long_loop import launch_validation
from experiments.post_training.long_loop.eval_runtime import judge
from experiments.post_training.long_loop.serving import serving_release
from tests.rl.micro210_fixture import policy_fixture_reference


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
        launch_validation.regional_preflight(document, (str(policy),))


@pytest.mark.timeout(120)
def test_judged_evaluation_forwards_configured_key_without_ambient_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.evaluation.micro210_policy.POLICY_REFERENCE", policy_fixture_reference())
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
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
    assert "JUDGE_API_KEY" not in environment.env_vars
