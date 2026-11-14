# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from experiments.post_training.long_loop import launch_validation
from experiments.post_training.long_loop.eval_runtime import judge
from experiments.post_training.long_loop.serving import serving_release


@pytest.mark.timeout(120)
def test_invalid_launch_fails_before_judge_allocation(monkeypatch):
    allocated = []

    def allocate(*args, **kwargs):
        allocated.append(True)
        raise AssertionError("Invalid config reached judge allocation")

    monkeypatch.setattr(judge, "remote_inference", allocate)
    config = SimpleNamespace(skyrl=SimpleNamespace(launch_config_yaml="unexpected_root_field: true\n"))
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
