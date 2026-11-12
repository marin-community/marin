# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
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
        assert command[command.index("--extra") + 1] == "vllm"
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
