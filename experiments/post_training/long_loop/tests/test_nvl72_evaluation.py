# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.rl import skyrl

from experiments.post_training.long_loop.nvl72_evaluation import NVL72EvaluationSpec, evaluation_from_spec


def test_adopted_evaluation_has_no_training_input_and_rejects_changed_data(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "outputs"))
    monkeypatch.setenv("MARIN_TEMP_PREFIX", str(tmp_path / "temporary"))
    model = tmp_path / "model"
    model.mkdir()
    manifest = json.dumps({"request": {"config": {"checkpoint": "fixture/step-400407"}}}).encode()
    config = json.dumps({"num_experts": 384, "max_position_embeddings": 262144}).encode()
    (model / "export-manifest.json").write_bytes(manifest)
    (model / "config.json").write_bytes(config)
    data = tmp_path / "data"
    data.mkdir()
    raw = b"pinned evaluation input"
    (data / "rollouts.parquet").write_bytes(raw)
    spec = NVL72EvaluationSpec(
        format="nvl72_eval",
        marin_revision="a" * 40,
        output_prefix=str(tmp_path / "outputs"),
        name="mochi-evaluation",
        model_uri=str(model),
        model_step=400407,
        model_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        model_config_sha256=hashlib.sha256(config).hexdigest(),
        tokenizer_uri="fixture/tokenizer",
        tokenizer_revision="b" * 40,
        data_uri=str(data),
        data_sha256=hashlib.sha256(raw).hexdigest(),
        input_version="2026.11.17.1",
        job_timeout_hours=2,
        context_tokens=65536,
        max_new_tokens=16384,
    )
    output = tmp_path / "launch.yaml"
    with build_context(BuildContext(versions=VersionCodex(default="2026.11.17.9"))):
        step = evaluation_from_spec(spec, output)
    launch = yaml.safe_load(output.read_text())
    assert launch["skyrl"]["entrypoint"] == "generate"
    assert launch["inputs"]["train_data"] == []
    assert launch["inputs"]["validation_data"][0]["uri"] == str(data)
    assert launch["inputs"]["model"]["uri"] == str(model)
    assert launch["skyrl"]["generator"]["trajectory_retention"]["model_checkpoint_step"] == 400407
    lowered = step.lower()
    assert lowered.name == step.name
    assert step.runtime_args["skyrl_execution"].coordinator_name == "mochi-mochi-evaluation-2026.11.17.9"

    class LaunchBoundaryReached(Exception):
        pass

    def launcher(command, *, env_vars):
        assert env_vars == {}
        assert command[-5:-1] == ["marinskyrl", "iris", "launch", "--config"]
        forwarded = yaml.safe_load(Path(command[-1]).read_text())
        assert forwarded["skyrl"]["entrypoint"] == "generate"
        assert forwarded["inputs"]["model"]["uri"] == str(model)
        raise LaunchBoundaryReached

    with monkeypatch.context() as patch:
        patch.setattr(skyrl, "_run_launcher", launcher)
        with pytest.raises(LaunchBoundaryReached):
            lowered.fn(step.path(str(tmp_path / "outputs")))
    with pytest.raises(ValueError, match="checkpoint step"):
        evaluation_from_spec(replace(spec, model_step=393491), output)
    (data / "rollouts.parquet").write_bytes(b"changed input")
    with pytest.raises(ValueError, match="micro evaluation data"):
        evaluation_from_spec(spec, output)
