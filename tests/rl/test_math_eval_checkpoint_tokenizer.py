# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import fsspec
import pytest

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval import checkpoint_tokenizer as tokenizer
from experiments.post_training.math_eval.calibration_preview import calibration_command
from experiments.post_training.math_eval.calibration_protocol import checkpoint_serving_configuration
from tests.rl.test_math_eval_export_binding import tokenizer_source


@pytest.mark.parametrize("poison", ["hash", "missing", "extra", "bound", "region", "contract"])
def test_tokenizer_source_refuses_poisoned_metadata(poison):
    source = tokenizer_source()
    if poison == "hash":
        source["files_sha256"] = "a" * 64
    elif poison == "missing":
        del source["files"]["tokenizer_config.json"]
    elif poison == "extra":
        source["files"]["../config.json"] = {"bytes": 1, "sha256": "a" * 64}
    elif poison == "bound":
        source["files"]["config.json"]["bytes"] = tokenizer.MAX_TOKENIZER_BYTES
    elif poison == "region":
        source["uri"] = "s3://other-region/model"
    else:
        source["files"]["tokenizer.json"]["sha256"] = "a" * 64
    if poison != "hash":
        source["files_sha256"] = audit.canonical_sha(source["files"])
        source["total_bytes"] = sum(v["bytes"] for v in source["files"].values())
    with pytest.raises(ValueError):
        tokenizer.validate_tokenizer_source(source)


def test_staging_hashes_original_metadata_and_refuses_changed_source(tmp_path, monkeypatch):
    filesystem = fsspec.filesystem("memory")
    root = "/tokenizer-staging-fixture"
    content = {"tokenizer.json": b"original", "tokenizer_config.json": b"{}", "config.json": b"{}"}
    for name, raw in content.items():
        filesystem.pipe(root + "/" + name, raw)
    monkeypatch.setenv("IRIS_TASK_ID", "fixture")
    monkeypatch.setattr(tokenizer, "TOKENIZER_STAGE_ROOT", tmp_path)
    monkeypatch.setattr(audit, "fs_path", lambda uri: (filesystem, root))
    monkeypatch.setitem(tokenizer.MODEL_PROFILES["qwen"], "tokenizer_sha256", hashlib.sha256(b"original").hexdigest())
    source = tokenizer.snapshot_tokenizer_source("s3://marin-us-east-02a/marin/original")
    staged = tokenizer.stage_tokenizer(source)
    assert Path(staged).joinpath("tokenizer.json").read_bytes() == b"original"
    Path(staged).joinpath("tokenizer.json").unlink()
    filesystem.pipe(root + "/tokenizer.json", b"ORIGINAL")
    with pytest.raises(ValueError, match="byte inventory"):
        tokenizer.stage_tokenizer(source)


def test_actual_vllm_command_uses_separate_original_tokenizer():
    files = {name: {"bytes": 1, "sha256": "a" * 64} for name in ("config.json", "tokenizer.json", "model.safetensors")}
    binding = {
        "schema": "math_eval_checkpoint_content_v1",
        "global_step": 96,
        "training_seed": 17,
        "model_uri": "s3://marin-us-east-02a/marin/exported/hf",
        "content": {"files": files, "total_bytes": 3, "files_sha256": audit.canonical_sha(files)},
        "tokenizer_source": tokenizer_source(),
    }
    binding["binding_sha256"] = audit.canonical_sha(binding)
    model, engine = checkpoint_serving_configuration(binding, expected_binding_sha256=binding["binding_sha256"])
    command = calibration_command(model, engine)
    for flag, value in {
        "--dtype": "bfloat16",
        "--max-model-len": "2048",
        "--max-num-seqs": "64",
        "--load-format": "runai_streamer",
        "--tensor-parallel-size": "1",
        "--served-model-name": model.model_id,
        "--seed": "17",
    }.items():
        assert command.count(flag) == 1
        assert command[command.index(flag) + 1] == value
    assert command.count("--tokenizer") == 1
    assert command[command.index("--tokenizer") + 1] == tokenizer.tokenizer_stage_path(binding["tokenizer_source"])
    assert command[command.index("serve") + 1] == binding["model_uri"]
    assert model.tokenizer != model.weights
    json.dumps(command)
