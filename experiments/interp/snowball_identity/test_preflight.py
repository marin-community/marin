# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The pre-load source and input checks."""

import hashlib
import json

import torch

from experiments.interp.snowball_identity.preflight import source_preflight


def test_source_preflight_lists_every_mismatch(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text(json.dumps({"qk_mult": 1.75, "max_position_embeddings": 262144}))
    (weights / "generation_config.json").write_text(json.dumps({"eos_token_id": [128001, 128009]}))
    (weights / "chat_template.jinja").write_text("template")
    (weights / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "0", "b": "0"}}))
    positions = tmp_path / "positions.pt"
    torch.save({"check|a": {"tokens": [1, 2, 3]}, "long|x": {"tokens": [4, 5]}}, positions)
    expect = {
        "files": {"chat_template.jinja": hashlib.sha256(b"template").hexdigest()},
        "config": {"qk_mult": 1.75, "max_position_embeddings": 262144},
        "eos_token_id": [128001, 128009],
        "index_keys": 2,
        "inputs": [
            {
                "path": str(positions),
                "sha256": hashlib.sha256(positions.read_bytes()).hexdigest(),
                "sequences": 2,
                "tokens": 5,
            }
        ],
    }
    assert source_preflight(str(weights), expect) == []
    wrong = {
        **expect,
        "files": {"chat_template.jinja": "0" * 64},
        "config": {"qk_mult": 1.0},
        "eos_token_id": [128001],
        "index_keys": 502,
        "inputs": [{**expect["inputs"][0], "tokens": 6}],
    }
    failures = source_preflight(str(weights), wrong)
    assert len(failures) == 5 and failures[-1].endswith("5 tokens, expected 6")
