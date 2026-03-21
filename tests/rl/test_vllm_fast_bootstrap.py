# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path

import numpy as np

from marin.rl.environments.inference_ctx import vllm as vllm_module


def test_discover_safetensor_shards_prefers_index(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    index_path = model_dir / vllm_module.SAFE_TENSORS_INDEX_NAME
    index_payload = {
        "weight_map": {
            "layers.0.weight": "model-00001-of-00002.safetensors",
            "layers.1.weight": "model-00002-of-00002.safetensors",
            "layers.0.bias": "model-00001-of-00002.safetensors",
        }
    }
    index_path.write_text(json.dumps(index_payload))

    fs, remote_path = vllm_module.url_to_fs(str(model_dir))
    shards = vllm_module._discover_safetensor_shards(fs, remote_path)
    assert shards == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]


def test_discover_safetensor_shards_falls_back_to_single_file(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / vllm_module.SAFE_TENSORS_MODEL).write_bytes(b"fake")

    fs, remote_path = vllm_module.url_to_fs(str(model_dir))
    shards = vllm_module._discover_safetensor_shards(fs, remote_path)
    assert shards == [vllm_module.SAFE_TENSORS_MODEL]


def test_stage_bootstrap_metadata_copies_required_files(tmp_path):
    remote_model_dir = tmp_path / "remote_model"
    remote_model_dir.mkdir()
    (remote_model_dir / "config.json").write_text("{}")
    (remote_model_dir / "tokenizer.json").write_text('{"tokenizer": true}')

    staged_dir = vllm_module._stage_bootstrap_metadata(str(remote_model_dir))
    try:
        staged_path = Path(staged_dir)
        assert staged_path.exists()
        staged_config = staged_path / "config.json"
        staged_tokenizer = staged_path / "tokenizer.json"
        assert staged_config.exists()
        assert staged_tokenizer.exists()
        assert staged_tokenizer.read_text() == '{"tokenizer": true}'
    finally:
        shutil.rmtree(staged_dir, ignore_errors=True)


def test_serialize_state_dict_for_rpc_preserves_non_arrays():
    input_state = {
        "array": np.arange(6, dtype=np.float32).reshape(2, 3),
        "non_array": {"x": 1},
    }
    serialized = vllm_module._serialize_state_dict_for_rpc(input_state)

    assert isinstance(serialized["array"], tuple)
    data_bytes, dtype_name, shape = serialized["array"]
    assert isinstance(data_bytes, bytes)
    assert dtype_name == "float32"
    assert shape == (2, 3)
    assert serialized["non_array"] == {"x": 1}
