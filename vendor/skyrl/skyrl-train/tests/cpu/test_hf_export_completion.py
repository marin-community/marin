import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from marinskyrl.model_manifest import HF_WEIGHT_INDEX_FILENAME, write_local_model_manifest
from skyrl_train.hf_model_io import has_completed_hf_model_export, verify_completed_hf_model_export


def test_completed_export_can_be_reused_and_partial_export_cannot(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "chat_templates").mkdir()
    (tmp_path / "chat_templates" / "tool.jinja").write_text("{{ messages }}")
    save_file({"weight": np.arange(4, dtype=np.float32)}, tmp_path / "model-00001-of-00001.safetensors")
    write_local_model_manifest(tmp_path)

    assert has_completed_hf_model_export(str(tmp_path))

    (tmp_path / HF_WEIGHT_INDEX_FILENAME).unlink()
    assert not has_completed_hf_model_export(str(tmp_path))


def test_completed_export_rejects_index_header_disagreement(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    save_file({"weight": np.arange(4, dtype=np.float32)}, tmp_path / "model-00001-of-00001.safetensors")
    write_local_model_manifest(tmp_path)
    index = tmp_path / HF_WEIGHT_INDEX_FILENAME
    document = json.loads(index.read_text())
    document["weight_map"] = {"different": "model-00001-of-00001.safetensors"}
    index.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="does not match its shards"):
        verify_completed_hf_model_export(str(tmp_path))
    assert not has_completed_hf_model_export(str(tmp_path))


def test_completed_export_rejects_shard_size_change(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    shard = tmp_path / "model-00001-of-00001.safetensors"
    save_file({"weight": np.arange(4, dtype=np.float32)}, shard)
    write_local_model_manifest(tmp_path)
    with shard.open("ab") as destination:
        destination.write(b"extra")

    with pytest.raises(RuntimeError, match="shard size differs from manifest"):
        verify_completed_hf_model_export(str(tmp_path))
    assert not has_completed_hf_model_export(str(tmp_path))
