import json

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file
from safetensors.torch import load_file

from marinskyrl.model_manifest import HF_WEIGHT_INDEX_FILENAME, write_local_model_manifest
from skyrl_train.hf_model_io import (
    has_completed_hf_model_export,
    local_hf_model_dir,
    verify_completed_hf_model_export,
    write_hf_weight_shards,
)


def test_completed_export_can_be_reused_and_partial_export_cannot(tmp_path):
    tied = torch.arange(4, dtype=torch.bfloat16)
    bank = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).T

    def weights():
        yield "input", tied
        yield "output", tied
        yield "bank", bank
        assert any(tmp_path.glob("*.safetensors")), "weights buffered until the end of conversion"
        yield "bias", tied + 1

    with local_hf_model_dir(str(tmp_path)) as work_dir:
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "tokenizer.json").write_text("{}")
        (tmp_path / "chat_templates").mkdir()
        (tmp_path / "chat_templates" / "tool.jinja").write_text("{{ messages }}")
        write_hf_weight_shards(weights(), work_dir, shard_size_bytes=16)

    assert has_completed_hf_model_export(str(tmp_path))
    index = json.loads((tmp_path / HF_WEIGHT_INDEX_FILENAME).read_text())["weight_map"]
    assert index["input"] == index["output"]
    assert len(set(index.values())) == 3
    for name, value in {"input": tied, "output": tied, "bank": bank, "bias": tied + 1}.items():
        torch.testing.assert_close(load_file(tmp_path / index[name])[name], value)

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
