import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file
from safetensors.torch import load_file

from marinskyrl.model_files import MODEL_MANIFEST_FILENAME
from marinskyrl.model_manifest import HF_WEIGHT_INDEX_FILENAME, write_local_model_manifest
from skyrl_train.hf_model_io import (
    has_completed_hf_model_export,
    local_hf_model_dir,
    save_megatron_hf_model,
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


@pytest.mark.parametrize(
    ("source_kind", "rank_zero"),
    [
        pytest.param("local", True, id="local-writer"),
        pytest.param("local", False, id="local-peer"),
        pytest.param("empty-local", True, id="local-without-weights"),
        pytest.param("range-backed", True, id="range-backed"),
        pytest.param("config-only", True, id="config-only"),
    ],
)
def test_megatron_export_preserves_local_bridge_validation(tmp_path, source_kind, rank_zero):
    class LocalSafetensorsSource:
        key_to_filename_map = {"expected.weight": "model.safetensors"} if source_kind == "local" else {}

    source = LocalSafetensorsSource() if source_kind in {"local", "empty-local"} else object()
    pretrained = SimpleNamespace(state=SimpleNamespace(source=source)) if source_kind != "config-only" else object()
    weight = torch.ones(2, dtype=torch.bfloat16)
    rejection = KeyError("unexpected.weight")

    bridge_calls = []

    def save_local_weights(_model, path):
        bridge_calls.append(path)
        if rank_zero:
            raise rejection

    def save_configs(path):
        (Path(path) / "config.json").write_text("{}")
        (Path(path) / "tokenizer.json").write_text("{}")

    bridge = SimpleNamespace(
        hf_pretrained=pretrained,
        export_hf_weights=lambda _model, *, cpu: iter([("unexpected.weight", weight)]),
        save_hf_weights=save_local_weights,
    )
    error = None
    try:
        save_megatron_hf_model(
            bridge,
            [],
            str(tmp_path),
            rank_zero=rank_zero,
            local_state_source_type=LocalSafetensorsSource,
            save_configs=save_configs,
        )
    except KeyError as exc:
        error = exc

    index = tmp_path / HF_WEIGHT_INDEX_FILENAME
    if source_kind == "local":
        assert not index.exists(), "published an HF index for incompatible local Bridge weights"
        assert not (tmp_path / MODEL_MANIFEST_FILENAME).exists()
        assert bridge_calls == [str(tmp_path)]
        assert error is (rejection if rank_zero else None)
    else:
        assert error is None
        assert has_completed_hf_model_export(str(tmp_path))
        shard = json.loads(index.read_text())["weight_map"]["unexpected.weight"]
        torch.testing.assert_close(load_file(tmp_path / shard)["unexpected.weight"], weight)


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
