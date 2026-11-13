# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import draccus
import pytest

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.post_training.long_loop.adopted_sft import verify_adopted_hf_export, verify_adopted_sft
from experiments.post_training.long_loop.run_spec import resolve_spec
from experiments.post_training.long_loop.sft import check_expert_tensors
from experiments.post_training.long_loop.sft_runtime.model import GrugModelConfig


@pytest.fixture
def recorded_selection(tmp_path):
    root = tmp_path / "source"
    checkpoint = root / "checkpoints/step-15"
    checkpoint.mkdir(parents=True)

    def write(path, value):
        path.write_text(json.dumps(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    # Verbatim metadata from the committed .8 step15. Other files are small
    # storage-contract fixtures, not a claimed restorable native checkpoint.
    metadata = {"step": 15, "timestamp": "2026-11-09T01:15:11.007938", "is_temporary": False}
    model = {"hidden_size": 4096}
    pin = {
        "uri": str(root),
        "step": 15,
        "version": "2026.11.08.8",
        "source_revision": "a" * 40,
        "source_job": "/test/source",
        "metadata_sha256": write(checkpoint / "metadata.json", metadata),
        "manifest_sha256": write(checkpoint / "manifest.json", {"format_version": 1, "arrays": [{"path": "model"}]}),
        "consumed_config_sha256": write(root / "consumed-config.json", {"model": model}),
        "selected_sha256": write(
            root / "selected.json", {"checkpoint": str(checkpoint), "step": 15, "metadata_digest": digest(metadata)}
        ),
    }
    lineage = {
        "version": pin["version"],
        "marin_revision": pin["source_revision"],
        "source_job": pin["source_job"],
        "tokenizer_revision": "b" * 40,
        "model_sha256": digest(model),
    }
    pin["lineage_uri"] = str(root / "lineage.json")
    pin["lineage_sha256"] = write(root / "lineage.json", lineage)
    (checkpoint / "manifest.ocdbt").write_bytes(b"fixture-payload")
    inventory = {
        "checkpoint": str(checkpoint),
        "files": [{"path": p.name, "bytes": p.stat().st_size} for p in sorted(checkpoint.iterdir())],
    }
    pin["payload_inventory_uri"] = str(root / "inventory.json")
    pin["payload_inventory_sha256"] = write(root / "inventory.json", inventory)
    return pin, model, checkpoint


def test_committed_selection_requires_exact_step_and_does_not_fall_back(recorded_selection, tmp_path):
    pin, model, _ = recorded_selection
    output = tmp_path / "verified"
    verify_adopted_sft(output=str(output), checkpoint_pin=pin, model=model, tokenizer_revision="b" * 40)
    assert json.loads((output / "selected.json").read_text())["step"] == 15
    with pytest.raises(ValueError, match="requested final checkpoint"):
        verify_adopted_sft(
            output=str(tmp_path / "wrong"), checkpoint_pin={**pin, "step": 746}, model=model, tokenizer_revision="b" * 40
        )
    assert not (tmp_path / "wrong/selected.json").exists()


def test_missing_payload_or_wrong_tokenizer_prevents_handoff(recorded_selection, tmp_path):
    pin, model, checkpoint = recorded_selection
    with pytest.raises(ValueError, match="lineage"):
        verify_adopted_sft(output=str(tmp_path / "wrong"), checkpoint_pin=pin, model=model, tokenizer_revision="c" * 40)
    (checkpoint / "manifest.ocdbt").unlink()
    with pytest.raises(ValueError, match="exact checkpoint file set"):
        verify_adopted_sft(
            output=str(tmp_path / "missing"), checkpoint_pin=pin, model=model, tokenizer_revision="b" * 40
        )
    assert not (tmp_path / "missing/selected.json").exists()


def test_changed_payload_size_prevents_handoff(recorded_selection, tmp_path):
    pin, model, checkpoint = recorded_selection
    (checkpoint / "manifest.ocdbt").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="payload is missing or changed"):
        verify_adopted_sft(
            output=str(tmp_path / "changed"), checkpoint_pin=pin, model=model, tokenizer_revision="b" * 40
        )
    assert not (tmp_path / "changed/selected.json").exists()


@pytest.fixture
def pinned_hf_export(tmp_path):
    """A small HF export at the loop recipe's geometry, stored as schema 1 (stacked experts)."""
    spec = Path(__file__).parents[1] / "specs/round1.yaml"
    model = resolve_spec(spec, smoke=False)["sft"]["model"]
    decoded = draccus.decode(GrugModelConfig, model)
    config = json.loads(json.dumps(decoded.to_hf_config(decoded.vocab_size).to_dict()))
    config["grugmoe_artifact_schema_version"] = 1
    shard = "model-00001-of-00001.safetensors"
    weight_map = {
        f"model.layers.{layer}.mlp.experts.{projection}.weight": shard
        for layer in range(config["num_hidden_layers"])
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    root = tmp_path / "export"
    root.mkdir()
    contents = {
        "config.json": json.dumps(config).encode(),
        "model.safetensors.index.json": json.dumps({"weight_map": weight_map}).encode(),
        "generation_config.json": json.dumps({"eos_token_id": [128001, 128009], "bos_token_id": 128000}).encode(),
        "tokenizer.json": b'{"fixture": "tokenizer"}',
        shard: b"fixture-shard",
    }
    for name, content in contents.items():
        (root / name).write_bytes(content)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_bytes(contents["tokenizer.json"])
    pin = {
        "kind": "hf_export",
        "repository": "test/export",
        "revision": "a" * 40,
        "uri": str(root),
        "files": {
            name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in contents.items()
        },
    }
    return pin, model, root, str(tokenizer)


def test_pinned_hf_export_is_verified_and_recorded(pinned_hf_export, tmp_path):
    pin, model, _, tokenizer = pinned_hf_export
    verify_adopted_hf_export(output=str(tmp_path / "ok"), export_pin=pin, model=model, tokenizer_uri=tokenizer)
    record = json.loads((tmp_path / "ok/adoption.json").read_text())
    assert record["artifact_schema_version"] == 1
    assert record["verified_bytes"] == sum(entry["bytes"] for entry in pin["files"].values())


def test_hf_export_rejects_changed_bytes_and_extra_objects(pinned_hf_export, tmp_path):
    pin, model, root, tokenizer = pinned_hf_export
    shard = root / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"fixture-shart")  # Same size, different bytes.
    with pytest.raises(ValueError, match="digest changed"):
        verify_adopted_hf_export(output=str(tmp_path / "a"), export_pin=pin, model=model, tokenizer_uri=tokenizer)
    shard.write_bytes(b"fixture-shard")
    (root / "extra.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="exact object set"):
        verify_adopted_hf_export(output=str(tmp_path / "b"), export_pin=pin, model=model, tokenizer_uri=tokenizer)
    assert not (tmp_path / "a/adoption.json").exists() and not (tmp_path / "b/adoption.json").exists()


def test_hf_export_rejects_geometry_tokenizer_and_short_revision(pinned_hf_export, tmp_path):
    pin, model, _, tokenizer = pinned_hf_export
    with pytest.raises(ValueError, match="model geometry"):
        verify_adopted_hf_export(
            output=str(tmp_path / "a"), export_pin=pin, model={**model, "num_layers": 2}, tokenizer_uri=tokenizer
        )
    (Path(tokenizer) / "tokenizer.json").write_bytes(b"other")
    with pytest.raises(ValueError, match="tokenizer"):
        verify_adopted_hf_export(output=str(tmp_path / "b"), export_pin=pin, model=model, tokenizer_uri=tokenizer)
    with pytest.raises(ValueError, match="40-character"):
        verify_adopted_hf_export(
            output=str(tmp_path / "c"), export_pin={**pin, "revision": "ac60d940"}, model=model, tokenizer_uri=tokenizer
        )


def test_expert_tensors_must_match_the_declared_schema():
    config = {"num_hidden_layers": 1, "num_experts": 2}
    stacked = {f"model.layers.0.mlp.experts.{p}.weight": "s" for p in ("gate_proj", "up_proj", "down_proj")}
    split = {
        f"model.layers.0.mlp.experts.{e}.{p}.weight": "s"
        for e in range(2)
        for p in ("gate_proj", "up_proj", "down_proj")
    }
    check_expert_tensors({**config, "grugmoe_artifact_schema_version": 1}, stacked)
    check_expert_tensors({**config, "grugmoe_artifact_schema_version": 2}, split)
    with pytest.raises(ValueError, match="schema-2 tensor missing"):
        check_expert_tensors({**config, "grugmoe_artifact_schema_version": 2}, stacked)
    with pytest.raises(ValueError, match="schema-1 tensor missing"):
        check_expert_tensors({**config, "grugmoe_artifact_schema_version": 1}, split)
