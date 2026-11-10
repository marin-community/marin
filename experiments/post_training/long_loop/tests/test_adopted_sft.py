# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.post_training.long_loop.adopted_sft import verify_adopted_sft


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
