# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import struct
from types import SimpleNamespace

import pytest

from experiments.grug.moe_hero_ep.ops.test_upload_hf import LICENSE_TEXT
from experiments.grug.moe_hero_ep.ops.upload_hf import RESPONSIBLE_USE, WARNING
from experiments.post_training.long_loop.hf_archive import HfUpload, upload_checkpoint, verify_hf_checkpoint


def upload_config(tmp_path, native):
    source = tmp_path / "hf-export"
    source.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (source / name).write_text("{}")
    header = json.dumps({"__metadata__": {"format": "pt"}}).encode()
    (source / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    card = tmp_path / "README.md"
    card.write_text(f"---\nlicense: openmdw-1.1\n---\n\n{WARNING}\n\n{RESPONSIBLE_USE}\n")
    license_path = tmp_path / "LICENSE"
    license_path.write_text(LICENSE_TEXT)
    return HfUpload(
        str(source), "open-athena/test", str(tmp_path / "scratch"), str(card), str(license_path), str(native), 2
    )


def test_upload_pins_complete_public_model_and_detects_remote_changes(tmp_path, hub):
    config = upload_config(tmp_path, tmp_path / "native")
    receipt = upload_checkpoint(config, timing_destination=str(tmp_path / "timing" / "upload.jsonl"))
    assert receipt["revision"] == hub.revision
    assert receipt["private"] is False
    request = json.loads(hub.files["long-loop-upload-request.json"])
    assert "source" not in request and "native_checkpoint" not in request
    assert str(tmp_path).encode() not in hub.files["long-loop-upload-request.json"]
    assert len(receipt["files"]) == 7
    assert verify_hf_checkpoint(receipt) == receipt
    # A resumed upload reuses the exact identity; another checkpoint cannot claim this repo.
    assert upload_checkpoint(config, timing_destination=str(tmp_path / "timing" / "resume.jsonl")) == receipt
    hub.files["model.safetensors"] = b"different weight bytes"
    with pytest.raises(ValueError, match="differs"):
        verify_hf_checkpoint(receipt)
    hub.private = True
    with pytest.raises(ValueError, match="public"):
        verify_hf_checkpoint(receipt)
    events = [json.loads(line) for line in (tmp_path / "timing" / "upload.jsonl").read_text().splitlines()]
    assert {event["kind"] for event in events if event["status"] == "completed"} == {
        "parallel_export_staging",
        "hf_parallel_upload",
        "hf_remote_hash_verification",
    }


def test_public_upload_excludes_extra_files_and_rejects_private_metadata(tmp_path, hub):
    config = upload_config(tmp_path, tmp_path / "native")
    source = tmp_path / "hf-export"
    (source / "internal-receipt.json").write_text('{"source":"s3://private/location"}')
    (source / "config.json").write_text('{"_name_or_path":"/tmp/internal/model"}')
    with pytest.raises(ValueError, match="Private location"):
        upload_checkpoint(config, timing_destination=str(tmp_path / "timing" / "bad.jsonl"))
    assert not hub.files
    (source / "config.json").write_text('{"base_url":"https://inference.example.com/v1"}')
    with pytest.raises(ValueError, match="Unreviewed public URL"):
        upload_checkpoint(config, timing_destination=str(tmp_path / "timing" / "external.jsonl"))
    (source / "config.json").write_text("{}")
    header = json.dumps({"__metadata__": {"endpoint": "http://10.1.2.3:8000"}}).encode()
    (source / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(ValueError, match="endpoint"):
        upload_checkpoint(config, timing_destination=str(tmp_path / "timing" / "header.jsonl"))
    assert not hub.files
    header = json.dumps({"__metadata__": {"format": "pt"}}).encode()
    (source / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    receipt = upload_checkpoint(config, timing_destination=str(tmp_path / "timing" / "clean.jsonl"))
    assert "internal-receipt.json" not in {item["path"] for item in receipt["files"]}
    hub.private = True
    with pytest.raises(ValueError, match="previously private"):
        upload_checkpoint(config, timing_destination=str(tmp_path / "timing" / "private.jsonl"))


def test_public_verification_rejects_private_metadata_in_old_commit(tmp_path, hub, monkeypatch):
    config = upload_config(tmp_path, tmp_path / "native")
    receipt = upload_checkpoint(config, timing_destination=str(tmp_path / "timing" / "history.jsonl"))
    old_card = tmp_path / "old-card.md"
    old_card.write_text("Native checkpoint: s3://internal/checkpoint")
    download = hub.download

    def historical_download(repo, name, **kwargs):
        if kwargs["revision"] == "b" * 40 and name == "README.md":
            return str(old_card)
        return download(repo, name, **kwargs)

    monkeypatch.setattr(
        hub,
        "list_repo_commits",
        lambda *a, **k: [SimpleNamespace(commit_id="b" * 40), SimpleNamespace(commit_id=hub.revision)],
    )
    monkeypatch.setattr("experiments.post_training.long_loop.hf_archive.hf_hub_download", historical_download)
    with pytest.raises(ValueError, match="Private location"):
        verify_hf_checkpoint(receipt)


@pytest.mark.parametrize("suffix", ["", "-of-00002"])
def test_upload_keeps_every_indexed_export_shard(tmp_path, hub, suffix):
    config = upload_config(tmp_path, tmp_path / "native")
    source = tmp_path / "hf-export"
    weight_bytes = (source / "model.safetensors").read_bytes()
    (source / "model.safetensors").unlink()
    shards = [f"model-{number:05d}{suffix}.safetensors" for number in (1, 2)]
    for name in shards:
        (source / name).write_bytes(weight_bytes)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer_a": shards[0], "layer_b": shards[1]}})
    )
    receipt = upload_checkpoint(config, timing_destination=str(tmp_path / "timing" / "sharded.jsonl"))
    published = {item["path"]: item["sha256"] for item in receipt["files"]}
    assert {name: published[name] for name in shards} == {
        name: hashlib.sha256(weight_bytes).hexdigest() for name in shards
    }
    assert all(hub.files[name] == weight_bytes for name in shards)
    assert verify_hf_checkpoint(receipt) == receipt


def test_upload_refuses_missing_indexed_shard_before_publication(tmp_path, hub):
    config = upload_config(tmp_path, tmp_path / "native")
    source = tmp_path / "hf-export"
    (source / "model.safetensors").rename(source / "model-00001.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer_a": "model-00001.safetensors", "layer_b": "model-00002.safetensors"}})
    )
    with pytest.raises(ValueError, match=r"missing weight shards.*model-00002"):
        upload_checkpoint(config, timing_destination=str(tmp_path / "timing" / "missing.jsonl"))
    assert not hub.files
