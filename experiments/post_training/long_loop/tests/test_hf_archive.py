# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.grug.moe_hero_ep.ops.test_upload_hf import LICENSE_TEXT
from experiments.grug.moe_hero_ep.ops.upload_hf import RESPONSIBLE_USE, WARNING
from experiments.post_training.long_loop.hf_archive import HfUpload, upload_checkpoint, verify_hf_checkpoint


def upload_config(tmp_path, native):
    source = tmp_path / "hf-export"
    source.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (source / name).write_text("{}")
    (source / "model.safetensors").write_bytes(b"small weight fixture")
    card = tmp_path / "README.md"
    card.write_text(f"---\nlicense: openmdw-1.1\n---\n\n{WARNING}\n\n{RESPONSIBLE_USE}\n")
    license_path = tmp_path / "LICENSE"
    license_path.write_text(LICENSE_TEXT)
    return HfUpload(
        str(source), "open-athena/test", str(tmp_path / "scratch"), str(card), str(license_path), str(native), 2
    )


def test_upload_pins_complete_private_model_and_detects_remote_changes(tmp_path, hub):
    config = upload_config(tmp_path, tmp_path / "native")
    receipt = upload_checkpoint(config, timing_destination=str(tmp_path / "timing" / "upload.jsonl"))
    assert receipt["revision"] == hub.revision
    assert len(receipt["files"]) == 7
    assert verify_hf_checkpoint(receipt) == receipt
    # A resumed upload reuses the exact identity; another checkpoint cannot claim this repo.
    assert upload_checkpoint(config, timing_destination=str(tmp_path / "timing" / "resume.jsonl")) == receipt
    hub.files["model.safetensors"] = b"different weight bytes"
    with pytest.raises(ValueError, match="differs"):
        verify_hf_checkpoint(receipt)
    hub.private = False
    with pytest.raises(ValueError, match="private"):
        verify_hf_checkpoint(receipt)
    events = [json.loads(line) for line in (tmp_path / "timing" / "upload.jsonl").read_text().splitlines()]
    assert {event["kind"] for event in events if event["status"] == "completed"} == {
        "parallel_export_staging",
        "hf_parallel_upload",
        "hf_remote_hash_verification",
    }
