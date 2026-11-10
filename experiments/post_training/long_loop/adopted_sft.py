# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify a committed SFT handoff without creating or replaying a training stage."""

import hashlib
import json

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest


def pinned_json(uri: str, sha256: str) -> dict:
    fs, path = filesystem_for(uri)
    content = fs.cat_file(path)
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError(f"Adopted SFT artifact digest changed: {uri}")
    return json.loads(content)


def verify_adopted_sft(*, output: str, checkpoint_pin: dict, model: dict, tokenizer_revision: str) -> None:
    """Publish a verified selection only after all pinned inputs and payloads exist."""
    root = checkpoint_pin["uri"]
    selected = pinned_json(prefix_join(root, "selected.json"), checkpoint_pin["selected_sha256"])
    checkpoint = prefix_join(root, f"checkpoints/step-{checkpoint_pin['step']}")
    if selected["checkpoint"] != checkpoint or selected["step"] != checkpoint_pin["step"]:
        raise ValueError("Adopted SFT selection is not the explicitly requested final checkpoint")
    metadata = pinned_json(prefix_join(checkpoint, "metadata.json"), checkpoint_pin["metadata_sha256"])
    if metadata["step"] != checkpoint_pin["step"] or metadata["is_temporary"] is not False:
        raise ValueError("Adopted SFT checkpoint is not permanently committed at the required step")
    if selected["metadata_digest"] != digest(metadata):
        raise ValueError("Adopted SFT selection metadata binding changed")
    manifest = pinned_json(prefix_join(checkpoint, "manifest.json"), checkpoint_pin["manifest_sha256"])
    if manifest["format_version"] != 1 or not manifest["arrays"]:
        raise ValueError("Adopted SFT array manifest is empty or unsupported")
    consumed = pinned_json(prefix_join(root, "consumed-config.json"), checkpoint_pin["consumed_config_sha256"])
    if consumed["model"] != model:
        raise ValueError("Adopted SFT consumed model configuration differs from the export recipe")
    lineage = pinned_json(checkpoint_pin["lineage_uri"], checkpoint_pin["lineage_sha256"])
    if (
        lineage["marin_revision"] != checkpoint_pin["source_revision"]
        or lineage["tokenizer_revision"] != tokenizer_revision
        or lineage["model_sha256"] != digest(model)
        or lineage["source_job"] != checkpoint_pin["source_job"]
        or lineage["version"] != checkpoint_pin["version"]
    ):
        raise ValueError("Adopted SFT model/tokenizer/source lineage does not match the pinned run")
    inventory = pinned_json(checkpoint_pin["payload_inventory_uri"], checkpoint_pin["payload_inventory_sha256"])
    if inventory["checkpoint"] != checkpoint or not inventory["files"]:
        raise ValueError("Adopted SFT payload inventory is empty or refers to a different checkpoint")
    fs, checkpoint_key = filesystem_for(checkpoint)
    expected_keys = {prefix_join(checkpoint_key, entry["path"]) for entry in inventory["files"]}
    objects = fs.find(checkpoint_key, detail=True)
    if set(objects) != expected_keys:
        raise ValueError("Adopted SFT payload inventory does not cover the exact checkpoint file set")
    payload_bytes = 0
    for entry in inventory["files"]:
        relative = entry["path"]
        if relative.startswith("/") or ".." in relative.split("/"):
            raise ValueError("Checkpoint payload inventory contains a non-relative path")
        info = objects[prefix_join(checkpoint_key, relative)]
        if info["type"] != "file" or info["size"] != entry["bytes"]:
            raise ValueError(f"Adopted SFT payload is missing or changed: {relative}")
        if relative not in {"metadata.json", "manifest.json"}:
            payload_bytes += entry["bytes"]
    if payload_bytes <= 0:
        raise ValueError("Adopted SFT checkpoint has no payload bytes")
    destination = StoragePath(output)
    destination.mkdirs(exist_ok=True)
    (destination / "selected.json").write_text(json.dumps(selected))
    (destination / "adoption.json").write_text(
        json.dumps({"checkpoint_pin": checkpoint_pin, "verified_payload_bytes": payload_bytes}, indent=2)
    )
