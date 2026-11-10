# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify a committed SFT handoff without creating or replaying a training stage.

Two forms are accepted. A loop-produced SFT run root is re-exported by the
graph. A pinned HF export (``kind: hf_export``) is verified byte for byte and
handed to the endpoint check and RL directly, with no export stage.
"""

import dataclasses
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import draccus
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.post_training.long_loop.sft import check_expert_tensors
from experiments.post_training.long_loop.sft_runtime.model import GrugModelConfig

HF_EXPORT_KIND = "hf_export"
# Kernel choices are runtime settings, not part of the HF checkpoint.
RUNTIME_ONLY_MODEL_FIELDS = ("attention_implementation", "moe_implementation")
GENERATION_EOS_TOKEN_IDS = {128001, 128009}


def pinned_json(uri: str, sha256: str) -> dict:
    content = StoragePath(uri).read_bytes()
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


@dataclass(frozen=True)
class PinnedFile:
    bytes: int
    sha256: str


@dataclass(frozen=True)
class HfExportPin:
    """An HF repository revision and the exact object set of its in-region copy."""

    kind: str
    repository: str
    revision: str
    uri: str
    files: dict[str, PinnedFile]

    def __post_init__(self) -> None:
        if self.kind != HF_EXPORT_KIND:
            raise ValueError(f"Adopted HF export pin must declare kind: {HF_EXPORT_KIND}")
        if re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise ValueError("Adopted HF export revision must be a full 40-character commit")
        for name in self.files:
            if not name or name.startswith("/") or ".." in name.split("/"):
                raise ValueError(f"Adopted HF export manifest contains a non-relative path: {name!r}")


def is_hf_export(pin: dict) -> bool:
    return pin.get("kind") == HF_EXPORT_KIND


def _file_sha256(uri: str) -> str:
    fs, key = filesystem_for(uri)
    with fs.open(key, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_adopted_hf_export(
    *, output: str, export_pin: dict, model: dict, tokenizer_uri: str, workers: int = 8
) -> None:
    """Publish an adoption record only after every pinned byte of the HF export matches."""
    pin = HfExportPin(**{**export_pin, "files": {k: PinnedFile(**v) for k, v in export_pin["files"].items()}})
    root = pin.uri
    files = pin.files
    fs, root_key = filesystem_for(root)
    if set(fs.find(root_key)) != {prefix_join(root_key, name) for name in files}:
        raise ValueError("Adopted HF export manifest does not cover the exact object set")
    for name, entry in files.items():
        info = fs.info(prefix_join(root_key, name))
        if info["type"] != "file" or info["size"] != entry.bytes:
            raise ValueError(f"Adopted HF export object is missing or changed size: {name}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        digests = dict(zip(files, pool.map(lambda name: _file_sha256(prefix_join(root, name)), files), strict=True))
    changed = sorted(name for name, entry in files.items() if digests[name] != entry.sha256)
    if changed:
        raise ValueError(f"Adopted HF export digest changed: {changed}")
    # The bytes are now pinned; the remaining checks read the verified objects.
    config = json.loads(fs.cat_file(prefix_join(root_key, "config.json")))
    expected = draccus.decode(GrugModelConfig, model)
    hf_config_class = type(expected.to_hf_config(expected.vocab_size))
    actual = GrugModelConfig.from_hf_config(hf_config_class(**config))
    actual = dataclasses.replace(actual, **{field: getattr(expected, field) for field in RUNTIME_ONLY_MODEL_FIELDS})
    if actual != expected:
        raise ValueError("Adopted HF export model geometry differs from the loop SFT recipe")
    weight_map = json.loads(fs.cat_file(prefix_join(root_key, "model.safetensors.index.json")))["weight_map"]
    check_expert_tensors(config, weight_map)
    if not set(weight_map.values()) <= files.keys():
        raise ValueError("Adopted HF export index names a shard outside the pinned manifest")
    tokenizer_fs, tokenizer_key = filesystem_for(prefix_join(tokenizer_uri, "tokenizer.json"))
    if hashlib.sha256(tokenizer_fs.cat_file(tokenizer_key)).hexdigest() != files["tokenizer.json"].sha256:
        raise ValueError("Adopted HF export tokenizer differs from the loop tokenizer")
    generation = json.loads(fs.cat_file(prefix_join(root_key, "generation_config.json")))
    if set(generation["eos_token_id"]) != GENERATION_EOS_TOKEN_IDS:
        raise ValueError("Adopted HF export does not stop on both loop EOS tokens")
    destination = StoragePath(output)
    destination.mkdirs(exist_ok=True)
    (destination / "adoption.json").write_text(
        json.dumps(
            {
                "export_pin": export_pin,
                "artifact_schema_version": config["grugmoe_artifact_schema_version"],
                "verified_bytes": sum(entry.bytes for entry in files.values()),
            },
            indent=2,
        )
    )
