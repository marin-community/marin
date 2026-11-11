# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload a completed Hero export to a private, checkpoint-specific HF branch."""

import hashlib
import json
import logging
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO

import draccus
from huggingface_hub import CommitOperationAdd, HfApi, RepoFile, hf_hub_download, snapshot_download
from levanter.compat.hf_export import HFShardRecord
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from rigging.secrets import require_secret

from experiments.grug.moe_hero_ep.ops.export_cpu import CpuExportConfig
from experiments.grug.moe_hero_ep.ops.export_metadata import (
    EXPORT_VERSION,
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    artifact_config,
    bf16_hf_config,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest

logger = logging.getLogger(__name__)
UPLOAD_REQUEST_FILENAME = "hf-upload-request.json"
UPLOAD_MANIFEST_FILENAME = "hf-upload-manifest.json"
LICENSE_SHA256 = "2ab44b68365473c112f5092211a38f231cb23e50de68b75a13369adbd76a74df"
WARNING = (
    "Research artifact. This model has not been properly tested or evaluated and is not necessarily secure. "
    "Use at your own risk in production settings."
)
RESPONSIBLE_USE = (
    "We release our tools to the public with the intent that they are used safely, ethically, and responsibly. "
    "They are meant to contribute to the public's understanding of science, and aid in its advancement. "
    "By using this Marin model, you agree to comply with all applicable law."
)


@dataclass(frozen=True)
class UploadConfig:
    export: CpuExportConfig
    expected_payload_bytes: int
    repo_id: str
    branch: str
    base_revision: str
    model_card: str
    license_file: str
    tokenizer: str
    tokenizer_revision: str
    staging_dir: str


@dataclass(frozen=True)
class _UploadPlan:
    root: StoragePath
    manifest: dict
    config_text: str
    index_text: str
    records: list[HFShardRecord]
    card: bytes
    license_text: bytes


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _status(config: UploadConfig, phase: str, **fields) -> None:
    value = {"phase": phase, "repo": config.repo_id, "branch": config.branch, **fields}
    root = StoragePath(config.export.destination)
    (root.parent / (root.name + ".upload-status.json")).write_text(json.dumps(value, indent=2))
    logger.info("Upload status: %s", json.dumps(value))


def validate_model_card(card: bytes, license_text: bytes) -> None:
    """Require the release notices and the official OpenMDW 1.1 license bytes."""
    text = card.decode()
    if "license: openmdw-1.1" not in text or text.count(WARNING) != 1 or text.count(RESPONSIBLE_USE) != 1:
        raise ValueError("Model card must contain the OpenMDW metadata and exactly one copy of each release notice")
    if hashlib.sha256(license_text).hexdigest() != LICENSE_SHA256:
        raise ValueError("License differs from the official OpenMDW 1.1 text")


def _payload(stream: BinaryIO, record: HFShardRecord) -> int:
    header_size = struct.unpack("<Q", stream.read(8))[0]
    if header_size > record.bytes - 8:
        raise ValueError(f"Invalid safetensors header size: {record.filename}")
    header = json.loads(stream.read(header_size))
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    if sorted(tensors) != record.tensor_names:
        raise ValueError(f"Shard tensor names differ: {record.filename}")
    if any(value["dtype"] != "BF16" for value in tensors.values()):
        raise ValueError(f"Non-BF16 tensor: {record.filename}")
    offsets = sorted(tuple(value["data_offsets"]) for value in tensors.values())
    end = 0
    for start, stop in offsets:
        if start != end or stop < start:
            raise ValueError(f"Invalid safetensors data offsets: {record.filename}")
        end = stop
    if 8 + header_size + end != record.bytes:
        raise ValueError(f"Safetensors payload size differs: {record.filename}")
    return end


def _repo_files(api: HfApi, config: UploadConfig, revision: str) -> dict[str, RepoFile]:
    return {
        item.path: item
        for item in api.list_repo_tree(config.repo_id, revision=revision, repo_type="model", recursive=True)
        if isinstance(item, RepoFile)
    }


def _download(api: HfApi, config: UploadConfig, filename: str, revision: str) -> bytes:
    return Path(hf_hub_download(config.repo_id, filename, revision=revision, token=api.token)).read_bytes()


def _verify_hub_shards(files: dict[str, RepoFile], records: list[HFShardRecord], *, complete: bool) -> None:
    shards = {name: item for name, item in files.items() if name.endswith(".safetensors")}
    expected = {record.filename: record for record in records}
    if not shards.keys() <= expected.keys() or (complete and shards.keys() != expected.keys()):
        raise ValueError("HF branch contains an unexpected shard set")
    for name, item in shards.items():
        record = expected[name]
        if item.size != record.bytes or item.lfs is None or item.lfs.sha256 != record.sha256:
            raise ValueError(f"HF shard identity differs: {name}")


def _verify_card(api: HfApi, config: UploadConfig, revision: str, card: bytes, license_text: bytes) -> None:
    uploaded_card = _download(api, config, "README.md", revision)
    uploaded_license = _download(api, config, "LICENSE", revision)
    validate_model_card(uploaded_card, uploaded_license)
    if uploaded_card != card or uploaded_license != license_text:
        raise ValueError("Published model card or license differs from the reviewed input")


def _upload_plan(config: UploadConfig) -> _UploadPlan:
    card = Path(config.model_card).read_bytes()
    license_text = Path(config.license_file).read_bytes()
    validate_model_card(card, license_text)
    if re.fullmatch(r"[0-9a-f]{40}", config.base_revision) is None:
        raise ValueError("base_revision must be an immutable HF commit SHA")
    root = StoragePath(config.export.destination)
    manifest = json.loads((root / MANIFEST_FILENAME).read_text())
    request = manifest["request"]
    expected_config = artifact_config(draccus.encode(config.export))
    if request["export_version"] != EXPORT_VERSION or request["config"] != expected_config:
        raise ValueError("Completed export differs from the pinned export configuration")
    expected_hf_config = bf16_hf_config(config.export.model)
    if request["hf_config"] != json.loads(json.dumps(expected_hf_config)):
        raise ValueError("Export HF configuration differs from the pinned model configuration")
    if manifest["export_id"] != digest(request):
        raise ValueError("Export identity differs from its request")
    config_text = (root / "config.json").read_text()
    if json.loads(config_text) != request["hf_config"]:
        raise ValueError("Export HF configuration differs from its request")
    index_text = (root / INDEX_FILENAME).read_text()
    index = json.loads(index_text)
    if index["metadata"]["total_size"] != config.expected_payload_bytes:
        raise ValueError("Export payload differs from the expected model size")
    records = [draccus.decode(HFShardRecord, item) for item in manifest["shards"]]
    if len({record.filename for record in records}) != len(records):
        raise ValueError("Duplicate shard filenames in export manifest")
    names = [name for record in records for name in record.tensor_names]
    weight_map = {name: record.filename for record in records for name in record.tensor_names}
    if len(names) != len(set(names)) or index["weight_map"] != weight_map:
        raise ValueError("Export index and shard manifest disagree")
    if any(record.export_id != manifest["export_id"] for record in records):
        raise ValueError("Shard receipts belong to another export")

    return _UploadPlan(
        root=root,
        manifest=manifest,
        config_text=config_text,
        index_text=index_text,
        records=records,
        card=card,
        license_text=license_text,
    )


def _prepare_branch(api: HfApi, config: UploadConfig, plan: _UploadPlan) -> dict[str, RepoFile]:
    records = plan.records
    request = plan.manifest["request"]
    card, license_text = plan.card, plan.license_text
    if not api.model_info(config.repo_id).private:
        raise ValueError("Model repository must remain private")
    baseline = _repo_files(api, config, config.base_revision)
    if baseline.keys() - {"README.md", "LICENSE", ".gitattributes"}:
        raise ValueError("New checkpoint branches require a metadata-only baseline")
    api.create_branch(
        config.repo_id, repo_type="model", branch=config.branch, revision=config.base_revision, exist_ok=True
    )
    files = _repo_files(api, config, config.branch)
    _verify_hub_shards(files, records, complete=False)
    upload_request = {
        "export_id": plan.manifest["export_id"],
        "export": request,
        "repo": config.repo_id,
        "branch": config.branch,
        "payload_bytes": config.expected_payload_bytes,
        "card_sha256": hashlib.sha256(card).hexdigest(),
        "license_sha256": LICENSE_SHA256,
        "tokenizer": config.tokenizer,
        "tokenizer_revision": config.tokenizer_revision,
    }
    if UPLOAD_REQUEST_FILENAME in files:
        if json.loads(_download(api, config, UPLOAD_REQUEST_FILENAME, config.branch)) != upload_request:
            raise ValueError("HF branch belongs to another upload request")
    else:
        if UPLOAD_MANIFEST_FILENAME in files or any(name.endswith(".safetensors") for name in files):
            raise ValueError("HF weights have no matching immutable upload request")
        pinned = api.create_commit(
            repo_id=config.repo_id,
            repo_type="model",
            revision=config.branch,
            commit_message=f"Pin {config.branch} checkpoint upload request",
            operations=[
                CommitOperationAdd(path_in_repo=UPLOAD_REQUEST_FILENAME, path_or_fileobj=_json_bytes(upload_request)),
                CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card),
                CommitOperationAdd(path_in_repo="LICENSE", path_or_fileobj=license_text),
            ],
        )
        _verify_card(api, config, pinned.oid, card, license_text)

    return files


def _upload_shards(
    api: HfApi, config: UploadConfig, plan: _UploadPlan, files: dict[str, RepoFile], scratch: Path
) -> None:
    root, records = plan.root, plan.records
    validated_payload = 0
    for number, record in enumerate(records, start=1):
        _status(config, "uploading", shard=number, total_shards=len(records), filename=record.filename)
        source = root / record.filename
        if record.filename in files:
            # Matching Hub SHA-256 verifies reuse; still validate the source
            # header so the total BF16 payload is checked on every resume.
            if source.size() != record.bytes:
                raise ValueError(f"Source shard size differs: {record.filename}")
            with source.open("rb") as stream:
                validated_payload += _payload(stream, record)
            continue
        local = scratch / record.filename
        source.download_to(str(local))
        if local.stat().st_size != record.bytes:
            raise ValueError(f"Source shard size differs: {record.filename}")
        with local.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record.sha256:
                raise ValueError(f"Source shard SHA-256 differs: {record.filename}")
            stream.seek(0)
            validated_payload += _payload(stream, record)
        api.upload_file(
            repo_id=config.repo_id,
            repo_type="model",
            revision=config.branch,
            path_in_repo=record.filename,
            path_or_fileobj=str(local),
            commit_message=f"Add {config.branch} BF16 shard {number}/{len(records)}",
        )
        remote = api.get_paths_info(config.repo_id, paths=[record.filename], repo_type="model", revision=config.branch)[
            0
        ]
        _verify_hub_shards({record.filename: remote}, [record], complete=True)
        local.unlink()
    if validated_payload != config.expected_payload_bytes:
        raise ValueError("Validated BF16 payload differs from the expected model size")


def _complete_upload(api: HfApi, config: UploadConfig, plan: _UploadPlan, scratch: Path) -> dict:
    records, manifest = plan.records, plan.manifest
    card, license_text = plan.card, plan.license_text
    config_text, index_text = plan.config_text, plan.index_text
    files = _repo_files(api, config, config.branch)
    _verify_hub_shards(files, records, complete=True)

    expected_result = {
        "repo": config.repo_id,
        "branch": config.branch,
        "export_id": manifest["export_id"],
        "shards": len(records),
        "payload_bytes": config.expected_payload_bytes,
        "tokenizer": config.tokenizer,
        "tokenizer_revision": config.tokenizer_revision,
    }
    if UPLOAD_MANIFEST_FILENAME in files:
        result = json.loads(_download(api, config, UPLOAD_MANIFEST_FILENAME, config.branch))
        if any(result.get(key) != value for key, value in expected_result.items()):
            raise ValueError("Completed HF upload differs from this request")
        _verify_hub_shards(_repo_files(api, config, result["weights_commit"]), records, complete=True)
        _verify_card(api, config, result["weights_commit"], card, license_text)
        _verify_card(api, config, config.branch, card, license_text)
        _status(config, "complete", **result)
        return result

    _status(config, "finalizing", verified_shards=len(records))
    metadata = scratch / "metadata"
    metadata.mkdir()
    snapshot_download(
        config.tokenizer,
        revision=config.tokenizer_revision,
        token=api.token,
        local_dir=metadata,
        allow_patterns=["tokenizer*", "special_tokens_map.json", "added_tokens.json", "chat_template*"],
    )
    (metadata / "README.md").write_bytes(card)
    (metadata / "LICENSE").write_bytes(license_text)
    (metadata / "config.json").write_text(config_text)
    (metadata / INDEX_FILENAME).write_text(index_text)
    (metadata / MANIFEST_FILENAME).write_bytes(_json_bytes(manifest))
    commit = api.create_commit(
        repo_id=config.repo_id,
        repo_type="model",
        revision=config.branch,
        commit_message=f"Complete verified {config.branch} BF16 checkpoint",
        operations=[
            CommitOperationAdd(path_in_repo=str(path.relative_to(metadata)), path_or_fileobj=str(path))
            for path in sorted(metadata.rglob("*"))
            if path.is_file() and ".cache" not in path.relative_to(metadata).parts
        ],
    )
    _verify_hub_shards(_repo_files(api, config, commit.oid), records, complete=True)
    _verify_card(api, config, commit.oid, card, license_text)
    result = {
        **expected_result,
        "weights_commit": commit.oid,
        "checks": [
            "BF16 headers",
            "tensor names",
            "payload bytes",
            "export SHA-256",
            "Hub sizes and SHA-256",
            "model card and license",
        ],
        "generation_parity_tested": False,
    }
    completion = api.upload_file(
        repo_id=config.repo_id,
        repo_type="model",
        revision=config.branch,
        path_in_repo=UPLOAD_MANIFEST_FILENAME,
        path_or_fileobj=_json_bytes(result),
        commit_message=f"Record verified {config.branch} upload",
    )
    if json.loads(_download(api, config, UPLOAD_MANIFEST_FILENAME, completion.oid)) != result:
        raise ValueError("Published upload completion marker differs")
    _verify_card(api, config, completion.oid, card, license_text)
    result["completion_commit"] = completion.oid
    _status(config, "complete", **result)
    return result


def upload(config: UploadConfig) -> dict:
    """Verify, resume and complete one private checkpoint upload using bounded disk staging."""
    plan = _upload_plan(config)
    api = HfApi(token=require_secret("HF_TOKEN"))
    files = _prepare_branch(api, config, plan)
    staging = Path(config.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="hero-hf-", dir=staging) as directory:
        scratch = Path(directory)
        _upload_shards(api, config, plan, files, scratch)
        return _complete_upload(api, config, plan, scratch)


@draccus.wrap()
def main(config: UploadConfig) -> None:
    configure_coreweave_s3()
    configure_logging(logging.INFO)
    upload(config)


if __name__ == "__main__":
    main()
