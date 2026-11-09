# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish selected HF weights from cluster-local scratch; archive only their receipt."""

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, hf_hub_download
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.secrets import require_secret

from experiments.grug.moe_hero_ep.ops.upload_hf import validate_model_card
from experiments.post_training.long_loop.timing import durable_event_log

logger = logging.getLogger(__name__)
UPLOAD_REQUEST = "long-loop-upload-request.json"


@dataclass(frozen=True)
class HfUpload:
    source: str
    repo_id: str
    local_directory: str
    model_card: str
    license_file: str
    native_checkpoint: str
    workers: int = 8


def file_identity(info: dict) -> dict:
    """Record provider identity without reading native checkpoint payloads."""
    identity = {
        key: str(info[key]) for key in ("ETag", "etag", "VersionId", "version_id", "mtime") if info.get(key) is not None
    }
    if not identity:
        raise ValueError("Storage object lacks a stable identity for reconciliation")
    return {"size": int(info["size"]), "identity": identity}


def _local_record(path: Path, name: str) -> dict:
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": name, "size": path.stat().st_size, "sha256": checksum}


def _download_file(source: str, destination: Path, name: str, events: str) -> dict:
    fs, key = filesystem_for(source)
    before = file_identity(fs.info(key))
    destination.parent.mkdir(parents=True, exist_ok=True)
    event_path = prefix_join(events, hashlib.sha256(name.encode()).hexdigest() + ".jsonl")
    with durable_event_log(event_path, source, "hf_upload", name) as log:
        with log.span("source_download_and_sha256"):
            checksum = hashlib.sha256()
            size = 0
            temporary = destination.with_name(destination.name + ".partial")
            with fs.open(key, "rb") as stream, temporary.open("wb") as output:
                while chunk := stream.read(8 * 1024 * 1024):
                    output.write(chunk)
                    checksum.update(chunk)
                    size += len(chunk)
            fs.invalidate_cache(key)
            if size != before["size"] or file_identity(fs.info(key)) != before:
                raise ValueError(f"HF export source changed while staging: {source}")
            temporary.replace(destination)
    return {"path": name, "size": size, "sha256": checksum.hexdigest()}


def verify_hf_checkpoint(receipt: dict) -> dict:
    """Verify every pinned Hub file and required notices without downloading LFS weights."""
    revision = receipt["revision"]
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("HF archive requires an immutable commit SHA")
    api = HfApi(token=require_secret("HF_TOKEN"))
    info = api.model_info(receipt["repo_id"], revision=revision, files_metadata=True)
    if info.sha != revision or not info.private:
        raise ValueError("Selected HF checkpoint must have the pinned revision and remain private")
    expected = {item["path"]: item for item in receipt["files"]}
    actual = {item.rfilename: item for item in info.siblings if item.rfilename != ".gitattributes"}
    if expected.keys() != actual.keys() or len(expected) != len(receipt["files"]):
        raise ValueError("HF archive file inventory differs")
    required = {"README.md", "LICENSE", "config.json", "tokenizer.json", "tokenizer_config.json", UPLOAD_REQUEST}
    if not required <= expected.keys() or not any(name.endswith(".safetensors") for name in expected):
        raise ValueError("HF archive lacks complete model/tokenizer/license payload")
    small = {}
    for name, record in expected.items():
        remote = actual[name]
        if remote.size != record["size"]:
            raise ValueError(f"HF file size differs: {name}")
        if remote.lfs is not None:
            checksum = remote.lfs.sha256
        else:
            if remote.size > 64 * 1024 * 1024:
                raise ValueError(f"Large HF file lacks content SHA-256 metadata: {name}")
            path = hf_hub_download(receipt["repo_id"], name, revision=revision, token=api.token)
            small[name] = Path(path).read_bytes()
            checksum = hashlib.sha256(small[name]).hexdigest()
        if checksum != record["sha256"]:
            raise ValueError(f"HF file SHA-256 differs: {name}")
    for name in ("README.md", "LICENSE", UPLOAD_REQUEST):
        if name not in small:
            small[name] = Path(
                hf_hub_download(receipt["repo_id"], name, revision=revision, token=api.token)
            ).read_bytes()
    validate_model_card(small["README.md"], small["LICENSE"])
    request = json.loads(small[UPLOAD_REQUEST])
    expected_request_files = [item for item in receipt["files"] if item["path"] != UPLOAD_REQUEST]
    if request["files"] != expected_request_files:
        raise ValueError("HF upload request and receipt inventories disagree")
    if request["source"] != receipt["source"] or request["native_checkpoint"] != receipt["native_checkpoint"]:
        raise ValueError("HF upload receipt does not bind the selected checkpoint")
    return receipt


def upload_checkpoint(config: HfUpload, *, timing_destination: str) -> dict:
    """Run beside the data, with concurrent staging and resumable Hub upload workers."""
    if not config.repo_id.startswith("open-athena/") or config.workers < 1:
        raise ValueError("Use an open-athena repository and positive upload worker count")
    card_fs, card_key = filesystem_for(config.model_card)
    license_fs, license_key = filesystem_for(config.license_file)
    card = card_fs.cat_file(card_key)
    license_text = license_fs.cat_file(license_key)
    validate_model_card(card, license_text)
    root = Path(config.local_directory)
    root.mkdir(parents=True, exist_ok=True)
    events = prefix_join(str(StoragePath(timing_destination).parent), "files")
    fs, prefix = filesystem_for(config.source)
    source_files = sorted(fs.find(prefix))
    names = [StoragePath(key).relative_to(StoragePath(prefix)) for key in source_files]
    if any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts for name in names):
        raise ValueError("Export contains a nonrelative path")
    # Source checkpoints can contain an old card; only the reviewed card is published.
    sources = [
        (key, name)
        for key, name in zip(source_files, names, strict=True)
        if name not in {"README.md", "LICENSE", UPLOAD_REQUEST}
    ]
    if any(name.startswith(".cache/") or name.endswith(".distcp") for _, name in sources):
        raise ValueError("HF export source contains cache or native optimizer payloads")
    with durable_event_log(timing_destination, config.repo_id, "hf_upload", "upload") as log:
        with log.span("parallel_export_staging"):
            with ThreadPoolExecutor(max_workers=config.workers) as pool:
                records = list(
                    pool.map(
                        lambda item: _download_file(fs.unstrip_protocol(item[0]), root / item[1], item[1], events),
                        sources,
                    )
                )
        (root / "README.md").write_bytes(card)
        (root / "LICENSE").write_bytes(license_text)
        records.extend([_local_record(root / name, name) for name in ("README.md", "LICENSE")])
        request = {
            "source": config.source,
            "native_checkpoint": config.native_checkpoint,
            "files": sorted(records, key=lambda item: item["path"]),
        }
        request_raw = json.dumps(request, sort_keys=True, indent=2).encode()
        (root / UPLOAD_REQUEST).write_bytes(request_raw)
        api = HfApi(token=require_secret("HF_TOKEN"))
        api.create_repo(config.repo_id, repo_type="model", private=True, exist_ok=True)
        if not api.model_info(config.repo_id).private:
            raise ValueError("Refuse to upload into a public repository")
        existing = set(api.list_repo_files(config.repo_id, repo_type="model"))
        if UPLOAD_REQUEST in existing:
            prior = Path(hf_hub_download(config.repo_id, UPLOAD_REQUEST, token=api.token)).read_bytes()
            if prior != request_raw:
                raise ValueError("Repository belongs to a different checkpoint upload")
        elif existing - {".gitattributes"}:
            raise ValueError("Refuse to overwrite a populated repository without matching upload identity")
        else:
            api.upload_file(
                repo_id=config.repo_id, repo_type="model", path_in_repo=UPLOAD_REQUEST, path_or_fileobj=request_raw
            )
        records.append(_local_record(root / UPLOAD_REQUEST, UPLOAD_REQUEST))
        with log.span("hf_parallel_upload"):
            api.upload_large_folder(
                repo_id=config.repo_id,
                repo_type="model",
                folder_path=root,
                private=True,
                allow_patterns=[item["path"] for item in records],
                num_workers=config.workers,
            )
        revision = api.model_info(config.repo_id).sha
        receipt = {
            "schema": 1,
            "repo_id": config.repo_id,
            "revision": revision,
            "private": True,
            "source": config.source,
            "native_checkpoint": config.native_checkpoint,
            "files": sorted(records, key=lambda item: item["path"]),
        }
        with log.span("hf_remote_hash_verification"):
            verify_hf_checkpoint(receipt)
        logger.info("Verified HF checkpoint %s at %s", config.repo_id, revision)
    return receipt
