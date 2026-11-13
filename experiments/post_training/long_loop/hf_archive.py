# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish selected HF weights from cluster-local scratch; archive only their receipt."""

import hashlib
import json
import logging
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from huggingface_hub import HfApi, hf_hub_download
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.secrets import require_secret

from experiments.grug.moe_hero_ep.ops.upload_hf import validate_model_card
from experiments.post_training.long_loop.timing import durable_event_log

logger = logging.getLogger(__name__)
UPLOAD_REQUEST = "long-loop-upload-request.json"
PUBLIC_SIDECARS = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "model.safetensors.index.json",
    ".marinskyrl-model-manifest.json",
}
PRIVATE_CONTENT = re.compile(
    rb"(?:s3|gs)://|/(?:Users|home|tmp|app)/|"
    rb"https?://(?:[^/\s]+\.)?(?:oa\.dev|internal)(?:[/:]|$)|"
    rb"https?://(?:[0-9]{1,3}\.){3}[0-9]{1,3}|"
    rb"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|"
    rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"
)


def public_weight_name(name: str) -> bool:
    return re.fullmatch(r"model(?:-[0-9]{5}-of-[0-9]{5})?\.safetensors", name) is not None


def audit_public_bytes(name: str, content: bytes) -> None:
    """Reject private locations and credential signatures without printing their values."""
    if PRIVATE_CONTENT.search(name.encode()) or PRIVATE_CONTENT.search(content):
        raise ValueError(f"Private location or credential-like content in public file: {name}")
    for url in re.findall(rb"https?://[^\s\"<>]+", content):
        host = urlsplit(url.decode()).hostname
        if host not in {"github.com", "huggingface.co", "openmdw.org", "www.openmdw.org"}:
            raise ValueError(f"Unreviewed public URL in file: {name}")
    if name.endswith(".json") and name != "tokenizer.json":
        audit_public_fields(name, json.loads(content))


def audit_public_fields(name: str, value: object) -> None:
    """Inspect configuration fields, never tokenizer vocabulary entries."""
    if isinstance(value, dict):
        for key, item in value.items():
            if item and re.search(
                r"(?:^|_)(?:api_key|access_token|auth_token|password|secret|authorization|endpoint|base_url|server_url|hostname)(?:$|_)",
                key.lower(),
            ):
                raise ValueError(f"Credential or endpoint field in public file: {name}")
            audit_public_fields(name, item)
    elif isinstance(value, list):
        for item in value:
            audit_public_fields(name, item)


def audit_public_file(path: Path, name: str) -> None:
    if public_weight_name(name):
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if header_size > 64 * 1024 * 1024:
                raise ValueError(f"Oversized safetensors header: {name}")
            header = stream.read(header_size)
            if len(header) != header_size:
                raise ValueError(f"Incomplete safetensors header: {name}")
            parsed = json.loads(header)
            audit_public_fields(name, parsed.get("__metadata__", {}))
            audit_public_bytes(name, header)
        return
    audit_public_bytes(name, path.read_bytes())


def location_commitment(source: str, native_checkpoint: str) -> str:
    return hashlib.sha256(
        json.dumps({"source": source, "native_checkpoint": native_checkpoint}, sort_keys=True).encode()
    ).hexdigest()


def audit_public_history(api: HfApi, repo_id: str) -> None:
    """Inspect every historical sidecar, so a clean tip cannot conceal private provenance."""
    seen = set()
    for commit in api.list_repo_commits(repo_id, repo_type="model"):
        info = api.model_info(repo_id, revision=commit.commit_id, files_metadata=True)
        for item in info.siblings:
            name = item.rfilename
            identity = (name, item.lfs.sha256 if item.lfs is not None else item.blob_id)
            if identity in seen:
                continue
            if public_weight_name(name):
                header = api.parse_safetensors_file_metadata(repo_id, name, revision=commit.commit_id)
                audit_public_fields(name, header.metadata)
                audit_public_bytes(
                    name, json.dumps({"metadata": header.metadata, "tensors": list(header.tensors)}).encode()
                )
            else:
                if name not in PUBLIC_SIDECARS | {"README.md", "LICENSE", UPLOAD_REQUEST, ".gitattributes"}:
                    raise ValueError(f"Unreviewed historical public filename: {name}")
                path = hf_hub_download(repo_id, name, revision=commit.commit_id, token=api.token)
                audit_public_file(Path(path), name)
            seen.add(identity)


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
    if info.sha != revision or info.private:
        raise ValueError("Selected HF checkpoint must have the pinned revision and be public")
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
    if request["location_commitment_sha256"] != location_commitment(receipt["source"], receipt["native_checkpoint"]):
        raise ValueError("HF upload receipt does not bind the selected checkpoint")
    audit_public_history(api, receipt["repo_id"])
    return receipt


def upload_checkpoint(config: HfUpload, *, timing_destination: str) -> dict:
    """Publish the selected HF export and return its verified commit/file receipt."""
    if not config.repo_id.startswith("open-athena/") or config.workers < 1:
        raise ValueError("Use an open-athena repository and positive upload worker count")
    card = StoragePath(config.model_card).read_bytes()
    license_text = StoragePath(config.license_file).read_bytes()
    validate_model_card(card, license_text)
    audit_public_bytes("README.md", card)
    audit_public_bytes("LICENSE", license_text)
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
        if name in PUBLIC_SIDECARS or public_weight_name(name)
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
        for item in records:
            audit_public_file(root / item["path"], item["path"])
        request = {
            "schema": 2,
            "location_commitment_sha256": location_commitment(config.source, config.native_checkpoint),
            "files": sorted(records, key=lambda item: item["path"]),
        }
        request_raw = json.dumps(request, sort_keys=True, indent=2).encode()
        (root / UPLOAD_REQUEST).write_bytes(request_raw)
        api = HfApi(token=require_secret("HF_TOKEN"))
        api.create_repo(config.repo_id, repo_type="model", private=False, exist_ok=True)
        if api.model_info(config.repo_id).private:
            raise ValueError("Refuse to publish a previously private repository; use a fresh public repo")
        audit_public_history(api, config.repo_id)
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
                private=False,
                allow_patterns=[item["path"] for item in records],
                num_workers=config.workers,
            )
        revision = api.model_info(config.repo_id).sha
        receipt = {
            "schema": 1,
            "repo_id": config.repo_id,
            "revision": revision,
            "private": False,
            "source": config.source,
            "native_checkpoint": config.native_checkpoint,
            "files": sorted(records, key=lambda item: item["path"]),
        }
        with log.span("hf_remote_hash_verification"):
            verify_hf_checkpoint(receipt)
        logger.info("Verified HF checkpoint %s at %s", config.repo_id, revision)
    return receipt
