#!/usr/bin/env python3
"""Move a validated checkpoint bundle through a content-addressed S3 blob."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from capability_pipeline.checkpoint_revalidation import validate_checkpoint_bundle

SCHEMA = "capability-checkpoint-revalidation-transport-v1"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _safe(name: str) -> None:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or str(path) != name or "\\" in name:
        raise ValueError("checkpoint transport member path is unsafe")


def pack(source: Path, archive: Path, transport: Path,
         manifest_sha256: str, request_sha256: str) -> dict:
    validate_checkpoint_bundle(source, expected_manifest_sha256=manifest_sha256,
                               expected_request_sha256=request_sha256)
    if archive.exists() or transport.exists():
        raise ValueError("checkpoint transport output already exists")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "x:gz") as stream:
        for path in files:
            name = path.relative_to(source).as_posix()
            _safe(name)
            entry = tarfile.TarInfo(name)
            entry.size = path.stat().st_size
            entry.mode = path.stat().st_mode & 0o777
            entry.mtime = 0
            with path.open("rb") as member:
                stream.addfile(entry, member)
    receipt = {"schema_version": SCHEMA, "archive_sha256": _sha(archive),
               "archive_bytes": archive.stat().st_size,
               "bundle_manifest_sha256": manifest_sha256,
               "request_sha256": request_sha256,
               "member_count": len(files)}
    transport.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def _receipt(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint transport receipt is missing or linked")
    value = json.loads(path.read_text())
    if (not isinstance(value, dict) or value.get("schema_version") != SCHEMA
            or not all(isinstance(value.get(key), str) and len(value[key]) == 64
                       and all(c in "0123456789abcdef" for c in value[key])
                       for key in ("archive_sha256", "bundle_manifest_sha256", "request_sha256"))
            or type(value.get("archive_bytes")) is not int or value["archive_bytes"] <= 0
            or type(value.get("member_count")) is not int or value["member_count"] <= 0):
        raise ValueError("checkpoint transport receipt is invalid")
    return value


def _verify_archive(archive: Path, receipt: dict) -> None:
    if archive.is_symlink() or not archive.is_file() or archive.stat().st_size != receipt["archive_bytes"] or _sha(archive) != receipt["archive_sha256"]:
        raise ValueError("checkpoint transport archive changed")


def restore(archive: Path, transport: Path, destination: Path,
            manifest_sha256: str, request_sha256: str) -> dict:
    receipt = _receipt(transport)
    if (receipt["bundle_manifest_sha256"] != manifest_sha256
            or receipt["request_sha256"] != request_sha256):
        raise ValueError("checkpoint transport belongs to another request")
    _verify_archive(archive, receipt)
    if destination.exists() or destination.is_symlink():
        raise ValueError("checkpoint restore destination already exists")
    with tempfile.TemporaryDirectory(prefix="checkpoint-restore-") as temporary:
        root = Path(temporary) / "bundle"
        root.mkdir()
        names = set()
        with tarfile.open(archive, "r:gz") as stream:
            members = stream.getmembers()
            if len(members) != receipt["member_count"]:
                raise ValueError("checkpoint archive member count changed")
            for member in members:
                _safe(member.name)
                if not member.isfile() or member.name in names:
                    raise ValueError("checkpoint archive has linked or duplicate members")
                names.add(member.name)
                path = root / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as output:
                    shutil.copyfileobj(stream.extractfile(member), output)
                path.chmod(member.mode & 0o777)
        validate_checkpoint_bundle(root, expected_manifest_sha256=manifest_sha256,
                                   expected_request_sha256=request_sha256)
        shutil.copytree(root, destination)
    return receipt


def _cloud(uri: str):
    import fsspec
    from rigging.filesystem.s3_compat import configure_coreweave_s3

    configure_coreweave_s3()
    return fsspec.core.url_to_fs(uri)


def upload(archive: Path, transport: Path, root: str, output_receipt: Path) -> dict:
    receipt = _receipt(transport)
    _verify_archive(archive, receipt)
    if not root.startswith("s3://"):
        raise ValueError("checkpoint transport requires S3 root")
    uri = root.rstrip("/") + f"/_checkpoint_revalidation_blobs/{receipt['archive_sha256']}/bundle.tar.gz"
    fs, key = _cloud(uri)
    if fs.exists(key):
        with fs.open(key, "rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != receipt["archive_sha256"]:
                raise ValueError("existing checkpoint blob has a conflicting digest")
    else:
        with archive.open("rb") as source, fs.open(key, "wb") as output:
            shutil.copyfileobj(source, output, length=1 << 20)
        with fs.open(key, "rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != receipt["archive_sha256"]:
                raise ValueError("checkpoint upload readback differs")
    value = {**receipt, "uri": uri}
    if output_receipt.exists():
        raise ValueError("checkpoint upload receipt already exists")
    output_receipt.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return value


def download(receipt_path: Path, destination: Path,
             manifest_sha256: str, request_sha256: str) -> dict:
    receipt = _receipt(receipt_path)
    uri = receipt.get("uri")
    if (receipt["bundle_manifest_sha256"] != manifest_sha256
            or receipt["request_sha256"] != request_sha256
            or not isinstance(uri, str) or not uri.startswith("s3://")
            or not uri.endswith(f"/_checkpoint_revalidation_blobs/{receipt['archive_sha256']}/bundle.tar.gz")):
        raise ValueError("checkpoint download receipt differs from frozen request")
    if destination.exists() or destination.is_symlink():
        raise ValueError("checkpoint download destination already exists")
    fs, key = _cloud(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".transport-part")
    if temporary.exists():
        raise ValueError("checkpoint transport temporary path already exists")
    try:
        with fs.open(key, "rb") as source, temporary.open("xb") as output:
            shutil.copyfileobj(source, output, length=1 << 20)
        _verify_archive(temporary, receipt)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("pack", "restore", "upload", "download"):
        command = sub.add_parser(name)
        if name in {"pack", "restore"}:
            command.add_argument("--archive", type=Path, required=True)
            command.add_argument("--transport", type=Path, required=True)
        if name == "pack":
            command.add_argument("--source", type=Path, required=True)
        if name == "restore":
            command.add_argument("--destination", type=Path, required=True)
        if name == "upload":
            command.add_argument("--archive", type=Path, required=True)
            command.add_argument("--transport", type=Path, required=True)
            command.add_argument("--root", required=True)
            command.add_argument("--receipt", type=Path, required=True)
        if name == "download":
            command.add_argument("--receipt", type=Path, required=True)
            command.add_argument("--destination", type=Path, required=True)
        if name in {"pack", "restore", "download"}:
            command.add_argument("--manifest-sha256", required=True)
            command.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    if args.command == "pack":
        result = pack(args.source, args.archive, args.transport, args.manifest_sha256, args.request_sha256)
    elif args.command == "restore":
        result = restore(args.archive, args.transport, args.destination, args.manifest_sha256, args.request_sha256)
    elif args.command == "upload":
        result = upload(args.archive, args.transport, args.root, args.receipt)
    else:
        result = download(args.receipt, args.destination, args.manifest_sha256, args.request_sha256)
    print(json.dumps({"state": args.command, "archive_sha256": result["archive_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
