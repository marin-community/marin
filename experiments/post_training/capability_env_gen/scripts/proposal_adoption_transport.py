#!/usr/bin/env python3
"""Portable immutable transport for a full proposal-adoption checkpoint."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

SCHEMA = "capability-proposal-adoption-transport-v1"
MAX_BYTES = 8 * 1024 * 1024 * 1024


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk); size += len(chunk)
            if size > MAX_BYTES: raise ValueError("proposal adoption member exceeds transport bound")
    return digest.hexdigest(), size


def safe(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or "\x00" in name or "\\" in name or path.as_posix() != name or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("proposal adoption transport has an unsafe member")
    return path


def files(root: Path, prefix: str) -> list[tuple[str, Path, bool]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("proposal checkpoint must be a real directory")
    result = [(prefix, root, True)]
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError("proposal checkpoint has an unsafe member")
        result.append((f"{prefix}/{path.relative_to(root).as_posix()}", path, stat.S_ISDIR(mode)))
    return result


def pack(checkpoint: Path, source_archive: Path, launch_receipt: Path, archive: Path, transport: Path) -> dict:
    if archive.exists() or archive.is_symlink() or transport.exists() or transport.is_symlink():
        raise ValueError("proposal adoption transport output must be fresh")
    if any(path.is_symlink() or not path.is_file() for path in (source_archive, launch_receipt)):
        raise ValueError("proposal adoption source inputs are missing or linked")
    entries = files(checkpoint, "checkpoint") + [
        ("source-archive.tar.gz", source_archive, False),
        ("launch-receipt.json", launch_receipt, False),
    ]
    members: list[dict] = []
    total = 0
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("xb") as target, gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as compressed, tarfile.open(fileobj=compressed, mode="w|") as output:
        for name, path, directory in entries:
            info = tarfile.TarInfo(name)
            info.mtime, info.mode = 0, stat.S_IMODE(path.stat().st_mode) & 0o777
            if directory:
                info.type = tarfile.DIRTYPE
                output.addfile(info)
                members.append({"path": name, "kind": "directory", "mode": info.mode})
            else:
                digest, size = file_hash(path)
                total += size
                if total > MAX_BYTES:
                    raise ValueError("proposal adoption input exceeds transport bound")
                info.size = size
                with path.open("rb") as source: output.addfile(info, source)
                members.append({"path": name, "kind": "file", "mode": info.mode, "bytes": size, "sha256": digest})
    archive_sha, archive_bytes = file_hash(archive)
    if archive_bytes > MAX_BYTES:
        raise ValueError("proposal adoption archive exceeds transport bound")
    capture, pull = checkpoint / "snapshot-capture.json", checkpoint / "pull-manifest.json"
    if not capture.is_file() or not pull.is_file():
        raise ValueError("proposal checkpoint lacks snapshot identity receipts")
    document = {
        "schema_version": SCHEMA,
        "archive_name": archive.name,
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "members": members,
        "snapshot_capture_sha256": file_hash(capture)[0],
        "snapshot_pull_sha256": file_hash(pull)[0],
        "source_archive_sha256": file_hash(source_archive)[0],
        "launch_receipt_sha256": file_hash(launch_receipt)[0],
    }
    transport.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def validate_transport(value: object) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise ValueError("invalid proposal adoption transport receipt")
    for key in ("archive_sha256", "snapshot_capture_sha256", "snapshot_pull_sha256", "source_archive_sha256", "launch_receipt_sha256"):
        if not isinstance(value.get(key), str) or len(value[key]) != 64:
            raise ValueError("invalid proposal adoption transport identity")
    if type(value.get("archive_bytes")) is not int or not 0 <= value["archive_bytes"] <= MAX_BYTES:
        raise ValueError("invalid proposal adoption transport size")
    if not isinstance(value.get("members"), list) or not value["members"]:
        raise ValueError("invalid proposal adoption transport members")
    return value


def _unpack(archive: Path, receipt: dict, destination: Path) -> None:
    archive_sha, archive_bytes = file_hash(archive)
    if archive_bytes != receipt["archive_bytes"] or archive_sha != receipt["archive_sha256"]:
        raise ValueError("proposal adoption archive fingerprint mismatch")
    if destination.exists() or destination.is_symlink():
        raise ValueError("proposal adoption destination must be fresh")
    expected = {member["path"]: member for member in receipt["members"]}
    if len(expected) != len(receipt["members"]):
        raise ValueError("proposal adoption transport has duplicate members")
    with tempfile.TemporaryDirectory(prefix="proposal-adoption-transport-") as temporary:
        root = Path(temporary) / "input"
        root.mkdir()
        total, seen = 0, set()
        with tarfile.open(archive, mode="r|gz") as source:
            for member in source:
                name = str(safe(member.name))
                declared = expected.get(name)
                if name in seen or declared is None or member.issym() or member.islnk() or member.isdev() or member.isfifo() or not (member.isfile() or member.isdir()):
                    raise ValueError("proposal adoption archive has an unsafe member")
                if (member.isdir() and declared.get("kind") != "directory") or (member.isfile() and declared.get("kind") != "file") or (member.mode & 0o777) != declared.get("mode"):
                    raise ValueError("proposal adoption archive member type differs")
                relative = safe(name)
                output = root.joinpath(*relative.parts)
                cursor = root
                for part in relative.parts:
                    cursor = cursor / part
                    if cursor.is_symlink():
                        raise ValueError("proposal adoption archive has a linked parent")
                if output.exists() or output.is_symlink():
                    raise ValueError("proposal adoption archive has a linked parent")
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=False)
                    os.chmod(output, member.mode & 0o777)
                else:
                    total += member.size
                    if total > MAX_BYTES:
                        raise ValueError("proposal adoption archive expanded beyond bound")
                    stream = source.extractfile(member)
                    if stream is None:
                        raise ValueError("proposal adoption archive member is unreadable")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open("xb") as handle:
                        digest, size = hashlib.sha256(), 0
                        while chunk := stream.read(1024 * 1024):
                            size += len(chunk)
                            if size > MAX_BYTES: raise ValueError("proposal adoption archive expanded beyond bound")
                            digest.update(chunk); handle.write(chunk)
                    if size != declared.get("bytes") or digest.hexdigest() != declared.get("sha256"):
                        raise ValueError("proposal adoption archive member fingerprint mismatch")
                    os.chmod(output, member.mode & 0o777)
                seen.add(name)
        if seen != set(expected):
            raise ValueError("proposal adoption archive member set differs")
        for name, declared in expected.items():
            if declared["kind"] == "directory":
                path = root.joinpath(*safe(name).parts)
                if not path.is_dir():
                    raise ValueError("proposal adoption logical directory is absent")
                os.chmod(path, declared["mode"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root, destination)
    if file_hash(destination / "checkpoint/snapshot-capture.json")[0] != receipt["snapshot_capture_sha256"] or file_hash(destination / "checkpoint/pull-manifest.json")[0] != receipt["snapshot_pull_sha256"] or file_hash(destination / "source-archive.tar.gz")[0] != receipt["source_archive_sha256"] or file_hash(destination / "launch-receipt.json")[0] != receipt["launch_receipt_sha256"]:
        raise ValueError("proposal adoption restored identity differs")


def cloud(uri: str):
    import fsspec
    from rigging.filesystem.s3_compat import configure_coreweave_s3
    configure_coreweave_s3()
    return fsspec.core.url_to_fs(uri)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("pack")
    create.add_argument("--checkpoint", type=Path, required=True); create.add_argument("--source-archive", type=Path, required=True); create.add_argument("--launch-receipt", type=Path, required=True); create.add_argument("--archive", type=Path, required=True); create.add_argument("--transport", type=Path, required=True)
    put = sub.add_parser("upload")
    put.add_argument("--archive", type=Path, required=True); put.add_argument("--transport", type=Path, required=True); put.add_argument("--root", required=True); put.add_argument("--receipt", type=Path, required=True)
    get = sub.add_parser("download")
    get.add_argument("--receipt", type=Path, required=True); get.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pack": pack(args.checkpoint, args.source_archive, args.launch_receipt, args.archive, args.transport)
    elif args.command == "upload":
        document = validate_transport(json.loads(args.transport.read_text())); digest, size = file_hash(args.archive)
        if digest != document["archive_sha256"] or size != document["archive_bytes"]: raise ValueError("proposal adoption archive changed before upload")
        uri = args.root.rstrip("/") + f"/_proposal_adoption_blobs/{document['archive_sha256']}/adoption.tar.gz"; fs, key = cloud(uri)
        if fs.exists(key):
            with fs.open(key, "rb") as source:
                remote = Path(tempfile.mkstemp(prefix="proposal-adoption-remote-")[1])
                try:
                    with remote.open("wb") as target: shutil.copyfileobj(source, target, 1024 * 1024)
                    if file_hash(remote) != (digest, size): raise ValueError("conflicting immutable proposal adoption object")
                finally: remote.unlink(missing_ok=True)
        else:
            with args.archive.open("rb") as source, fs.open(key, "wb") as target: shutil.copyfileobj(source, target, 1024 * 1024)
        with fs.open(key, "rb") as source:
            remote = Path(tempfile.mkstemp(prefix="proposal-adoption-verify-")[1])
            try:
                with remote.open("wb") as target: shutil.copyfileobj(source, target, 1024 * 1024)
                if file_hash(remote) != (digest, size): raise ValueError("proposal adoption upload checksum mismatch")
            finally: remote.unlink(missing_ok=True)
        args.receipt.write_text(json.dumps({**document, "uri": uri}, indent=2, sort_keys=True) + "\n")
    else:
        receipt = validate_transport(json.loads(args.receipt.read_text())); uri = receipt.get("uri")
        if not isinstance(uri, str) or not uri.endswith(f"/{receipt['archive_sha256']}/adoption.tar.gz"): raise ValueError("invalid proposal adoption object URI")
        fs, key = cloud(uri)
        with tempfile.TemporaryDirectory(prefix="proposal-adoption-download-") as temporary:
            archive = Path(temporary) / "adoption.tar.gz"
            with fs.open(key, "rb") as source, archive.open("xb") as target: shutil.copyfileobj(source, target, 1024 * 1024)
            _unpack(archive, receipt, args.destination)
    return 0

if __name__ == "__main__": raise SystemExit(main())
