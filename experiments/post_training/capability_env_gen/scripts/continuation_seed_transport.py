#!/usr/bin/env python3
"""Move a large, manifest-bound continuation archive outside an Iris bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SCHEMA = "capability-continuation-seed-transport-v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def receipt_for(
    root: str, archive: Path, source_manifest_sha256: str
) -> tuple[str, dict[str, object]]:
    if len(source_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_manifest_sha256
    ):
        raise ValueError("invalid continuation source manifest digest")
    data = archive.read_bytes()
    digest = sha256(data)
    uri = root.rstrip("/") + f"/_continuation_seed_blobs/{digest}/restore-seed.tar.gz"
    return uri, {
        "schema_version": SCHEMA,
        "uri": uri,
        "sha256": digest,
        "bytes": len(data),
        "source_manifest_sha256": source_manifest_sha256,
    }


def validate_receipt(receipt: object) -> dict[str, object]:
    if not isinstance(receipt, dict) or receipt.get("schema_version") != SCHEMA:
        raise ValueError("invalid continuation seed transport receipt")
    uri, digest, size, source_manifest = (
        receipt.get("uri"),
        receipt.get("sha256"),
        receipt.get("bytes"),
        receipt.get("source_manifest_sha256"),
    )
    if (
        not isinstance(uri, str)
        or not uri.startswith("s3://")
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or type(size) is not int
        or size < 0
        or not isinstance(source_manifest, str)
        or len(source_manifest) != 64
        or any(character not in "0123456789abcdef" for character in source_manifest)
        or not uri.endswith(f"/_continuation_seed_blobs/{digest}/restore-seed.tar.gz")
    ):
        raise ValueError("invalid continuation seed transport receipt")
    return receipt


def require_source_manifest(receipt: dict[str, object], expected: str) -> None:
    if receipt["source_manifest_sha256"] != expected:
        raise ValueError(
            "continuation seed receipt belongs to a different source manifest"
        )


def upload(fs, key: str, archive: Path, receipt: dict[str, object]) -> None:
    data = archive.read_bytes()
    if sha256(data) != receipt["sha256"] or len(data) != receipt["bytes"]:
        raise ValueError("continuation seed changed while preparing transport")
    if fs.exists(key):
        existing = fs.cat(key)
        if sha256(existing) != receipt["sha256"] or len(existing) != receipt["bytes"]:
            raise ValueError(
                "existing continuation seed object has a conflicting digest"
            )
        return
    fs.pipe(key, data)
    published = fs.cat(key)
    if sha256(published) != receipt["sha256"] or len(published) != receipt["bytes"]:
        raise ValueError("continuation seed upload checksum mismatch")


def download(fs, key: str, destination: Path, receipt: dict[str, object]) -> None:
    if destination.exists() or destination.is_symlink():
        raise ValueError("continuation seed destination already exists")
    data = fs.cat(key)
    if sha256(data) != receipt["sha256"] or len(data) != receipt["bytes"]:
        raise ValueError("continuation seed download checksum mismatch")
    # Do not follow a staged symlink to publish a transport object elsewhere.
    for parent in (destination.parent, *destination.parents):
        if parent.is_symlink():
            raise ValueError("continuation seed destination has a symlinked parent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("continuation seed destination has a symlinked parent")
    temporary = destination.with_name(destination.name + ".transport-part")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("continuation seed temporary destination already exists")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
        # link(2) gives a no-replace final install: unlike replace(), it cannot
        # silently overwrite a destination introduced after the first check.
        os.link(temporary, destination)
    except FileExistsError as error:
        raise ValueError("continuation seed destination already exists") from error
    finally:
        temporary.unlink(missing_ok=True)


def cloud(uri: str):
    import fsspec
    from rigging.filesystem.s3_compat import configure_coreweave_s3

    configure_coreweave_s3()
    return fsspec.core.url_to_fs(uri)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    put = sub.add_parser("upload")
    put.add_argument("--archive", type=Path, required=True)
    put.add_argument("--root", required=True)
    put.add_argument("--receipt", type=Path, required=True)
    put.add_argument("--source-manifest-sha256", required=True)
    get = sub.add_parser("download")
    get.add_argument("--receipt", type=Path, required=True)
    get.add_argument("--destination", type=Path, required=True)
    get.add_argument("--expected-source-manifest-sha256", required=True)
    args = parser.parse_args()
    if args.command == "upload":
        if not args.archive.is_file():
            raise ValueError("continuation seed archive is missing")
        uri, receipt = receipt_for(args.root, args.archive, args.source_manifest_sha256)
        fs, key = cloud(uri)
        upload(fs, key, args.archive, receipt)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    else:
        receipt = validate_receipt(json.loads(args.receipt.read_text()))
        require_source_manifest(receipt, args.expected_source_manifest_sha256)
        fs, key = cloud(str(receipt["uri"]))
        download(fs, key, args.destination, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
