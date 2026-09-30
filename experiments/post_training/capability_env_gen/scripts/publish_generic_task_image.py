#!/usr/bin/env python3
"""Publish one GLM-reviewed generic rootfs from an isolated trusted worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

from capability_pipeline.generic_image_publication import (
    publication_registry_host,
    publish_generic_capture,
)
from capability_pipeline.oci_registry import RegistryClient

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _capture_object(receipt_path: Path) -> tuple[str, str, int, str, dict]:
    """Return only a complete, content-addressed object named by a capture receipt."""
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("capture receipt is unavailable")
    receipt = json.loads(receipt_path.read_text())
    archive = receipt.get("capture") if isinstance(receipt, dict) else None
    process = archive.get("capture") if isinstance(archive, dict) else None
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != "capability-rootfs-capture-v1"
        or receipt.get("state") != "captured_pending_privacy_and_publication"
        or not isinstance(archive, dict)
        or not isinstance(process, dict)
        or archive.get("ok") is not True
        or not isinstance(archive.get("object"), str)
        or not isinstance(archive.get("object_key"), str)
        or type(archive.get("object_bytes")) is not int
        or archive["object_bytes"] <= 0
        or not isinstance(archive.get("sha256"), str)
        or _SHA256.fullmatch(archive["sha256"]) is None
        or process.get("compressed_bytes") != archive["object_bytes"]
        or process.get("sha256") != archive["sha256"]
        or process.get("tar_exit") != 0
        or process.get("gzip_exit") != 0
    ):
        raise ValueError("capture receipt lacks a complete sanitized object identity")
    parsed = urlsplit(archive["object"])
    key = parsed.path.removeprefix("/")
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not key
        or key != archive["object_key"]
        or "\\" in key
        or "\0" in key
        or any(part in {"", ".", ".."} for part in Path(key).parts)
    ):
        raise ValueError("capture receipt object identity is unsafe")
    return parsed.netloc, key, archive["object_bytes"], archive["sha256"], receipt


def download_captured_layer(
    *, capture_receipt: Path, download_dir: Path, capture_tools: Path,
    presigner_factory=None,
) -> tuple[Path, dict]:
    """Stream the receipt-bound S3 object into a private trusted-worker file."""
    bucket, key, expected_bytes, expected_sha256, _receipt = _capture_object(capture_receipt)
    if capture_tools.is_symlink() or not capture_tools.is_dir():
        raise ValueError("staged capture tools are unavailable")
    if download_dir.is_symlink():
        raise ValueError("layer download directory is linked")
    download_dir.mkdir(parents=True, exist_ok=True)
    if not download_dir.is_dir():
        raise ValueError("layer download directory is unavailable")
    if presigner_factory is None:
        sys.path.insert(0, str(capture_tools.resolve()))
        from cw_presign import (
            presigner as presigner_factory,  # type: ignore[import-not-found]
        )

    signer = presigner_factory()
    head = signer.head(bucket, key)
    if not isinstance(head, dict) or head.get("ContentLength") != expected_bytes:
        raise ValueError("capture object size differs before download")
    response = signer.c.get_object(Bucket=bucket, Key=key)
    body = response.get("Body") if isinstance(response, dict) else None
    if body is None or not hasattr(body, "read"):
        raise ValueError("capture object body is unavailable")
    target = download_dir / (expected_sha256 + ".tar.gz")
    temporary = target.with_name(target.name + ".part")
    if target.exists() or target.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise ValueError("layer download target already exists or is linked")
    total, checksum = 0, hashlib.sha256()
    try:
        with temporary.open("xb") as stream:
            while chunk := body.read(1 << 20):
                if not isinstance(chunk, bytes):
                    raise TypeError("capture object stream is invalid")
                total += len(chunk)
                if total > expected_bytes:
                    raise ValueError("capture object exceeds receipt byte count")
                checksum.update(chunk)
                stream.write(chunk)
        if total != expected_bytes or checksum.hexdigest() != expected_sha256:
            raise ValueError("capture object byte count or SHA-256 differs")
        temporary.chmod(0o400)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    transport = {
        "schema_version": "capability-captured-layer-transport-v1",
        "capture_receipt_sha256": _sha256(capture_receipt),
        "object": f"s3://{bucket}/{key}",
        "object_bucket": bucket,
        "object_key": key,
        "expected_bytes": expected_bytes,
        "expected_sha256": expected_sha256,
        "downloaded_bytes": total,
        "downloaded_sha256": checksum.hexdigest(),
        "local_filename": target.name,
    }
    return target, transport


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--capture-tools", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--builder-session-id", action="append", default=[])
    parser.add_argument("--capture-receipt", type=Path, required=True)
    layer_source = parser.add_mutually_exclusive_group(required=True)
    layer_source.add_argument("--layer", type=Path)
    layer_source.add_argument("--download-layer-dir", type=Path)
    parser.add_argument("--max-uncompressed-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--credentials-file", type=Path,
        help="absolute path to the explicitly mounted trusted-publisher registry secret",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("publication receipt already exists")
    receipt = publish_role(
        plan=args.plan, workspace=args.workspace, capture_tools=args.capture_tools,
        approval=args.approval, builder_session_ids=set(args.builder_session_id),
        capture_receipt=args.capture_receipt, layer=args.layer,
        download_layer_dir=args.download_layer_dir,
        max_uncompressed_bytes=args.max_uncompressed_bytes,
        credentials_file=args.credentials_file, execute=args.execute,
    )
    write_receipt(receipt, args.output)
    print(json.dumps({"state": receipt["state"], "role": receipt["role"], "receipt": str(args.output)}))
    return 0


def publish_role(
    *, plan: Path, workspace: Path, capture_tools: Path, approval: Path,
    builder_session_ids: set[str], capture_receipt: Path, layer: Path | None = None,
    download_layer_dir: Path | None = None, max_uncompressed_bytes: int,
    credentials_file: Path | None = None, execute: bool = False,
    presigner_factory=None, registry_client_factory=RegistryClient,
) -> dict:
    """Review (and with ``execute``, publish) one role; shared by the CLI,
    the one-off worker and the long-running publisher service.

    ``presigner_factory`` replaces the packet's staged ``cw_presign`` with
    pinned trusted code; ``registry_client_factory(host, repository, user,
    password)`` builds the registry client (a fake in dry runs).
    """
    if (layer is None) == (download_layer_dir is None):
        raise ValueError("exactly one layer source is required")
    if download_layer_dir is not None and not execute:
        raise ValueError("download-layer-dir requires trusted publisher --execute")
    client = None
    if execute:
        if (
            credentials_file is None
            or not credentials_file.is_absolute()
            or credentials_file.is_symlink()
            or not credentials_file.is_file()
        ):
            raise ValueError("isolated trusted publisher credentials are absent")
        credentials = json.loads(credentials_file.read_text())
        plan_document = json.loads(plan.read_text())
        role = json.loads(capture_receipt.read_text())["role"]
        image = next(row for row in plan_document["images"] if row["role"] == role)
        try:
            # Equal hosts, or an object-store plan host corrected to the
            # credential's registry (recorded in the receipt by the publisher).
            publication_registry_host(plan_document["registry_host"], credentials.get("registry"))
        except ValueError as error:
            raise ValueError("publisher credential host differs from reviewed plan") from error
        client = registry_client_factory(
            credentials["registry"], image["repository"],
            credentials["user"], credentials["password"],
        )
    transport = None
    if download_layer_dir is not None:
        layer, transport = download_captured_layer(
            capture_receipt=capture_receipt,
            download_dir=download_layer_dir,
            capture_tools=capture_tools,
            presigner_factory=presigner_factory,
        )
    receipt = publish_generic_capture(
        plan_path=plan, workspace=workspace,
        capture_tools=capture_tools, approval_path=approval,
        builder_session_ids=builder_session_ids,
        capture_path=capture_receipt, layer_path=layer,
        max_uncompressed_bytes=max_uncompressed_bytes,
        registry_client=client,
    )
    if transport is not None:
        receipt["layer_transport"] = transport
    return receipt


def write_receipt(receipt: dict, output: Path) -> None:
    """Write a receipt once, in the canonical byte form every caller shares."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - never print credential-bearing provider text.
        print(json.dumps({"state": "failed", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
