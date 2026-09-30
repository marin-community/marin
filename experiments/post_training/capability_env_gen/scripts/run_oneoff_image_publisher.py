#!/usr/bin/env python3
"""Trusted Kubernetes worker for one frozen reviewed image handoff.

``open_packet`` and ``publish_packet_role`` are the single packet-validation
and publication path, shared with the long-running publisher service
(``scripts/run_image_publisher_service.py``).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit

from capability_pipeline.image_publication_handoff import unpack_handoff

try:  # Package import (tests, service) or script-directory import (one-off Job).
    from scripts.publish_generic_task_image import publish_role, write_receipt
except ImportError:  # pragma: no cover - exercised only as a bare script
    from publish_generic_task_image import (  # type: ignore[no-redef]
        publish_role,
        write_receipt,
    )

PACKET_ROLES = frozenset({"candidate", "private_verifier"})


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def open_packet(packet_archive: Path, packet: Path) -> dict:
    """Unpack and validate a handoff exactly as the publisher requires."""
    unpack_handoff(packet_archive, packet)
    manifest = json.loads((packet / "manifest.json").read_text())
    roles = manifest["roles"]
    if not roles or any(role not in PACKET_ROLES for role in roles):
        raise ValueError("publisher packet has invalid roles")
    return manifest


def publish_packet_role(packet: Path, manifest: dict, role: str, output_root: Path,
                        max_rootfs_bytes: int, credentials_file: Path, *,
                        presigner_factory=None, registry_client_factory=None) -> Path:
    """Publish one role of an opened packet; return its canonical receipt path."""
    if role not in manifest["roles"]:
        raise ValueError("publisher role is absent from its packet")
    if not credentials_file.is_absolute() or credentials_file.is_symlink() or not credentials_file.is_file():
        raise ValueError("publisher credential mount is unavailable")
    if type(max_rootfs_bytes) is not int or max_rootfs_bytes <= 0:
        raise ValueError("publisher rootfs limit is invalid")
    output = output_root / f"publication-{role}.json"
    layer_dir = output_root / f"layer-{role}"
    extra = {} if registry_client_factory is None else {"registry_client_factory": registry_client_factory}
    try:
        receipt = publish_role(
            plan=packet / "plan.json", workspace=packet / "workspace",
            capture_tools=packet / "tools/capture-tools",
            approval=packet / "review/approval.json",
            builder_session_ids=set(manifest["builder_session_ids"]),
            capture_receipt=packet / "captures" / f"{role}.json",
            download_layer_dir=layer_dir, max_uncompressed_bytes=max_rootfs_bytes,
            credentials_file=credentials_file, execute=True,
            presigner_factory=presigner_factory, **extra,
        )
    finally:
        shutil.rmtree(layer_dir, ignore_errors=True)
    if receipt.get("state") != "published_pending_cold_pull" or receipt.get("role") != role:
        raise ValueError("publisher did not produce a pending-cold-pull receipt")
    write_receipt(receipt, output)
    return output


def _object(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/") or parsed.query or parsed.fragment:
        raise ValueError("publisher return URI is invalid")
    return parsed.netloc, parsed.path.lstrip("/")


def _put_new(client, uri: str, path: Path) -> None:
    bucket, key = _object(uri)
    from botocore.exceptions import ClientError

    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 404:
            raise RuntimeError("publisher could not confirm fresh return object") from None
    else:
        raise ValueError("publisher return object already exists")
    with path.open("rb") as source:
        client.put_object(Bucket=bucket, Key=key, Body=source)
    observed = client.head_object(Bucket=bucket, Key=key)
    if observed.get("ContentLength") != path.stat().st_size:
        raise ValueError("publisher return object size differs")


def run(packet_archive: Path, packet_sha256: str, source_archive: Path,
        source_sha256: str, output_root: Path, return_prefix: str,
        max_rootfs_bytes: int, credentials_file: Path) -> dict:
    if _sha(packet_archive) != packet_sha256 or _sha(source_archive) != source_sha256:
        raise ValueError("publisher staged archive changed")
    if output_root.exists():
        raise ValueError("publisher output already exists")
    output_root.mkdir(parents=True)
    packet = output_root / "packet"
    manifest = open_packet(packet_archive, packet)
    roles = manifest["roles"]
    from capability_pipeline.publication_exchange import ObjectStoreReader, s3_client

    # Pinned reader, never the packet's staged cw_presign; the one-off keeps
    # its historical unrestricted bucket scope (operator-prepared packets).
    reader = ObjectStoreReader(s3_client(), ["s3://marin-us-east-02a"])
    receipts = {}
    for role in roles:
        output = publish_packet_role(packet, manifest, role, output_root, max_rootfs_bytes,
                                     credentials_file, presigner_factory=lambda: reader)
        receipts[role] = {"sha256": _sha(output), "bytes": output.stat().st_size,
                          "uri": return_prefix.rstrip("/") + f"/publication-{role}.json"}
    import botocore.session
    from botocore.config import Config

    client = botocore.session.get_session().create_client(
        "s3", region_name="auto", endpoint_url="https://cwobject.com",
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )
    for role in roles:
        _put_new(client, receipts[role]["uri"], output_root / f"publication-{role}.json")
    result = {"schema_version": "capability-oneoff-image-publisher-return-v1",
              "packet_sha256": packet_sha256, "source_sha256": source_sha256,
              "packet_manifest_sha256": _sha(packet / "manifest.json"), "receipts": receipts}
    result_path = output_root / "return-manifest.json"
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    _put_new(client, return_prefix.rstrip("/") + "/return-manifest.json", result_path)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 9:
        raise SystemExit("expected packet, packet SHA, source, source SHA, output, return prefix, limit, credentials")
    try:
        run(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4],
            Path(sys.argv[5]), sys.argv[6], int(sys.argv[7]), Path(sys.argv[8]))
        print(json.dumps({"state": "receipts_uploaded"}))
    except Exception as error:  # noqa: BLE001 - redact provider and credential text.
        print(json.dumps({"state": "failed", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
