#!/usr/bin/env python3
"""Fetch and verify exact one-off publisher receipts from CW S3."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from capability_pipeline.image_publication_handoff import import_publication

_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_s3(uri: str) -> bytes:
    import fsspec
    from rigging.filesystem.s3_compat import configure_coreweave_s3

    configure_coreweave_s3()
    fs, key = fsspec.core.url_to_fs(uri)
    with fs.open(key, "rb") as stream:
        return stream.read()


def fetch(preparation_path: Path, output: Path, *, reader=_read_s3) -> dict:
    preparation = json.loads(preparation_path.read_text())
    if (preparation.get("schema_version") != "capability-oneoff-image-publisher-preparation-v1"
            or output.exists() or output.is_symlink()):
        raise ValueError("publisher preparation or fresh output is invalid")
    prefix = preparation["return_prefix"]
    raw_manifest = reader(prefix + "/return-manifest.json")
    manifest = json.loads(raw_manifest)
    roles = preparation["roles"]
    if (manifest.get("schema_version") != "capability-oneoff-image-publisher-return-v1"
            or manifest.get("packet_sha256") != preparation["packet_archive_sha256"]
            or manifest.get("source_sha256") != preparation["source_archive_sha256"]
            or manifest.get("packet_manifest_sha256") != preparation["packet_manifest_sha256"]
            or set(manifest.get("receipts", {})) != set(roles)):
        raise ValueError("publisher return manifest differs from frozen preparation")
    receipts = {}
    for role in roles:
        record = manifest["receipts"][role]
        uri = prefix + f"/publication-{role}.json"
        if (not isinstance(record, dict) or record.get("uri") != uri
                or not isinstance(record.get("sha256"), str) or not _HEX.fullmatch(record["sha256"])
                or type(record.get("bytes")) is not int or record["bytes"] <= 0):
            raise ValueError("publisher return receipt identity is invalid")
        data = reader(uri)
        if len(data) != record["bytes"] or _sha(data) != record["sha256"]:
            raise ValueError("publisher return receipt bytes differ")
        value = json.loads(data)
        if value.get("role") != role or value.get("state") != "published_pending_cold_pull":
            raise ValueError("publisher role was not published")
        receipts[role] = data
    output.mkdir(parents=True)
    (output / "return-manifest.json").write_bytes(raw_manifest)
    for role, data in receipts.items():
        (output / f"publication-{role}.json").write_bytes(data)
    return {"state": "verified", "roles": roles, "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--handoff", type=Path)
    args = parser.parse_args()
    if bool(args.status) != bool(args.handoff):
        parser.error("--status and --handoff must be supplied together for import")
    result = fetch(args.preparation, args.output)
    if args.status:
        for role in result["roles"]:
            import_publication(args.status, args.handoff, role,
                               args.output / f"publication-{role}.json")
        result["state"] = "imported"
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - no provider or credential text.
        print(json.dumps({"state": "failed", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
