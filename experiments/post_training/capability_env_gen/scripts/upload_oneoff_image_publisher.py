#!/usr/bin/env python3
"""Upload exact reviewed packet and trusted source objects before Job creation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _cloud(uri: str):
    import fsspec
    from rigging.filesystem.s3_compat import configure_coreweave_s3

    configure_coreweave_s3()
    return fsspec.core.url_to_fs(uri)


def upload(preparation_path: Path, packet: Path, *, cloud=_cloud) -> dict:
    preparation = json.loads(preparation_path.read_text())
    if preparation.get("schema_version") != "capability-oneoff-image-publisher-preparation-v1":
        raise ValueError("publisher preparation is invalid")
    source = preparation_path.parent / "publisher-source.tar.gz"
    objects = (
        (packet, preparation["packet_archive_sha256"], preparation["packet_uri"]),
        (source, preparation["source_archive_sha256"], preparation["source_uri"]),
    )
    for path, expected, uri in objects:
        if path.is_symlink() or not path.is_file() or _sha(path) != expected:
            raise ValueError("publisher input changed after preparation")
        fs, key = cloud(uri)
        if not fs.exists(key):
            with path.open("rb") as input_stream, fs.open(key, "wb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=1 << 20)
        with fs.open(key, "rb") as remote_stream:
            if hashlib.file_digest(remote_stream, "sha256").hexdigest() != expected:
                raise ValueError("publisher S3 object differs from frozen input")
    return {"state": "uploaded_verified", "job": preparation["job_name"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(upload(args.preparation, args.handoff), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - no provider or credential text.
        print(json.dumps({"state": "failed", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
