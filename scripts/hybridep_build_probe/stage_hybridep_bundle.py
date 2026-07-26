# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pack the validated HybridEP source tree and JAX bridge extension."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
from pathlib import Path

import fsspec
import zstandard

SOURCE_ROOT = Path("/tmp/DeepEP")
OUTPUT_PATH = Path("/tmp/ep64-hybridep-jax-preallocated-v7.tar.zst")


def _include(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    relative = Path(info.name)
    if ".git" in relative.parts or "build" in relative.parts or "__pycache__" in relative.parts:
        return None
    return info


def _pack_bundle() -> None:
    extensions = list(SOURCE_ROOT.glob("hybrid_ep_cpp*.so"))
    if len(extensions) != 1:
        raise RuntimeError(f"Expected one HybridEP extension in {SOURCE_ROOT}, found {extensions}")

    with OUTPUT_PATH.open("wb") as raw_output:
        compressor = zstandard.ZstdCompressor(level=3, threads=-1)
        with compressor.stream_writer(raw_output) as compressed_output:
            with tarfile.open(fileobj=compressed_output, mode="w|") as archive:
                archive.add(
                    SOURCE_ROOT,
                    arcname=SOURCE_ROOT.name,
                    recursive=True,
                    filter=_include,
                )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _upload_bundle(uri: str) -> None:
    filesystem, remote_path = fsspec.core.url_to_fs(uri)
    with OUTPUT_PATH.open("rb") as local_input:
        with filesystem.open(remote_path, "wb") as remote_output:
            shutil.copyfileobj(local_input, remote_output, length=8 * 1024 * 1024)
    local_size = OUTPUT_PATH.stat().st_size
    remote_size = filesystem.size(remote_path)
    if remote_size != local_size:
        raise RuntimeError(f"Uploaded bundle size mismatch: local={local_size}, remote={remote_size}")


def main() -> None:
    _pack_bundle()
    checksum = _sha256(OUTPUT_PATH)
    upload_uri = os.environ.get("HYBRID_EP_BUNDLE_UPLOAD_URI")
    if upload_uri is not None:
        _upload_bundle(upload_uri)
    print(
        f"HYBRID_EP_BUNDLE path={OUTPUT_PATH} size={OUTPUT_PATH.stat().st_size} "
        f"sha256={checksum} upload_uri={upload_uri}",
        flush=True,
    )


if __name__ == "__main__":
    main()
