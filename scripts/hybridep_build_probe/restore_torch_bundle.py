# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restore the CUDA Torch runtime used by the HybridEP JAX bridge."""

from __future__ import annotations

import shutil
import sysconfig
import tarfile
import tempfile
from pathlib import Path

import fsspec
import zstandard

TORCH_BUNDLE_URI = "s3://marin-us-east-02a/tmp/ttl=7d/wheels/ep64-torch-cu128-sitepackages.tar.zst"
CUDA13_BUNDLE_URI = "s3://marin-us-east-02a/tmp/ttl=7d/wheels/ep64-cuda13-toolkit.tar.zst"
HYBRIDEP_BUNDLE_URI = "s3://marin-us-east-02a/tmp/ttl=7d/wheels/ep64-hybridep-jax-metadata-headroom-v8.tar.zst"


def _extract_bundle(uri: str, site_packages: Path) -> None:
    with fsspec.open(uri, "rb") as compressed_input:
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(compressed_input) as tar_input:
            with tarfile.open(fileobj=tar_input, mode="r|") as archive:
                archive.extractall(site_packages, filter="data")


def restore_hybridep_bundle(destination: Path) -> None:
    _extract_bundle(HYBRIDEP_BUNDLE_URI, destination)


def restore_torch_runtime() -> None:
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    _extract_bundle(TORCH_BUNDLE_URI, site_packages)


def restore_cuda13_toolkit() -> None:
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    with tempfile.TemporaryDirectory(prefix="hybridep-cuda13-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        _extract_bundle(CUDA13_BUNDLE_URI, temporary_root)
        staged_cuda = temporary_root / "nvidia" / "cu13"
        cuda_home = site_packages / "nvidia" / "cu13"
        if cuda_home.exists():
            shutil.rmtree(cuda_home)
        cuda_home.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(staged_cuda, cuda_home)


def main() -> None:
    restore_torch_runtime()
    restore_cuda13_toolkit()


if __name__ == "__main__":
    main()
