# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pack the CUDA 13 toolkit already installed in a GB200 development pod."""

from __future__ import annotations

import sysconfig
import tarfile
from pathlib import Path

import zstandard

OUTPUT_PATH = Path("/tmp/ep64-cuda13-toolkit.tar.zst")


def main() -> None:
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    cuda_root = site_packages / "nvidia" / "cu13"
    if not (cuda_root / "bin" / "nvcc").exists():
        raise RuntimeError(f"CUDA compiler is missing from {cuda_root}")

    with OUTPUT_PATH.open("wb") as raw_output:
        compressor = zstandard.ZstdCompressor(level=3, threads=-1)
        with compressor.stream_writer(raw_output) as compressed_output:
            with tarfile.open(fileobj=compressed_output, mode="w|") as archive:
                archive.dereference = True
                archive.add(
                    cuda_root,
                    arcname=cuda_root.relative_to(site_packages),
                    recursive=True,
                    filter=lambda info: None if "__pycache__" in Path(info.name).parts else info,
                )
    print(f"{OUTPUT_PATH} {OUTPUT_PATH.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
