# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pack the CUDA Torch runtime already installed in a GB200 development pod."""

from __future__ import annotations

import sysconfig
import tarfile
from pathlib import Path

import zstandard

OUTPUT_PATH = Path("/tmp/ep64-torch-cu128-sitepackages.tar.zst")


def _include(path: Path, site_packages: Path) -> bool:
    relative = path.relative_to(site_packages)
    if relative.parts[:2] == ("nvidia", "cu13"):
        return False
    return "__pycache__" not in relative.parts


def main() -> None:
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    roots = [
        site_packages / "torch",
        site_packages / "torchgen",
        site_packages / "functorch",
        site_packages / "nvidia",
        site_packages / "cuda",
        site_packages / "triton",
    ]
    roots.extend(site_packages.glob("torch-*.dist-info"))
    roots.extend(site_packages.glob("cuda_*-*.dist-info"))
    roots.extend(site_packages.glob("nvidia_*_cu12-*.dist-info"))
    roots.extend(site_packages.glob("triton-*.dist-info"))
    roots = sorted(set(roots))
    missing = [path for path in roots[:6] if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing CUDA Torch runtime paths: {missing}")

    with OUTPUT_PATH.open("wb") as raw_output:
        compressor = zstandard.ZstdCompressor(level=3, threads=-1)
        with compressor.stream_writer(raw_output) as compressed_output:
            with tarfile.open(fileobj=compressed_output, mode="w|") as archive:
                for root in roots:
                    archive.add(
                        root,
                        arcname=root.relative_to(site_packages),
                        recursive=True,
                        filter=lambda info: info if _include(Path(info.name), Path(".")) else None,
                    )
    print(f"{OUTPUT_PATH} {OUTPUT_PATH.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
