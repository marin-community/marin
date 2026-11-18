# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pin the NCCL library, runtime version and transport environment of a worker."""

import hashlib
import os
from pathlib import Path

import torch


def runtime_receipt() -> dict:
    """Capture versions and transport settings, excluding unrelated environment values."""
    libraries = []
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        if "libnccl.so" in line and line.split()[-1].startswith("/"):
            paths.add(line.split()[-1])
    for path in sorted(paths):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        libraries.append({"path": path, "sha256": digest.hexdigest()})
    config_files = []
    for path in dict.fromkeys(["/etc/nccl.conf", str(Path.home() / ".nccl.conf"), os.environ.get("NCCL_CONF_FILE", "")]):
        if not path or not Path(path).is_file():
            continue
        raw = Path(path).read_bytes()
        config_files.append(
            {
                "path": path,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "settings": [
                    line for line in raw.decode(errors="replace").splitlines() if line.strip().startswith("NCCL_")
                ],
            }
        )
    return {
        "torch_version": str(torch.__version__),
        "torch_git": torch.version.git_version,
        "torch_cuda": torch.version.cuda,
        "nccl_version": torch.cuda.nccl.version(),
        "loaded_nccl_libraries": libraries,
        "nccl_config_files": config_files,
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("NCCL_", "TORCH_NCCL_", "TORCH_FR_"))
            or key in ("CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF")
        },
    }
