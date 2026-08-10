# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile the CUDA 13 headers used by generated typed-FFI handlers."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

CUDA_HEADER_SOURCE = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <nv/target>

__global__ void shuttle_cuda_header_smoke(__half* fp16, __nv_bfloat16* bf16) {
  const int index = static_cast<int>(threadIdx.x);
  fp16[index] = __float2half(1.0f);
  bf16[index] = __float2bfloat16(1.0f);
}
""".lstrip()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--architecture", default="compute_90")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if platform.system() != "Linux":
        raise RuntimeError("the CUDA header smoke must run on Linux")
    if not arguments.nvcc.is_file():
        raise FileNotFoundError(arguments.nvcc)

    arguments.build_directory.mkdir(parents=True, exist_ok=True)
    source_path = arguments.build_directory / "cuda13_header_smoke.cu"
    object_path = arguments.build_directory / "cuda13_header_smoke.o"
    source_path.write_text(CUDA_HEADER_SOURCE)
    command = (
        str(arguments.nvcc),
        "--std=c++17",
        "--compile",
        f"--gpu-architecture={arguments.architecture}",
        str(source_path),
        "--output-file",
        str(object_path),
    )
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if not object_path.is_file() or object_path.stat().st_size == 0:
        raise RuntimeError(f"NVCC did not produce {object_path}")

    compiler_version = subprocess.run(
        (str(arguments.nvcc), "--version"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(
        json.dumps(
            {
                "architecture": arguments.architecture,
                "cccl": importlib.metadata.version("nvidia-cuda-cccl"),
                "command": command,
                "nvcc": str(arguments.nvcc.resolve()),
                "nvcc_version": compiler_version,
                "object_bytes": object_path.stat().st_size,
                "status": "cuda13_headers_compiled",
                "stderr": completed.stderr,
                "stdout": completed.stdout,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
