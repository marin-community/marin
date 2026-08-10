#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile, load, and register generated CUDA typed-FFI sources without devices."""

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import jax
import jaxlib


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    return parser.parse_args()


def _package_version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def _shared_library(directory: Path, name: str) -> Path:
    unversioned = directory / f"lib{name}.so"
    if unversioned.is_file():
        return unversioned
    candidates = sorted(directory.glob(f"lib{name}.so.*"))
    if not candidates:
        raise RuntimeError(f"missing lib{name}.so under {directory}")
    return candidates[-1]


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["architecture"] != args.architecture:
        raise RuntimeError("manifest and requested CUDA architecture differ")
    if not args.nvcc.is_file():
        raise RuntimeError(f"missing NVCC: {args.nvcc}")
    cuda_root = args.nvcc.parent.parent
    cuda_include = cuda_root / "include"
    cuda_library = cuda_root / "lib"
    jaxlib_include = Path(jaxlib.__file__).resolve().parent / "include"
    if not cuda_include.is_dir() or not cuda_library.is_dir() or not jaxlib_include.is_dir():
        raise RuntimeError("CUDA or jaxlib include/library directories are incomplete")
    cudart = _shared_library(cuda_library, "cudart")
    cublas = _shared_library(cuda_library, "cublas")
    args.build_directory.mkdir(parents=True, exist_ok=True)
    records = []
    loaded_libraries = []
    for handler in manifest["handlers"]:
        source = args.manifest.parent / handler["source"]
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if source_digest != handler["source_sha256"]:
            raise RuntimeError(f"source digest mismatch for {source.name}")
        library = args.build_directory / f"{handler['handler_symbol']}.so"
        command = (
            str(args.nvcc),
            "-std=c++17",
            "-O3",
            f"-arch={args.architecture}",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-I",
            str(jaxlib_include),
            "-I",
            str(cuda_include),
            str(source),
            "-o",
            str(library),
            "-cudart=none",
            "-L",
            str(cuda_library),
            "-Xlinker",
            "-rpath",
            "-Xlinker",
            str(cuda_library),
            "-Xlinker",
            str(cudart),
            "-Xlinker",
            str(cublas),
        )
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            print(error.stdout, file=sys.stderr)
            print(error.stderr, file=sys.stderr)
            raise
        loaded = ctypes.CDLL(str(library))
        symbol = getattr(loaded, handler["handler_symbol"])
        symbol.restype = ctypes.c_void_p
        jax.ffi.register_ffi_target(
            handler["target"],
            jax.ffi.pycapsule(symbol),
            platform="CUDA",
            api_version=1,
        )
        loaded_libraries.append(loaded)
        records.append(
            {
                "target": handler["target"],
                "handler_symbol": handler["handler_symbol"],
                "source_sha256": source_digest,
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "command": list(command),
                "compiler_stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                "compiler_stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
                "compiler_stderr_lines": len(completed.stderr.splitlines()),
                "loaded": True,
                "registered": True,
            }
        )
    nvcc_version = subprocess.run(
        (str(args.nvcc), "--version"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    result = {
        "status": "cpu_linux_compile_load_register_passed",
        "device_query_or_execution": False,
        "architecture": args.architecture,
        "source_revision": manifest["source_revision"],
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "handlers": records,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "nvidia-cuda-nvcc": _package_version("nvidia-cuda-nvcc"),
            "nvidia-cuda-cccl": _package_version("nvidia-cuda-cccl"),
            "nvidia-cuda-crt": _package_version("nvidia-cuda-crt"),
            "nvidia-cuda-nvrtc": _package_version("nvidia-cuda-nvrtc"),
            "nvidia-cuda-runtime": _package_version("nvidia-cuda-runtime"),
            "nvidia-cublas": _package_version("nvidia-cublas"),
            "nvidia-nvvm": _package_version("nvidia-nvvm"),
            "nvcc_version": nvcc_version,
            "executable": sys.executable,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
