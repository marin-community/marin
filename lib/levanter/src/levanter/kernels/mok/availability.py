# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Cheap environment checks for the optional Mixture-of-Kittens runtime."""

from __future__ import annotations

import importlib
import re
import shutil
import subprocess
from dataclasses import dataclass


_REQUIRED_TORCH_VERSION = (2, 11)
_REQUIRED_TORCH_CUDA = (13, 0)
_NATIVE_SYMBOLS = (
    "levanter_mok_ffi_abi_version",
    "levanter_mok_bf16_forward",
    "levanter_mok_bf16_backward",
    "levanter_mok_bf16_forward_scratch_bytes_v1",
    "levanter_mok_bf16_backward_scratch_bytes_v1",
    "levanter_mok_register_workspace_v1",
    "levanter_mok_close_workspace_v1",
)


@dataclass(frozen=True)
class MokPreflightStatus:
    """Result of probing the host without initializing CUDA or distributed state."""

    torch_version: str | None
    torch_cuda_version: str | None
    nvcc_path: str | None
    nvcc_version: tuple[int, int] | None
    native_extension_loaded: bool
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _major_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"\s*(\d+)\.(\d+)", version)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _nvcc_release(output: str) -> tuple[int, int] | None:
    match = re.search(r"\brelease\s+(\d+)\.(\d+)\b", output)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def mok_preflight_status() -> MokPreflightStatus:
    """Check the pinned Torch/CUDA toolchain and MoK native ABI.

    This function deliberately imports Torch and MoK only when called. Importing
    :mod:`levanter.kernels.mok` itself is therefore safe in CPU and TPU jobs.
    """

    errors: list[str] = []
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    native_extension_loaded = False

    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        errors.append(f"Torch could not be loaded; MoK requires torch 2.11 built for CUDA 13.0: {exc}")
    else:
        torch_version = str(torch.__version__)
        parsed_torch_version = _major_minor(torch_version)
        if parsed_torch_version != _REQUIRED_TORCH_VERSION:
            errors.append(f"MoK requires torch 2.11, found {torch_version}")

        raw_cuda_version = torch.version.cuda
        torch_cuda_version = None if raw_cuda_version is None else str(raw_cuda_version)
        parsed_cuda_version = None if torch_cuda_version is None else _major_minor(torch_cuda_version)
        if parsed_cuda_version != _REQUIRED_TORCH_CUDA:
            errors.append(
                f"MoK requires the torch CUDA 13.0 (cu130) build, found torch.version.cuda={torch_cuda_version!r}"
            )

    nvcc_path = shutil.which("nvcc")
    nvcc_version: tuple[int, int] | None = None
    if nvcc_path is None:
        errors.append("nvcc is not on PATH; MoK requires the CUDA 13 toolkit")
    else:
        try:
            result = subprocess.run(
                [nvcc_path, "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"failed to run {nvcc_path} --version: {exc}")
        else:
            nvcc_version = _nvcc_release(result.stdout + result.stderr)
            if nvcc_version is None:
                errors.append(f"could not parse the CUDA release from {nvcc_path} --version")
            elif nvcc_version != _REQUIRED_TORCH_CUDA:
                errors.append(
                    "MoK requires nvcc to match torch.version.cuda=13.0, "
                    f"found {nvcc_version[0]}.{nvcc_version[1]} at {nvcc_path}"
                )

    try:
        native = importlib.import_module("mok._C")
    except (ImportError, OSError) as exc:
        errors.append(f"the MoK native extension is unavailable: {exc}")
    else:
        missing_symbols = tuple(name for name in _NATIVE_SYMBOLS if not hasattr(native, name))
        if missing_symbols:
            errors.append("the MoK native extension is missing the Levanter ABI: " + ", ".join(missing_symbols))
        else:
            abi_version = int(native.levanter_mok_ffi_abi_version())
            if abi_version != 1:
                errors.append(f"the MoK native extension has FFI ABI {abi_version}; Levanter requires ABI 1")
            else:
                native_extension_loaded = True

    return MokPreflightStatus(
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        nvcc_path=nvcc_path,
        nvcc_version=nvcc_version,
        native_extension_loaded=native_extension_loaded,
        errors=tuple(errors),
    )


def require_mok_available() -> MokPreflightStatus:
    """Return a successful preflight status or raise one actionable error."""

    status = mok_preflight_status()
    if status.errors:
        details = "\n  - ".join(status.errors)
        raise RuntimeError(f"Mixture-of-Kittens preflight failed:\n  - {details}")
    return status
