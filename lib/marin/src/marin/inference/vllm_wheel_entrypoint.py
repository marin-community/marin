# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify a Marin vLLM wheel before invoking its normal CLI.

The launcher already names one promoted wheel URL, so startup only checks what building that command
cannot settle: the PEP 610 record identifies the promoted wheel, ``vllm._C`` comes from inside that
distribution, and the GPU this process was scheduled onto is one the wheel was compiled for. The
first two are a pair. A leaked ``PYTHONPATH`` entry holding a complete vLLM install would satisfy the
extension check on its own, because its metadata and its extension agree with each other.
"""

import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

_SELECTED_SENTINEL = "MARIN_VLLM_WHEEL_SELECTED="
_VERIFIED_SENTINEL = "MARIN_VLLM_WHEEL_VERIFIED="
_DEEP_GEMM_DISTRIBUTION = "deep-gemm"
_DEEP_GEMM_NVRTC_ENV_VAR = "DG_JIT_USE_NVRTC"
_NVCC_PREPEND_FLAGS_ENV_VAR = "NVCC_PREPEND_FLAGS"
_CUDA_HOME_ENV_VAR = "CUDA_HOME"
_CUDA_COMPAT_DIRECTORY = "marin-deep-gemm-cuda"


@dataclass(frozen=True)
class _WheelProvenance:
    release_tag: str
    sm_targets: tuple[str, ...]
    source_commit: str
    version: str
    wheel_sha256: str
    wheel_url: str

    @classmethod
    def from_json(cls, value: str) -> "_WheelProvenance":
        payload = json.loads(value)
        return cls(
            release_tag=payload["release_tag"],
            sm_targets=tuple(payload["sm_targets"]),
            source_commit=payload["source_commit"],
            version=payload["version"],
            wheel_sha256=payload["wheel_sha256"],
            wheel_url=payload["wheel_url"],
        )

    def record(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def installed_wheel_url_matches(direct_url: dict, expected_wheel_url: str) -> bool:
    """Whether a PEP 610 record identifies ``expected_wheel_url``.

    The URL is the only artifact identity available at runtime: uv drops the ``#sha256=`` fragment of
    a remote direct reference and writes an empty ``archive_info``, so there is no installed digest
    to compare against. Do not add a digest check here.
    """
    installed_url = urlsplit(direct_url["url"])
    installed_url_without_fragment = urlunsplit(installed_url._replace(fragment=""))
    return unquote(installed_url_without_fragment) == unquote(expected_wheel_url)


def _link_directory_entries(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source_entry in sorted(source.iterdir()):
        destination_entry = destination / source_entry.name
        if destination_entry.is_symlink():
            if destination_entry.resolve() != source_entry.resolve():
                raise RuntimeError(f"Conflicting CUDA compatibility link: {destination_entry}")
            continue
        if destination_entry.exists():
            raise RuntimeError(f"CUDA compatibility path is not a symlink: {destination_entry}")
        destination_entry.symlink_to(source_entry, target_is_directory=source_entry.is_dir())


def deep_gemm_cuda_environment(nvidia_roots: tuple[Path, ...], temporary_root: Path) -> dict[str, str]:
    """Build the packaged CUDA toolkit view expected by DeepGEMM and FlashInfer JITs."""
    compiler_roots = tuple(
        candidate
        for nvidia_root in nvidia_roots
        for candidate in sorted(nvidia_root.glob("cu*"))
        if (candidate / "bin" / "nvcc").is_file()
    )
    cccl_roots = tuple(
        candidate
        for nvidia_root in nvidia_roots
        for candidate in (nvidia_root / "cuda_cccl" / "include",)
        if (candidate / "nv" / "target").is_file() and (candidate / "cuda" / "std" / "type_traits").is_file()
    )
    cublas_roots = tuple(
        candidate
        for nvidia_root in nvidia_roots
        for candidate in (nvidia_root / "cublas",)
        if (candidate / "include" / "cublasLt.h").is_file() and (candidate / "lib").is_dir()
    )
    cuda_runtime_roots = tuple(
        candidate
        for nvidia_root in nvidia_roots
        for candidate in (nvidia_root / "cuda_runtime",)
        if (candidate / "lib").is_dir()
    )
    if len(compiler_roots) != 1:
        raise RuntimeError(f"Expected one packaged CUDA compiler root, found {compiler_roots}")
    if len(cccl_roots) != 1:
        raise RuntimeError(f"Expected one packaged CUDA CCCL include root, found {cccl_roots}")
    if len(cublas_roots) != 1:
        raise RuntimeError(f"Expected one packaged cuBLAS root, found {cublas_roots}")
    if len(cuda_runtime_roots) != 1:
        raise RuntimeError(f"Expected one packaged CUDA runtime root, found {cuda_runtime_roots}")

    compiler_root = compiler_roots[0]
    cccl_root = cccl_roots[0]
    cublas_root = cublas_roots[0]
    cuda_runtime_root = cuda_runtime_roots[0]
    source_identity = "\0".join(
        str(path.resolve()) for path in (compiler_root, cccl_root, cublas_root, cuda_runtime_root)
    )
    identity = hashlib.sha256(source_identity.encode()).hexdigest()[:16]
    compatibility_home = temporary_root / _CUDA_COMPAT_DIRECTORY / identity
    compatibility_bin = compatibility_home / "bin"
    compatibility_include = compatibility_home / "include"
    compatibility_lib = compatibility_home / "lib64"

    nvvm_root = compiler_root / "nvvm"
    if not (nvvm_root / "bin" / "cicc").is_file():
        raise RuntimeError(f"Packaged CUDA compiler is missing NVVM cicc: {nvvm_root}")

    _link_directory_entries(compiler_root / "bin", compatibility_bin)
    _link_directory_entries(compiler_root / "include", compatibility_include)
    _link_directory_entries(cublas_root / "include", compatibility_include)
    _link_directory_entries(cuda_runtime_root / "lib", compatibility_lib)
    _link_directory_entries(cublas_root / "lib", compatibility_lib)

    compatibility_nvvm = compatibility_home / "nvvm"
    if compatibility_nvvm.is_symlink():
        if compatibility_nvvm.resolve() != nvvm_root.resolve():
            raise RuntimeError(f"Conflicting CUDA NVVM compatibility link: {compatibility_nvvm}")
    elif compatibility_nvvm.exists():
        raise RuntimeError(f"CUDA NVVM compatibility path is not a symlink: {compatibility_nvvm}")
    else:
        compatibility_nvvm.symlink_to(nvvm_root, target_is_directory=True)

    namespaced_cccl = compatibility_include / "cccl"
    if namespaced_cccl.exists() or namespaced_cccl.is_symlink():
        if not namespaced_cccl.is_symlink() or namespaced_cccl.resolve() != cccl_root.resolve():
            raise RuntimeError(f"Conflicting CUDA CCCL compatibility path: {namespaced_cccl}")
    else:
        namespaced_cccl.symlink_to(cccl_root, target_is_directory=True)

    return {
        _CUDA_HOME_ENV_VAR: str(compatibility_home),
        _DEEP_GEMM_NVRTC_ENV_VAR: "0",
        _NVCC_PREPEND_FLAGS_ENV_VAR: f"-I{compatibility_include} -I{cccl_root}",
    }


def _configure_deep_gemm_cuda_environment() -> None:
    try:
        importlib.metadata.distribution(_DEEP_GEMM_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return

    nvidia = importlib.import_module("nvidia")
    nvidia_roots = tuple(Path(path) for path in nvidia.__path__)
    environment = deep_gemm_cuda_environment(nvidia_roots, Path(tempfile.gettempdir()))
    os.environ.update(environment)
def main() -> None:
    expected = _WheelProvenance.from_json(sys.argv.pop(1))
    print(f"{_SELECTED_SENTINEL}{json.dumps(expected.record(), sort_keys=True)}", flush=True)
    _configure_deep_gemm_cuda_environment()

    # The installed version is not checked separately: promotion requires the release URL to carry the
    # declared version in its filename, so a distribution installed from that URL has that version.
    distribution = importlib.metadata.distribution("vllm")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("Installed vLLM does not record direct wheel provenance")
    direct_url = json.loads(direct_url_text)
    if not installed_wheel_url_matches(direct_url, expected.wheel_url):
        raise RuntimeError(f"Installed vLLM URL {direct_url['url']} does not match {expected.wheel_url}")

    torch = importlib.import_module("torch")
    major, minor = torch.cuda.get_device_capability()
    compute_capability = f"{major}.{minor}"
    if compute_capability not in expected.sm_targets:
        raise RuntimeError(
            f"GPU compute capability {compute_capability} is not supported by verified wheel "
            f"targets {expected.sm_targets}"
        )

    vllm_extension = importlib.import_module("vllm._C")
    extension_path = vllm_extension.__file__
    assert extension_path is not None
    resolved_extension_path = Path(extension_path).resolve()
    distribution_root = Path(distribution.locate_file("")).resolve()
    if not resolved_extension_path.is_relative_to(distribution_root):
        raise RuntimeError(f"vllm._C loaded outside the verified distribution: {resolved_extension_path}")

    provenance = {
        **expected.record(),
        "compute_capability": compute_capability,
        "extension_path": str(resolved_extension_path),
    }
    print(f"{_VERIFIED_SENTINEL}{json.dumps(provenance, sort_keys=True)}", flush=True)
    vllm_cli = importlib.import_module("vllm.entrypoints.cli.main")
    vllm_cli.main()


if __name__ == "__main__":
    main()
