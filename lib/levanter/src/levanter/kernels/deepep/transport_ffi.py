# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX FFI wrappers for DeepEP's intranode dispatch/combine kernels."""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import importlib.machinery
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

_DISPATCH_TARGET = "levanter_deepep_dispatch_intranode"
_COMBINE_TARGET = "levanter_deepep_combine_intranode"
_INIT_SYMBOL = "levanter_deepep_init_intranode_runtime"
_SHUTDOWN_SYMBOL = "levanter_deepep_shutdown_intranode_runtime"
_LAST_ERROR_SYMBOL = "levanter_deepep_last_error"
_PROBE_DISPATCH_SYMBOL = "levanter_deepep_probe_dispatch_kernel_attributes"
_RUN_HOST_DISPATCH_SYMBOL = "levanter_deepep_run_host_dispatch_round"
_DEEPEP_SRC_ENV = "DEEPEP_SRC_ROOT"
_DEEPEP_CUDA_ARCH_ENV = "DEEPEP_CUDA_ARCH"
_DISABLE_SM90_ENV = "DISABLE_SM90_FEATURES"
_BUILD_WITH_TORCH_EXTENSION_ENV = "DEEPEP_BUILD_WITH_TORCH_EXTENSION"
_LOAD_AS_PYTHON_MODULE_ENV = "DEEPEP_LOAD_AS_PYTHON_MODULE"
_EXTENDED_INTRNODE_DISPATCH_MACRO = "LEVANTER_DEEPEP_EXTENDED_INTRNODE_DISPATCH"
_PYEXT_MODULE_NAME_MACRO = "LEVANTER_DEEPEP_PYEXT_MODULE_NAME"
_BUILD_CACHE_SCHEMA_VERSION = "transport_ffi_raw_dlink_v4"
_LIBRARY_DLOPEN_MODE = getattr(os, "RTLD_NOW", 0) | getattr(ctypes, "RTLD_GLOBAL", 0)


@dataclass(frozen=True)
class IntranodeConfig:
    num_sms: int
    num_max_send_tokens: int
    num_max_recv_tokens: int


@dataclass(frozen=True)
class BuildArtifact:
    library_path: Path
    module_name: str | None


_DEFAULT_DISPATCH_CONFIGS = {
    2: IntranodeConfig(num_sms=20, num_max_send_tokens=24, num_max_recv_tokens=256),
    4: IntranodeConfig(num_sms=20, num_max_send_tokens=6, num_max_recv_tokens=256),
    8: IntranodeConfig(num_sms=20, num_max_send_tokens=6, num_max_recv_tokens=256),
}

_DEFAULT_COMBINE_CONFIGS = {
    2: IntranodeConfig(num_sms=20, num_max_send_tokens=10, num_max_recv_tokens=256),
    4: IntranodeConfig(num_sms=20, num_max_send_tokens=9, num_max_recv_tokens=256),
    8: IntranodeConfig(num_sms=20, num_max_send_tokens=4, num_max_recv_tokens=256),
}


def _jaxlib_include_dir() -> Path:
    return Path(jaxlib.__file__).resolve().parent / "include"


def _ffi_source() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "deepep_transport_ffi.cu"


def _python_extension_source() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "deepep_transport_pyext.cc"


def _cuda_sources(deepep_root: Path) -> tuple[Path, ...]:
    return (
        _ffi_source(),
        deepep_root / "csrc" / "kernels" / "runtime.cu",
        deepep_root / "csrc" / "kernels" / "layout.cu",
        deepep_root / "csrc" / "kernels" / "intranode.cu",
    )


def _deepep_source_root() -> Path:
    raw = os.environ.get(_DEEPEP_SRC_ENV)
    if not raw:
        raise RuntimeError(
            f"{_DEEPEP_SRC_ENV} must point at a DeepEP checkout before using the DeepEP JAX FFI transport kernels."
        )
    root = Path(raw).expanduser().resolve()
    required = (
        root / "csrc" / "kernels" / "layout.cu",
        root / "csrc" / "kernels" / "runtime.cu",
        root / "csrc" / "kernels" / "intranode.cu",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"{_DEEPEP_SRC_ENV}={root} does not look like a DeepEP source checkout")
    return root


def _cache_root() -> Path:
    return Path.home() / ".cache" / "marin" / "deepep_transport_ffi"


def _python_extension_suffix() -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if suffix:
        return suffix
    if not importlib.machinery.EXTENSION_SUFFIXES:
        raise RuntimeError("Could not determine the Python extension suffix for DeepEP transport FFI")
    return importlib.machinery.EXTENSION_SUFFIXES[0]


def _python_include_dirs() -> tuple[Path, ...]:
    include_dirs: list[Path] = []
    for key in ("include", "platinclude"):
        raw = sysconfig.get_paths().get(key)
        if not raw:
            continue
        path = Path(raw)
        if path not in include_dirs:
            include_dirs.append(path)
    if not include_dirs:
        raise RuntimeError("Could not determine Python include directories for DeepEP transport FFI")
    return tuple(include_dirs)


def _cuda_arch_flag() -> list[str]:
    arch = os.environ.get(_DEEPEP_CUDA_ARCH_ENV, "sm_90").strip()
    if not arch.startswith("sm_"):
        raise RuntimeError(f"{_DEEPEP_CUDA_ARCH_ENV} must look like sm_90 or sm_90a, got {arch!r}")
    compute = arch.replace("sm_", "compute_", 1)
    return [f"-gencode=arch={compute},code={arch}"]


def _sm90_compile_flags() -> list[str]:
    if int(os.environ.get(_DISABLE_SM90_ENV, "0")):
        return ["-DDISABLE_SM90_FEATURES"]
    return []


def _use_torch_extension_build() -> bool:
    return bool(int(os.environ.get(_BUILD_WITH_TORCH_EXTENSION_ENV, "0")))


def _load_as_python_module() -> bool:
    return bool(int(os.environ.get(_LOAD_AS_PYTHON_MODULE_ENV, "0")))


def _torch_cuda_arch_list() -> str:
    if int(os.environ.get(_DISABLE_SM90_ENV, "0")):
        return "8.0"
    arch = os.environ.get(_DEEPEP_CUDA_ARCH_ENV, "sm_90").strip()
    if arch == "sm_90":
        return "9.0"
    if arch == "sm_90a":
        return "9.0a"
    raise RuntimeError(f"Unsupported {_DEEPEP_CUDA_ARCH_ENV}={arch!r} for torch extension build")


def _preload_torch_shared_libraries() -> None:
    import torch

    lib_dir = Path(torch.__file__).resolve().parent / "lib"
    if not lib_dir.is_dir():
        raise RuntimeError(f"Could not find Torch shared libraries under {lib_dir}")
    torch_libraries = (
        "libc10.so",
        "libc10_cuda.so",
        "libtorch_cpu.so",
        "libtorch_cuda.so",
        "libtorch.so",
        "libtorch_python.so",
    )
    for library_name in torch_libraries:
        library_path = lib_dir / library_name
        if not library_path.is_file():
            continue
        ctypes.CDLL(str(library_path), mode=_LIBRARY_DLOPEN_MODE)


def _build_artifact() -> BuildArtifact:
    deepep_root = _deepep_source_root()
    key = hashlib.sha256()
    sources = _cuda_sources(deepep_root) + (
        deepep_root / "csrc" / "config.hpp",
        deepep_root / "csrc" / "kernels" / "api.cuh",
        deepep_root / "csrc" / "kernels" / "configs.cuh",
    )
    for path in sources:
        key.update(path.read_bytes())
    if _load_as_python_module():
        key.update(_python_extension_source().read_bytes())
    key.update(Path(__file__).read_bytes())
    key.update(str(_jaxlib_include_dir()).encode("utf-8"))
    key.update(str(deepep_root).encode("utf-8"))
    key.update(_BUILD_CACHE_SCHEMA_VERSION.encode("utf-8"))
    key.update(" ".join(_cuda_arch_flag()).encode("utf-8"))
    key.update(" ".join(_sm90_compile_flags()).encode("utf-8"))
    key.update(str(int(_has_extended_intranode_dispatch_signature(deepep_root))).encode("utf-8"))
    key.update(str(int(_use_torch_extension_build())).encode("utf-8"))
    key.update(str(int(_load_as_python_module())).encode("utf-8"))
    digest = key.hexdigest()[:16]
    out_dir = _cache_root() / digest
    out_dir.mkdir(parents=True, exist_ok=True)
    if _load_as_python_module():
        module_name = f"deepep_transport_ffi_{digest}"
        return BuildArtifact(
            library_path=out_dir / f"{module_name}{_python_extension_suffix()}",
            module_name=module_name,
        )
    return BuildArtifact(library_path=out_dir / "libdeepep_transport_ffi.so", module_name=None)


def _shared_library_path() -> Path:
    return _build_artifact().library_path


def _nvcc_common_flags(deepep_root: Path, compatibility_flags: list[str]) -> list[str]:
    return [
        "-std=c++17",
        "-Xcompiler",
        "-fPIC",
        "--expt-relaxed-constexpr",
        "-O3",
        "-DDISABLE_NVSHMEM",
        "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
        *compatibility_flags,
        *_sm90_compile_flags(),
        *_cuda_arch_flag(),
        "-I",
        str(_jaxlib_include_dir()),
        "-I",
        str(deepep_root),
        "-I",
        str(deepep_root / "csrc"),
    ]


def _build_object_files(
    *,
    build_dir: Path,
    deepep_root: Path,
    compatibility_flags: list[str],
) -> list[Path]:
    objects_dir = build_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)
    common_flags = _nvcc_common_flags(deepep_root, compatibility_flags)
    compile_flags = [
        "nvcc",
        *common_flags,
        "-rdc=true",
        "--ptxas-options=--register-usage-level=10",
        "-c",
    ]

    object_paths: list[Path] = []
    for source in _cuda_sources(deepep_root):
        object_path = objects_dir / f"{source.stem}.o"
        cmd = [
            *compile_flags,
            str(source),
            "-o",
            str(object_path),
        ]
        subprocess.run(cmd, check=True)
        object_paths.append(object_path)
    return object_paths


def _build_python_extension_shim_object(*, build_dir: Path, module_name: str) -> Path:
    object_path = build_dir / "deepep_transport_pyext.o"
    include_flags: list[str] = []
    for include_dir in _python_include_dirs():
        include_flags.extend(["-I", str(include_dir)])
    cmd = [
        "c++",
        "-std=c++17",
        "-O3",
        "-fPIC",
        f"-D{_PYEXT_MODULE_NAME_MACRO}={module_name}",
        *include_flags,
        "-c",
        str(_python_extension_source()),
        "-o",
        str(object_path),
    ]
    subprocess.run(cmd, check=True)
    return object_path


def _device_link_objects(
    *,
    build_dir: Path,
    deepep_root: Path,
    compatibility_flags: list[str],
    object_paths: list[Path],
) -> Path:
    dlink_object = build_dir / "deepep_transport_ffi.dlink.o"
    common_flags = _nvcc_common_flags(deepep_root, compatibility_flags)
    cmd = [
        "nvcc",
        *common_flags,
        "-dlink",
        *[str(path) for path in object_paths],
        "-o",
        str(dlink_object),
    ]
    subprocess.run(cmd, check=True)
    return dlink_object


def _link_shared_library(
    *,
    out_path: Path,
    object_paths: list[Path],
    dlink_object: Path,
    extra_object_paths: list[Path] | None = None,
) -> None:
    all_object_paths = [*object_paths, *(extra_object_paths or [])]
    cmd = [
        "nvcc",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "--cudart=shared",
        *_cuda_arch_flag(),
        *[str(path) for path in all_object_paths],
        str(dlink_object),
        "-lcuda",
        "-o",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def _build_raw_shared_library(artifact: BuildArtifact, deepep_root: Path, compatibility_flags: list[str]) -> None:
    if _use_torch_extension_build() and _load_as_python_module():
        raise RuntimeError(
            f"{_BUILD_WITH_TORCH_EXTENSION_ENV}=1 is not yet supported with {_LOAD_AS_PYTHON_MODULE_ENV}=1"
        )
    out_path = artifact.library_path
    build_dir = out_path.parent / "raw_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    object_paths = _build_object_files(
        build_dir=build_dir,
        deepep_root=deepep_root,
        compatibility_flags=compatibility_flags,
    )
    dlink_object = _device_link_objects(
        build_dir=build_dir,
        deepep_root=deepep_root,
        compatibility_flags=compatibility_flags,
        object_paths=object_paths,
    )
    extra_object_paths: list[Path] = []
    if artifact.module_name is not None:
        extra_object_paths.append(
            _build_python_extension_shim_object(build_dir=build_dir, module_name=artifact.module_name)
        )
    _link_shared_library(
        out_path=out_path,
        object_paths=object_paths,
        dlink_object=dlink_object,
        extra_object_paths=extra_object_paths,
    )


def _build_shared_library(artifact: BuildArtifact) -> None:
    deepep_root = _deepep_source_root()
    compatibility_flags = _compatibility_compile_flags(deepep_root)
    out_path = artifact.library_path
    if _use_torch_extension_build() and _load_as_python_module():
        raise RuntimeError(
            f"{_BUILD_WITH_TORCH_EXTENSION_ENV}=1 is not yet supported with {_LOAD_AS_PYTHON_MODULE_ENV}=1"
        )
    if _use_torch_extension_build():
        _build_with_torch_extension(out_path, deepep_root, compatibility_flags)
        return
    _build_raw_shared_library(artifact, deepep_root, compatibility_flags)


def _build_with_torch_extension(out_path: Path, deepep_root: Path, compatibility_flags: list[str]) -> None:
    from torch.utils import cpp_extension

    build_dir = out_path.parent
    name = f"deepep_transport_ffi_{build_dir.name}"
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = _torch_cuda_arch_list()
    try:
        cpp_extension.load(
            name=name,
            sources=[
                str(_ffi_source()),
                str(deepep_root / "csrc" / "kernels" / "runtime.cu"),
                str(deepep_root / "csrc" / "kernels" / "layout.cu"),
                str(deepep_root / "csrc" / "kernels" / "intranode.cu"),
            ],
            extra_cuda_cflags=[
                "-O3",
                "-DDISABLE_NVSHMEM",
                "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
                *compatibility_flags,
                "-rdc=true",
                "--ptxas-options=--register-usage-level=10",
                *_sm90_compile_flags(),
            ],
            extra_include_paths=[
                str(_jaxlib_include_dir()),
                str(deepep_root),
                str(deepep_root / "csrc"),
            ],
            build_directory=str(build_dir),
            verbose=True,
            with_cuda=True,
            is_python_module=False,
        )
    finally:
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list

    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    candidates = []
    for suffix in suffixes:
        candidates.extend(build_dir.glob(f"{name}*{suffix}"))
    if not candidates:
        raise RuntimeError(f"torch cpp_extension did not produce a shared library in {build_dir}")
    built_path = max(candidates, key=lambda path: path.stat().st_mtime)
    if built_path != out_path:
        shutil.copy2(built_path, out_path)


def _load_torch_extension_python_module(artifact: BuildArtifact):
    if artifact.module_name is None:
        raise RuntimeError("Torch extension Python-module load requires a module-named build artifact")

    cached_module = getattr(_load_torch_extension_python_module, "_module", None)
    cached_path = getattr(_load_torch_extension_python_module, "_path", None)
    if cached_module is not None and cached_path == artifact.library_path:
        return cached_module

    deepep_root = _deepep_source_root()
    compatibility_flags = _compatibility_compile_flags(deepep_root)
    build_dir = artifact.library_path.parent
    build_dir.mkdir(parents=True, exist_ok=True)
    setup_path = build_dir / "setup.py"
    setup_path.write_text(
        "\n".join(
            (
                "from setuptools import setup",
                "from torch.utils.cpp_extension import BuildExtension, CUDAExtension",
                "",
                f"MODULE_NAME = {artifact.module_name!r}",
                f"SOURCES = {([str(_python_extension_source()), str(_ffi_source()), *[str(path) for path in _cuda_sources(deepep_root)[1:]]])!r}",
                f"INCLUDE_DIRS = {[str(_jaxlib_include_dir()), str(deepep_root), str(deepep_root / 'csrc')]!r}",
                f"CXX_FLAGS = {['-O3', f'-D{_PYEXT_MODULE_NAME_MACRO}={artifact.module_name}']!r}",
                (
                    "NVCC_FLAGS = "
                    + repr(
                        [
                            "-O3",
                            "-DDISABLE_NVSHMEM",
                            "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
                            *compatibility_flags,
                            "-rdc=true",
                            "--ptxas-options=--register-usage-level=10",
                            *_sm90_compile_flags(),
                        ]
                    )
                ),
                "NVCC_DLINK_FLAGS = ['-dlink']",
                "",
                "setup(",
                "    name=MODULE_NAME,",
                "    ext_modules=[",
                "        CUDAExtension(",
                "            name=MODULE_NAME,",
                "            sources=SOURCES,",
                "            include_dirs=INCLUDE_DIRS,",
                "            extra_compile_args={",
                "                'cxx': CXX_FLAGS,",
                "                'nvcc': NVCC_FLAGS,",
                "                'nvcc_dlink': NVCC_DLINK_FLAGS,",
                "            },",
                "            extra_link_args=['-lcuda'],",
                "            dlink=True,",
                "        )",
                "    ],",
                "    cmdclass={'build_ext': BuildExtension},",
                ")",
            )
        )
        + "\n"
    )
    build_env = os.environ.copy()
    build_env["TORCH_CUDA_ARCH_LIST"] = _torch_cuda_arch_list()
    subprocess.run(
        [sys.executable, str(setup_path), "build_ext", "--inplace"],
        cwd=build_dir,
        env=build_env,
        check=True,
    )
    if not artifact.library_path.exists():
        candidates = sorted(build_dir.glob(f"{artifact.module_name}*{_python_extension_suffix()}"))
        if not candidates:
            raise RuntimeError(f"CUDAExtension build did not produce {artifact.library_path.name} in {build_dir}")
        if candidates[-1] != artifact.library_path:
            shutil.copy2(candidates[-1], artifact.library_path)

    _preload_torch_shared_libraries()
    module = _load_python_module(artifact)
    module_path = Path(module.__file__).resolve()
    _load_torch_extension_python_module._module = module
    _load_torch_extension_python_module._path = module_path
    return module


def _has_extended_intranode_dispatch_signature(deepep_root: Path) -> bool:
    api_header = deepep_root / "csrc" / "kernels" / "api.cuh"
    return "recv_x_sf_scale_for_nvfp4" in api_header.read_text()


def _compatibility_compile_flags(deepep_root: Path) -> list[str]:
    if _has_extended_intranode_dispatch_signature(deepep_root):
        return [f"-D{_EXTENDED_INTRNODE_DISPATCH_MACRO}=1"]
    return []


def _load_library() -> ctypes.CDLL:
    artifact = _build_artifact()
    if _use_torch_extension_build() and _load_as_python_module():
        module = _load_torch_extension_python_module(artifact)
        return ctypes.CDLL(str(Path(module.__file__).resolve()), mode=_LIBRARY_DLOPEN_MODE)
    if not artifact.library_path.exists():
        _build_shared_library(artifact)
    if artifact.module_name is not None:
        module = _load_python_module(artifact)
        return ctypes.CDLL(str(Path(module.__file__).resolve()), mode=_LIBRARY_DLOPEN_MODE)
    return ctypes.CDLL(str(artifact.library_path), mode=_LIBRARY_DLOPEN_MODE)


def _load_python_module(artifact: BuildArtifact):
    if artifact.module_name is None:
        raise RuntimeError("Build artifact does not describe a Python extension module")
    cached_module = getattr(_load_python_module, "_module", None)
    cached_path = getattr(_load_python_module, "_path", None)
    if cached_module is not None and cached_path == artifact.library_path:
        return cached_module

    spec = importlib.util.spec_from_file_location(artifact.module_name, artifact.library_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create a Python extension spec for {artifact.library_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[artifact.module_name] = module
    spec.loader.exec_module(module)
    _load_python_module._module = module
    _load_python_module._path = artifact.library_path
    return module


def _register_targets() -> None:
    if getattr(_register_targets, "_done", False):
        return
    library = _load_library()
    for target in (_DISPATCH_TARGET, _COMBINE_TARGET):
        handler = getattr(library, target)
        handler.restype = ctypes.c_void_p
        jax.ffi.register_ffi_target(
            target,
            jax.ffi.pycapsule(handler),
            platform="CUDA",
            api_version=1,
        )
        jax.ffi.register_ffi_target_as_batch_partitionable(target)
    _register_targets._done = True


def _library_function(name: str):
    library = _load_library()
    return getattr(library, name)


def _default_dispatch_config(num_ranks: int) -> IntranodeConfig:
    if num_ranks not in _DEFAULT_DISPATCH_CONFIGS:
        raise ValueError(f"Unsupported DeepEP intranode dispatch rank count: {num_ranks}")
    return _DEFAULT_DISPATCH_CONFIGS[num_ranks]


def _default_combine_config(num_ranks: int) -> IntranodeConfig:
    if num_ranks not in _DEFAULT_COMBINE_CONFIGS:
        raise ValueError(f"Unsupported DeepEP intranode combine rank count: {num_ranks}")
    return _DEFAULT_COMBINE_CONFIGS[num_ranks]


def shutdown_intranode_runtime() -> None:
    if getattr(ensure_intranode_runtime, "_signature", None) is None:
        return
    shutdown = _library_function(_SHUTDOWN_SYMBOL)
    shutdown.argtypes = []
    shutdown.restype = None
    shutdown()
    setattr(ensure_intranode_runtime, "_signature", None)


atexit.register(shutdown_intranode_runtime)


def ensure_intranode_runtime(
    *,
    num_ranks: int,
    hidden_bytes: int,
    dispatch_config: IntranodeConfig | None = None,
    combine_config: IntranodeConfig | None = None,
) -> None:
    _register_targets()
    dispatch = dispatch_config or _default_dispatch_config(num_ranks)
    combine = combine_config or _default_combine_config(num_ranks)
    signature = (num_ranks, hidden_bytes, dispatch, combine)
    if getattr(ensure_intranode_runtime, "_signature", None) == signature:
        return

    local_gpu_devices = [device for device in jax.local_devices() if device.platform == "gpu"]
    if len(local_gpu_devices) != num_ranks:
        raise RuntimeError(
            f"DeepEP JAX intranode runtime currently expects the expert group to span all visible local GPUs; "
            f"got num_ranks={num_ranks} and visible_gpus={len(local_gpu_devices)}."
        )

    init = _library_function(_INIT_SYMBOL)
    init.argtypes = [
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    init.restype = ctypes.c_int

    status = init(
        num_ranks,
        hidden_bytes,
        dispatch.num_sms,
        dispatch.num_max_send_tokens,
        dispatch.num_max_recv_tokens,
        combine.num_sms,
        combine.num_max_send_tokens,
        combine.num_max_recv_tokens,
    )
    if status != 0:
        last_error = _library_function(_LAST_ERROR_SYMBOL)
        last_error.argtypes = []
        last_error.restype = ctypes.c_char_p
        message = last_error()
        text = message.decode("utf-8") if message else "unknown error"
        raise RuntimeError(f"Failed to initialize DeepEP intranode JAX runtime: {text}")
    ensure_intranode_runtime._signature = signature


def probe_dispatch_kernel_attributes() -> dict[str, int | str]:
    """Call the DeepEP dispatch host wrapper outside XLA execution."""
    probe = _library_function(_PROBE_DISPATCH_SYMBOL)
    probe.argtypes = []
    probe.restype = ctypes.c_int

    status = probe()
    result: dict[str, int | str] = {"probe_status_code": int(status)}
    last_error = _library_function(_LAST_ERROR_SYMBOL)
    last_error.argtypes = []
    last_error.restype = ctypes.c_char_p
    message = last_error()
    text = message.decode("utf-8") if message else ""
    result["last_error"] = text
    if status != 0:
        raise RuntimeError(f"Failed to probe DeepEP dispatch kernel attributes: {text}")
    return result


def run_host_dispatch_round(
    *,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
) -> dict[str, int | str]:
    """Run a same-process all-ranks dispatch round outside XLA execution."""
    run = _library_function(_RUN_HOST_DISPATCH_SYMBOL)
    run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    run.restype = ctypes.c_int

    status = run(num_tokens, hidden, num_experts, num_topk)
    result: dict[str, int | str] = {
        "host_dispatch_status_code": int(status),
        "num_tokens": int(num_tokens),
        "hidden": int(hidden),
        "num_experts": int(num_experts),
        "num_topk": int(num_topk),
    }
    last_error = _library_function(_LAST_ERROR_SYMBOL)
    last_error.argtypes = []
    last_error.restype = ctypes.c_char_p
    message = last_error()
    text = message.decode("utf-8") if message else ""
    result["last_error"] = text
    if status != 0:
        raise RuntimeError(f"Failed to run DeepEP host dispatch round: {text}")
    return result


def deepep_dispatch_intranode(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    num_tokens_per_rank: jax.Array,
    num_tokens_per_expert: jax.Array,
    is_token_in_rank: jax.Array,
    *,
    num_experts: int,
    dispatch_config: IntranodeConfig | None = None,
    combine_config: IntranodeConfig | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    _register_targets()
    num_ranks = int(num_tokens_per_rank.shape[0])
    hidden_bytes = x.shape[1] * max(jnp.dtype(x.dtype).itemsize, 2)
    ensure_intranode_runtime(
        num_ranks=num_ranks,
        hidden_bytes=hidden_bytes,
        dispatch_config=dispatch_config,
        combine_config=combine_config,
    )

    x_bf16 = jnp.asarray(x, dtype=jnp.bfloat16)
    topk_idx_i64 = jnp.asarray(topk_idx, dtype=jnp.int64)
    topk_weights_f32 = jnp.asarray(topk_weights, dtype=jnp.float32)
    num_tokens_per_rank_i32 = jnp.asarray(num_tokens_per_rank, dtype=jnp.int32)
    num_tokens_per_expert_i32 = jnp.asarray(num_tokens_per_expert, dtype=jnp.int32)
    local_experts = num_experts // num_ranks
    max_recv_tokens = x_bf16.shape[0] * num_ranks
    num_channels = (dispatch_config or _default_dispatch_config(num_ranks)).num_sms // 2
    topk = topk_idx_i64.shape[1]
    result_shape_dtypes = (
        jax.ShapeDtypeStruct((max_recv_tokens, x_bf16.shape[1]), x_bf16.dtype),
        jax.ShapeDtypeStruct((max_recv_tokens, topk), jnp.int64),
        jax.ShapeDtypeStruct((max_recv_tokens, topk), jnp.float32),
        jax.ShapeDtypeStruct((max_recv_tokens,), jnp.int32),
        jax.ShapeDtypeStruct((num_ranks, num_ranks), jnp.int32),
        jax.ShapeDtypeStruct((num_ranks, num_channels), jnp.int32),
        jax.ShapeDtypeStruct((x_bf16.shape[0], num_ranks), jnp.int32),
        jax.ShapeDtypeStruct((local_experts,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    return jax.ffi.ffi_call(
        _DISPATCH_TARGET,
        result_shape_dtypes,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        x_bf16,
        topk_idx_i64,
        topk_weights_f32,
        num_tokens_per_rank_i32,
        num_tokens_per_expert_i32,
        is_token_in_rank,
        num_experts=np.int32(num_experts),
    )


def deepep_combine_intranode(
    recv_x: jax.Array,
    recv_topk_weights: jax.Array,
    recv_src_idx: jax.Array,
    rank_prefix_matrix: jax.Array,
    channel_prefix_matrix: jax.Array,
    send_head: jax.Array,
    num_recv_tokens: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    _register_targets()
    recv_x_bf16 = jnp.asarray(recv_x, dtype=jnp.bfloat16)
    recv_topk_weights_f32 = jnp.asarray(recv_topk_weights, dtype=jnp.float32)
    recv_src_idx_i32 = jnp.asarray(recv_src_idx, dtype=jnp.int32)
    rank_prefix_matrix_i32 = jnp.asarray(rank_prefix_matrix, dtype=jnp.int32)
    channel_prefix_matrix_i32 = jnp.asarray(channel_prefix_matrix, dtype=jnp.int32)
    send_head_i32 = jnp.asarray(send_head, dtype=jnp.int32)
    num_recv_tokens_i32 = jnp.asarray(num_recv_tokens, dtype=jnp.int32)
    topk = recv_topk_weights_f32.shape[1]
    result_shape_dtypes = (
        jax.ShapeDtypeStruct((send_head_i32.shape[0], recv_x_bf16.shape[1]), recv_x_bf16.dtype),
        jax.ShapeDtypeStruct((send_head_i32.shape[0], topk), jnp.float32),
        jax.ShapeDtypeStruct(send_head_i32.shape, send_head_i32.dtype),
    )
    combined_x, combined_topk_weights, _ = jax.ffi.ffi_call(
        _COMBINE_TARGET,
        result_shape_dtypes,
        has_side_effect=True,
        vmap_method="broadcast_all",
        input_output_aliases={5: 2},
    )(
        recv_x_bf16,
        recv_topk_weights_f32,
        recv_src_idx_i32,
        rank_prefix_matrix_i32,
        channel_prefix_matrix_i32,
        send_head_i32,
        num_recv_tokens_i32,
    )
    return combined_x, combined_topk_weights
