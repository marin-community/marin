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
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from levanter.kernels.deepep.availability import (
    BUILD_WITH_TORCH_EXTENSION_ENV,
    DISABLE_SM90_ENV,
    LOAD_AS_PYTHON_MODULE_ENV,
    TRANSPORT_REQUIRED_FILES,
    deepep_cache_root,
    deepep_cuda_arch,
    deepep_cuda_arch_flag,
    deepep_layout_source,
    deepep_source_root,
    deepep_torch_cuda_arch_list,
    env_flag,
)

_DISPATCH_TARGET = "levanter_deepep_dispatch_intranode"
_DISPATCH_CACHED_TARGET = "levanter_deepep_dispatch_intranode_cached"
_COMBINE_TARGET = "levanter_deepep_combine_intranode"
_INIT_SYMBOL = "levanter_deepep_init_intranode_runtime"
_SHUTDOWN_SYMBOL = "levanter_deepep_shutdown_intranode_runtime"
_LAST_ERROR_SYMBOL = "levanter_deepep_last_error"
_PROBE_DISPATCH_SYMBOL = "levanter_deepep_probe_dispatch_kernel_attributes"
_RUN_HOST_DISPATCH_SYMBOL = "levanter_deepep_run_host_dispatch_round"
_EXTENDED_INTRNODE_DISPATCH_MACRO = "LEVANTER_DEEPEP_EXTENDED_INTRNODE_DISPATCH"
_PYEXT_MODULE_NAME_MACRO = "LEVANTER_DEEPEP_PYEXT_MODULE_NAME"
_DISPATCH_THREADS_ENV = "DEEPEP_DISPATCH_NUM_THREADS"
_BUILD_CACHE_SCHEMA_VERSION = "transport_ffi_raw_dlink_v18"
_LIBRARY_DLOPEN_MODE = getattr(os, "RTLD_NOW", 0) | getattr(ctypes, "RTLD_GLOBAL", 0)
_SM100_TMA_DISPATCH_THREADS = 512
_UPSTREAM_DISPATCH_THREADS = 768


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


def _launch_compat_header() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "deepep_launch_compat.cuh"


def _intranode_source(deepep_root: Path) -> Path:
    return deepep_root / "csrc" / "kernels" / "intranode.cu"


def _cuda_sources(deepep_root: Path) -> tuple[Path, ...]:
    return (
        _ffi_source(),
        deepep_root / "csrc" / "kernels" / "runtime.cu",
        deepep_layout_source(deepep_root),
        _intranode_source(deepep_root),
    )


def _deepep_source_root() -> Path:
    return deepep_source_root(
        required_files=TRANSPORT_REQUIRED_FILES,
        purpose="the DeepEP JAX FFI transport kernels",
        requires_layout_source=True,
    )


def _cache_root() -> Path:
    return deepep_cache_root("deepep_transport_ffi")


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
    return deepep_cuda_arch_flag()


def _sm90_compile_flags(*, include_launch_compat: bool = True) -> list[str]:
    flags: list[str] = []
    if include_launch_compat:
        flags.extend(["-include", str(_launch_compat_header())])
    if env_flag(DISABLE_SM90_ENV):
        flags.append("-DDISABLE_SM90_FEATURES")
    return flags


def _use_torch_extension_build() -> bool:
    return env_flag(BUILD_WITH_TORCH_EXTENSION_ENV)


def _load_as_python_module() -> bool:
    return env_flag(LOAD_AS_PYTHON_MODULE_ENV)


def _torch_cuda_arch_list() -> str:
    return deepep_torch_cuda_arch_list()


def _dispatch_thread_override() -> int | None:
    raw = os.environ.get(_DISPATCH_THREADS_ENV)
    if raw is not None:
        try:
            threads = int(raw)
        except ValueError as exc:
            raise RuntimeError(f"{_DISPATCH_THREADS_ENV} must be an integer, got {raw!r}") from exc
        if threads < 256 or threads % 32 != 0:
            raise RuntimeError(f"{_DISPATCH_THREADS_ENV} must be a multiple of 32 and at least 256, got {threads}")
        return threads
    if deepep_cuda_arch() == "sm_100" and not env_flag(DISABLE_SM90_ENV):
        return _SM100_TMA_DISPATCH_THREADS
    return None


def _intranode_source_bytes(deepep_root: Path) -> bytes:
    source = _intranode_source(deepep_root)
    text = source.read_text()
    dispatch_threads = _dispatch_thread_override()
    if dispatch_threads is not None and dispatch_threads != _UPSTREAM_DISPATCH_THREADS:
        dispatch_start = text.find("\nvoid dispatch(")
        combine_start = text.find("\nvoid combine(", dispatch_start)
        needle = f"    constexpr int kNumThreads = {_UPSTREAM_DISPATCH_THREADS};"
        threads_start = text.find(needle, dispatch_start)
        if dispatch_start < 0 or threads_start < 0 or (combine_start >= 0 and threads_start > combine_start):
            raise RuntimeError("Could not patch DeepEP intranode dispatch thread count for this source tree")
        text = (
            text[:threads_start]
            + f"    constexpr int kNumThreads = {dispatch_threads};"
            + text[threads_start + len(needle) :]
        )

    replacements = (
        (
            "    auto kernel = dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>; \\\n"
            "    SET_SHARED_MEMORY_FOR_TMA(kernel); \\",
            "    SET_SHARED_MEMORY_FOR_TMA((dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>)); \\\n"
            "    auto kernel = dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>; \\",
        ),
        (
            "    auto kernel = combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp>; \\\n"
            "    SET_SHARED_MEMORY_FOR_TMA(kernel); \\",
            "    SET_SHARED_MEMORY_FOR_TMA((combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp>)); \\\n"
            "    auto kernel = combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp>; \\",
        ),
    )
    for old, new in replacements:
        if old not in text and not env_flag(DISABLE_SM90_ENV):
            raise RuntimeError("Could not patch DeepEP intranode TMA launch pattern for this source tree")
        text = text.replace(old, new, 1)

    return text.encode("utf-8")


def _prepare_intranode_source(build_dir: Path, deepep_root: Path) -> Path:
    dispatch_threads = _dispatch_thread_override()
    if dispatch_threads is None or dispatch_threads == _UPSTREAM_DISPATCH_THREADS:
        return _intranode_source(deepep_root)
    patched_source = build_dir / "generated" / "intranode.cu"
    patched_source.parent.mkdir(parents=True, exist_ok=True)
    patched_source.write_bytes(_intranode_source_bytes(deepep_root))
    return patched_source


def _prepared_cuda_sources(build_dir: Path, deepep_root: Path) -> tuple[Path, ...]:
    return (
        _ffi_source(),
        deepep_root / "csrc" / "kernels" / "runtime.cu",
        deepep_layout_source(deepep_root),
        _prepare_intranode_source(build_dir, deepep_root),
    )


def _preload_torch_shared_libraries() -> None:
    import torch  # noqa: PLC0415  # optional dep: torch

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
        key.update(str(path).encode("utf-8"))
        if path == _intranode_source(deepep_root):
            key.update(_intranode_source_bytes(deepep_root))
        else:
            key.update(path.read_bytes())
    if _load_as_python_module():
        key.update(_python_extension_source().read_bytes())
    key.update(_launch_compat_header().read_bytes())
    key.update(Path(__file__).read_bytes())
    key.update(str(_jaxlib_include_dir()).encode("utf-8"))
    key.update(str(deepep_root).encode("utf-8"))
    key.update(_BUILD_CACHE_SCHEMA_VERSION.encode("utf-8"))
    key.update(" ".join(_cuda_arch_flag()).encode("utf-8"))
    key.update(" ".join(_sm90_compile_flags()).encode("utf-8"))
    key.update(str(_dispatch_thread_override()).encode("utf-8"))
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


def _nvcc_common_flags(
    deepep_root: Path,
    compatibility_flags: list[str],
    *,
    include_launch_compat: bool = True,
) -> list[str]:
    return [
        "-std=c++17",
        "-Xcompiler",
        "-fPIC",
        "--expt-relaxed-constexpr",
        "-O3",
        "-DDISABLE_NVSHMEM",
        "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
        *compatibility_flags,
        *_sm90_compile_flags(include_launch_compat=include_launch_compat),
        *_cuda_arch_flag(),
        "-I",
        str(_jaxlib_include_dir()),
        "-I",
        str(deepep_root),
        "-I",
        str(deepep_root / "csrc"),
        "-I",
        str(deepep_root / "csrc" / "kernels"),
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
    for source in _prepared_cuda_sources(objects_dir, deepep_root):
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
    common_flags = _nvcc_common_flags(deepep_root, compatibility_flags, include_launch_compat=False)
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
            f"{BUILD_WITH_TORCH_EXTENSION_ENV}=1 is not yet supported with {LOAD_AS_PYTHON_MODULE_ENV}=1"
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
            f"{BUILD_WITH_TORCH_EXTENSION_ENV}=1 is not yet supported with {LOAD_AS_PYTHON_MODULE_ENV}=1"
        )
    if _use_torch_extension_build():
        _build_with_torch_extension(out_path, deepep_root, compatibility_flags)
        return
    _build_raw_shared_library(artifact, deepep_root, compatibility_flags)


def _build_with_torch_extension(out_path: Path, deepep_root: Path, compatibility_flags: list[str]) -> None:
    from torch.utils import cpp_extension  # noqa: PLC0415  # optional dep: torch

    build_dir = out_path.parent
    name = f"deepep_transport_ffi_{build_dir.name}"
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = _torch_cuda_arch_list()
    try:
        cpp_extension.load(
            name=name,
            sources=[
                *[str(source) for source in _prepared_cuda_sources(build_dir, deepep_root)],
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
                f"SOURCES = {([str(_python_extension_source()), str(_ffi_source()), *[str(path) for path in _prepared_cuda_sources(build_dir, deepep_root)[1:]]])!r}",
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
    for target in (_DISPATCH_TARGET, _DISPATCH_CACHED_TARGET, _COMBINE_TARGET):
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


def _read_last_error(default: str = "") -> str:
    """Return the message recorded by the most recent failed DeepEP host call."""
    last_error = _library_function(_LAST_ERROR_SYMBOL)
    last_error.argtypes = []
    last_error.restype = ctypes.c_char_p
    message = last_error()
    return message.decode("utf-8") if message else default


def _materialize_cotangent(
    cotangent: jax.Array | jax.custom_derivatives.SymbolicZero,
    *,
    dtype: jnp.dtype,
    shape: tuple[int, ...] | None = None,
    reference: jax.Array | None = None,
) -> jax.Array:
    if reference is None and shape is None:
        raise ValueError("Either reference or shape must be provided when materializing a cotangent.")
    if isinstance(cotangent, jax.custom_derivatives.SymbolicZero):
        if reference is not None:
            return jnp.zeros_like(reference, dtype=dtype)
        return jnp.zeros(shape, dtype=dtype)
    return jnp.asarray(cotangent, dtype=dtype)


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
        text = _read_last_error("unknown error")
        raise RuntimeError(f"Failed to initialize DeepEP intranode JAX runtime: {text}")
    ensure_intranode_runtime._signature = signature


def probe_dispatch_kernel_attributes() -> dict[str, int | str]:
    """Call the DeepEP dispatch host wrapper outside XLA execution."""
    probe = _library_function(_PROBE_DISPATCH_SYMBOL)
    probe.argtypes = []
    probe.restype = ctypes.c_int

    status = probe()
    result: dict[str, int | str] = {"probe_status_code": int(status)}
    text = _read_last_error()
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
    text = _read_last_error()
    result["last_error"] = text
    if status != 0:
        raise RuntimeError(f"Failed to run DeepEP host dispatch round: {text}")
    return result


def _resolve_runtime(
    *,
    x: jax.Array,
    num_ranks: int,
    dispatch_config: IntranodeConfig | None,
    combine_config: IntranodeConfig | None,
) -> IntranodeConfig:
    _register_targets()
    hidden_bytes = x.shape[1] * max(jnp.dtype(x.dtype).itemsize, 2)
    ensure_intranode_runtime(
        num_ranks=num_ranks,
        hidden_bytes=hidden_bytes,
        dispatch_config=dispatch_config,
        combine_config=combine_config,
    )
    return dispatch_config or _default_dispatch_config(num_ranks)


def _dispatch_intranode_impl(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    num_tokens_per_rank: jax.Array,
    num_tokens_per_expert: jax.Array,
    is_token_in_rank: jax.Array,
    *,
    num_experts: int,
    dispatch_config: IntranodeConfig | None,
    combine_config: IntranodeConfig | None,
    max_recv_tokens: int | None,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]:
    num_ranks = int(num_tokens_per_rank.shape[0])
    resolved_dispatch_config = _resolve_runtime(
        x=x,
        num_ranks=num_ranks,
        dispatch_config=dispatch_config,
        combine_config=combine_config,
    )

    x_bf16 = jnp.asarray(x, dtype=jnp.bfloat16)
    topk_idx_i32 = jnp.asarray(topk_idx, dtype=jnp.int32)
    topk_weights_f32 = jnp.asarray(topk_weights, dtype=jnp.float32)
    num_tokens_per_rank_i32 = jnp.asarray(num_tokens_per_rank, dtype=jnp.int32)
    num_tokens_per_expert_i32 = jnp.asarray(num_tokens_per_expert, dtype=jnp.int32)
    local_experts = num_experts // num_ranks
    if max_recv_tokens is None:
        max_recv_tokens = x_bf16.shape[0] * num_ranks
    elif max_recv_tokens <= 0:
        raise ValueError(f"max_recv_tokens must be positive, got {max_recv_tokens}")
    num_channels = resolved_dispatch_config.num_sms // 2
    topk = topk_idx_i32.shape[1]
    result_shape_dtypes = (
        jax.ShapeDtypeStruct((max_recv_tokens, x_bf16.shape[1]), x_bf16.dtype),
        jax.ShapeDtypeStruct((max_recv_tokens, topk), jnp.int32),
        jax.ShapeDtypeStruct((max_recv_tokens, topk), jnp.float32),
        jax.ShapeDtypeStruct((max_recv_tokens,), jnp.int32),
        jax.ShapeDtypeStruct((num_ranks, num_ranks), jnp.int32),
        jax.ShapeDtypeStruct((num_ranks, num_channels), jnp.int32),
        jax.ShapeDtypeStruct((num_ranks, num_channels), jnp.int32),
        jax.ShapeDtypeStruct((x_bf16.shape[0], num_ranks), jnp.int32),
        jax.ShapeDtypeStruct((local_experts,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((x_bf16.shape[0], topk * 2), jnp.int32),
        jax.ShapeDtypeStruct((max_recv_tokens, topk * 2), jnp.int32),
    )
    results = jax.ffi.ffi_call(
        _DISPATCH_TARGET,
        result_shape_dtypes,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        x_bf16,
        topk_idx_i32,
        topk_weights_f32,
        num_tokens_per_rank_i32,
        num_tokens_per_expert_i32,
        is_token_in_rank,
        num_experts=np.int32(num_experts),
    )
    return (
        results[0],
        results[1],
        results[2],
        results[3],
        results[4],
        results[5],
        results[6],
        results[7],
        results[8],
        results[9],
    )


def _dispatch_intranode_cached_impl(
    x: jax.Array,
    is_token_in_rank: jax.Array,
    rank_prefix_matrix: jax.Array,
    channel_prefix_matrix: jax.Array,
    num_recv_tokens: jax.Array,
    *,
    dispatch_config: IntranodeConfig | None,
    combine_config: IntranodeConfig | None,
    max_recv_tokens: int,
) -> jax.Array:
    if max_recv_tokens <= 0:
        raise ValueError(f"max_recv_tokens must be positive, got {max_recv_tokens}")
    num_ranks = int(rank_prefix_matrix.shape[0])
    resolved_dispatch_config = _resolve_runtime(
        x=x,
        num_ranks=num_ranks,
        dispatch_config=dispatch_config,
        combine_config=combine_config,
    )

    x_bf16 = jnp.asarray(x, dtype=jnp.bfloat16)
    is_token_in_rank_bool = jnp.asarray(is_token_in_rank, dtype=jnp.bool_)
    rank_prefix_matrix_i32 = jnp.asarray(rank_prefix_matrix, dtype=jnp.int32)
    channel_prefix_matrix_i32 = jnp.asarray(channel_prefix_matrix, dtype=jnp.int32)
    num_recv_tokens_i32 = jnp.asarray(num_recv_tokens, dtype=jnp.int32)
    num_channels = resolved_dispatch_config.num_sms // 2
    result_shape_dtypes = (
        jax.ShapeDtypeStruct((max_recv_tokens, x_bf16.shape[1]), x_bf16.dtype),
        jax.ShapeDtypeStruct((max_recv_tokens,), jnp.int32),
        jax.ShapeDtypeStruct((num_ranks, num_channels), jnp.int32),
        jax.ShapeDtypeStruct((x_bf16.shape[0], num_ranks), jnp.int32),
    )
    recv_x, _, _, _ = jax.ffi.ffi_call(
        _DISPATCH_CACHED_TARGET,
        result_shape_dtypes,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        x_bf16,
        is_token_in_rank_bool,
        rank_prefix_matrix_i32,
        channel_prefix_matrix_i32,
        num_recv_tokens_i32,
    )
    recv_token_limit = jnp.squeeze(num_recv_tokens_i32, axis=0)
    recv_valid = jnp.arange(max_recv_tokens, dtype=jnp.int32) < recv_token_limit
    return jnp.where(recv_valid[:, None], recv_x, 0)


def _combine_intranode_impl(
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


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9))
def _dispatch_intranode_with_vjp(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    num_tokens_per_rank: jax.Array,
    num_tokens_per_expert: jax.Array,
    is_token_in_rank: jax.Array,
    num_experts: int,
    dispatch_config: IntranodeConfig | None,
    combine_config: IntranodeConfig | None,
    max_recv_tokens: int | None,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]:
    return _dispatch_intranode_impl(
        x,
        topk_idx,
        topk_weights,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        num_experts=num_experts,
        dispatch_config=dispatch_config,
        combine_config=combine_config,
        max_recv_tokens=max_recv_tokens,
    )


def _dispatch_intranode_with_vjp_fwd(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    num_tokens_per_rank: jax.Array,
    num_tokens_per_expert: jax.Array,
    is_token_in_rank: jax.Array,
    num_experts: int,
    dispatch_config: IntranodeConfig | None,
    combine_config: IntranodeConfig | None,
    max_recv_tokens: int | None,
):
    outputs = _dispatch_intranode_impl(
        x,
        topk_idx,
        topk_weights,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        num_experts=num_experts,
        dispatch_config=dispatch_config,
        combine_config=combine_config,
        max_recv_tokens=max_recv_tokens,
    )
    (
        recv_x,
        _,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        _,
        recv_channel_prefix_matrix,
        send_head,
        _,
        num_recv_tokens,
    ) = outputs
    residuals = (
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
    )
    return outputs, residuals


def _dispatch_intranode_with_vjp_bwd(
    num_experts: int,
    dispatch_config: IntranodeConfig | None,
    combine_config: IntranodeConfig | None,
    max_recv_tokens: int | None,
    residuals,
    cotangents,
):
    (
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
    ) = residuals
    grad_recv_x = _materialize_cotangent(cotangents[0], dtype=recv_x.dtype, reference=recv_x)
    grad_recv_topk_weights = _materialize_cotangent(
        cotangents[2],
        dtype=recv_topk_weights.dtype,
        reference=recv_topk_weights,
    )
    grad_x, grad_topk_weights = _combine_intranode_impl(
        grad_recv_x,
        grad_recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
    )
    return grad_x, None, grad_topk_weights, None, None, None


_dispatch_intranode_with_vjp.defvjp(
    _dispatch_intranode_with_vjp_fwd,
    _dispatch_intranode_with_vjp_bwd,
)


@jax.custom_vjp
def _combine_intranode_with_vjp(
    recv_x: jax.Array,
    recv_topk_weights: jax.Array,
    recv_src_idx: jax.Array,
    rank_prefix_matrix: jax.Array,
    dispatch_channel_prefix_matrix: jax.Array,
    recv_channel_prefix_matrix: jax.Array,
    send_head: jax.Array,
    num_recv_tokens: jax.Array,
    is_token_in_rank: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return _combine_intranode_impl(
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
    )


def _combine_intranode_with_vjp_fwd(
    recv_x: jax.Array,
    recv_topk_weights: jax.Array,
    recv_src_idx: jax.Array,
    rank_prefix_matrix: jax.Array,
    dispatch_channel_prefix_matrix: jax.Array,
    recv_channel_prefix_matrix: jax.Array,
    send_head: jax.Array,
    num_recv_tokens: jax.Array,
    is_token_in_rank: jax.Array,
):
    outputs = _combine_intranode_impl(
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
    )
    residuals = (
        recv_x,
        recv_topk_weights,
        rank_prefix_matrix,
        dispatch_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        is_token_in_rank,
    )
    return outputs, residuals


def _combine_intranode_with_vjp_bwd(residuals, cotangents):
    (
        recv_x,
        recv_topk_weights,
        rank_prefix_matrix,
        dispatch_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        is_token_in_rank,
    ) = residuals
    grad_combined_x = _materialize_cotangent(
        cotangents[0],
        dtype=recv_x.dtype,
        shape=(send_head.shape[0], recv_x.shape[1]),
    )
    grad_recv_x = _dispatch_intranode_cached_impl(
        grad_combined_x,
        is_token_in_rank,
        rank_prefix_matrix,
        dispatch_channel_prefix_matrix,
        num_recv_tokens,
        dispatch_config=None,
        combine_config=None,
        max_recv_tokens=recv_x.shape[0],
    )
    return (
        grad_recv_x,
        jnp.zeros_like(recv_topk_weights),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_combine_intranode_with_vjp.defvjp(
    _combine_intranode_with_vjp_fwd,
    _combine_intranode_with_vjp_bwd,
)


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
    max_recv_tokens: int | None = None,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]:
    return _dispatch_intranode_with_vjp(
        x,
        topk_idx,
        topk_weights,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        num_experts,
        dispatch_config,
        combine_config,
        max_recv_tokens,
    )


def deepep_combine_intranode(
    recv_x: jax.Array,
    recv_topk_weights: jax.Array,
    recv_src_idx: jax.Array,
    rank_prefix_matrix: jax.Array,
    dispatch_channel_prefix_matrix: jax.Array,
    recv_channel_prefix_matrix: jax.Array,
    send_head: jax.Array,
    num_recv_tokens: jax.Array,
    is_token_in_rank: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return _combine_intranode_with_vjp(
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        dispatch_channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        is_token_in_rank,
    )
