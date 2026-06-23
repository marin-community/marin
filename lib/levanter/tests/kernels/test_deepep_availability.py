# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import os
import sys
from typing import Callable

import jax
import jax.numpy as jnp
import pytest

import levanter.kernels.deepep.availability as deepep_availability
from levanter.kernels.deepep import transport_ffi
from levanter.kernels.deepep.availability import (
    DEEPEP_CUDA_ARCH_ENV,
    DEEPEP_KNOWN_GOOD_COMMIT,
    DEEPEP_RDMA_INCLUDE_DIR_ENV,
    DEEPEP_SRC_ENV,
    INTERNODE_TRANSPORT_REQUIRED_FILES,
    TRANSPORT_REQUIRED_FILES,
    deepep_layout_source,
    deepep_nvcc_path,
    deepep_nvshmem_config,
    deepep_nvshmem_status,
    deepep_preflight_status,
    missing_deepep_rdma_headers,
    deepep_source_root,
    deepep_cuda_include_dirs,
    deepep_cuda_library_dirs,
)


def _write(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("// test source\n")
    return path


def _write_transport_sources(root: Path) -> None:
    for relative in TRANSPORT_REQUIRED_FILES:
        _write(root, relative)


def _write_internode_transport_sources(root: Path) -> None:
    for relative in INTERNODE_TRANSPORT_REQUIRED_FILES:
        _write(root, relative)


def _write_fake_nvshmem_package(root: Path, *, host_name: str = "libnvshmem_host.so.3") -> Path:
    package_dir = root / "nvidia" / "nvshmem"
    package_dir.mkdir(parents=True)
    (root / "nvidia" / "__init__.py").write_text("")
    (package_dir / "__init__.py").write_text("")
    (package_dir / "include").mkdir()
    (package_dir / "include" / "nvshmem.h").write_text("// nvshmem test header\n")
    (package_dir / "lib" / host_name).parent.mkdir(parents=True)
    (package_dir / "lib" / host_name).write_text("")
    (package_dir / "lib" / "libnvshmem_device.a").write_text("")
    return package_dir


def _clear_fake_nvshmem_modules() -> None:
    sys.modules.pop("nvidia.nvshmem", None)
    sys.modules.pop("nvidia", None)


def _clear_fake_nvcc_modules() -> None:
    sys.modules.pop("nvidia.cuda_nvcc", None)
    sys.modules.pop("nvidia", None)


def test_deepep_layout_source_accepts_legacy_layout_path(tmp_path: Path) -> None:
    root = tmp_path / "DeepEP"
    layout_source = _write(root, "csrc/kernels/legacy/layout.cu")

    assert deepep_layout_source(root) == layout_source


def test_deepep_nvcc_path_falls_back_to_python_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_dir = tmp_path / "nvidia" / "cuda_nvcc"
    nvcc = package_dir / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    (tmp_path / "nvidia" / "__init__.py").write_text("")
    (package_dir / "__init__.py").write_text("")
    nvcc.write_text("#!/bin/sh\n")
    _clear_fake_nvcc_modules()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(deepep_availability.shutil, "which", lambda name: None)

    assert deepep_nvcc_path() == str(nvcc)


def test_deepep_nvcc_path_accepts_nvidia_cuda_nvcc_wheel_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nvcc = tmp_path / "nvidia" / "cu13" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\n")

    class FakeDistribution:
        def locate_file(self, relative: str) -> Path:
            return tmp_path / relative

    monkeypatch.setattr(deepep_availability.shutil, "which", lambda name: None)
    monkeypatch.setattr(deepep_availability, "_SYSTEM_NVCC_CANDIDATES", ())
    monkeypatch.setattr(deepep_availability.importlib.metadata, "distribution", lambda name: FakeDistribution())

    assert deepep_nvcc_path() == str(nvcc)


def test_deepep_nvcc_path_falls_back_to_cuda_image_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nvcc = tmp_path / "usr" / "local" / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\n")
    monkeypatch.setattr(deepep_availability.shutil, "which", lambda name: None)
    monkeypatch.setattr(deepep_availability, "_SYSTEM_NVCC_CANDIDATES", (nvcc,))

    assert deepep_nvcc_path() == str(nvcc)


def test_deepep_cuda_include_dirs_find_nvcc_and_cccl_wheel_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nvcc = tmp_path / "nvidia" / "cu13" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\n")
    nvcc_include = tmp_path / "nvidia" / "cu13" / "include"
    nvcc_include.mkdir()
    cccl_include = tmp_path / "nvidia" / "cuda_cccl" / "include"
    (cccl_include / "nv").mkdir(parents=True)
    (cccl_include / "nv" / "target").write_text("// target\n")

    class FakeDistribution:
        def locate_file(self, relative: str) -> Path:
            return tmp_path / relative

    def distribution(name: str) -> FakeDistribution:
        if name in {"nvidia-cuda-cccl-cu13", "nvidia-cuda-cccl-cu12"}:
            return FakeDistribution()
        raise deepep_availability.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(deepep_availability, "deepep_nvcc_path", lambda: str(nvcc))
    monkeypatch.setattr(deepep_availability.importlib.metadata, "distribution", distribution)

    assert deepep_cuda_include_dirs() == (nvcc_include, cccl_include)


def test_deepep_cuda_library_dirs_find_runtime_wheel_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_lib = tmp_path / "nvidia" / "cuda_runtime" / "lib"
    runtime_lib.mkdir(parents=True)
    (runtime_lib / "libcudart.so.12").write_text("")

    class FakeDistribution:
        def locate_file(self, relative: str) -> Path:
            return tmp_path / relative

    def distribution(name: str) -> FakeDistribution:
        if name in {"nvidia-cuda-runtime-cu13", "nvidia-cuda-runtime-cu12"}:
            return FakeDistribution()
        raise deepep_availability.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(deepep_availability.importlib.metadata, "distribution", distribution)

    assert deepep_cuda_library_dirs() == (runtime_lib,)


def test_internode_dispatch_clean_buffer_size_hint_catches_d2560_topk2_regression() -> None:
    nvl_bytes, rdma_bytes = transport_ffi.internode_dispatch_clean_buffer_size_hint(
        hidden=2560,
        topk=2,
        num_rdma_ranks=2,
    )

    assert nvl_bytes > 64 * 1024 * 1024
    assert nvl_bytes <= 256 * 1024 * 1024
    assert rdma_bytes <= 64 * 1024 * 1024


def test_internode_dispatch_clean_buffer_size_hint_rejects_non_int4_hidden() -> None:
    with pytest.raises(ValueError, match="hidden bf16 bytes divisible by int4"):
        transport_ffi.internode_dispatch_clean_buffer_size_hint(
            hidden=2559,
            topk=2,
            num_rdma_ranks=2,
        )


def test_deepep_source_root_accepts_legacy_layout_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "DeepEP"
    _write_transport_sources(root)
    _write(root, "csrc/kernels/legacy/layout.cu")
    monkeypatch.setenv(DEEPEP_SRC_ENV, str(root))

    assert (
        deepep_source_root(
            required_files=TRANSPORT_REQUIRED_FILES,
            purpose="test",
            requires_layout_source=True,
        )
        == root.resolve()
    )


def test_deepep_preflight_reports_missing_layout_source_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "DeepEP"
    _write_transport_sources(root)
    monkeypatch.setenv(DEEPEP_SRC_ENV, str(root))
    monkeypatch.setenv(DEEPEP_CUDA_ARCH_ENV, "sm_100")

    status = deepep_preflight_status(required_files=TRANSPORT_REQUIRED_FILES)

    assert any("layout source" in error and "csrc/kernels/legacy/layout.cu" in error for error in status.errors)


def test_deepep_preflight_warns_for_unverified_source_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "DeepEP"
    _write_transport_sources(root)
    _write(root, "csrc/kernels/legacy/layout.cu")
    monkeypatch.setenv(DEEPEP_SRC_ENV, str(root))
    monkeypatch.setenv(DEEPEP_CUDA_ARCH_ENV, "sm_100")

    status = deepep_preflight_status(required_files=TRANSPORT_REQUIRED_FILES)

    assert status.source_revision is None
    assert any(DEEPEP_KNOWN_GOOD_COMMIT in warning for warning in status.warnings)


def test_deepep_nvshmem_status_detects_python_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_dir = _write_fake_nvshmem_package(tmp_path)
    _clear_fake_nvshmem_modules()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("NVSHMEM_DIR", raising=False)

    nvshmem_dir, host_lib, errors = deepep_nvshmem_status()

    assert nvshmem_dir == package_dir
    assert host_lib == "libnvshmem_host.so.3"
    assert errors == ()


def test_deepep_nvshmem_config_returns_build_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_dir = _write_fake_nvshmem_package(tmp_path)
    _clear_fake_nvshmem_modules()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("NVSHMEM_DIR", raising=False)

    config = deepep_nvshmem_config()

    assert config.root == package_dir
    assert config.host_library_name == "libnvshmem_host.so.3"
    assert config.device_library_name == "libnvshmem_device.a"
    assert config.include_dirs == (package_dir / "include",)
    assert config.library_dirs == (package_dir / "lib",)


def test_deepep_nvshmem_status_prefers_cuda13_python_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cu12_package = _write_fake_nvshmem_package(tmp_path / "site-cu12", host_name="libnvshmem_host.so.12")
    cu13_package = _write_fake_nvshmem_package(tmp_path / "site-cu13", host_name="libnvshmem_host.so.13")
    monkeypatch.delenv("NVSHMEM_DIR", raising=False)
    monkeypatch.setattr(
        deepep_availability,
        "_nvshmem_python_package_roots",
        lambda: (cu13_package, cu12_package),
    )

    nvshmem_dir, host_lib, errors = deepep_nvshmem_status()

    assert nvshmem_dir == cu13_package.resolve()
    assert host_lib == "libnvshmem_host.so.13"
    assert errors == ()


def test_deepep_nvshmem_config_prefers_cuda13_nested_libraries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_dir = tmp_path / "nvidia" / "nvshmem"
    (package_dir / "include").mkdir(parents=True)
    (package_dir / "include" / "nvshmem.h").write_text("// nvshmem test header\n")
    (package_dir / "toolkit" / "r12.9" / "main_nvshmem" / "lib").mkdir(parents=True)
    (package_dir / "toolkit" / "r13.0" / "main_nvshmem" / "lib").mkdir(parents=True)
    (package_dir / "toolkit" / "r12.9" / "main_nvshmem" / "lib" / "libnvshmem_host.so.3").write_text("")
    (package_dir / "toolkit" / "r12.9" / "main_nvshmem" / "lib" / "libnvshmem_device.a").write_text("")
    (package_dir / "toolkit" / "r13.0" / "main_nvshmem" / "lib" / "libnvshmem_host.so.3").write_text("")
    (package_dir / "toolkit" / "r13.0" / "main_nvshmem" / "lib" / "libnvshmem_device.a").write_text("")
    monkeypatch.delenv("NVSHMEM_DIR", raising=False)
    monkeypatch.setattr(deepep_availability, "_nvshmem_python_package_roots", lambda: (package_dir,))

    config = deepep_nvshmem_config()

    assert "r13.0" in str(config.host_library_path)
    assert "r13.0" in str(config.device_library_path)


def test_deepep_internode_preflight_requires_nvshmem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "DeepEP"
    _write_internode_transport_sources(root)
    _write(root, "csrc/kernels/legacy/layout.cu")
    monkeypatch.setenv(DEEPEP_SRC_ENV, str(root))
    monkeypatch.setenv(DEEPEP_CUDA_ARCH_ENV, "sm_90")
    monkeypatch.delenv("NVSHMEM_DIR", raising=False)
    monkeypatch.setattr(deepep_availability, "_nvshmem_python_package_roots", lambda: ())

    status = deepep_preflight_status(
        required_files=INTERNODE_TRANSPORT_REQUIRED_FILES,
        requires_nvshmem=True,
    )

    assert any("NVSHMEM_DIR is unset" in error for error in status.errors)


def test_deepep_internode_preflight_requires_rdma_headers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "DeepEP"
    _write_internode_transport_sources(root)
    _write(root, "csrc/kernels/legacy/layout.cu")
    _write_fake_nvshmem_package(tmp_path)
    _clear_fake_nvshmem_modules()
    empty_include = tmp_path / "empty-include"
    empty_include.mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(DEEPEP_SRC_ENV, str(root))
    monkeypatch.setenv(DEEPEP_CUDA_ARCH_ENV, "sm_90")
    monkeypatch.setenv(DEEPEP_RDMA_INCLUDE_DIR_ENV, str(empty_include))
    monkeypatch.setattr(
        "levanter.kernels.deepep.availability.deepep_rdma_include_dirs",
        lambda: (empty_include,),
    )
    monkeypatch.delenv("NVSHMEM_DIR", raising=False)

    status = deepep_preflight_status(
        required_files=INTERNODE_TRANSPORT_REQUIRED_FILES,
        requires_nvshmem=True,
        requires_rdma=True,
    )

    assert status.missing_rdma_headers == ("infiniband/mlx5dv.h",)
    assert any("RDMA development headers" in error for error in status.errors)


def test_missing_deepep_rdma_headers_accepts_custom_include_dir(tmp_path: Path) -> None:
    include_dir = tmp_path / "include"
    (include_dir / "infiniband").mkdir(parents=True)
    (include_dir / "infiniband" / "mlx5dv.h").write_text("// rdma test header\n")

    assert missing_deepep_rdma_headers((include_dir,)) == ()


def test_deepep_intranode_configs_accept_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPEP_DISPATCH_NUM_SMS", "40")
    monkeypatch.setenv("DEEPEP_DISPATCH_MAX_SEND_TOKENS", "8")
    monkeypatch.setenv("DEEPEP_DISPATCH_MAX_RECV_TOKENS", "384")
    monkeypatch.setenv("DEEPEP_COMBINE_NUM_SMS", "60")
    monkeypatch.setenv("DEEPEP_COMBINE_MAX_SEND_TOKENS", "10")
    monkeypatch.setenv("DEEPEP_COMBINE_MAX_RECV_TOKENS", "512")

    assert transport_ffi._default_dispatch_config(8) == transport_ffi.IntranodeConfig(
        num_sms=40,
        num_max_send_tokens=8,
        num_max_recv_tokens=384,
    )
    assert transport_ffi._default_combine_config(8) == transport_ffi.IntranodeConfig(
        num_sms=60,
        num_max_send_tokens=10,
        num_max_recv_tokens=512,
    )


def test_deepep_intranode_config_rejects_invalid_num_sms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPEP_DISPATCH_NUM_SMS", "41")

    with pytest.raises(RuntimeError, match="DEEPEP_DISPATCH_NUM_SMS"):
        transport_ffi._default_dispatch_config(8)


def test_prepare_intranode_source_compiles_generated_launch_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "DeepEP"
    intranode_source = root / "csrc" / "kernels" / "intranode.cu"
    intranode_source.parent.mkdir(parents=True, exist_ok=True)
    intranode_source.write_text(
        "\nvoid dispatch() {\n"
        "    constexpr int kNumThreads = 768;\n"
        "    auto kernel = dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>; \\\n"
        "    SET_SHARED_MEMORY_FOR_TMA(kernel); \\\n"
        "}\n"
        "\nvoid combine() {\n"
        "    auto kernel = combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp>; \\\n"
        "    SET_SHARED_MEMORY_FOR_TMA(kernel); \\\n"
        "}\n"
    )
    monkeypatch.setattr(transport_ffi, "_add_assignment_dispatch_source", lambda text, *, dispatch_threads: text)

    prepared = transport_ffi._prepare_intranode_source(tmp_path / "build", root)

    assert prepared == tmp_path / "build" / "generated" / "intranode.cu"
    prepared_text = prepared.read_text()
    assert "SET_SHARED_MEMORY_FOR_TMA((dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>));" in prepared_text
    assert "SET_SHARED_MEMORY_FOR_TMA((combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp>));" in prepared_text
    assert "SET_SHARED_MEMORY_FOR_TMA(kernel);" not in prepared_text


def test_transport_internode_sources_include_upstream_internode_kernels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "DeepEP"
    _write_internode_transport_sources(root)
    layout_source = _write(root, "csrc/kernels/legacy/layout.cu")
    patched_intranode = tmp_path / "generated" / "intranode.cu"
    monkeypatch.setattr(transport_ffi, "_prepare_intranode_source", lambda _build_dir, _root: patched_intranode)

    sources = transport_ffi._prepared_cuda_sources(
        tmp_path / "build", root, transport_ffi.TransportBuildMode.INTERNODE
    )

    assert sources == (
        transport_ffi._ffi_source(),
        root / "csrc" / "kernels" / "runtime.cu",
        layout_source,
        patched_intranode,
        root / "csrc" / "kernels" / "internode.cu",
        root / "csrc" / "kernels" / "internode_ll.cu",
        root / "csrc" / "kernels" / "pcie.cu",
    )


def test_transport_internode_flags_enable_nvshmem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_dir = _write_fake_nvshmem_package(tmp_path)
    _clear_fake_nvshmem_modules()
    root = tmp_path / "DeepEP"
    cuda_include = tmp_path / "cuda" / "include"
    cuda_include.mkdir(parents=True)
    _write_transport_sources(root)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("NVSHMEM_DIR", raising=False)
    monkeypatch.setenv(DEEPEP_CUDA_ARCH_ENV, "sm_90")
    monkeypatch.setattr(transport_ffi, "deepep_cuda_include_dirs", lambda: (cuda_include,))

    intranode_flags = transport_ffi._nvcc_common_flags(root, [], build_mode=transport_ffi.TransportBuildMode.INTRANODE)
    internode_flags = transport_ffi._nvcc_common_flags(root, [], build_mode=transport_ffi.TransportBuildMode.INTERNODE)
    internode_link_flags = transport_ffi._nvshmem_link_flags(transport_ffi.TransportBuildMode.INTERNODE)
    internode_device_link_flags = transport_ffi._nvshmem_device_link_flags(transport_ffi.TransportBuildMode.INTERNODE)

    assert "-DDISABLE_NVSHMEM" in intranode_flags
    assert "-DDISABLE_NVSHMEM" not in internode_flags
    assert str(cuda_include) in intranode_flags
    assert str(package_dir / "include") in internode_flags
    assert str(cuda_include) in internode_flags
    assert str(package_dir / "lib") in internode_link_flags
    assert "-l:libnvshmem_host.so.3" in internode_link_flags
    assert str(package_dir / "lib" / "libnvshmem_device.a") in internode_device_link_flags


def test_transport_link_includes_cuda_runtime_library_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cuda_lib = tmp_path / "cuda" / "lib"
    cuda_lib.mkdir(parents=True)
    (cuda_lib / "libcudart.so.12").write_text("")
    out_path = tmp_path / "libdeepep_transport_ffi.so"
    object_path = tmp_path / "object.o"
    dlink_object = tmp_path / "dlink.o"
    commands: list[list[str]] = []
    monkeypatch.setattr(transport_ffi, "_require_nvcc", lambda: "/cuda/bin/nvcc")
    monkeypatch.setattr(transport_ffi, "deepep_cuda_library_dirs", lambda: (cuda_lib,))
    monkeypatch.setattr(transport_ffi.subprocess, "run", lambda cmd, check: commands.append(cmd))

    transport_ffi._link_shared_library(
        out_path=out_path,
        object_paths=[object_path],
        dlink_object=dlink_object,
        build_mode=transport_ffi.TransportBuildMode.INTRANODE,
    )

    assert commands
    assert "-L" in commands[0]
    assert str(cuda_lib) in commands[0]
    assert "--cudart=none" in commands[0]
    assert "-l:libcudart.so.12" in commands[0]
    assert "-rpath" in commands[0]


def test_transport_internode_final_link_includes_nvshmem_device_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    out_path = tmp_path / "libdeepep_transport_ffi.so"
    object_path = tmp_path / "ffi.o"
    dlink_path = tmp_path / "ffi.dlink.o"
    device_archive = tmp_path / "libnvshmem_device.a"

    monkeypatch.setattr(transport_ffi, "_cuda_arch_flag", lambda: ["--gpu-architecture=sm_90"])
    monkeypatch.setattr(transport_ffi, "_require_nvcc", lambda: "/cuda/bin/nvcc")
    monkeypatch.setattr(
        transport_ffi,
        "_nvshmem_device_link_flags",
        lambda build_mode: [str(device_archive)] if build_mode is transport_ffi.TransportBuildMode.INTERNODE else [],
    )
    monkeypatch.setattr(transport_ffi, "_nvshmem_link_flags", lambda build_mode: ["-lnvshmem_host"])
    monkeypatch.setattr(transport_ffi.subprocess, "run", lambda cmd, check: commands.append(cmd))

    transport_ffi._link_shared_library(
        out_path=out_path,
        object_paths=[object_path],
        dlink_object=dlink_path,
        build_mode=transport_ffi.TransportBuildMode.INTERNODE,
    )

    assert commands == [
        [
            "/cuda/bin/nvcc",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "--cudart=none",
            "--gpu-architecture=sm_90",
            str(object_path),
            str(dlink_path),
            str(device_archive),
            "-lcudart",
            "-lcuda",
            "-lnvshmem_host",
            "-o",
            str(out_path),
        ]
    ]


def test_transport_local_device_id_uses_internode_library(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, transport_ffi.TransportBuildMode]] = []

    def fake_library_function(name: str, build_mode: transport_ffi.TransportBuildMode):
        calls.append((name, build_mode))

        def get_device_id(device_id) -> int:
            device_id._obj.value = 3
            return 0

        return get_device_id

    monkeypatch.setattr(transport_ffi, "_library_function", fake_library_function)

    assert transport_ffi.local_device_id() == 3
    assert calls == [
        ("levanter_deepep_get_local_device_id", transport_ffi.TransportBuildMode.INTERNODE),
    ]


def test_transport_local_nvshmem_unique_id_reads_two_step_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"nvshmem-id"

    def fake_library_function(name: str, build_mode: transport_ffi.TransportBuildMode) -> Callable:
        assert build_mode is transport_ffi.TransportBuildMode.INTERNODE

        def get_size(size_ptr) -> int:
            size_ptr._obj.value = len(payload)
            return 0

        def get_unique_id(buffer, capacity: int, written_ptr) -> int:
            assert capacity == len(payload)
            for idx, value in enumerate(payload):
                buffer[idx] = value
            written_ptr._obj.value = len(payload)
            return 0

        if name == "levanter_deepep_get_local_nvshmem_unique_id_size":
            return get_size
        if name == "levanter_deepep_get_local_nvshmem_unique_id":
            return get_unique_id
        raise AssertionError(f"unexpected symbol {name}")

    monkeypatch.setattr(transport_ffi, "_library_function", fake_library_function)

    assert transport_ffi.local_nvshmem_unique_id() == payload


def test_internode_bootstrap_metadata_round_trips_bytes() -> None:
    metadata = transport_ffi.InternodeProcessBootstrapMetadata(
        process_index=2,
        local_device_ids=(0, 1),
        nvshmem_unique_id=b"nvshmem-root",
        local_ipc_handles=(b"ipc0", b"ipc1"),
    )

    assert transport_ffi._metadata_from_json(transport_ffi._metadata_to_json(metadata)) == metadata


def test_local_internode_bootstrap_metadata_uses_root_nvshmem_id(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDevice:
        platform = "gpu"

        def __init__(self, device_id: int) -> None:
            self.id = device_id

    monkeypatch.setattr(transport_ffi.jax, "local_devices", lambda: [FakeDevice(4), FakeDevice(5)])
    monkeypatch.setattr(transport_ffi.jax, "process_index", lambda: 0)
    monkeypatch.setattr(transport_ffi, "local_nvshmem_unique_id", lambda: b"root-id")

    assert transport_ffi.local_internode_bootstrap_metadata() == transport_ffi.InternodeProcessBootstrapMetadata(
        process_index=0,
        local_device_ids=(4, 5),
        nvshmem_unique_id=b"root-id",
    )


def test_local_internode_bootstrap_metadata_skips_nonroot_nvshmem_id(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDevice:
        platform = "gpu"
        id = 7

    def fail_unique_id() -> bytes:
        raise AssertionError("non-root process should not request an NVSHMEM unique id")

    monkeypatch.setattr(transport_ffi.jax, "local_devices", lambda: [FakeDevice()])
    monkeypatch.setattr(transport_ffi.jax, "process_index", lambda: 1)
    monkeypatch.setattr(transport_ffi, "local_nvshmem_unique_id", fail_unique_id)

    assert transport_ffi.local_internode_bootstrap_metadata() == transport_ffi.InternodeProcessBootstrapMetadata(
        process_index=1,
        local_device_ids=(7,),
        nvshmem_unique_id=None,
    )


def test_current_internode_process_topology_detects_process_per_gpu_ep16() -> None:
    topology = transport_ffi.current_internode_process_topology(
        ranks_per_node=8,
        process_index=9,
        process_count=16,
        visible_local_gpus=1,
    )

    assert topology == transport_ffi.InternodeProcessTopology(
        process_index=9,
        process_count=16,
        process_model="process_per_gpu",
        node_rank=1,
        node_count=2,
        ranks_per_node=8,
        visible_local_gpus=1,
        local_rank=1,
    )


def test_current_internode_process_topology_detects_process_per_node() -> None:
    topology = transport_ffi.current_internode_process_topology(
        ranks_per_node=8,
        process_index=1,
        process_count=2,
        visible_local_gpus=8,
    )

    assert topology == transport_ffi.InternodeProcessTopology(
        process_index=1,
        process_count=2,
        process_model="process_per_node",
        node_rank=1,
        node_count=2,
        ranks_per_node=8,
        visible_local_gpus=8,
        local_rank=None,
    )


def test_preflight_internode_process_topology_accepts_process_per_gpu_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(transport_ffi.jax, "process_index", lambda: 9)
    monkeypatch.setattr(transport_ffi.jax, "process_count", lambda: 16)
    monkeypatch.setattr(transport_ffi, "_visible_local_gpu_count", lambda: 1)

    topology = transport_ffi.preflight_internode_process_topology(ranks_per_node=8)

    assert topology.process_model == "process_per_gpu"
    assert topology.node_rank == 1
    assert topology.local_rank == 1


def test_local_internode_bootstrap_metadata_carries_explicit_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDevice:
        platform = "gpu"
        id = 0

    calls: list[str] = []

    def fake_local_ipc_handle(*, topology: transport_ffi.InternodeProcessTopology, num_nvl_bytes: int) -> bytes:
        assert topology.local_rank == 1
        assert num_nvl_bytes == 4096
        calls.append("ipc")
        return b"ipc-rank-9"

    monkeypatch.setattr(transport_ffi.jax, "local_devices", lambda: [FakeDevice()])
    monkeypatch.setattr(transport_ffi.jax, "process_index", lambda: 9)
    monkeypatch.setattr(transport_ffi.jax, "process_count", lambda: 16)
    monkeypatch.setattr(transport_ffi, "local_nvshmem_unique_id", lambda: b"should-not-be-used")
    monkeypatch.setattr(transport_ffi, "local_internode_ipc_handle", fake_local_ipc_handle)
    monkeypatch.setenv("DEEPEP_RANKS_PER_NODE", "8")

    assert transport_ffi.local_internode_bootstrap_metadata(
        num_nvl_bytes=4096
    ) == transport_ffi.InternodeProcessBootstrapMetadata(
        process_index=9,
        local_device_ids=(0,),
        nvshmem_unique_id=None,
        local_ipc_handles=(b"ipc-rank-9",),
        node_rank=1,
        local_rank=1,
        ranks_per_node=8,
        process_model="process_per_gpu",
    )
    assert calls == ["ipc"]


def test_local_internode_bootstrap_metadata_process_per_gpu_node0_publishes_nvshmem_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDevice:
        platform = "gpu"
        id = 0

    monkeypatch.setattr(transport_ffi.jax, "local_devices", lambda: [FakeDevice()])
    monkeypatch.setattr(transport_ffi.jax, "process_index", lambda: 6)
    monkeypatch.setattr(transport_ffi.jax, "process_count", lambda: 16)
    monkeypatch.setattr(transport_ffi, "local_nvshmem_unique_id", lambda: b"root-local-rank-6")
    monkeypatch.setattr(transport_ffi, "local_internode_ipc_handle", lambda **_: b"ipc-rank-6")
    monkeypatch.setenv("DEEPEP_RANKS_PER_NODE", "8")

    metadata = transport_ffi.local_internode_bootstrap_metadata(num_nvl_bytes=4096)

    assert metadata.nvshmem_unique_id == b"root-local-rank-6"
    assert metadata.local_ipc_handles == (b"ipc-rank-6",)


def test_exchange_internode_bootstrap_metadata_uses_jax_distributed_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.values: dict[str, str] = {}
            self.barriers: list[str] = []

        def key_value_set(self, key: str, value: str) -> None:
            self.values[key] = value

        def wait_at_barrier(self, key: str, *, timeout_in_ms: int) -> None:
            assert timeout_in_ms == 123000
            self.barriers.append(key)

        def blocking_key_value_get(self, key: str, *, timeout_in_ms: int) -> str:
            assert timeout_in_ms == 123000
            return self.values[key]

    local = transport_ffi.InternodeProcessBootstrapMetadata(
        process_index=0,
        local_device_ids=(0,),
        nvshmem_unique_id=b"root",
        local_ipc_handles=(b"ipc0",),
    )
    remote = transport_ffi.InternodeProcessBootstrapMetadata(
        process_index=1,
        local_device_ids=(0,),
        nvshmem_unique_id=None,
        local_ipc_handles=(b"ipc1",),
    )
    client = FakeClient()
    monkeypatch.setattr(transport_ffi.jax, "process_count", lambda: 2)
    monkeypatch.setattr(transport_ffi, "_internode_exchange_counter", 0)
    monkeypatch.setattr(transport_ffi.jax_distributed.global_state, "client", client)
    client.values["levanter_deepep_internode_bootstrap_0_1"] = transport_ffi._metadata_to_json(remote)

    assert transport_ffi.exchange_internode_bootstrap_metadata(local, timeout=123.0) == (local, remote)
    assert client.barriers == ["levanter_deepep_internode_bootstrap_0_barrier"]


def test_root_nvshmem_unique_id_requires_exactly_one_root() -> None:
    metadata = (
        transport_ffi.InternodeProcessBootstrapMetadata(
            process_index=0,
            local_device_ids=(0,),
            nvshmem_unique_id=b"root",
        ),
        transport_ffi.InternodeProcessBootstrapMetadata(
            process_index=1,
            local_device_ids=(0,),
            nvshmem_unique_id=None,
        ),
    )

    assert transport_ffi._root_nvshmem_unique_id(metadata) == b"root"

    with pytest.raises(RuntimeError, match="exactly one root"):
        transport_ffi._root_nvshmem_unique_id((metadata[1],))


def test_root_nvshmem_unique_id_process_per_gpu_uses_node0_matching_local_rank() -> None:
    metadata = tuple(
        transport_ffi.InternodeProcessBootstrapMetadata(
            process_index=process_index,
            local_device_ids=(0,),
            nvshmem_unique_id=f"root-{process_index}".encode() if process_index < 8 else None,
            local_ipc_handles=(f"ipc-{process_index}".encode(),),
            node_rank=process_index // 8,
            local_rank=process_index % 8,
            ranks_per_node=8,
            process_model="process_per_gpu",
        )
        for process_index in range(16)
    )
    topology = transport_ffi.InternodeProcessTopology(
        process_index=13,
        process_count=16,
        process_model="process_per_gpu",
        node_rank=1,
        node_count=2,
        ranks_per_node=8,
        visible_local_gpus=1,
        local_rank=5,
    )

    assert transport_ffi._root_nvshmem_unique_id(metadata, topology=topology) == b"root-5"


def test_ensure_internode_runtime_calls_internode_init(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int, int, int, bytes, int]] = []

    def fake_library_function(name: str, build_mode: transport_ffi.TransportBuildMode):
        assert build_mode is transport_ffi.TransportBuildMode.INTERNODE

        def init(
            process_rank, process_count, num_local_ranks, num_nvl_bytes, root_id, root_id_size, num_rdma_bytes
        ) -> int:
            calls.append(
                (
                    int(process_rank),
                    int(process_count),
                    int(num_local_ranks),
                    int(num_nvl_bytes),
                    bytes(root_id[:root_id_size]),
                    int(num_rdma_bytes),
                )
            )
            return 0

        if name == "levanter_deepep_init_internode_runtime":
            return init
        raise AssertionError(f"unexpected symbol {name}")

    metadata = (
        transport_ffi.InternodeProcessBootstrapMetadata(
            process_index=0,
            local_device_ids=(0,),
            nvshmem_unique_id=b"root-id",
        ),
        transport_ffi.InternodeProcessBootstrapMetadata(
            process_index=1,
            local_device_ids=(0,),
            nvshmem_unique_id=None,
        ),
    )
    monkeypatch.setattr(transport_ffi, "_library_function", fake_library_function)
    monkeypatch.setattr(transport_ffi.jax, "process_index", lambda: 1)
    monkeypatch.setattr(transport_ffi.jax, "process_count", lambda: 2)
    monkeypatch.setattr(transport_ffi.ensure_internode_runtime, "_signature", None, raising=False)
    for env_name in (
        "NVSHMEM_IB_ENABLE_IBGDA",
        "NVSHMEM_IBGDA_NUM_RC_PER_PE",
        "NVSHMEM_QP_DEPTH",
        "NVSHMEM_CUMEM_GRANULARITY",
    ):
        monkeypatch.delenv(env_name, raising=False)

    transport_ffi.ensure_internode_runtime(
        num_nvl_bytes=2048,
        num_rdma_bytes=1024,
        configure_nvshmem_env=True,
        metadata=metadata,
    )

    assert calls == [(1, 2, 1, 2048, b"root-id", 1024)]
    assert os.environ["NVSHMEM_IB_ENABLE_IBGDA"] == "1"
    assert os.environ["NVSHMEM_IBGDA_NUM_RC_PER_PE"] == "24"
    assert os.environ["NVSHMEM_QP_DEPTH"] == "1024"
    assert os.environ["NVSHMEM_CUMEM_GRANULARITY"] == str(2**29)


def test_ensure_internode_runtime_calls_process_per_gpu_init(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int, int, int, int, int, int, bytes, bytes, int]] = []

    def fake_library_function(name: str, build_mode: transport_ffi.TransportBuildMode):
        assert build_mode is transport_ffi.TransportBuildMode.INTERNODE

        def init_process(
            global_rank,
            global_rank_count,
            node_rank,
            node_count,
            local_rank,
            ranks_per_node,
            num_nvl_bytes,
            ipc_handles,
            ipc_handles_size,
            root_id,
            root_id_size,
            num_rdma_bytes,
        ) -> int:
            calls.append(
                (
                    int(global_rank),
                    int(global_rank_count),
                    int(node_rank),
                    int(node_count),
                    int(local_rank),
                    int(ranks_per_node),
                    int(num_nvl_bytes),
                    bytes(ipc_handles[:ipc_handles_size]),
                    bytes(root_id[:root_id_size]),
                    int(num_rdma_bytes),
                )
            )
            return 0

        if name == "levanter_deepep_init_internode_process_runtime":
            return init_process
        raise AssertionError(f"unexpected symbol {name}")

    metadata = tuple(
        transport_ffi.InternodeProcessBootstrapMetadata(
            process_index=process_index,
            local_device_ids=(0,),
            nvshmem_unique_id=f"root-{process_index}".encode() if process_index < 8 else None,
            local_ipc_handles=(bytes([process_index]) * 4,),
            node_rank=process_index // 8,
            local_rank=process_index % 8,
            ranks_per_node=8,
            process_model="process_per_gpu",
        )
        for process_index in range(16)
    )
    monkeypatch.setattr(transport_ffi, "_library_function", fake_library_function)
    monkeypatch.setattr(transport_ffi.jax, "process_index", lambda: 9)
    monkeypatch.setattr(transport_ffi.jax, "process_count", lambda: 16)
    monkeypatch.setattr(transport_ffi, "_visible_local_gpu_count", lambda: 1)
    monkeypatch.setattr(transport_ffi.ensure_internode_runtime, "_signature", None, raising=False)

    transport_ffi.ensure_internode_runtime(
        num_nvl_bytes=2048,
        num_rdma_bytes=1024,
        metadata=metadata,
    )

    assert calls == [
        (
            9,
            16,
            1,
            2,
            1,
            8,
            2048,
            b"".join(bytes([process_index]) * 4 for process_index in range(16)),
            b"root-1",
            1024,
        )
    ]


def test_internode_runtime_status_uses_internode_library(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, transport_ffi.TransportBuildMode]] = []

    def fake_library_function(name: str, build_mode: transport_ffi.TransportBuildMode):
        calls.append((name, build_mode))

        def status(
            initialized,
            process_rank,
            process_count,
            num_local_ranks,
            num_global_ranks,
            num_nvl_bytes,
            num_rdma_bytes,
        ) -> int:
            initialized._obj.value = 1
            process_rank._obj.value = 3
            process_count._obj.value = 4
            num_local_ranks._obj.value = 8
            num_global_ranks._obj.value = 32
            num_nvl_bytes._obj.value = 2048
            num_rdma_bytes._obj.value = 4096
            return 0

        return status

    monkeypatch.setattr(transport_ffi, "_library_function", fake_library_function)

    assert transport_ffi.internode_runtime_status() == transport_ffi.InternodeRuntimeStatus(
        initialized=True,
        process_rank=3,
        process_count=4,
        num_local_ranks=8,
        num_global_ranks=32,
        num_nvl_bytes=2048,
        num_rdma_bytes=4096,
    )
    assert calls == [
        ("levanter_deepep_internode_runtime_status", transport_ffi.TransportBuildMode.INTERNODE),
    ]


def test_run_internode_mapped_counter_smoke_uses_internode_library(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, transport_ffi.TransportBuildMode]] = []

    def fake_library_function(name: str, build_mode=transport_ffi.TransportBuildMode.INTRANODE):
        calls.append((name, build_mode))

        def run(num_checked):
            num_checked._obj.value = 8
            return 0

        def last_error():
            return b""

        if name == "levanter_deepep_run_internode_mapped_counter_smoke":
            return run
        if name == "levanter_deepep_last_error":
            return last_error
        raise AssertionError(name)

    monkeypatch.setattr(transport_ffi, "_library_function", fake_library_function)

    assert transport_ffi.run_internode_mapped_counter_smoke() == {
        "internode_mapped_counter_smoke_status_code": 0,
        "num_checked": 8,
        "last_error": "",
    }
    assert calls == [
        ("levanter_deepep_run_internode_mapped_counter_smoke", transport_ffi.TransportBuildMode.INTERNODE),
        ("levanter_deepep_last_error", transport_ffi.TransportBuildMode.INTERNODE),
    ]


def test_deepep_dispatch_internode_exposes_static_jax_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_ffi_call(target, result_shape_dtypes, **kwargs):
        captured["target"] = target
        captured["result_shape_dtypes"] = result_shape_dtypes
        captured["kwargs"] = kwargs

        def call(*args, **attrs):
            captured["attrs"] = attrs
            return tuple(jnp.zeros(shape_dtype.shape, dtype=shape_dtype.dtype) for shape_dtype in result_shape_dtypes)

        return call

    monkeypatch.setattr(transport_ffi, "_register_internode_targets", lambda: None)
    monkeypatch.setattr(transport_ffi, "_register_targets", lambda: None)
    monkeypatch.setattr(transport_ffi.jax.ffi, "ffi_call", fake_ffi_call)

    dispatch = transport_ffi.deepep_dispatch_internode(
        jnp.zeros((3, 16), dtype=jnp.bfloat16),
        jnp.zeros((3, 2), dtype=jnp.int32),
        jnp.zeros((3, 2), dtype=jnp.float32),
        jnp.zeros((16,), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((32,), dtype=jnp.int32),
        jnp.zeros((3, 16), dtype=jnp.bool_),
        num_experts=32,
        max_recv_tokens=12,
        max_rdma_recv_tokens=10,
        source_meta_bytes=16,
        num_local_ranks=8,
    )

    assert captured["target"] == "levanter_deepep_dispatch_internode"
    attrs = captured["attrs"]
    assert isinstance(attrs, dict)
    assert int(attrs["num_experts"]) == 32
    assert int(attrs["num_sms"]) == 24
    assert int(attrs["num_max_nvl_chunked_send_tokens"]) == 8
    assert int(attrs["num_max_nvl_chunked_recv_tokens"]) == 512
    assert int(attrs["num_max_rdma_chunked_send_tokens"]) == 16
    assert int(attrs["num_max_rdma_chunked_recv_tokens"]) == 128
    assert dispatch.recv_x.shape == (12, 16)
    assert dispatch.recv_topk_idx.shape == (12, 2)
    assert dispatch.recv_topk_idx.dtype == jnp.int32
    assert dispatch.is_token_in_rank.shape == (3, 16)
    assert dispatch.recv_src_meta.shape == (12, 16)
    assert dispatch.rdma_channel_prefix_matrix.shape == (2, 12)
    assert dispatch.gbl_channel_prefix_matrix.shape == (16, 12)
    assert dispatch.send_rdma_head.shape == (3, 2)
    assert dispatch.send_nvl_head.shape == (10, 8)
    assert dispatch.local_expert_counts.shape == (2,)
    assert dispatch.num_recv_tokens.shape == (1,)
    assert dispatch.num_recv_rdma_tokens.shape == (1,)
    assert dispatch.local_group_sizes.shape == (2,)
    assert dispatch.x_dispatch.shape == (24, 16)
    assert dispatch.assignment_weights.shape == (24,)
    assert dispatch.recv_token_indices.shape == (24,)
    assert dispatch.assignment_destinations.shape == (24,)


def test_internode_dispatch_config_accepts_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPEP_INTERNODE_DISPATCH_NUM_SMS", "8")
    monkeypatch.setenv("DEEPEP_INTERNODE_DISPATCH_MAX_NVL_SEND_TOKENS", "4")
    monkeypatch.setenv("DEEPEP_INTERNODE_DISPATCH_MAX_NVL_RECV_TOKENS", "128")
    monkeypatch.setenv("DEEPEP_INTERNODE_DISPATCH_MAX_RDMA_SEND_TOKENS", "8")
    monkeypatch.setenv("DEEPEP_INTERNODE_DISPATCH_MAX_RDMA_RECV_TOKENS", "64")

    config = transport_ffi._default_internode_dispatch_config()

    assert config == transport_ffi.InternodeConfig(
        num_sms=8,
        num_max_nvl_chunked_send_tokens=4,
        num_max_nvl_chunked_recv_tokens=128,
        num_max_rdma_chunked_send_tokens=8,
        num_max_rdma_chunked_recv_tokens=64,
    )


def test_internode_dispatch_config_rejects_odd_num_sms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPEP_INTERNODE_DISPATCH_NUM_SMS", "7")

    with pytest.raises(RuntimeError, match="DEEPEP_INTERNODE_DISPATCH_NUM_SMS"):
        transport_ffi._default_internode_dispatch_config()


def test_deepep_dispatch_internode_backward_uses_combine_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    targets: list[str] = []

    def fake_ffi_call(target, result_shape_dtypes, **_kwargs):
        targets.append(target)

        def call(*_args, **_attrs):
            return tuple(jnp.zeros(shape_dtype.shape, dtype=shape_dtype.dtype) for shape_dtype in result_shape_dtypes)

        return call

    monkeypatch.setattr(transport_ffi, "_register_internode_targets", lambda: None)
    monkeypatch.setattr(transport_ffi, "_register_targets", lambda: None)
    monkeypatch.setattr(transport_ffi.jax.ffi, "ffi_call", fake_ffi_call)

    topk_idx = jnp.zeros((3, 2), dtype=jnp.int32)
    num_tokens_per_rank = jnp.zeros((16,), dtype=jnp.int32)
    num_tokens_per_rdma_rank = jnp.zeros((2,), dtype=jnp.int32)
    num_tokens_per_expert = jnp.zeros((32,), dtype=jnp.int32)
    is_token_in_rank = jnp.zeros((3, 16), dtype=jnp.bool_)

    def loss_fn(x: jax.Array, topk_weights: jax.Array) -> jax.Array:
        dispatch = transport_ffi.deepep_dispatch_internode(
            x,
            topk_idx,
            topk_weights,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            num_experts=32,
            max_recv_tokens=12,
            max_rdma_recv_tokens=10,
            source_meta_bytes=16,
            num_local_ranks=8,
        )
        return jnp.sum(dispatch.recv_x.astype(jnp.float32)) + jnp.sum(dispatch.recv_topk_weights)

    grad_x, grad_topk_weights = jax.grad(loss_fn, argnums=(0, 1))(
        jnp.zeros((3, 16), dtype=jnp.bfloat16),
        jnp.zeros((3, 2), dtype=jnp.float32),
    )

    assert grad_x.shape == (3, 16)
    assert grad_topk_weights.shape == (3, 2)
    assert targets == [
        "levanter_deepep_dispatch_internode",
        "levanter_deepep_combine_internode",
    ]


def test_deepep_dispatch_internode_backward_can_use_assignment_gradient_ffi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets: list[str] = []

    def fake_ffi_call(target, result_shape_dtypes, **_kwargs):
        targets.append(target)

        def call(*_args, **_attrs):
            return tuple(jnp.zeros(shape_dtype.shape, dtype=shape_dtype.dtype) for shape_dtype in result_shape_dtypes)

        return call

    monkeypatch.setenv("LEVANTER_DEEPEP_INTERNODE_ASSIGNMENT_GRADIENT_MODE", "ffi")
    monkeypatch.setattr(transport_ffi, "_register_internode_targets", lambda: None)
    monkeypatch.setattr(transport_ffi, "_register_targets", lambda: None)
    monkeypatch.setattr(transport_ffi.jax.ffi, "ffi_call", fake_ffi_call)

    topk_idx = jnp.zeros((3, 2), dtype=jnp.int32)
    num_tokens_per_rank = jnp.zeros((16,), dtype=jnp.int32)
    num_tokens_per_rdma_rank = jnp.zeros((2,), dtype=jnp.int32)
    num_tokens_per_expert = jnp.zeros((32,), dtype=jnp.int32)
    is_token_in_rank = jnp.zeros((3, 16), dtype=jnp.bool_)

    def loss_fn(x: jax.Array, topk_weights: jax.Array) -> jax.Array:
        dispatch = transport_ffi.deepep_dispatch_internode(
            x,
            topk_idx,
            topk_weights,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            num_experts=32,
            max_recv_tokens=12,
            max_rdma_recv_tokens=10,
            source_meta_bytes=16,
            num_local_ranks=8,
        )
        return jnp.sum(dispatch.x_dispatch.astype(jnp.float32)) + jnp.sum(
            dispatch.assignment_weights.astype(jnp.float32)
        )

    grad_x, grad_topk_weights = jax.grad(loss_fn, argnums=(0, 1))(
        jnp.zeros((3, 16), dtype=jnp.bfloat16),
        jnp.zeros((3, 2), dtype=jnp.float32),
    )

    assert grad_x.shape == (3, 16)
    assert grad_topk_weights.shape == (3, 2)
    assert targets == [
        "levanter_deepep_dispatch_internode",
        "levanter_deepep_assignment_gradients",
        "levanter_deepep_combine_internode",
    ]


def test_deepep_dispatch_internode_backward_can_use_fused_assignment_gradient_combine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets: list[str] = []

    def fake_ffi_call(target, result_shape_dtypes, **_kwargs):
        targets.append(target)

        def call(*_args, **_attrs):
            return tuple(jnp.zeros(shape_dtype.shape, dtype=shape_dtype.dtype) for shape_dtype in result_shape_dtypes)

        return call

    monkeypatch.setenv("LEVANTER_DEEPEP_INTERNODE_ASSIGNMENT_GRADIENT_MODE", "fused")
    monkeypatch.setattr(transport_ffi, "_register_internode_targets", lambda: None)
    monkeypatch.setattr(transport_ffi, "_register_targets", lambda: None)
    monkeypatch.setattr(transport_ffi.jax.ffi, "ffi_call", fake_ffi_call)

    topk_idx = jnp.zeros((3, 2), dtype=jnp.int32)
    num_tokens_per_rank = jnp.zeros((16,), dtype=jnp.int32)
    num_tokens_per_rdma_rank = jnp.zeros((2,), dtype=jnp.int32)
    num_tokens_per_expert = jnp.zeros((32,), dtype=jnp.int32)
    is_token_in_rank = jnp.zeros((3, 16), dtype=jnp.bool_)

    def loss_fn(x: jax.Array, topk_weights: jax.Array) -> jax.Array:
        dispatch = transport_ffi.deepep_dispatch_internode(
            x,
            topk_idx,
            topk_weights,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            num_experts=32,
            max_recv_tokens=12,
            max_rdma_recv_tokens=10,
            source_meta_bytes=16,
            num_local_ranks=8,
        )
        return jnp.sum(dispatch.x_dispatch.astype(jnp.float32)) + jnp.sum(
            dispatch.assignment_weights.astype(jnp.float32)
        )

    grad_x, grad_topk_weights = jax.grad(loss_fn, argnums=(0, 1))(
        jnp.zeros((3, 16), dtype=jnp.bfloat16),
        jnp.zeros((3, 2), dtype=jnp.float32),
    )

    assert grad_x.shape == (3, 16)
    assert grad_topk_weights.shape == (3, 2)
    assert targets == [
        "levanter_deepep_dispatch_internode",
        "levanter_deepep_dispatch_internode_bwd_fused",
    ]


def test_deepep_combine_internode_exposes_static_jax_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_ffi_call(target, result_shape_dtypes, **kwargs):
        captured["target"] = target
        captured["result_shape_dtypes"] = result_shape_dtypes
        captured["kwargs"] = kwargs

        def call(*args, **attrs):
            captured["attrs"] = attrs
            return tuple(jnp.zeros(shape_dtype.shape, dtype=shape_dtype.dtype) for shape_dtype in result_shape_dtypes)

        return call

    monkeypatch.setattr(transport_ffi, "_register_internode_targets", lambda: None)
    monkeypatch.setattr(transport_ffi.jax.ffi, "ffi_call", fake_ffi_call)

    combined_x, combined_topk_weights = transport_ffi.deepep_combine_internode(
        jnp.zeros((12, 16), dtype=jnp.bfloat16),
        jnp.zeros((12, 2), dtype=jnp.float32),
        jnp.zeros((3, 16), dtype=jnp.bool_),
        jnp.zeros((12, 16), dtype=jnp.uint8),
        jnp.zeros((2, 12), dtype=jnp.int32),
        jnp.zeros((2, 12), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((16, 12), dtype=jnp.int32),
        jnp.zeros((16, 12), dtype=jnp.int32),
        jnp.zeros((16,), dtype=jnp.int32),
        jnp.zeros((3, 2), dtype=jnp.int32),
        jnp.zeros((10, 8), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
    )

    assert captured["target"] == "levanter_deepep_combine_internode"
    attrs = captured["attrs"]
    assert isinstance(attrs, dict)
    assert int(attrs["num_sms"]) == 24
    assert int(attrs["num_max_nvl_chunked_send_tokens"]) == 8
    assert int(attrs["num_max_nvl_chunked_recv_tokens"]) == 512
    assert int(attrs["num_max_rdma_chunked_send_tokens"]) == 16
    assert int(attrs["num_max_rdma_chunked_recv_tokens"]) == 128
    assert combined_x.shape == (3, 16)
    assert combined_x.dtype == jnp.bfloat16
    assert combined_topk_weights.shape == (3, 2)
    assert combined_topk_weights.dtype == jnp.float32


def test_deepep_combine_internode_x_only_exposes_static_jax_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_ffi_call(target, result_shape_dtypes, **kwargs):
        captured["target"] = target
        captured["result_shape_dtypes"] = result_shape_dtypes
        captured["kwargs"] = kwargs

        def call(*args, **attrs):
            captured["arg_shapes"] = tuple(arg.shape for arg in args)
            captured["attrs"] = attrs
            return tuple(jnp.zeros(shape_dtype.shape, dtype=shape_dtype.dtype) for shape_dtype in result_shape_dtypes)

        return call

    monkeypatch.setattr(transport_ffi, "_register_internode_targets", lambda: None)
    monkeypatch.setattr(transport_ffi.jax.ffi, "ffi_call", fake_ffi_call)

    combined_x = transport_ffi.deepep_combine_internode_x_only(
        jnp.zeros((12, 16), dtype=jnp.bfloat16),
        jnp.zeros((3, 16), dtype=jnp.bool_),
        jnp.zeros((12, 16), dtype=jnp.uint8),
        jnp.zeros((2, 12), dtype=jnp.int32),
        jnp.zeros((2, 12), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((16, 12), dtype=jnp.int32),
        jnp.zeros((16, 12), dtype=jnp.int32),
        jnp.zeros((16,), dtype=jnp.int32),
        jnp.zeros((3, 2), dtype=jnp.int32),
        jnp.zeros((10, 8), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
        num_topk=2,
    )

    assert captured["target"] == "levanter_deepep_combine_internode_x_only"
    assert captured["arg_shapes"] == (
        (12, 16),
        (3, 16),
        (12, 16),
        (2, 12),
        (2,),
        (16, 12),
        (3, 2),
        (10, 8),
        (1,),
        (1,),
    )
    attrs = captured["attrs"]
    assert isinstance(attrs, dict)
    assert int(attrs["num_sms"]) == 24
    assert int(attrs["num_max_nvl_chunked_send_tokens"]) == 8
    assert int(attrs["num_max_nvl_chunked_recv_tokens"]) == 512
    assert int(attrs["num_max_rdma_chunked_send_tokens"]) == 16
    assert int(attrs["num_max_rdma_chunked_recv_tokens"]) == 128
    assert combined_x.shape == (3, 16)
    assert combined_x.dtype == jnp.bfloat16


def test_deepep_collapse_local_assignments_internode_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    targets: list[str] = []

    def fake_ffi_call(target, result_shape_dtypes, **_kwargs):
        targets.append(target)

        def call(*_args, **_attrs):
            if isinstance(result_shape_dtypes, tuple):
                return tuple(
                    jnp.zeros(shape_dtype.shape, dtype=shape_dtype.dtype) for shape_dtype in result_shape_dtypes
                )
            return jnp.zeros(result_shape_dtypes.shape, dtype=result_shape_dtypes.dtype)

        return call

    monkeypatch.setattr(transport_ffi, "_register_internode_targets", lambda: None)
    monkeypatch.setattr(transport_ffi.jax.ffi, "ffi_call", fake_ffi_call)

    def loss_fn(out_dispatch: jax.Array, assignment_weights: jax.Array) -> jax.Array:
        recv_out = transport_ffi.deepep_collapse_local_assignments(
            out_dispatch,
            assignment_weights,
            jnp.arange(20, dtype=jnp.int32) % 12,
            jnp.arange(24, dtype=jnp.int32) % 12,
            jnp.array([20, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32),
            jnp.array([12], dtype=jnp.int32),
            recv_capacity=12,
            internode=True,
        )
        return jnp.sum(recv_out.astype(jnp.float32))

    grad_out_dispatch, grad_assignment_weights = jax.grad(loss_fn, argnums=(0, 1))(
        jnp.zeros((20, 16), dtype=jnp.bfloat16),
        jnp.zeros((20,), dtype=jnp.bfloat16),
    )

    assert grad_out_dispatch.shape == (20, 16)
    assert grad_assignment_weights.shape == (20,)
    assert targets == [
        "levanter_deepep_collapse_local_assignments_internode",
        "levanter_deepep_collapse_local_assignments_internode_bwd",
    ]


def test_deepep_combine_internode_backward_uses_cached_dispatch_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    targets: list[str] = []

    def fake_ffi_call(target, result_shape_dtypes, **_kwargs):
        targets.append(target)

        def call(*_args, **_attrs):
            return tuple(jnp.zeros(shape_dtype.shape, dtype=shape_dtype.dtype) for shape_dtype in result_shape_dtypes)

        return call

    monkeypatch.setattr(transport_ffi, "_register_internode_targets", lambda: None)
    monkeypatch.setattr(transport_ffi.jax.ffi, "ffi_call", fake_ffi_call)

    def loss_fn(recv_x: jax.Array, recv_topk_weights: jax.Array) -> jax.Array:
        combined_x, combined_topk_weights = transport_ffi.deepep_combine_internode(
            recv_x,
            recv_topk_weights,
            jnp.zeros((3, 16), dtype=jnp.bool_),
            jnp.zeros((12, 16), dtype=jnp.uint8),
            jnp.zeros((2, 12), dtype=jnp.int32),
            jnp.zeros((2, 12), dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((16, 12), dtype=jnp.int32),
            jnp.zeros((16, 12), dtype=jnp.int32),
            jnp.zeros((16,), dtype=jnp.int32),
            jnp.zeros((3, 2), dtype=jnp.int32),
            jnp.zeros((10, 8), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
        )
        return jnp.sum(combined_x.astype(jnp.float32)) + jnp.sum(combined_topk_weights)

    grad_recv_x, grad_recv_topk_weights = jax.grad(loss_fn, argnums=(0, 1))(
        jnp.zeros((12, 16), dtype=jnp.bfloat16),
        jnp.zeros((12, 2), dtype=jnp.float32),
    )

    assert grad_recv_x.shape == (12, 16)
    assert grad_recv_topk_weights.shape == (12, 2)
    assert targets == ["levanter_deepep_combine_internode", "levanter_deepep_dispatch_internode_cached"]


def test_deepep_combine_internode_x_only_backward_uses_cached_dispatch_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets: list[str] = []
    attrs_by_target: dict[str, list[dict[str, object]]] = {}

    def fake_ffi_call(target, result_shape_dtypes, **_kwargs):
        targets.append(target)

        def call(*_args, **attrs):
            attrs_by_target.setdefault(target, []).append(attrs)
            if isinstance(result_shape_dtypes, tuple):
                return tuple(
                    jnp.zeros(shape_dtype.shape, dtype=shape_dtype.dtype) for shape_dtype in result_shape_dtypes
                )
            return jnp.zeros(result_shape_dtypes.shape, dtype=result_shape_dtypes.dtype)

        return call

    monkeypatch.setattr(transport_ffi, "_register_internode_targets", lambda: None)
    monkeypatch.setattr(transport_ffi.jax.ffi, "ffi_call", fake_ffi_call)

    def loss_fn(recv_x: jax.Array) -> jax.Array:
        combined_x = transport_ffi.deepep_combine_internode_x_only(
            recv_x,
            jnp.zeros((3, 16), dtype=jnp.bool_),
            jnp.zeros((12, 16), dtype=jnp.uint8),
            jnp.zeros((2, 12), dtype=jnp.int32),
            jnp.zeros((2, 12), dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((16, 12), dtype=jnp.int32),
            jnp.zeros((16, 12), dtype=jnp.int32),
            jnp.zeros((16,), dtype=jnp.int32),
            jnp.zeros((3, 2), dtype=jnp.int32),
            jnp.zeros((10, 8), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
            num_topk=2,
        )
        return jnp.sum(combined_x.astype(jnp.float32))

    grad_recv_x = jax.grad(loss_fn)(jnp.zeros((12, 16), dtype=jnp.bfloat16))

    assert grad_recv_x.shape == (12, 16)
    assert targets == ["levanter_deepep_combine_internode_x_only", "levanter_deepep_dispatch_internode_cached"]
    cached_dispatch_attrs = attrs_by_target["levanter_deepep_dispatch_internode_cached"][0]
    assert int(cached_dispatch_attrs["num_topk"]) == 2


def test_deepep_combine_internode_with_local_collapse_exposes_static_jax_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_ffi_call(target, result_shape_dtypes, **kwargs):
        captured["target"] = target
        captured["result_shape_dtypes"] = result_shape_dtypes
        captured["kwargs"] = kwargs

        def call(*args, **attrs):
            captured["arg_shapes"] = tuple(arg.shape for arg in args)
            captured["attrs"] = attrs
            return tuple(jnp.zeros(shape_dtype.shape, dtype=shape_dtype.dtype) for shape_dtype in result_shape_dtypes)

        return call

    monkeypatch.setattr(transport_ffi, "_register_internode_targets", lambda: None)
    monkeypatch.setattr(transport_ffi.jax.ffi, "ffi_call", fake_ffi_call)

    combined_x, combined_topk_weights = transport_ffi.deepep_combine_internode_with_local_collapse(
        jnp.zeros((20, 16), dtype=jnp.bfloat16),
        jnp.zeros((20,), dtype=jnp.bfloat16),
        jnp.zeros((20,), dtype=jnp.int32),
        jnp.zeros((24,), dtype=jnp.int32),
        jnp.array([20, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32),
        jnp.zeros((12, 2), dtype=jnp.float32),
        jnp.zeros((3, 16), dtype=jnp.bool_),
        jnp.zeros((12, 16), dtype=jnp.uint8),
        jnp.zeros((2, 12), dtype=jnp.int32),
        jnp.zeros((2, 12), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((16, 12), dtype=jnp.int32),
        jnp.zeros((16, 12), dtype=jnp.int32),
        jnp.zeros((16,), dtype=jnp.int32),
        jnp.zeros((3, 2), dtype=jnp.int32),
        jnp.zeros((10, 8), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
    )

    assert captured["target"] == "levanter_deepep_combine_internode_with_local_collapse"
    attrs = captured["attrs"]
    assert isinstance(attrs, dict)
    assert int(attrs["num_sms"]) == 24
    assert int(attrs["num_max_nvl_chunked_send_tokens"]) == 8
    assert int(attrs["num_max_nvl_chunked_recv_tokens"]) == 512
    assert int(attrs["num_max_rdma_chunked_send_tokens"]) == 16
    assert int(attrs["num_max_rdma_chunked_recv_tokens"]) == 128
    assert captured["arg_shapes"] == (
        (20, 16),
        (20,),
        (24,),
        (1,),
        (12, 2),
        (3, 16),
        (12, 16),
        (2, 12),
        (2,),
        (16, 12),
        (3, 2),
        (10, 8),
        (1,),
        (1,),
    )
    assert combined_x.shape == (3, 16)
    assert combined_x.dtype == jnp.bfloat16
    assert combined_topk_weights.shape == (3, 2)
    assert combined_topk_weights.dtype == jnp.float32


def test_deepep_combine_internode_with_local_collapse_backward_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets: list[str] = []

    def fake_ffi_call(target, result_shape_dtypes, **_kwargs):
        targets.append(target)

        def call(*_args, **_attrs):
            return tuple(jnp.zeros(shape_dtype.shape, dtype=shape_dtype.dtype) for shape_dtype in result_shape_dtypes)

        return call

    monkeypatch.setattr(transport_ffi, "_register_internode_targets", lambda: None)
    monkeypatch.setattr(transport_ffi.jax.ffi, "ffi_call", fake_ffi_call)

    def loss_fn(out_dispatch: jax.Array, assignment_weights: jax.Array) -> jax.Array:
        combined_x, combined_topk_weights = transport_ffi.deepep_combine_internode_with_local_collapse(
            out_dispatch,
            assignment_weights,
            jnp.arange(20, dtype=jnp.int32) % 12,
            jnp.arange(24, dtype=jnp.int32) % 12,
            jnp.array([20, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32),
            jnp.zeros((12, 2), dtype=jnp.float32),
            jnp.zeros((3, 16), dtype=jnp.bool_),
            jnp.zeros((12, 16), dtype=jnp.uint8),
            jnp.zeros((2, 12), dtype=jnp.int32),
            jnp.zeros((2, 12), dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((16, 12), dtype=jnp.int32),
            jnp.zeros((16, 12), dtype=jnp.int32),
            jnp.zeros((16,), dtype=jnp.int32),
            jnp.zeros((3, 2), dtype=jnp.int32),
            jnp.zeros((10, 8), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
        )
        return jnp.sum(combined_x.astype(jnp.float32)) + jnp.sum(combined_topk_weights)

    grad_out_dispatch, grad_assignment_weights = jax.grad(loss_fn, argnums=(0, 1))(
        jnp.zeros((20, 16), dtype=jnp.bfloat16),
        jnp.zeros((20,), dtype=jnp.bfloat16),
    )

    assert grad_out_dispatch.shape == (20, 16)
    assert grad_assignment_weights.shape == (20,)
    assert targets == [
        "levanter_deepep_combine_internode_with_local_collapse",
        "levanter_deepep_dispatch_internode_cached",
        "levanter_deepep_collapse_local_assignments_internode_bwd",
    ]


def test_deepep_combine_internode_x_only_with_local_collapse_exposes_static_jax_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_ffi_call(target, result_shape_dtypes, **kwargs):
        captured["target"] = target
        captured["result_shape_dtypes"] = result_shape_dtypes
        captured["kwargs"] = kwargs

        def call(*args, **attrs):
            captured["arg_shapes"] = tuple(arg.shape for arg in args)
            captured["attrs"] = attrs
            return jnp.zeros(result_shape_dtypes.shape, dtype=result_shape_dtypes.dtype)

        return call

    monkeypatch.setattr(transport_ffi, "_register_internode_targets", lambda: None)
    monkeypatch.setattr(transport_ffi.jax.ffi, "ffi_call", fake_ffi_call)

    combined_x = transport_ffi.deepep_combine_internode_x_only_with_local_collapse(
        jnp.zeros((20, 16), dtype=jnp.bfloat16),
        jnp.zeros((20,), dtype=jnp.bfloat16),
        jnp.zeros((20,), dtype=jnp.int32),
        jnp.zeros((24,), dtype=jnp.int32),
        jnp.array([20, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32),
        jnp.zeros((3, 16), dtype=jnp.bool_),
        jnp.zeros((12, 16), dtype=jnp.uint8),
        jnp.zeros((2, 12), dtype=jnp.int32),
        jnp.zeros((2, 12), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((16, 12), dtype=jnp.int32),
        jnp.zeros((16, 12), dtype=jnp.int32),
        jnp.zeros((16,), dtype=jnp.int32),
        jnp.zeros((3, 2), dtype=jnp.int32),
        jnp.zeros((10, 8), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
        num_topk=2,
    )

    assert captured["target"] == "levanter_deepep_combine_internode_x_only_with_local_collapse"
    attrs = captured["attrs"]
    assert isinstance(attrs, dict)
    assert int(attrs["num_topk"]) == 2
    assert int(attrs["num_sms"]) == 24
    assert int(attrs["num_max_nvl_chunked_send_tokens"]) == 8
    assert int(attrs["num_max_nvl_chunked_recv_tokens"]) == 512
    assert int(attrs["num_max_rdma_chunked_send_tokens"]) == 16
    assert int(attrs["num_max_rdma_chunked_recv_tokens"]) == 128
    assert captured["arg_shapes"] == (
        (20, 16),
        (20,),
        (24,),
        (1,),
        (3, 16),
        (12, 16),
        (2, 12),
        (2,),
        (16, 12),
        (3, 2),
        (10, 8),
        (1,),
        (1,),
    )
    assert combined_x.shape == (3, 16)
    assert combined_x.dtype == jnp.bfloat16


def test_deepep_combine_internode_x_only_with_local_collapse_backward_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets: list[str] = []
    attrs_by_target: dict[str, list[dict[str, object]]] = {}

    def fake_ffi_call(target, result_shape_dtypes, **_kwargs):
        targets.append(target)

        def call(*_args, **attrs):
            attrs_by_target.setdefault(target, []).append(attrs)
            if isinstance(result_shape_dtypes, tuple):
                return tuple(
                    jnp.zeros(shape_dtype.shape, dtype=shape_dtype.dtype) for shape_dtype in result_shape_dtypes
                )
            return jnp.zeros(result_shape_dtypes.shape, dtype=result_shape_dtypes.dtype)

        return call

    monkeypatch.setattr(transport_ffi, "_register_internode_targets", lambda: None)
    monkeypatch.setattr(transport_ffi.jax.ffi, "ffi_call", fake_ffi_call)

    def loss_fn(out_dispatch: jax.Array, assignment_weights: jax.Array) -> jax.Array:
        combined_x = transport_ffi.deepep_combine_internode_x_only_with_local_collapse(
            out_dispatch,
            assignment_weights,
            jnp.arange(20, dtype=jnp.int32) % 12,
            jnp.arange(24, dtype=jnp.int32) % 12,
            jnp.array([20, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32),
            jnp.zeros((3, 16), dtype=jnp.bool_),
            jnp.zeros((12, 16), dtype=jnp.uint8),
            jnp.zeros((2, 12), dtype=jnp.int32),
            jnp.zeros((2, 12), dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((16, 12), dtype=jnp.int32),
            jnp.zeros((16, 12), dtype=jnp.int32),
            jnp.zeros((16,), dtype=jnp.int32),
            jnp.zeros((3, 2), dtype=jnp.int32),
            jnp.zeros((10, 8), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
            num_topk=2,
        )
        return jnp.sum(combined_x.astype(jnp.float32))

    grad_out_dispatch, grad_assignment_weights = jax.grad(loss_fn, argnums=(0, 1))(
        jnp.zeros((20, 16), dtype=jnp.bfloat16),
        jnp.zeros((20,), dtype=jnp.bfloat16),
    )

    assert grad_out_dispatch.shape == (20, 16)
    assert grad_assignment_weights.shape == (20,)
    assert targets == [
        "levanter_deepep_combine_internode_x_only_with_local_collapse",
        "levanter_deepep_dispatch_internode_cached",
        "levanter_deepep_collapse_local_assignments_internode_bwd",
    ]
    cached_dispatch_attrs = attrs_by_target["levanter_deepep_dispatch_internode_cached"][0]
    assert int(cached_dispatch_attrs["num_topk"]) == 2
