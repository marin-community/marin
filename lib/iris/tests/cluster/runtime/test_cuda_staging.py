# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the CUDA toolchain setup script.

The script runs in real bash, so these tests execute it against a fake venv tree
and inspect the resulting symlinks and staged files rather than asserting on its
source text. The GPU extra wires it into the resolved setup scripts.
"""

import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest
from iris.cluster.setup_scripts import CUDA_TOOLCHAIN_PACKAGE, cuda_toolchain_setup_script, wants_gpu_extra
from iris.cluster.types import EnvironmentSpec

_CUDA_13_TEST_PACKAGES = (
    ("nvidia-cudnn-cu13", "9.19.0.56", "nvidia/cudnn/lib/libcudnn.so.9"),
    ("nvidia-nccl-cu13", "2.28.9", "nvidia/nccl/lib/libnccl.so.2"),
)


def _add_cuda_toolchain(
    site_packages: Path, *, cuda_major: str, with_ptxas: bool, with_libdevice: bool, tool_mode: int = 0o755
) -> None:
    cuda = site_packages / "nvidia" / cuda_major
    (cuda / "bin").mkdir(parents=True)
    if with_ptxas:
        for tool in ("ptxas", "nvlink"):
            tool_path = cuda / "bin" / tool
            tool_path.write_text("#!/bin/sh\n")
            tool_path.chmod(tool_mode)
    if with_libdevice:
        libdevice = cuda / "nvvm" / "libdevice"
        libdevice.mkdir(parents=True)
        (libdevice / "libdevice.10.bc").write_bytes(b"BC\xc0\xde")


def _make_venv(tmp_path: Path, *, cuda_major: str, with_ptxas: bool, with_libdevice: bool) -> Path:
    """Create a fake venv tree mirroring what jax[cuda*] installs."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    site_packages = venv / "lib" / "python3.12" / "site-packages"
    _add_cuda_toolchain(
        site_packages,
        cuda_major=cuda_major,
        with_ptxas=with_ptxas,
        with_libdevice=with_libdevice,
    )
    return venv


def _site_packages(venv: Path) -> Path:
    return next((venv / "lib").glob("python*/site-packages"))


def _write_test_wheel(
    wheelhouse: Path,
    package: str,
    version: str,
    files: dict[str, str],
    *,
    executable_files: tuple[str, ...] = (),
) -> Path:
    normalized_package = package.replace("-", "_")
    dist_info = f"{normalized_package}-{version}.dist-info"
    wheel = wheelhouse / f"{normalized_package}-{version}-py3-none-any.whl"
    wheelhouse.mkdir(exist_ok=True)
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr(
            f"{dist_info}/METADATA",
            "\n".join(
                [
                    "Metadata-Version: 2.1",
                    f"Name: {package}",
                    f"Version: {version}",
                    "",
                ]
            ),
        )
        zf.writestr(
            f"{dist_info}/WHEEL",
            "\n".join(
                [
                    "Wheel-Version: 1.0",
                    "Generator: iris-test",
                    "Root-Is-Purelib: true",
                    "Tag: py3-none-any",
                    "",
                ]
            ),
        )
        for path, content in files.items():
            if path in executable_files:
                file_info = zipfile.ZipInfo(path)
                file_info.external_attr = 0o755 << 16
                zf.writestr(file_info, content)
            else:
                zf.writestr(path, content)
        zf.writestr(f"{dist_info}/RECORD", "")
    return wheel


def _write_cuda_toolchain_wheel(wheelhouse: Path, version: str) -> Path:
    tool_paths = tuple(f"nvidia/cu13/bin/{tool}" for tool in ("ptxas", "nvlink"))
    return _write_test_wheel(
        wheelhouse,
        CUDA_TOOLCHAIN_PACKAGE,
        version,
        {path: "#!/bin/sh\n" for path in tool_paths},
        executable_files=tool_paths,
    )


@dataclass(frozen=True)
class _TestWheelEnvironment:
    venv: Path
    workdir: Path
    offline_env: dict[str, str]


def _install_test_wheel(tmp_path: Path, wheel: Path) -> _TestWheelEnvironment:
    venv = tmp_path / "venv"
    subprocess.run(["uv", "venv", "--python", sys.executable, str(venv)], capture_output=True, text=True, check=True)
    # A copy lets tests change installed files without changing uv's cached wheel.
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--no-cache",
            "--link-mode",
            "copy",
            "--python",
            str(venv / "bin" / "python"),
            str(wheel),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    workdir = tmp_path / "work"
    workdir.mkdir()
    offline_env = {
        "UV_FIND_LINKS": str(wheel.parent),
        "UV_OFFLINE": "1",
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
    }
    return _TestWheelEnvironment(venv=venv, workdir=workdir, offline_env=offline_env)


def _run_script(script: str, venv: Path, workdir: Path, *, path: str = "/usr/bin:/bin", extra_env=None) -> None:
    env = {"IRIS_VENV": str(venv), "IRIS_WORKDIR": str(workdir), "PATH": path}
    if extra_env:
        env.update(extra_env)
    subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=True)


def _run_setup(venv: Path, workdir: Path) -> None:
    _run_script(cuda_toolchain_setup_script(), venv, workdir)


# cu12 and cu13 exercise the version-agnostic glob: the same script must stage
# either CUDA major with no change.
@pytest.mark.parametrize("cuda_major", ["cu12", "cu13"])
def test_stages_toolchain_when_present(tmp_path, cuda_major):
    venv = _make_venv(tmp_path, cuda_major=cuda_major, with_ptxas=True, with_libdevice=True)
    workdir = tmp_path / "work"
    workdir.mkdir()

    _run_setup(venv, workdir)

    ptxas = venv / "bin" / "ptxas"
    assert ptxas.is_symlink()
    assert ptxas.resolve().is_file()
    assert (venv / "bin" / "nvlink").is_symlink()
    # libdevice staged into XLA's default data dir and the working directory.
    assert (workdir / "cuda_sdk_lib" / "nvvm" / "libdevice" / "libdevice.10.bc").is_file()
    assert (workdir / "libdevice.10.bc").is_file()


def test_noop_when_toolchain_absent(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    workdir = tmp_path / "work"
    workdir.mkdir()

    _run_setup(venv, workdir)

    assert not (venv / "bin" / "ptxas").exists()
    assert not (workdir / "libdevice.10.bc").exists()
    assert not (workdir / "cuda_sdk_lib").exists()


def test_noop_when_ptxas_missing(tmp_path):
    # cu13/bin exists but carries no compiler — a partial install stages nothing.
    venv = _make_venv(tmp_path, cuda_major="cu13", with_ptxas=False, with_libdevice=True)
    workdir = tmp_path / "work"
    workdir.mkdir()

    _run_setup(venv, workdir)

    assert not (venv / "bin" / "ptxas").exists()
    assert not (workdir / "libdevice.10.bc").exists()


def test_fails_when_ptxas_is_not_executable(tmp_path):
    # An install that drops file modes leaves ptxas unrunnable; staging it would
    # only move the failure to the first kernel compile.
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _add_cuda_toolchain(
        venv / "lib" / "python3.12" / "site-packages",
        cuda_major="cu13",
        with_ptxas=True,
        with_libdevice=True,
        tool_mode=0o644,
    )
    workdir = tmp_path / "work"
    workdir.mkdir()

    with pytest.raises(subprocess.CalledProcessError):
        _run_setup(venv, workdir)

    assert not (venv / "bin" / "ptxas").exists()


def test_repairs_installed_cuda_toolchain_when_files_are_missing(tmp_path):
    version = "13.2.78"
    wheelhouse = tmp_path / "wheelhouse"
    wheel = _write_cuda_toolchain_wheel(wheelhouse, version)
    environment = _install_test_wheel(tmp_path, wheel)
    shutil.rmtree(_site_packages(environment.venv) / "nvidia" / "cu13" / "bin")

    _run_script(
        cuda_toolchain_setup_script(),
        environment.venv,
        environment.workdir,
        path=os.environ["PATH"],
        extra_env=environment.offline_env,
    )

    assert (environment.venv / "bin" / "ptxas").resolve().is_file()
    assert (environment.venv / "bin" / "nvlink").resolve().is_file()


def test_fails_when_cuda_toolchain_repair_does_not_restore_ptxas(tmp_path):
    version = "13.2.78"
    wheelhouse = tmp_path / "wheelhouse"
    wheel = _write_test_wheel(wheelhouse, CUDA_TOOLCHAIN_PACKAGE, version, {})
    environment = _install_test_wheel(tmp_path, wheel)

    with pytest.raises(subprocess.CalledProcessError):
        _run_script(
            cuda_toolchain_setup_script(),
            environment.venv,
            environment.workdir,
            path=os.environ["PATH"],
            extra_env=environment.offline_env,
        )


def test_stages_when_libdevice_missing(tmp_path):
    # ptxas present but libdevice absent: still symlink the toolchain, skip copies.
    venv = _make_venv(tmp_path, cuda_major="cu13", with_ptxas=True, with_libdevice=False)
    workdir = tmp_path / "work"
    workdir.mkdir()

    _run_setup(venv, workdir)

    assert (venv / "bin" / "ptxas").is_symlink()
    assert not (workdir / "libdevice.10.bc").exists()


# The restore must not depend on staging: a venv without a usable toolchain still
# needs the CUDA 13 NCCL, or its processes can load different NCCL builds.
@pytest.mark.parametrize("with_toolchain", [True, False])
@pytest.mark.parametrize(("package", "version", "library_path"), _CUDA_13_TEST_PACKAGES)
def test_restores_cuda13_shared_library_package_when_present(tmp_path, package, version, library_path, with_toolchain):
    wheelhouse = tmp_path / "wheelhouse"
    wheel = _write_test_wheel(wheelhouse, package, version, {library_path: "cu13\n"})
    environment = _install_test_wheel(tmp_path, wheel)
    site_packages = _site_packages(environment.venv)
    if with_toolchain:
        _add_cuda_toolchain(site_packages, cuda_major="cu13", with_ptxas=True, with_libdevice=True)

    shared_library = site_packages / library_path
    shared_library.write_text("cu12\n")

    _run_script(
        cuda_toolchain_setup_script(),
        environment.venv,
        environment.workdir,
        path=os.environ["PATH"],
        extra_env=environment.offline_env,
    )

    assert shared_library.read_text() == "cu13\n"


def test_wants_gpu_extra():
    assert wants_gpu_extra(["gpu"])
    assert wants_gpu_extra(["marin:gpu"])
    assert not wants_gpu_extra(["cpu"])
    assert not wants_gpu_extra(["tpu", "vllm"])


def test_gpu_extra_appends_a_real_staging_step(tmp_path):
    """A GPU job appends exactly one setup step over the CPU baseline, and that
    step actually stages the toolchain — verified by effect, not string identity."""
    cpu_scripts = list(EnvironmentSpec(extras=["cpu"]).to_proto().setup_scripts)
    gpu_scripts = list(EnvironmentSpec(extras=["gpu"]).to_proto().setup_scripts)

    appended = gpu_scripts[len(cpu_scripts) :]
    assert len(appended) == 1

    venv = _make_venv(tmp_path, cuda_major="cu13", with_ptxas=True, with_libdevice=True)
    workdir = tmp_path / "work"
    workdir.mkdir()
    _run_script(appended[0], venv, workdir)
    assert (venv / "bin" / "ptxas").is_symlink()


def test_custom_setup_scripts_skip_cuda_staging():
    # An explicit setup_scripts list is used verbatim even with the gpu extra:
    # no staging step is appended, so a bring-your-own setup must stage itself.
    scripts = list(EnvironmentSpec(extras=["gpu"], setup_scripts=["echo hi\n"]).to_proto().setup_scripts)
    assert scripts == ["echo hi\n"]
