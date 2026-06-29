# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the CUDA toolchain setup script.

The script runs in real bash, so these tests execute it against a fake venv tree
and inspect the resulting symlinks and staged files rather than asserting on its
source text. The GPU extra wires it into the resolved setup scripts.
"""

import subprocess
from pathlib import Path

import pytest
from iris.cluster.setup import cuda_toolchain_setup_script, wants_gpu_extra
from iris.cluster.types import EnvironmentSpec


def _make_venv(tmp_path: Path, *, cuda_major: str, with_ptxas: bool, with_libdevice: bool) -> Path:
    """Create a fake venv tree mirroring what jax[cuda*] installs."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text(f'VIRTUAL_ENV="{venv}"\nexport VIRTUAL_ENV\n')
    cuda = venv / "lib" / "python3.12" / "site-packages" / "nvidia" / cuda_major
    (cuda / "bin").mkdir(parents=True)
    if with_ptxas:
        for tool in ("ptxas", "nvlink"):
            tool_path = cuda / "bin" / tool
            tool_path.write_text("#!/bin/sh\n")
            tool_path.chmod(0o755)
    if with_libdevice:
        libdevice = cuda / "nvvm" / "libdevice"
        libdevice.mkdir(parents=True)
        (libdevice / "libdevice.10.bc").write_bytes(b"BC\xc0\xde")
    return venv


def _add_nvshmem_runtime(venv: Path) -> Path:
    """Create the wheel layout for nvidia-nvshmem-cu13."""
    nvshmem_lib = venv / "lib" / "python3.12" / "site-packages" / "nvidia" / "nvshmem" / "lib"
    nvshmem_lib.mkdir(parents=True)
    (nvshmem_lib / "libnvshmem_host.so.3").write_bytes(b"NVSHMEM")
    return nvshmem_lib


def _run_script(script: str, venv: Path, workdir: Path) -> None:
    env = {"IRIS_VENV": str(venv), "IRIS_WORKDIR": str(workdir), "PATH": "/usr/bin:/bin"}
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


def test_stages_when_libdevice_missing(tmp_path):
    # ptxas present but libdevice absent: still symlink the toolchain, skip copies.
    venv = _make_venv(tmp_path, cuda_major="cu13", with_ptxas=True, with_libdevice=False)
    workdir = tmp_path / "work"
    workdir.mkdir()

    _run_setup(venv, workdir)

    assert (venv / "bin" / "ptxas").is_symlink()
    assert not (workdir / "libdevice.10.bc").exists()


def test_exposes_wheel_provided_nvshmem_runtime_when_present(tmp_path):
    venv = _make_venv(tmp_path, cuda_major="cu13", with_ptxas=True, with_libdevice=True)
    nvshmem_lib = _add_nvshmem_runtime(venv)
    workdir = tmp_path / "work"
    workdir.mkdir()

    _run_setup(venv, workdir)

    nvshmem_alias = nvshmem_lib / "libnvshmem.so"
    assert nvshmem_alias.is_symlink()
    assert nvshmem_alias.resolve() == nvshmem_lib / "libnvshmem_host.so.3"

    activation_probe = subprocess.run(
        ["bash", "-c", f"source {venv / 'bin' / 'activate'} && printf '%s' \"$LD_LIBRARY_PATH\""],
        capture_output=True,
        text=True,
        check=True,
    )
    assert str(nvshmem_lib) in activation_probe.stdout.split(":")


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
