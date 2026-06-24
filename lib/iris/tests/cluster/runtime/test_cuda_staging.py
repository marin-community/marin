# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the CUDA toolchain setup script.

The script runs in real bash, so these tests execute it against a fake venv tree
and inspect the resulting symlinks and staged files rather than asserting on its
source text. The GPU extra wires it into the resolved setup scripts.
"""

import subprocess
from pathlib import Path

from iris.cluster.setup import cuda_toolchain_setup_script, wants_gpu_extra
from iris.cluster.types import EnvironmentSpec


def _make_venv(tmp_path: Path, *, cuda_major: str, with_ptxas: bool, with_libdevice: bool) -> Path:
    """Create a fake venv tree mirroring what jax[cuda*] installs."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
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


def _run_setup(venv: Path, workdir: Path) -> None:
    env = {"IRIS_VENV": str(venv), "IRIS_WORKDIR": str(workdir), "PATH": "/usr/bin:/bin"}
    subprocess.run(["bash", "-c", cuda_toolchain_setup_script()], env=env, capture_output=True, text=True, check=True)


def test_stages_toolchain_when_present(tmp_path):
    venv = _make_venv(tmp_path, cuda_major="cu13", with_ptxas=True, with_libdevice=True)
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


def test_version_agnostic(tmp_path):
    # The same script handles a different CUDA major (cu12) with no change.
    venv = _make_venv(tmp_path, cuda_major="cu12", with_ptxas=True, with_libdevice=True)
    workdir = tmp_path / "work"
    workdir.mkdir()

    _run_setup(venv, workdir)

    assert (venv / "bin" / "ptxas").is_symlink()
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


def test_wants_gpu_extra():
    assert wants_gpu_extra(["gpu"])
    assert wants_gpu_extra(["marin:gpu"])
    assert not wants_gpu_extra(["cpu"])
    assert not wants_gpu_extra(["tpu", "vllm"])


def test_gpu_extra_appends_cuda_setup_script():
    scripts = list(EnvironmentSpec(extras=["gpu"]).to_proto().setup_scripts)
    assert cuda_toolchain_setup_script() in scripts


def test_non_gpu_extra_has_no_cuda_setup_script():
    scripts = list(EnvironmentSpec(extras=["cpu"]).to_proto().setup_scripts)
    assert cuda_toolchain_setup_script() not in scripts


def test_custom_setup_scripts_skip_cuda_staging():
    # An explicit setup_scripts list is used verbatim; no CUDA staging is added.
    scripts = list(EnvironmentSpec(extras=["gpu"], setup_scripts=["echo hi\n"]).to_proto().setup_scripts)
    assert scripts == ["echo hi\n"]
