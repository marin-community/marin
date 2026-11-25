from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).parents[3]
BOOTSTRAP_SCRIPT = REPOSITORY_ROOT / "cloud" / "iris" / "bootstrap_runtime.sh"
FIXTURES = Path(__file__).with_name("bootstrap_fixtures")
RUNTIME_FILE = "runtime.env"


def _write_module(site_packages: Path, relative_path: str, source: Path | None = None) -> None:
    path = site_packages / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if source is None:
        path.touch()
    else:
        shutil.copyfile(source, path)


def _fake_frozen_runtime(
    tmp_path: Path, *, managed_python_pin_is_unresolvable: bool = False
) -> tuple[Path, dict[str, str]]:
    environment = tmp_path / "runtime"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    site_packages = next((environment / "lib").glob("python*/site-packages"))

    (site_packages / "nvidia/cu13/lib").mkdir(parents=True)
    for package in (
        "shellbox",
        "shellbox/backends",
        "megatron",
        "shellbox/backends/daytona",
        "shellbox/backends/shellsim",
        "rolloutengine",
        "quack",
        "skyrl_train",
        "skyrl_train/models",
        "transformer_engine",
        "vllm/model_executor",
    ):
        _write_module(site_packages, f"{package}/__init__.py")

    _write_module(site_packages, "quack/activation.py")
    _write_module(site_packages, "flash_attn/__init__.py")
    _write_module(site_packages, "flash_attn/cute/__init__.py")
    _write_module(site_packages, "memray.py")
    _write_module(site_packages, "vllm/_C_stable_libtorch.py")
    _write_module(site_packages, "vllm/cumem_allocator.py")
    for path in (
        "flash_attn/cute/interface.py",
        "flash_attn_interface.py",
        "megatron/bridge/__init__.py",
        "torch.py",
        "vllm/__init__.py",
        "transformer_engine/common/__init__.py",
        "vllm/model_executor/models.py",
        "skyrl_train/models/grug_moe.py",
        "shellbox/backends/daytona/machine.py",
        "shellbox/backends/shellsim/machine.py",
        "rolloutengine/engine.py",
    ):
        _write_module(site_packages, path, FIXTURES / "dependency.py")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "python3.12").symlink_to(sys.executable)
    uv = fake_bin / "uv"
    shutil.copyfile(FIXTURES / "uv.sh", uv)
    uv.chmod(0o755)
    process_environment = os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}"}
    if managed_python_pin_is_unresolvable:
        process_environment["EXPECTED_SYSTEM_PYTHON"] = str(fake_bin / "python3.12")
    return environment, process_environment


def _run_bootstrap(
    environment: Path, process_environment: dict[str, str], profile: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(BOOTSTRAP_SCRIPT),
            str(REPOSITORY_ROOT),
            str(environment),
            str(environment / RUNTIME_FILE),
            profile,
            "production",
        ],
        cwd=REPOSITORY_ROOT,
        env=process_environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("architecture", "missing_module", "import_name"),
    [
        ("x86_64", "flash_attn_interface.py", "flash_attn_interface"),
        ("aarch64", "flash_attn/cute/interface.py", "flash_attn.cute.interface"),
    ],
)
def test_policy_bootstrap_rejects_runtime_without_flash_attention_extension(
    tmp_path: Path, architecture: str, missing_module: str, import_name: str
) -> None:
    environment, process_environment = _fake_frozen_runtime(tmp_path)
    site_packages = next((environment / "lib").glob("python*/site-packages"))
    (site_packages / missing_module).unlink()
    _write_module(site_packages, "sitecustomize.py", FIXTURES / "sitecustomize.py")
    process_environment["FIXTURE_ARCHITECTURE"] = architecture
    # Give the architecture shim precedence over a system sitecustomize module.
    process_environment["PYTHONPATH"] = str(site_packages)

    result = _run_bootstrap(environment, process_environment, "megatron")

    assert result.returncode != 0
    assert f"No module named '{import_name}'" in result.stderr


def test_export_bootstrap_does_not_require_rollout_or_telemetry_packages(tmp_path: Path) -> None:
    environment, process_environment = _fake_frozen_runtime(tmp_path)
    site_packages = next((environment / "lib").glob("python*/site-packages"))
    _write_module(site_packages, "ray.py")
    _write_module(site_packages, "skyrl_train/checkpoint_exporter.py", FIXTURES / "dependency.py")
    (site_packages / "shellbox/backends/daytona/machine.py").unlink()
    (site_packages / "memray.py").unlink()
    (site_packages / "rolloutengine/engine.py").unlink()
    shutil.rmtree(site_packages / "vllm")

    result = _run_bootstrap(environment, process_environment, "megatron-export")

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("publisher_installed", [False, True])
def test_hub_publish_bootstrap_checks_cpu_publisher_without_cuda(tmp_path: Path, publisher_installed: bool) -> None:
    environment, process_environment = _fake_frozen_runtime(tmp_path)
    site_packages = next((environment / "lib").glob("python*/site-packages"))
    shutil.rmtree(site_packages / "nvidia")
    shutil.rmtree(site_packages / "flash_attn")
    (site_packages / "flash_attn_interface.py").unlink()
    if publisher_installed:
        _write_module(site_packages, "skyrl_train/hf_publisher.py", FIXTURES / "dependency.py")

    project = tmp_path / "skyrl"
    scripts = project / "cloud/iris"
    scripts.mkdir(parents=True)
    for name in ("setup_task_runtime.sh", "bootstrap_runtime.sh", "load_runtime_environment.sh", "runtime_checks.py"):
        (scripts / name).symlink_to(BOOTSTRAP_SCRIPT.with_name(name))
    (project / ".uv-version").symlink_to(REPOSITORY_ROOT / ".uv-version")
    result = subprocess.run(
        ["bash", scripts / "setup_task_runtime.sh", "hub-publish", "printenv", "PATH"],
        env=process_environment | {"IRIS_VENV": str(environment)},
        capture_output=True,
        text=True,
        timeout=15,
    )

    if publisher_installed:
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == f"{environment / 'bin'}{os.pathsep}{process_environment['PATH']}"
        assert not (environment / "lib/libcudart.so").exists()
    else:
        assert result.returncode != 0
        assert "skyrl_train.hf_publisher" in result.stderr


def test_bootstrap_uses_system_python_when_managed_pin_is_unresolvable(tmp_path: Path) -> None:
    environment, process_environment = _fake_frozen_runtime(tmp_path, managed_python_pin_is_unresolvable=True)

    result = _run_bootstrap(environment, process_environment, "megatron")

    assert result.returncode == 0, result.stderr


def test_bootstrap_activation_exposes_runtime_commands(tmp_path: Path) -> None:
    environment, process_environment = _fake_frozen_runtime(tmp_path)
    ninja = environment / "bin" / "ninja"
    ninja.touch(mode=0o755)
    process_environment["LD_LIBRARY_PATH"] = str(tmp_path / "existing library path")

    result = _run_bootstrap(environment, process_environment, "megatron")
    activation = subprocess.run(
        [
            "bash",
            FIXTURES / "activation.sh",
            BOOTSTRAP_SCRIPT.with_name("load_runtime_environment.sh"),
            environment / RUNTIME_FILE,
        ],
        env=process_environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0, result.stderr
    binary, libraries, path = activation.stdout.splitlines()
    assert binary == str(ninja)
    assert path == f"{environment / 'bin'}{os.pathsep}{process_environment['PATH']}"
    assert libraries.split(os.pathsep)[-1] == process_environment["LD_LIBRARY_PATH"]


def test_bootstrap_exposes_cuda_linker_compatibility_paths(tmp_path: Path) -> None:
    environment, process_environment = _fake_frozen_runtime(tmp_path)
    site_packages = next((environment / "lib").glob("python*/site-packages"))
    nvrtc = site_packages / "nvidia" / "cu13" / "lib" / "libnvrtc.so.13"
    nvrtc.touch()

    result = _run_bootstrap(environment, process_environment, "megatron")

    assert result.returncode == 0, result.stderr
    assert (site_packages / "nvidia" / "cu13" / "lib64").resolve() == site_packages / "nvidia" / "cu13" / "lib"
    assert (environment / "lib" / "libnvrtc.so").resolve() == nvrtc


@pytest.mark.parametrize(
    ("missing_module", "expected_error"),
    [
        ("shellbox/backends/daytona/machine.py", "shellbox.backends.daytona.machine"),
        ("rolloutengine/engine.py", "rolloutengine.engine"),
        ("memray.py", "No module named 'memray'"),
        ("megatron/bridge", "megatron.bridge"),
        ("transformer_engine/common", "transformer_engine.common"),
    ],
)
def test_bootstrap_rejects_incomplete_agentic_debug_runtime(
    tmp_path: Path, missing_module: str, expected_error: str
) -> None:
    environment, process_environment = _fake_frozen_runtime(tmp_path)
    site_packages = next((environment / "lib").glob("python*/site-packages"))
    missing_path = site_packages / missing_module
    if missing_path.is_dir():
        shutil.rmtree(missing_path)
    else:
        missing_path.unlink()

    result = _run_bootstrap(environment, process_environment, "megatron")

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_parallel_bootstrap_waits_for_every_check_and_reports_failure(tmp_path):
    environment, process_environment = _fake_frozen_runtime(tmp_path)
    barrier = tmp_path / "imports-ready"
    os.mkfifo(barrier)
    process_environment["FIXTURE_IMPORT_BARRIER"] = str(barrier)
    descriptor = os.open(barrier, os.O_RDWR | os.O_NONBLOCK)
    try:
        result = _run_bootstrap(environment, process_environment, "megatron")
    finally:
        os.close(descriptor)
    assert result.returncode != 0, result.stderr
    assert "flash-attention import failed" in result.stderr
    assert "megatron import failed" in result.stderr
