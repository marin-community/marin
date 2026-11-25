from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from cloud.iris.tests.bootstrap_runtime_fixtures import FIXTURES, fake_frozen_runtime

REPOSITORY_ROOT = Path(__file__).parents[3]
BOOTSTRAP_SCRIPT = REPOSITORY_ROOT / "cloud" / "iris" / "bootstrap_runtime.sh"
RUNTIME_FILE = "runtime.env"


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
    environment, process_environment = fake_frozen_runtime(tmp_path)
    site_packages = next((environment / "lib").glob("python*/site-packages"))
    (site_packages / missing_module).unlink()
    for shim in ("fixture-architecture.pth", "_fixture_architecture.py"):
        shutil.copyfile(FIXTURES / shim, site_packages / shim)
    process_environment["FIXTURE_ARCHITECTURE"] = architecture
    process_environment["PYTHONPATH"] = str(FIXTURES / "site-packages")

    result = _run_bootstrap(environment, process_environment, "megatron")

    assert result.returncode != 0
    assert f"No module named '{import_name}'" in result.stderr


def test_export_bootstrap_does_not_require_rollout_or_telemetry_packages(tmp_path: Path) -> None:
    environment, process_environment = fake_frozen_runtime(tmp_path)
    site_packages = next((environment / "lib").glob("python*/site-packages"))
    (site_packages / "shellbox/backends/daytona/machine.py").unlink()
    (site_packages / "memray.py").unlink()
    (site_packages / "rolloutengine/engine.py").unlink()
    shutil.rmtree(site_packages / "vllm")

    result = _run_bootstrap(environment, process_environment, "megatron-export")

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("publisher_installed", [False, True])
def test_hub_publish_bootstrap_checks_cpu_publisher_without_cuda(tmp_path: Path, publisher_installed: bool) -> None:
    environment, process_environment = fake_frozen_runtime(tmp_path)
    site_packages = next((environment / "lib").glob("python*/site-packages"))
    shutil.rmtree(site_packages / "nvidia")
    shutil.rmtree(site_packages / "flash_attn")
    (site_packages / "flash_attn_interface.py").unlink()
    if not publisher_installed:
        (site_packages / "skyrl_train/hf_publisher.py").unlink()

    project = tmp_path / "skyrl"
    scripts = project / "cloud/iris"
    scripts.mkdir(parents=True)
    for name in (
        "setup_task_runtime.sh",
        "bootstrap_runtime.sh",
        "load_runtime_environment.sh",
        "runtime_checks.py",
        "runtime_environment.py",
    ):
        (scripts / name).symlink_to(BOOTSTRAP_SCRIPT.with_name(name))
    for relative in (".uv-version", ".python-version", "marinskyrl/environment_contract.py"):
        target = project / relative
        target.parent.mkdir(exist_ok=True)
        target.symlink_to(REPOSITORY_ROOT / relative)
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
    environment, process_environment = fake_frozen_runtime(tmp_path, managed_python_pin_is_unresolvable=True)

    result = _run_bootstrap(environment, process_environment, "megatron")

    assert result.returncode == 0, result.stderr


def test_bootstrap_activation_exposes_runtime_commands(tmp_path: Path) -> None:
    environment, process_environment = fake_frozen_runtime(tmp_path)
    ninja = environment / "bin" / "ninja"
    ninja.touch(mode=0o755)
    for variable in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        process_environment[variable] = str(tmp_path / f"existing {variable} path")

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
    binary, libraries, linker, path = activation.stdout.splitlines()
    assert binary == str(ninja)
    assert path == f"{environment / 'bin'}{os.pathsep}{process_environment['PATH']}"
    site_packages = next((environment / "lib").glob("python*/site-packages"))
    assert libraries.split(os.pathsep) == [
        str(site_packages / "nvidia/cu13/lib"),
        process_environment["LD_LIBRARY_PATH"],
    ]
    assert linker.split(os.pathsep) == [str(environment / "lib"), process_environment["LIBRARY_PATH"]]
    runtime_file = environment / RUNTIME_FILE
    complete = runtime_file.read_bytes()
    runtime_file.write_bytes(complete[:-1])
    failed = subprocess.run(activation.args, env=process_environment, capture_output=True, text=True, timeout=5)
    assert failed.returncode != 0
    assert "Unterminated SkyRL runtime assignment" in failed.stderr
    runtime_file.write_bytes(complete.split(b"\0", 1)[0] + b"\0")
    partial = subprocess.run(activation.args, env=process_environment, capture_output=True, text=True, timeout=5)
    assert partial.returncode != 0
    assert "Incomplete SkyRL runtime assignments" in partial.stderr


def test_bootstrap_exposes_cuda_linker_compatibility_paths(tmp_path: Path) -> None:
    environment, process_environment = fake_frozen_runtime(tmp_path)
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
    environment, process_environment = fake_frozen_runtime(tmp_path)
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
    environment, process_environment = fake_frozen_runtime(tmp_path)
    site_packages = next((environment / "lib").glob("python*/site-packages"))
    for module in ("flash_attn_interface.py", "flash_attn/cute/interface.py"):
        shutil.copyfile(FIXTURES / "flash_attention_failure.py", site_packages / module)
    shutil.copyfile(FIXTURES / "megatron_failure.py", site_packages / "transformer_engine/common/__init__.py")
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
