from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).parents[3]
BOOTSTRAP_SCRIPT = REPOSITORY_ROOT / "cloud" / "iris" / "bootstrap_runtime.sh"
RUNTIME_FILE = "runtime.env"
FIXTURES = Path(__file__).with_name("bootstrap_fixtures")


def fake_frozen_runtime(
    tmp_path: Path, *, managed_python_pin_is_unresolvable: bool = False
) -> tuple[Path, dict[str, str]]:
    environment = tmp_path / "runtime"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    site_packages = next((environment / "lib").glob("python*/site-packages"))

    (site_packages / "nvidia/cu13/lib").mkdir(parents=True)
    shutil.copytree(FIXTURES / "site-packages", site_packages, dirs_exist_ok=True)

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

    result = subprocess.run(
        ["bash", BOOTSTRAP_SCRIPT.with_name("setup_task_runtime.sh"), "hub-publish", "printenv", "PATH"],
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
    runtime_file.write_bytes(b"replace:EUID=1\0" + complete)
    rejected = subprocess.run(activation.args, env=process_environment, capture_output=True, text=True, timeout=5)
    assert rejected.returncode != 0
    assert "EUID: readonly variable" in rejected.stderr


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


def test_node_entrypoint_checks_transport_before_starting_controller(tmp_path):
    app = tmp_path / "app"
    skyrl = app / "vendor/skyrl"
    launcher = skyrl / "cloud/iris"
    launcher.mkdir(parents=True)
    for name in ("node_entrypoint.sh", "load_runtime_environment.sh", "bootstrap_runtime.sh",
                 "runtime_checks.py", "setup_task_runtime.sh"):
        (launcher / name).symlink_to(BOOTSTRAP_SCRIPT.with_name(name))
    for relative in (".uv-version", ".python-version", "marinskyrl/environment_contract.py"):
        target = skyrl / relative
        target.parent.mkdir(exist_ok=True)
        target.symlink_to(REPOSITORY_ROOT / relative)
    stub = Path(__file__).with_name("node_command_fixture.py")
    (launcher / "task_runtime.py").symlink_to(stub)
    probe = app / "experiments/post_training/nccl_transport_probe.py"
    probe.parent.mkdir(parents=True)
    probe.symlink_to(stub)
    runtime, environment = fake_frozen_runtime(tmp_path)
    environment["IRIS_VENV"] = str(runtime)
    subprocess.run(["bash", launcher / "setup_task_runtime.sh", "megatron"], env=environment, check=True, timeout=30)

    command = ["bash", launcher / "node_entrypoint.sh", "launch.yaml", "s3://test/interconnect-gate", "--pynccl-init"]
    result = subprocess.run(command, env=environment, check=True, capture_output=True, text=True, timeout=15)
    transport, controller = map(json.loads, result.stdout.splitlines())
    for record in (transport, controller):
        assert record["cwd"] == record["skyrl_home"] == str(skyrl)
        assert record["max_jobs"] == "8" and record["vllm_use_v1"] == "1"
        assert record["python_path"].split(os.pathsep)[:3] == [str(app), str(skyrl), str(skyrl / "skyrl-train")]
    assert transport["argv"] == ["--mode", "gate", "--gate-coverage", "all-gpus", "--gate-report-prefix",
                                 "s3://test/interconnect-gate", "--pynccl-init"]
    assert controller["argv"] == ["--config", "launch.yaml", "--environment-prepared"]
    failed = subprocess.run(command, env=environment | {"EXIT_TRANSPORT_FIXTURE": "1"},
                            capture_output=True, text=True, timeout=15)
    assert failed.returncode == 19 and len(failed.stdout.splitlines()) == 1
