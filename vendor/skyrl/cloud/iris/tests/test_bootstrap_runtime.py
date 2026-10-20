from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).parents[3]
BOOTSTRAP_SCRIPT = REPOSITORY_ROOT / "cloud" / "iris" / "bootstrap_runtime.sh"


def _write_module(site_packages: Path, relative_path: str, source: str = "") -> None:
    path = site_packages / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


def _fake_frozen_runtime(
    tmp_path: Path, *, managed_python_pin_is_unresolvable: bool = False
) -> tuple[Path, dict[str, str]]:
    environment = tmp_path / "runtime"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    site_packages = Path(
        subprocess.run(
            [environment / "bin" / "python", "-c", "import site; print(site.getsitepackages()[0])"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    for package in (
        "shellbox/backends/daytona",
        "shellbox/backends/shellsim",
        "rolloutengine",
        "megatron/bridge",
        "nvidia/cu13/lib",
        "quack",
        "skyrl_train/models",
        "transformer_engine/common",
        "vllm/model_executor",
    ):
        (site_packages / package).mkdir(parents=True, exist_ok=True)
    for package in (
        "shellbox",
        "shellbox/backends",
        "megatron",
        "shellbox/backends/daytona",
        "shellbox/backends/shellsim",
        "rolloutengine",
        "megatron/bridge",
        "quack",
        "skyrl_train",
        "skyrl_train/models",
        "transformer_engine",
        "transformer_engine/common",
        "vllm/model_executor",
    ):
        _write_module(site_packages, f"{package}/__init__.py")

    _write_module(site_packages, "quack/activation.py")
    _write_module(site_packages, "flash_attn.py", "__version__ = '2.8.4'\n")
    _write_module(site_packages, "flash_attn_2_cuda.py")
    _write_module(site_packages, "memray.py")
    _write_module(site_packages, "megatron/bridge/__init__.py", "class AutoBridge: pass\n")
    _write_module(site_packages, "torch.py", "__version__ = '2.13.0+cu132'\n")
    _write_module(site_packages, "vllm/__init__.py", "__version__ = 'test'\n")
    _write_module(site_packages, "vllm/_C_stable_libtorch.py")
    _write_module(site_packages, "vllm/cumem_allocator.py")
    _write_module(
        site_packages,
        "transformer_engine/common/__init__.py",
        "import os\nfrom pathlib import Path\nassert (Path(os.environ['NVRTC_HOME']) / 'lib').is_dir()\n",
    )
    _write_module(
        site_packages,
        "vllm/model_executor/models.py",
        "class ModelRegistry:\n"
        "    @staticmethod\n"
        "    def get_supported_archs():\n"
        "        return {'GrugMoeForCausalLM'}\n",
    )
    _write_module(
        site_packages,
        "skyrl_train/models/grug_moe.py",
        "GRUG_MOE_ARCHITECTURE = 'GrugMoeForCausalLM'\n",
    )
    _write_module(site_packages, "shellbox/backends/daytona/machine.py", "class DaytonaMachineFactory: pass\n")
    _write_module(site_packages, "shellbox/backends/shellsim/machine.py", "class ShellSimMachineFactory: pass\n")
    _write_module(site_packages, "rolloutengine/engine.py", "class RolloutEngine: pass\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "python3.12").symlink_to(sys.executable)
    uv = fake_bin / "uv"
    if managed_python_pin_is_unresolvable:
        uv.write_text(
            "#!/bin/sh\n"
            "selected_python=\n"
            "downloads_disabled=false\n"
            'while [ "$#" -gt 0 ]; do\n'
            '  case "$1" in\n'
            "    --python) selected_python=$2; shift 2 ;;\n"
            "    --no-python-downloads) downloads_disabled=true; shift ;;\n"
            "    *) shift ;;\n"
            "  esac\n"
            "done\n"
            'if [ "$selected_python" = "$EXPECTED_SYSTEM_PYTHON" ] && "$downloads_disabled"; then\n'
            "  exit 0\n"
            "fi\n"
            "echo 'error: No interpreter found for Python 3.12.13' >&2\n"
            "exit 2\n"
        )
    else:
        uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    return environment, os.environ | {
        "EXPECTED_SYSTEM_PYTHON": str(fake_bin / "python3.12"),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PYTHONPATH": "",
    }


def _run_bootstrap(
    environment: Path, process_environment: dict[str, str], profile: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(BOOTSTRAP_SCRIPT),
            str(REPOSITORY_ROOT),
            str(environment),
            str(environment / "runtime.sh"),
            profile,
            "production",
        ],
        cwd=REPOSITORY_ROOT,
        env=process_environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
def test_policy_bootstrap_rejects_runtime_without_flash_attention_extension(tmp_path: Path, architecture: str) -> None:
    environment, process_environment = _fake_frozen_runtime(tmp_path)
    site_packages = next((environment / "lib").glob("python*/site-packages"))
    (site_packages / "flash_attn_2_cuda.py").unlink()
    _write_module(site_packages, "sitecustomize.py", f"import platform\nplatform.machine = lambda: {architecture!r}\n")

    result = _run_bootstrap(environment, process_environment, "megatron")

    assert result.returncode != 0
    assert "No module named 'flash_attn_2_cuda'" in result.stderr


def test_export_bootstrap_does_not_require_rollout_or_telemetry_packages(tmp_path: Path) -> None:
    environment, process_environment = _fake_frozen_runtime(tmp_path)
    site_packages = next((environment / "lib").glob("python*/site-packages"))
    _write_module(site_packages, "flash_attn_2_cuda.py")
    _write_module(site_packages, "ray.py")
    _write_module(site_packages, "skyrl_train/checkpoint_exporter.py", "class CheckpointExporter: pass\n")
    (site_packages / "shellbox/backends/daytona/machine.py").unlink()
    (site_packages / "memray.py").unlink()
    (site_packages / "rolloutengine/engine.py").unlink()

    result = _run_bootstrap(environment, process_environment, "megatron-export")

    assert result.returncode == 0, result.stderr


def test_bootstrap_uses_system_python_when_managed_pin_is_unresolvable(tmp_path: Path) -> None:
    environment, process_environment = _fake_frozen_runtime(tmp_path, managed_python_pin_is_unresolvable=True)

    result = _run_bootstrap(environment, process_environment, "megatron")

    assert result.returncode == 0, result.stderr


def test_bootstrap_activation_exposes_runtime_commands(tmp_path: Path) -> None:
    environment, process_environment = _fake_frozen_runtime(tmp_path)
    ninja = environment / "bin" / "ninja"
    ninja.write_text("#!/bin/sh\nexit 0\n")
    ninja.chmod(0o755)

    result = _run_bootstrap(environment, process_environment, "megatron")
    activation = subprocess.run(
        ["bash", "-c", 'source "$1"; command -v ninja', "bash", environment / "runtime.sh"],
        env=process_environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0, result.stderr
    assert activation.stdout.strip() == str(ninja)


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
