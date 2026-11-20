from __future__ import annotations

import os
import json
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
    _write_module(site_packages, "flash_attn/__init__.py")
    _write_module(site_packages, "flash_attn/cute/__init__.py")
    _write_module(site_packages, "flash_attn/cute/interface.py", "flash_attn_varlen_func = object()\n")
    _write_module(site_packages, "flash_attn_interface.py", "flash_attn_varlen_func = object()\n")
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
    _write_module(
        site_packages,
        "skyrl_train/workers/levanter/levanter_worker.py",
        "class LevanterRayActorGroup: pass\n",
    )

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
    }


def _run_bootstrap(
    environment: Path,
    process_environment: dict[str, str],
    profile: str,
    *,
    project_root: Path = REPOSITORY_ROOT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(BOOTSTRAP_SCRIPT),
            str(project_root),
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
        timeout=30,
    )


def test_levanter_bootstrap_keeps_torch_and_jax_environments_separate(tmp_path: Path) -> None:
    environment, process_environment = _fake_frozen_runtime(tmp_path)
    workspace = tmp_path / "workspace"
    project = workspace / "vendor" / "skyrl"
    (project / "marinskyrl").mkdir(parents=True)
    shutil.copy2(REPOSITORY_ROOT / ".uv-version", project / ".uv-version")
    shutil.copy2(
        REPOSITORY_ROOT / "marinskyrl/environment_contract.py", project / "marinskyrl/environment_contract.py"
    )
    child = workspace / ".levanter-policy-env"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", child], check=True)
    site_packages = next((child / "lib").glob("python*/site-packages"))
    receipt = tmp_path / "jax-runtime.json"
    process_environment.update(
        JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu", JAX_CUDA_VISIBLE_DEVICES="7", CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    )
    # The bootstrap imports each runtime in its actual venv. The child checks
    # toolchain isolation while the parent still needs its frozen CUDA paths.
    _write_module(
        site_packages,
        "jax.py",
        "import json, os, sys\nfrom pathlib import Path\n"
        f"Path({str(receipt)!r}).write_text(json.dumps({{'prefix': sys.prefix, "
        "'inherited_cuda_paths': [name for name in ('CUDA_HOME', 'NVRTC_HOME', 'LIBRARY_PATH', "
        "'LD_LIBRARY_PATH', 'JAX_PLATFORM_NAME', 'JAX_CUDA_VISIBLE_DEVICES') if name in os.environ], "
        "'jax_platforms': os.environ.get('JAX_PLATFORMS'), 'gpu_visibility': os.environ.get('CUDA_VISIBLE_DEVICES')}))\n"
        "__version__ = 'test'\n"
        "def default_backend(): return 'cpu' if os.environ.get('JAX_PLATFORMS') == 'cpu' else 'gpu'\n"
        "def local_devices(): return []\n",
    )
    _write_module(site_packages, "marin/__init__.py")
    _write_module(site_packages, "marin/rl/__init__.py")
    _write_module(site_packages, "marin/rl/levanter_worker_runtime.py", "def main(): pass\n")
    _write_module(site_packages, "levanter/__init__.py")
    _write_module(site_packages, "levanter/policy_learner.py", "class SpmdPolicyLearner: pass\n")

    result = _run_bootstrap(environment, process_environment, "levanter", project_root=project)

    assert result.returncode == 0, result.stderr
    assert json.loads(receipt.read_text()) == {
        "prefix": str(child), "inherited_cuda_paths": [], "jax_platforms": "cuda,cpu", "gpu_visibility": "0,1,2,3,4,5,6,7"
    }
    activation = subprocess.run(
        ["bash", "-c", 'source "$1"; "$2" -c "import os; assert os.path.isdir(os.environ[\'CUDA_HOME\'])"',
         "bash", environment / "runtime.sh", environment / "bin" / "python"],
        env=process_environment,
        capture_output=True,
        text=True,
    )
    assert activation.returncode == 0, activation.stderr


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
    _write_module(site_packages, "sitecustomize.py", f"import platform\nplatform.machine = lambda: {architecture!r}\n")

    result = _run_bootstrap(environment, process_environment, "megatron")

    assert result.returncode != 0
    assert f"No module named '{import_name}'" in result.stderr


def test_export_bootstrap_does_not_require_rollout_or_telemetry_packages(tmp_path: Path) -> None:
    environment, process_environment = _fake_frozen_runtime(tmp_path)
    site_packages = next((environment / "lib").glob("python*/site-packages"))
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


def test_parallel_bootstrap_waits_for_every_check_and_reports_failure(tmp_path):
    environment, process_environment = _fake_frozen_runtime(tmp_path)
    site_packages = next((environment / "lib").glob("python*/site-packages"))
    barrier = tmp_path / "imports-ready"
    os.mkfifo(barrier)
    transformer_engine = site_packages / "transformer_engine/common/__init__.py"
    transformer_engine.write_text(
        transformer_engine.read_text()
        + "import select\n"
        + f"with open(os.open({str(barrier)!r}, os.O_RDONLY | os.O_NONBLOCK), 'rb', buffering=0) as gate:\n"
        + "    assert select.select([gate], [], [], 8)[0], 'concurrent import never started'\n"
        + "    assert gate.read(1) == b'r'\n"
        + "raise RuntimeError('megatron import failed')\n"
    )
    for module in ("flash_attn_interface.py", "flash_attn/cute/interface.py"):
        _write_module(
            site_packages,
            module,
            f"with open({str(barrier)!r}, 'wb', buffering=0) as gate:\n    gate.write(b'r')\n"
            + "raise RuntimeError('flash-attention import failed')\n",
        )
    descriptor = os.open(barrier, os.O_RDWR | os.O_NONBLOCK)
    try:
        result = _run_bootstrap(environment, process_environment, "megatron")
    finally:
        os.close(descriptor)
    assert result.returncode != 0, result.stderr
    assert "flash-attention import failed" in result.stderr
    assert "megatron import failed" in result.stderr
