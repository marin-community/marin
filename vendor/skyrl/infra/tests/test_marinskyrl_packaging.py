"""Contracts for the root ``marinskyrl`` distribution."""

from __future__ import annotations

import subprocess
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path

from packaging.markers import Marker
from packaging.requirements import Requirement
import pytest


REPOSITORY_ROOT = Path(__file__).parents[2]


@dataclass(frozen=True)
class BuiltWheel:
    names: set[str]
    metadata: str
    entry_points: str


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> BuiltWheel:
    output = tmp_path_factory.mktemp("marinskyrl-wheel")
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(output)], cwd=REPOSITORY_ROOT, check=True)
    wheel = next(output.glob("marinskyrl-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        entry_points_name = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        metadata = archive.read(metadata_name).decode()
        entry_points = archive.read(entry_points_name).decode()
    return BuiltWheel(names=names, metadata=metadata, entry_points=entry_points)


def test_root_wheel_owns_launcher_and_training_packages(built_wheel: BuiltWheel) -> None:
    assert Parser().parsestr(built_wheel.metadata)["Name"] == "marinskyrl"
    assert "marinskyrl = cloud.iris.launch:main" in built_wheel.entry_points
    assert "cloud/iris/launch.py" in built_wheel.names
    assert "cloud/iris/runtime_bundle_files.txt" in built_wheel.names
    assert "chat_templates/delphi_v0.jinja2" in built_wheel.names
    assert "skyrl_gym/__init__.py" in built_wheel.names
    assert "skyrl_train/__init__.py" in built_wheel.names
    assert "skyrl_train/config/ppo_base_config.yaml" in built_wheel.names
    assert "skyrl_train/inline_harbor.py" in built_wheel.names
    assert "skyrl_gym/envs/nemotron_ultra/abstention_prompt.txt" in built_wheel.names
    assert "skyrl_gym/envs/nemotron_ultra/jailbreak_verifiers.yaml" in built_wheel.names


def test_base_dependencies_are_cpu_only(built_wheel: BuiltWheel) -> None:
    requirements = Parser().parsestr(built_wheel.metadata).get_all("Requires-Dist", [])
    base_requirements = {requirement.partition(";")[0].strip().split("[")[0].split()[0].lower() for requirement in requirements if "extra ==" not in requirement}

    assert base_requirements.isdisjoint({"flash-attn", "torch", "transformer-engine", "vllm", "harbor", "harbor-config"})


def test_training_extras_publish_hardware_policy_and_rollout_requirements(built_wheel: BuiltWheel) -> None:
    metadata = Parser().parsestr(built_wheel.metadata)
    extras = set(metadata.get_all("Provides-Extra", []))
    requirements = metadata.get_all("Requires-Dist", [])

    assert {"cpu", "cuda", "vllm", "megatron", "telemetry"}.issubset(extras)
    assert "agentic" not in extras
    assert any(requirement.startswith("torch==") and "extra == 'cpu'" in requirement for requirement in requirements)
    assert any(requirement.startswith("torch==") and "extra == 'cuda'" in requirement for requirement in requirements)
    assert any(requirement.startswith("vllm==") and "extra == 'vllm'" in requirement for requirement in requirements)
    base_requirements = [Requirement(value) for value in requirements if "extra ==" not in value]
    assert {"marin-rolloutengine", "marin-shellbox", "marin-taskcompendium", "marin-verifyit"} <= {
        requirement.name for requirement in base_requirements
    }
    task_packages = {"marin-rolloutengine", "marin-shellbox", "marin-taskcompendium", "marin-verifyit"}
    for requirement in base_requirements:
        if requirement.name in task_packages:
            assert requirement.url is None, f"Task runtime must resolve {requirement.name} from PyPI"
        elif requirement.name.startswith("marin-"):
            assert requirement.url is None
    assert any(requirement.startswith("torch==") and "extra == 'vllm'" in requirement for requirement in requirements)
    assert any(requirement.startswith("memray") and "extra == 'telemetry'" in requirement for requirement in requirements)
    assert any(
        requirement.startswith("torchvision==") and "extra == 'megatron'" in requirement
        for requirement in requirements
    )
    assert any(
        requirement.startswith("megatron-core") and "extra == 'megatron'" in requirement
        for requirement in requirements
    )

    conflict = subprocess.run(
        ["uv", "sync", "--frozen", "--dry-run", "--extra", "cpu", "--extra", "cuda"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert conflict.returncode != 0
    assert "Extras `cpu` and `cuda` are incompatible" in conflict.stderr


def test_rollout_runtime_exports_vendored_harbor_for_linux_training() -> None:
    platform = {"sys_platform": "linux", "platform_machine": "x86_64"}
    harbor = next(line for line in _exported_lines(("megatron", "vllm")) if line.startswith("./harbor ;"))
    assert Marker(harbor.partition(";")[2].strip()).evaluate(platform)
    assert (REPOSITORY_ROOT / "harbor").resolve() == (REPOSITORY_ROOT.parent / "harbor").resolve()


def _exported_lines(extras: tuple[str, ...]) -> list[str]:
    command = ["uv", "export", "--frozen", "--no-annotate", "--no-dev", "--no-hashes"]
    for extra in extras:
        command.extend(("--extra", extra))
    exported = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return [line for line in exported if line and not line.startswith(("#", "-e "))]


def _exported_requirements(extras: tuple[str, ...]) -> list[Requirement]:
    return [Requirement(line) for line in _exported_lines(extras) if not line.startswith("./")]


@pytest.mark.parametrize("extras", [("cuda",), ("megatron", "vllm")])
def test_gpu_profiles_use_one_cuda132_runtime(extras: tuple[str, ...]) -> None:
    platform = {"sys_platform": "linux", "platform_machine": "x86_64"}
    requirements = _exported_requirements(extras)
    runtimes = {
        (requirement.name, str(requirement.specifier))
        for requirement in requirements
        if requirement.name.startswith("nvidia-cuda-runtime")
        and (requirement.marker is None or requirement.marker.evaluate(platform))
    }

    assert runtimes == {("nvidia-cuda-runtime", "==13.2.75")}


@pytest.mark.parametrize(
    ("extras", "architecture", "required", "forbidden"),
    [
        (
            ("megatron", "vllm"),
            "x86_64",
            {"causal-conv1d", "flash-attn", "mamba-ssm", "megatron-core", "transformer-engine-torch"},
            {"fast-hadamard-transform"},
        ),
    ],
)
def test_policy_closures_match_supported_architectures(
    extras: tuple[str, ...], architecture: str, required: set[str], forbidden: set[str]
) -> None:
    platform = {"sys_platform": "linux", "platform_machine": architecture}
    exported = [
        requirement
        for requirement in _exported_requirements(extras)
        if requirement.marker is None or requirement.marker.evaluate(platform)
    ]
    names = {requirement.name for requirement in exported}

    assert {"vllm", "torch", "torchaudio", "flashinfer-python", "nvidia-cuda-nvcc", *required}.issubset(names)
    assert names.isdisjoint(forbidden)
    vllm = next(requirement for requirement in exported if requirement.name == "vllm")
    assert vllm.url is not None and vllm.url.endswith(f"manylinux_2_28_{architecture}.whl")
