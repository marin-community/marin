# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run pinned R2E-Gym repairs through the complete Harbor lifecycle."""

import gzip
import json
import shlex
import subprocess
from pathlib import Path

import pytest

from taskcompendium.execution import (
    HarborExecutionConfig,
    HarborLaunchConfig,
    HarborTaskBinding,
    HarnessToolBinding,
    environment_for_requirements,
)
from taskcompendium.harbor.runner import run_trial
from taskcompendium.importers.r2egym import SNAPSHOT_EXCLUSIONS, import_row, source_image
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import ContainerRuntime, FinalState, ImageOverlay, Rejected, Rendering

pytestmark = [pytest.mark.docker, pytest.mark.harbor_conformance]
FIXTURE = Path(__file__).parent / "fixtures/r2egym/rows.json.gz"
SYMPY_FIXTURE = Path(__file__).parent / "fixtures/r2egym/broadened_rows.json.gz"
RUNTIME_DOCKERFILE = Path(__file__).parents[1] / "src/taskcompendium/harbor/r2e_runtime.Dockerfile"


@pytest.fixture(scope="session")
def r2e_runtime_images() -> dict[int, str]:
    rows = json.loads(gzip.decompress(FIXTURE.read_bytes()))
    runtimes = {}
    for index, row in enumerate(rows[:2]):
        image = source_image(row)
        assert image is not None, f"row {index} has no pinned source image"
        tag = f"taskcompendium-r2e-runtime:{index}"
        subprocess.run(
            [
                "docker",
                "build",
                "-q",
                "--platform",
                "linux/amd64",
                "--build-arg",
                f"R2E_SOURCE_IMAGE={image}",
                "-t",
                tag,
                "-f",
                str(RUNTIME_DOCKERFILE),
                str(RUNTIME_DOCKERFILE.parent),
            ],
            check=True,
        )
        runtimes[index] = subprocess.check_output(
            ["docker", "image", "inspect", tag, "--format", "{{.Id}}"], text=True
        ).strip()
    return runtimes


def _repair_commands(row: dict, attempt: str) -> list[str]:
    if attempt == "empty":
        return ["rm -rf Orange"]
    if attempt == "bad":
        return []
    parsed = json.loads(row["parsed_commit_content"])
    commands = []
    for diff in parsed["file_diffs"]:
        path = diff["header"]["file"]["path"]
        commands.append(f"mkdir -p {shlex.quote(str(Path(path).parent))}")
        commands.append(f"printf '%s' {shlex.quote(diff['new_file_content'])} > {shlex.quote(path)}")
    return commands


@pytest.fixture(scope="session")
def sympy_runtime_image() -> str:
    row = json.loads(gzip.decompress(SYMPY_FIXTURE.read_bytes()))[0]
    image = source_image(row)
    assert image is not None
    tag = "taskcompendium-r2e-sympy500-runtime:validation"
    subprocess.run(
        [
            "docker",
            "build",
            "-q",
            "--platform",
            "linux/amd64",
            "--build-arg",
            f"R2E_SOURCE_IMAGE={image}",
            "-t",
            tag,
            "-f",
            str(RUNTIME_DOCKERFILE),
            str(RUNTIME_DOCKERFILE.parent),
        ],
        check=True,
    )
    return subprocess.check_output(["docker", "image", "inspect", tag, "--format", "{{.Id}}"], text=True).strip()


@pytest.mark.timeout(900)
@pytest.mark.parametrize("row_index", [0, 1])
@pytest.mark.parametrize("attempt,reward", [("good", 1.0), ("bad", 0.0), ("empty", 0.0)])
async def test_real_r2egym_row_through_harbor(tmp_path, r2e_runtime_images, row_index, attempt, reward):
    row = json.loads(gzip.decompress(FIXTURE.read_bytes()))[row_index]
    runtime = ContainerRuntime(
        r2e_runtime_images[row_index],
        timeout=300,
        workspace=ImageOverlay((".venv",)),
        supervisor_python="/usr/local/bin/python3",
    )
    specification = import_row(row, verifier_runtime=runtime)
    assert not isinstance(specification, Rejected), specification
    protocol = Rendering(
        specification.id,
        FinalState((".",), excluded_paths=SNAPSHOT_EXCLUSIONS),
    )
    binding = HarborTaskBinding(
        environment_for_requirements(specification.requirements),
        (HarnessToolBinding("terminal", "docker"),),
    )
    task = lower_to_harbor(
        specification,
        (protocol,),
        binding,
        tmp_path / "task",
        reference_execution=HarborExecutionConfig(binding, HarborLaunchConfig("replay")),
        agent_kwargs={"commands": _repair_commands(row, attempt)},
    )
    execution = json.loads((task / "reference-execution.json").read_text())
    result = await run_trial(task, execution, tmp_path / "trials", attempt)

    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": reward}


@pytest.mark.timeout(900)
@pytest.mark.parametrize("attempt,reward", [("good", 1.0), ("bad", 0.0)])
async def test_real_sympy_r2egym_row_through_harbor(tmp_path, sympy_runtime_image, attempt, reward):
    row = json.loads(gzip.decompress(SYMPY_FIXTURE.read_bytes()))[0]
    runtime = ContainerRuntime(
        sympy_runtime_image,
        timeout=300,
        workspace=ImageOverlay((".venv",)),
        supervisor_python="/usr/local/bin/python3",
    )
    specification = import_row(row, verifier_runtime=runtime)
    assert not isinstance(specification, Rejected), specification
    rendering = Rendering(
        specification.id,
        FinalState((".",), excluded_paths=SNAPSHOT_EXCLUSIONS),
    )
    binding = HarborTaskBinding(
        environment_for_requirements(specification.requirements),
        (HarnessToolBinding("terminal", "docker"),),
    )
    task = lower_to_harbor(
        specification,
        (rendering,),
        binding,
        tmp_path / "task",
        reference_execution=HarborExecutionConfig(binding, HarborLaunchConfig("replay")),
        agent_kwargs={"commands": _repair_commands(row, attempt)},
    )
    execution = json.loads((task / "reference-execution.json").read_text())
    result = await run_trial(task, execution, tmp_path / "trials", attempt)

    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": reward}
