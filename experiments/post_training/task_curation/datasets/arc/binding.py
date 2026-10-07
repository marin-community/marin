# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind preserved TaskTrove and Ultra ARC scorers to private Shellbox machines."""

import asyncio
import hashlib
import json
import tomllib
from collections.abc import Callable
from dataclasses import replace
from enum import StrEnum
from functools import partial
from pathlib import Path

from shellbox.machine import Backend, MachineFactory, MachineSpec
from taskcompendium.datasets.executable_tasks import SubmissionControl, executable_checks
from taskcompendium.datasets.nemotron_ultra.normalization import VERIFIER_REVISION
from taskcompendium.datasets.reasoning_tasks import snapshot_file
from taskcompendium.datasets.source_definitions import archive_resources
from taskcompendium.grader import GraderPackage, grader_config, native_command_package
from taskcompendium.models import AnswerType, EnvironmentRequirements, TaskSpec
from taskcompendium.native_grader import NativeCommandSpec
from taskcompendium.pipeline.execution_binding import bound_grader_task
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    DatasetRecipe,
    ImportFailureKind,
    ImportRejection,
    NormalizedTask,
    RawRow,
    VerificationReport,
)
from taskcompendium.pipeline.verification import control_result
from taskcompendium.runtime.grading import grade_submission
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from taskcompendium.runtime.shell import machine_spec_identity
from verifyit.execution import source_callable

RUNNER = Path(__file__).with_name("runner.py")
SOURCE_CALLABLE = Path(source_callable.__file__)
BACKENDS = (Backend.GVISOR, Backend.QEMU)
TASKTROVE_PIN = ("open-thoughts/TaskTrove", "02923004846e4e73862c20962f823a6d05100e7a")
ULTRA_PIN = ("nvidia/Nemotron-RL-Ultra-Training-Blends", "482392c14c6418e26804ea2e5d10359df9877df4")
ANSWER_PATH = "/app/answer.txt"
AGENTS = ("nvarc_inductive_simple_agent", "nvarc_transductive_simple_agent")


class ArcSource(StrEnum):
    TASKTROVE = "tasktrove"
    ULTRA = "ultra"


def original_package(config: dict) -> GraderPackage:
    """Declare the image-installed SkyRL scorer and official NeMo Skills server."""
    if config["contract"]["agent_ref"]["name"] == AGENTS[1]:
        result_path = "/logs/verifier/reward.json"
        return native_command_package(
            NativeCommandSpec(
                argv=(
                    "python3",
                    "/tests/source_callable.py",
                    "/tests/invocation.json",
                    "/tests/arc_contract.json",
                    ANSWER_PATH,
                    result_path,
                ),
                cwd="/",
                env={},
                result_format="score_json",
                result_path=result_path,
                timeout=45,
            ),
            (
                inline_resource("source_callable.py", SOURCE_CALLABLE.read_bytes()),
                inline_resource(
                    "invocation.json",
                    json.dumps(
                        {
                            "function": "skyrl_gym.envs.nemotron_ultra.nvarc:grade_transductive_arc",
                            "args": ["answer", "contract"],
                        },
                        allow_nan=False,
                    ).encode(),
                ),
                inline_resource("arc_contract.json", json.dumps(config, allow_nan=False).encode()),
            ),
        )
    return native_command_package(
        NativeCommandSpec(
            argv=("python3", "/tests/grade.py"),
            cwd="/",
            env={},
            result_format="score_json",
            result_path="/logs/verifier/reward.json",
            timeout=45,
        ),
        (
            inline_resource("grade.py", RUNNER.read_bytes()),
            inline_resource("arc_contract.json", json.dumps(config, allow_nan=False).encode()),
        ),
    )


def normalize_isolated(
    row: RawRow,
    *,
    normalize_task: Callable[[RawRow], TaskSpec | NormalizedTask | ImportRejection],
    image: str,
    source: ArcSource,
) -> TaskSpec | NormalizedTask | ImportRejection:
    """Preserve public requests and original evidence while binding the private scorer."""
    tasktrove = source == ArcSource.TASKTROVE
    pin = TASKTROVE_PIN if tasktrove else ULTRA_PIN
    if (row.source.dataset, row.source.revision) != pin:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="unsupported_arc_grader_source_revision",
            detail="Only the declared pinned ARC source is bound",
        )
    if not tasktrove:
        result = normalize_task(row)
        if isinstance(result, ImportRejection):
            return result
        task = result.task if isinstance(result, NormalizedTask) else result
        config = grader_config(task)
        if config["source_revision"] != VERIFIER_REVISION or config["contract"]["agent_ref"]["name"] not in AGENTS:
            return ImportRejection(
                kind=ImportFailureKind.UNSUPPORTED,
                reason="unsupported_native_text_agent",
                detail=f"Only the pinned {AGENTS} evaluators are bound",
            )
        if task.interaction_tools or task.final_tools:
            return ImportRejection(
                kind=ImportFailureKind.UNSUPPORTED,
                reason="unsupported_native_text_tool_request",
                detail="This original stateless evaluator accepts terminal text only",
            )
        package = original_package(config)
        task = bound_grader_task(task, package=package, image=image).model_copy(update={"answer_type": AnswerType.TEXT})
        task = task.model_copy(
            update={
                "verifier": task.verifier.model_copy(
                    update={
                        "environment_requirements": EnvironmentRequirements(
                            docker_image=image, compatible_backends=BACKENDS
                        )
                    }
                )
            }
        )
        return replace(result, task=task) if isinstance(result, NormalizedTask) else task
    result = normalize_task(row)
    if isinstance(result, ImportRejection):
        return result
    task = result.task if isinstance(result, NormalizedTask) else result
    files = row.data.get("files")
    if (
        row.data.get("archive_links")
        or not isinstance(files, dict)
        or any(path not in files for path in ("tests/test.sh", "tests/verifier.py", "task.toml"))
    ):
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="missing_original_arc_command",
            detail="The source row needs tests/test.sh, tests/verifier.py, task.toml and no archive links",
        )
    archive = archive_resources(row.data)
    source_task = snapshot_file(row, "task.toml")
    assert source_task is not None
    source_timeout = float(tomllib.loads(source_task.decode())["verifier"]["timeout_sec"])
    package = native_command_package(
        NativeCommandSpec(
            argv=("bash", "/tests/test.sh"),
            cwd="/",
            env={},
            result_format="reward_file",
            result_path="/logs/verifier/reward.txt",
            timeout=source_timeout,
        ),
        archive.verifier,
    )
    task = task.model_copy(
        update={
            "verifier": package.verifier.model_copy(
                update={
                    "environment_requirements": EnvironmentRequirements(docker_image=image, compatible_backends=BACKENDS)
                }
            ),
            "resources": task.resources.model_copy(
                update={
                    "worker": archive.worker,
                    "verifier": package.resources,
                    "oracle": archive.oracle,
                }
            ),
        }
    )
    return replace(result, task=task) if isinstance(result, NormalizedTask) else task


async def isolated_checks(
    task: TaskSpec, *, source: ArcSource, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
) -> VerificationReport:
    if source == ArcSource.TASKTROVE:
        inductive = "/app/solution.py" in task.output_paths
        output = "/app/solution.py" if inductive else ANSWER_PATH
        wrong = (
            b"def transform(grid):\n    raise RuntimeError('__negative_control__')\n"
            if inductive
            else b"__incorrect_grid__\n"
        )
        return await executable_checks(
            task,
            factory=factory,
            machine_spec=machine_spec,
            timeout=timeout,
            controls=(
                SubmissionControl("missing_submission", {}, 0.0),
                SubmissionControl("empty_submission", {output: b""}, 0.0),
                SubmissionControl("wrong_submission", {output: wrong}, 0.0),
            ),
        )
    contract = json.loads(
        resource_bytes(next(resource for resource in task.resources.verifier if resource.path == "arc_contract.json"))
    )["contract"]
    inductive = contract["agent_ref"]["name"] == AGENTS[0]
    controls = [("empty", "", 0.0)]
    if inductive:
        controls.append(
            (
                "native_runtime",
                "```python\ndef transform(grid):\n    raise RuntimeError('__negative_control__')\n```",
                0.0,
            )
        )
    else:
        grid = contract["expected_output"]
        witness = "\n".join(" ".join(str(cell) for cell in row) for row in grid)
        wrong = [row[:] for row in grid]
        wrong[0][0] = (wrong[0][0] + 1) % 10
        controls.extend(
            [
                ("reference", witness, 1.0),
                ("wrong_grid", "\n".join(" ".join(str(cell) for cell in row) for row in wrong), 0.0),
            ]
        )
    checks = []
    for name, answer, expected in controls:
        result = await grade_submission(
            task, {ANSWER_PATH: answer.encode()}, factory, machine_spec=machine_spec, timeout=timeout
        )
        check = control_result(result, name, expected)
        if result.error:
            check = check.model_copy(update={"detail": check.detail + "; " + result.error})
        checks.append(check)
    if inductive:
        checks.append(
            CheckResult(
                check="positive_witness",
                status=CheckStatus.SKIPPED,
                detail="Source supplies held-out grids but no golden transform program",
            )
        )
    return VerificationReport(checks)


def isolated_verification(
    task: TaskSpec, *, source: ArcSource, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
) -> VerificationReport:
    return asyncio.run(isolated_checks(task, source=source, factory=factory, machine_spec=machine_spec, timeout=timeout))


def bind(
    recipe: DatasetRecipe,
    *,
    image: str,
    factory: MachineFactory,
    machine_spec: MachineSpec,
    worker_image: str | None,
    timeout: float,
    source: ArcSource,
) -> DatasetRecipe:
    """Bind original source scoring and fresh negative controls without inventing goldens."""
    if factory.backend not in BACKENDS:
        raise ValueError("Original ARC execution requires Shellbox QEMU or gVisor")
    tasktrove = source == ArcSource.TASKTROVE
    if tasktrove:
        machine_spec = replace(machine_spec, cpus=1, memory_mb=4096)
    machine = machine_spec_identity(machine_spec)
    suite = CheckSuite(
        id="original-native-arc-controls",
        revision=TASKTROVE_PIN[1] if tasktrove else VERIFIER_REVISION,
        parameters={
            "image": image,
            "backend": factory.backend.value,
            "machine": machine,
            "worker_image": worker_image,
            "timeout": timeout,
            "source_environment": {"cpus": 1, "memory_mb": 4096, "storage_mb": 10240} if tasktrove else {},
            "storage_budget_enforced": False if tasktrove else None,
            "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
            "source_callable_sha256": hashlib.sha256(SOURCE_CALLABLE.read_bytes()).hexdigest(),
        },
        run=partial(isolated_verification, source=source, factory=factory, machine_spec=machine_spec, timeout=timeout),
    )
    return replace(
        recipe,
        policy=replace(
            recipe.policy,
            normalize=partial(normalize_isolated, normalize_task=recipe.policy.normalize, image=image, source=source),
            check_suite=suite,
        ),
    )
