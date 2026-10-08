# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind original Reasoning Gym source contracts to private native scorer runtimes."""

import asyncio
import json
import tomllib
from collections.abc import Callable
from dataclasses import replace
from enum import StrEnum
from functools import partial
from pathlib import Path

from shellbox.machine import MachineFactory, MachineSpec
from taskcompendium.datasets import reasoning_tasks
from taskcompendium.datasets.source_definitions import archive_resources
from taskcompendium.grader import GraderPackage, grader_config
from taskcompendium.models import FileReward, RewardFile, RewardFileFormat, ScriptGrader, TaskSpec, TextMessage
from taskcompendium.pipeline.execution_binding import (
    bind_grader_recipe,
    bound_grader_task,
    grading_environment,
    native_runtime_report,
)
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    DatasetRecipe,
    ImportFailureKind,
    ImportRejection,
    NormalizedTask,
    RawRow,
    VerificationReport,
)
from taskcompendium.pipeline.verification import control_result
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.task_curation.datasets.shared import grade_final_message

RUNNER = Path(__file__).with_name("runner.py")
GENERATED_REVISION = "49b07130b3fcd12f2d064bba7c43869543a0e7e7"
ULTRA_REVISION = "d8b6e8c163def3660e9d3072c1c174226a1709fa"
TASKTROVE_REVISION = "02923004846e4e73862c20962f823a6d05100e7a"


class ReasoningContract(StrEnum):
    GENERATED = "generated"
    TASKTROVE = "tasktrove"
    ULTRA = "ultra"


def tasktrove_grader(package_path: str, timeout: float, image: str) -> ScriptGrader:
    """Run the archived grader with the source package selected for python3."""
    return ScriptGrader(
        argv=("bash", "/tests/test.sh"),
        cwd="/",
        env={"PYTHONPATH": package_path},
        environment=grading_environment(image),
        reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),)),
        timeout=timeout,
    )


def normalize_isolated(
    row: RawRow,
    *,
    image: str,
    contract: ReasoningContract,
    package_path: str,
    normalize_task: Callable[[RawRow], TaskSpec | NormalizedTask | ImportRejection],
) -> TaskSpec | NormalizedTask | ImportRejection:
    result = normalize_task(row)
    if isinstance(result, ImportRejection):
        return result
    task = result.task if isinstance(result, NormalizedTask) else result
    config = grader_config(task)
    if contract == ReasoningContract.ULTRA:
        if (
            config["source_revision"] != ULTRA_REVISION
            or config["contract"]["agent_ref"]["name"] != "reasoning_gym_simple_agent"
        ):
            return ImportRejection(
                kind=ImportFailureKind.UNSUPPORTED,
                reason="unsupported_reasoning_contract",
                detail="Only the pinned original Ultra reasoning_gym_simple_agent is bound",
            )
        if task.interaction_tools or task.final_tools:
            return ImportRejection(
                kind=ImportFailureKind.UNSUPPORTED,
                reason="unsupported_reasoning_tools",
                detail="Original Reasoning Gym grading requires terminal text",
            )
    elif contract == ReasoningContract.GENERATED and row.source.revision != GENERATED_REVISION:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="unsupported_generator_revision",
            detail="Native regeneration requires the exact pinned generator",
        )
    elif contract == ReasoningContract.TASKTROVE:
        if row.data.get("archive_links"):
            return ImportRejection(
                kind=ImportFailureKind.UNSUPPORTED,
                reason="unsupported_reasoning_archive_links",
                detail="The source archive contains links that cannot be mounted as regular files",
            )
        required = ("tests/test.sh", "tests/verifier.py", "task.toml")
        if any(reasoning_tasks.snapshot_file(row, path) is None for path in required):
            return ImportRejection(
                kind=ImportFailureKind.UNSUPPORTED,
                reason="missing_original_reasoning_grader",
                detail="The source row lacks tests/test.sh, tests/verifier.py, or task.toml",
            )
        source_task = reasoning_tasks.snapshot_file(row, "task.toml")
        assert source_task is not None
        source_timeout = float(tomllib.loads(source_task.decode())["verifier"]["timeout_sec"])
        archive = archive_resources(row.data)
        task = bound_grader_task(
            task, GraderPackage(tasktrove_grader(package_path, source_timeout, image), archive.verifier)
        )
        task = task.model_copy(
            update={"resources": task.resources.model_copy(update={"worker": archive.worker, "oracle": archive.oracle})}
        )
        return replace(result, task=task) if isinstance(result, NormalizedTask) else task
    package = GraderPackage(
        ScriptGrader(
            argv=("python3", "/tests/grade.py"),
            cwd="/",
            env={"PYTHONPATH": package_path + ":/opt/skyrl_gym"},
            environment=grading_environment(image),
            reward=FileReward(files=(RewardFile(path="/logs/verifier/score.json", format=RewardFileFormat.JSON),)),
            timeout=60,
        ),
        (
            inline_resource("grade.py", RUNNER.read_bytes()),
            inline_resource(
                "reasoning_contract.json", json.dumps({"mode": contract.value, "contract": config["contract"]}).encode()
            ),
        ),
    )
    task = bound_grader_task(task, package)
    return replace(result, task=task) if isinstance(result, NormalizedTask) else task


async def isolated_checks(
    task: TaskSpec, *, contract: ReasoningContract, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
) -> VerificationReport:
    resource_name = "verifier_data.json" if contract == ReasoningContract.TASKTROVE else "reasoning_contract.json"
    source = json.loads(resource_bytes(next(item for item in task.resources.verifier if item.path == resource_name)))
    record = source if contract == ReasoningContract.TASKTROVE else source["contract"]
    recorded = record.get("recorded_pinned_generator_controls") if contract == ReasoningContract.GENERATED else None
    entry = record["entry"] if contract == ReasoningContract.GENERATED else record
    witness = recorded["positive"]["candidate"] if recorded else entry.get("answer")
    if not isinstance(witness, str):
        diagnostic = await grade_final_message(
            task, TextMessage(role="assistant", content="Runtime diagnostic response."), factory, machine_spec, timeout
        )
        return native_runtime_report(diagnostic, "No native passing witness is present in this source entry")
    positive = await grade_final_message(
        task, TextMessage(role="assistant", content=witness), factory, machine_spec, timeout
    )
    checks = [control_result(positive, "positive_witness", 1.0)]
    if recorded and recorded["negative"]["reward"] is not None:
        negative = recorded["negative"]
        result = await grade_final_message(
            task, TextMessage(role="assistant", content=negative["candidate"]), factory, machine_spec, timeout
        )
        check = control_result(result, "recorded_negative", negative["reward"])
        if negative["reward"] >= 1.0:
            check = CheckResult(check=check.check, status=CheckStatus.FAIL, detail="Recorded negative is accepted fully")
        checks.append(check)
    else:
        checks.append(
            CheckResult(
                check="negative_witness", status=CheckStatus.SKIPPED, detail="No scored negative witness is recorded"
            )
        )
    return VerificationReport(checks=checks)


def verification_report(
    task: TaskSpec, *, contract: ReasoningContract, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
):
    return asyncio.run(
        isolated_checks(task, contract=contract, factory=factory, machine_spec=machine_spec, timeout=timeout)
    )


def bind(
    recipe: DatasetRecipe,
    *,
    contract: ReasoningContract,
    package_path: str,
    image: str,
    factory: MachineFactory,
    machine_spec: MachineSpec,
    worker_image: str | None,
    timeout: float,
) -> DatasetRecipe:
    """Bind an explicit original source policy without changing review criteria."""
    revision = {
        ReasoningContract.GENERATED: GENERATED_REVISION,
        ReasoningContract.TASKTROVE: TASKTROVE_REVISION,
        ReasoningContract.ULTRA: ULTRA_REVISION,
    }[contract]
    return bind_grader_recipe(
        recipe,
        normalize=partial(
            normalize_isolated,
            image=image,
            contract=contract,
            package_path=package_path,
            normalize_task=recipe.policy.normalize,
        ),
        verification=partial(
            verification_report, contract=contract, factory=factory, machine_spec=machine_spec, timeout=timeout
        ),
        suite_id=f"original-{contract.value}-reasoning-gym-controls",
        grader_bytes=RUNNER.read_bytes()
        + (
            tasktrove_grader(package_path, timeout, image).model_dump_json().encode()
            if contract == ReasoningContract.TASKTROVE
            else package_path.encode()
        ),
        verifier_revision=revision,
        image=image,
        factory=factory,
        machine_spec=machine_spec,
        worker_image=worker_image,
        timeout=timeout,
    )
