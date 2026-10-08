# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the pinned SkyRL IFEval predicates without changing their fractional reward."""

import asyncio
import json
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from shellbox.machine import MachineFactory, MachineSpec
from taskcompendium.datasets.skyrl_ifeval_mapping import nemotron_constraint
from taskcompendium.grader import GraderPackage, grader_config
from taskcompendium.grading_result import Outcome
from taskcompendium.models import FileReward, RewardFile, RewardFileFormat, ScriptGrader, TaskSpec, TextMessage
from taskcompendium.pipeline.execution_binding import grading_environment
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    DatasetRecipe,
    ImportFailureKind,
    ImportRejection,
    RawRow,
    VerificationReport,
)
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.shell import machine_spec_identity
from verifyit.execution import source_callable

from experiments.post_training.task_curation.datasets.shared import grade_final_message

UPSTREAM_REVISION = "544d5d6f14116a06bde0209352585903133bd618"
EXECUTION_REVISION = "3-native-image"
LANGUAGE_DEPENDENCY = "langdetect==1.0.9"
UTILS_SHA256 = "3194b7a44ada0a4cd185ab3c85d883f85da8641970fef3dfc4b66d13780079fc"
SCORER_PATH = "/opt/skyrl_gym/skyrl_gym/envs/ifeval/utils.py"
ANSWER_PATH = "/app/answer.txt"
REWARD_PATH = "/logs/verifier/reward.json"
SOURCE_CALLABLE = Path(source_callable.__file__)


def grader_package(constraints: dict | list[dict], input_format: str, image: str) -> GraderPackage:
    """Run the installed upstream scorer with normalized private constraints."""
    config = {
        "constraints": constraints,
        "input_format": input_format,
        "upstream_revision": UPSTREAM_REVISION,
        "execution_revision": EXECUTION_REVISION,
        "language_dependency": LANGUAGE_DEPENDENCY,
        "scorer_path": SCORER_PATH,
        "scorer_sha256": UTILS_SHA256,
    }
    invocation = {
        "function": "pinned_skyrl_ifeval:compute_score",
        "source_path": SCORER_PATH,
        "source_sha256": UTILS_SHA256,
        "args": ["answer", "contract.constraints"],
        "reward_key": "score",
    }
    return GraderPackage(
        ScriptGrader(
            argv=(
                "python3",
                "/tests/source_callable.py",
                "/tests/invocation.json",
                "/tests/config.json",
                ANSWER_PATH,
                REWARD_PATH,
            ),
            cwd="/",
            environment=grading_environment(image),
            answer_path=ANSWER_PATH,
            reward=FileReward(files=(RewardFile(path=REWARD_PATH, format=RewardFileFormat.JSON),)),
            timeout=40,
        ),
        (
            inline_resource("config.json", json.dumps(config, allow_nan=False).encode()),
            inline_resource("invocation.json", json.dumps(invocation, allow_nan=False).encode()),
            inline_resource("source_callable.py", SOURCE_CALLABLE.read_bytes()),
        ),
    )


def _bound_task(task: TaskSpec | ImportRejection, input_format: str, image: str) -> TaskSpec | ImportRejection:
    if isinstance(task, ImportRejection):
        return task
    contract = grader_config(task)["contract"]
    prompt = next(
        event.content
        for event in reversed(task.context.events)
        if isinstance(event, TextMessage) and event.role == "user"
    )
    constraints = contract["constraints"]
    if input_format == "nemotron":
        try:
            constraints = [
                nemotron_constraint(name, arguments, prompt)
                for name, arguments in zip(
                    constraints["instruction_id_list"], constraints["instruction_kwargs"], strict=True
                )
            ]
        except ValueError as error:
            return ImportRejection(
                kind=ImportFailureKind.UNSUPPORTED, reason="unsupported_ifeval_constraint", detail=str(error)
            )
    package = grader_package(constraints, input_format, image)
    return task.model_copy(
        update={
            "grader": package.grader,
            "resources": task.resources.model_copy(update={"verifier": package.resources}),
        }
    )


def verification_report(task: TaskSpec) -> VerificationReport:
    config = grader_config(task)
    constraints = config["constraints"]
    specs = constraints if isinstance(constraints, list) else [constraints]
    language_constraint = any(spec["func_name"] == "validate_response_language" for spec in specs)
    if language_constraint:
        try:
            installed = version("langdetect")
        except PackageNotFoundError:
            installed = None
        if f"langdetect=={installed}" != LANGUAGE_DEPENDENCY:
            return VerificationReport(
                checks=[
                    CheckResult(check="language_dependency", status=CheckStatus.UNSUPPORTED, detail=LANGUAGE_DEPENDENCY)
                ]
            )
    return VerificationReport(
        checks=[
            CheckResult(
                check="positive_witness",
                status=CheckStatus.SKIPPED,
                detail="Pinned native predicates are packaged; the source supplies no passing response witness",
            )
        ]
    )


def normalize_isolated(
    row: RawRow,
    *,
    normalize_task: Callable[[RawRow], TaskSpec | ImportRejection],
    input_format: Literal["nemotron", "rlvr"],
    image: str,
) -> TaskSpec | ImportRejection:
    return _bound_task(normalize_task(row), input_format, image)


async def isolated_checks(
    task: TaskSpec, *, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
) -> VerificationReport:
    """Probe execution without asserting a semantically correct response."""
    result = await grade_final_message(
        task, TextMessage(role="assistant", content="Runtime diagnostic response."), factory, machine_spec, timeout
    )
    diagnostic = result.error
    if not diagnostic and result.detail:
        errors = {key: result.detail[key] for key in ("error_type", "error_message") if key in result.detail}
        if errors:
            diagnostic = json.dumps(errors, ensure_ascii=True)[:1000]
    if result.status == Outcome.GRADED and result.reward is not None and 0 <= result.reward <= 1:
        status = CheckStatus.PASS
        detail = f"Original predicates executed; diagnostic reward={result.reward}; not a correctness witness"
    elif diagnostic and LANGUAGE_DEPENDENCY in diagnostic:
        status = CheckStatus.UNSUPPORTED
        detail = diagnostic
    else:
        status = CheckStatus.INFRA_ERROR
        detail = diagnostic or f"Native runtime did not return a fractional reward: {result.status}"
    return VerificationReport(
        checks=[
            CheckResult(check="native_runtime", status=status, detail=detail),
            CheckResult(
                check="positive_witness",
                status=CheckStatus.SKIPPED,
                detail="The source supplies no passing response witness",
            ),
        ]
    )


def isolated_verification(
    task: TaskSpec, *, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
) -> VerificationReport:
    return asyncio.run(isolated_checks(task, factory=factory, machine_spec=machine_spec, timeout=timeout))


def bind(
    recipe: DatasetRecipe,
    *,
    input_format: Literal["nemotron", "rlvr"],
    image: str,
    factory: MachineFactory,
    machine_spec: MachineSpec,
    worker_image: str | None,
    timeout: float,
) -> DatasetRecipe:
    """Bind original predicates and fresh runtime diagnostics to an explicit private grader image."""
    machine = machine_spec_identity(machine_spec)
    suite = CheckSuite(
        id="skyrl-native-ifeval-runtime",
        revision=EXECUTION_REVISION,
        parameters={
            "image": image,
            "backend": factory.backend.value,
            "machine": machine,
            "worker_image": worker_image,
            "timeout": timeout,
            "input_format": input_format,
        },
        run=partial(isolated_verification, factory=factory, machine_spec=machine_spec, timeout=timeout),
    )
    return replace(
        recipe,
        policy=replace(
            recipe.policy,
            normalize=partial(
                normalize_isolated, normalize_task=recipe.policy.normalize, input_format=input_format, image=image
            ),
            check_suite=suite,
        ),
    )
