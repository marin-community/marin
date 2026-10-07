# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the original stateless Ultra calendar evaluator to a isolated machine."""

import asyncio
from collections.abc import Callable
from functools import partial

from shellbox.machine import MachineFactory, MachineSpec
from taskcompendium.datasets.nemotron_ultra.normalization import VERIFIER_REVISION
from taskcompendium.grader import GraderPackage
from taskcompendium.models import AnswerType, TaskSpec
from taskcompendium.pipeline.execution_binding import bind_grader_recipe, native_runtime_report
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    ImportRejection,
    NormalizedTask,
    RawRow,
    VerificationReport,
)
from taskcompendium.runtime.grading import grade_submission

from experiments.post_training.task_curation.datasets.nemotron_ultra.grading.binding import (
    ANSWER_EXTRACTOR,
    ANSWER_PATH,
    invocation_bytes,
    normalize_terminal_grader,
    score_package,
)

CALL = {
    "function": "skyrl_gym.envs.nemotron_ultra.calendar:grade_calendar",
    "args": ["answer", "contract.exp_cal_state"],
    "answer_extractor": ANSWER_EXTRACTOR,
}


def grader_package(config: dict) -> GraderPackage:
    return score_package(config, invocation=CALL, timeout=60)


def normalize_isolated(
    row: RawRow,
    *,
    image: str,
    normalize_task: Callable[[RawRow], TaskSpec | NormalizedTask | ImportRejection],
) -> TaskSpec | NormalizedTask | ImportRejection:
    return normalize_terminal_grader(
        row,
        image=image,
        normalize_task=normalize_task,
        allowed_agents=("calendar_simple_agent",),
        package=grader_package,
        answer_type=AnswerType.TEXT,
    )


async def isolated_checks(
    task: TaskSpec, *, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
) -> VerificationReport:
    """Check runtime without assuming any response universally fails the source constraints."""
    diagnostic = await grade_submission(task, {ANSWER_PATH: b"[]"}, factory, machine_spec=machine_spec, timeout=timeout)
    return native_runtime_report(diagnostic, "The source supplies constraints without a passing schedule witness")


def verification_report(
    task: TaskSpec, *, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
) -> VerificationReport:
    return asyncio.run(isolated_checks(task, factory=factory, machine_spec=machine_spec, timeout=timeout))


def bind(
    recipe: DatasetRecipe,
    *,
    image: str,
    factory: MachineFactory,
    machine_spec: MachineSpec,
    worker_image: str | None,
    timeout: float,
) -> DatasetRecipe:
    """Bind original calendar checks without changing the public request or quality policy."""
    return bind_grader_recipe(
        recipe,
        normalize=partial(normalize_isolated, image=image, normalize_task=recipe.policy.normalize),
        verification=partial(verification_report, factory=factory, machine_spec=machine_spec, timeout=timeout),
        suite_id="original-ultra-calendar-controls",
        grader_bytes=invocation_bytes(CALL),
        image=image,
        factory=factory,
        machine_spec=machine_spec,
        worker_image=worker_image,
        timeout=timeout,
        verifier_revision=VERIFIER_REVISION,
    )
