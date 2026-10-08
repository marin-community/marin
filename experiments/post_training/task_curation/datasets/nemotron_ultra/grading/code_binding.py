# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the original Ultra competitive-code grader and its pinned local LCB child."""

import asyncio
from collections.abc import Callable
from dataclasses import replace
from functools import partial

from shellbox.machine import MachineFactory, MachineSpec
from taskcompendium.datasets.code_contracts import has_code_block, validate_code_cases
from taskcompendium.datasets.nemotron_ultra.normalization import VERIFIER_REVISION
from taskcompendium.grader import GraderPackage, grader_config
from taskcompendium.models import AnswerType, TaskSpec, TextMessage
from taskcompendium.pipeline.execution_binding import bind_grader_recipe
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

from experiments.post_training.task_curation.datasets.nemotron_ultra.grading.binding import (
    ANSWER_EXTRACTOR,
    invocation_bytes,
    normalize_terminal_grader,
    score_package,
)
from experiments.post_training.task_curation.datasets.shared import grade_final_message
from experiments.post_training.task_curation.datasets.skyrl.code_sql.binding import THREAD_ENVIRONMENT

CALL = {
    "function": "skyrl_gym.envs.nemotron_ultra.code_gen:grade_code",
    "args": ["answer", "contract"],
    "answer_extractor": ANSWER_EXTRACTOR,
    "kwargs": {"assistant_message": "terminal_message"},
}
SCRIPT_TIMEOUT = 330.0


def original_package(config: dict, image: str) -> GraderPackage:
    return score_package(config, invocation=CALL, timeout=SCRIPT_TIMEOUT, image=image, state_path="/app/state.json")


def normalize_isolated(
    row: RawRow,
    *,
    image: str,
    normalize_task: Callable[[RawRow], TaskSpec | NormalizedTask | ImportRejection],
) -> TaskSpec | NormalizedTask | ImportRejection:
    result = normalize_terminal_grader(
        row,
        image=image,
        normalize_task=normalize_task,
        allowed_agents=("code_gen_simple_agent",),
        package=original_package,
        answer_type=AnswerType.TEXT,
    )
    if isinstance(result, ImportRejection):
        return result
    try:
        validate_code_cases(row.data["verifier_metadata"]["unit_tests"])
    except (KeyError, TypeError, ValueError) as error:
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT, reason="invalid_original_code_tests", detail=str(error)
        )
    return result


def positive_witness(config: dict) -> str | None:
    """Only a source-provided golden program can be a positive native control."""
    solution = config["contract"].get("gold_standard_solution")
    if not isinstance(solution, str) or not solution.strip():
        return None
    if has_code_block(solution):
        return solution
    return f"```python\n{solution}\n```"


async def isolated_checks(task: TaskSpec, *, factory: MachineFactory, machine_spec: MachineSpec, timeout: float):
    controls = [
        ("empty_submission", "", 0.0),
        ("native_runtime", '```python\nraise RuntimeError("negative control")\n```', 0.0),
    ]
    witness = positive_witness(grader_config(task))
    if witness is not None:
        controls.append(("positive_witness", witness, 1.0))
    checks = []
    for name, answer, reward in controls:
        result = await grade_final_message(
            task, TextMessage(role="assistant", content=answer), factory, machine_spec, timeout
        )
        check = control_result(result, name, reward)
        if result.error:
            check = check.model_copy(update={"detail": check.detail + "; " + result.error})
        checks.append(check)
    if witness is None:
        checks.append(
            CheckResult(check="positive_witness", status=CheckStatus.SKIPPED, detail="No source-provided golden program")
        )
    return VerificationReport(checks)


def verification_report(task: TaskSpec, *, factory: MachineFactory, machine_spec: MachineSpec, timeout: float):
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
    """Preserve original binary code scoring in a network-disabled isolated machine."""
    machine_spec = replace(machine_spec, env={**machine_spec.env, **THREAD_ENVIRONMENT})
    return bind_grader_recipe(
        recipe,
        normalize=partial(normalize_isolated, image=image, normalize_task=recipe.policy.normalize),
        verification=partial(verification_report, factory=factory, machine_spec=machine_spec, timeout=timeout),
        suite_id="original-ultra-competitive-code-controls",
        grader_bytes=invocation_bytes(CALL),
        image=image,
        factory=factory,
        machine_spec=machine_spec,
        worker_image=worker_image,
        timeout=timeout,
        verifier_revision=VERIFIER_REVISION,
    )
