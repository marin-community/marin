# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind Ultra's original NVIDIA instruction registry and per-row reward modes."""

import asyncio
from functools import partial

from shellbox.machine import MachineFactory, MachineSpec
from taskcompendium.datasets.nemotron_ultra.normalization import VERIFIER_REVISION
from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, TaskSpec, TextMessage
from taskcompendium.pipeline.execution_binding import bind_grader_recipe
from taskcompendium.pipeline.models import CheckResult, CheckStatus, DatasetRecipe, VerificationReport

from experiments.post_training.task_curation.datasets.nemotron_ultra.grading.binding import (
    ANSWER_EXTRACTOR,
    invocation_bytes,
    normalize_terminal_grader,
    score_package,
)
from experiments.post_training.task_curation.datasets.shared import grade_final_message

CALL = {
    "function": "skyrl_gym.envs.nemotron_ultra.instruction_following:grade_instruction_following",
    "args": ["answer", "contract"],
    "answer_extractor": ANSWER_EXTRACTOR,
}


def grader_package(config: dict, image: str) -> GraderPackage:
    return score_package(config, invocation=CALL, timeout=60, image=image)


async def isolated_checks(
    task: TaskSpec, *, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
) -> VerificationReport:
    """Probe fractional runtime output without asserting a passing response witness."""
    result = await grade_final_message(
        task, TextMessage(role="assistant", content="Runtime diagnostic response."), factory, machine_spec, timeout
    )
    details = result.detail or {}
    errors = details.get("instruction_errors", [])
    if any(error is not None and error.startswith(("KeyError:", "TypeError:")) for error in errors):
        status = CheckStatus.FAIL
    elif any(error is not None for error in errors):
        status = CheckStatus.UNSUPPORTED
    elif result.status == Outcome.GRADED and result.reward is not None and 0 <= result.reward <= 1:
        status = CheckStatus.PASS
    elif result.status == Outcome.INFRA_ERROR:
        status = CheckStatus.INFRA_ERROR
    else:
        status = CheckStatus.FAIL
    return VerificationReport(
        checks=[
            CheckResult(
                check="native_runtime",
                status=status,
                detail=result.error
                or (
                    f"Diagnostic status={result.status}; reward={result.reward}; "
                    f"predicate_errors={errors}; not a witness"
                ),
            ),
            CheckResult(
                check="positive_witness",
                status=CheckStatus.SKIPPED,
                detail="The source supplies predicate constraints without a passing response witness",
            ),
        ]
    )


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
    """Bind original NVIDIA predicates; keep the row's binary or fractional grading mode."""
    return bind_grader_recipe(
        recipe,
        normalize=partial(
            normalize_terminal_grader,
            image=image,
            normalize_task=recipe.policy.normalize,
            allowed_agents=("instruction_following_simple_agent",),
            package=grader_package,
            answer_type=AnswerType.TEXT,
        ),
        verification=partial(verification_report, factory=factory, machine_spec=machine_spec, timeout=timeout),
        suite_id="original-ultra-instruction-following-runtime",
        grader_bytes=invocation_bytes(CALL),
        image=image,
        factory=factory,
        machine_spec=machine_spec,
        worker_image=worker_image,
        timeout=timeout,
        verifier_revision=VERIFIER_REVISION,
    )
