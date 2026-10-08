# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the original stateless Ultra MCQA evaluator to a isolated Shellbox machine."""

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import replace
from functools import partial

from shellbox.machine import MachineFactory, MachineSpec
from taskcompendium.datasets.nemotron_ultra.normalization import (
    VERIFIER_REVISION,
)
from taskcompendium.grader import GraderPackage, grader_config
from taskcompendium.models import AnswerType, TaskSpec, TextMessage
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    DatasetRecipe,
    ImportRejection,
    NormalizedTask,
    RawRow,
    VerificationReport,
)
from taskcompendium.pipeline.verification import control_result
from taskcompendium.runtime.shell import machine_spec_identity

from experiments.post_training.task_curation.datasets.nemotron_ultra.grading.binding import (
    ANSWER_EXTRACTOR,
    invocation_bytes,
    normalize_terminal_grader,
    score_package,
)
from experiments.post_training.task_curation.datasets.shared import grade_final_message

CALL = {
    "function": "skyrl_gym.envs.nemotron_ultra.mcqa:grade_mcqa",
    "args": ["answer", "contract"],
    "answer_extractor": ANSWER_EXTRACTOR,
}


def grader_package(config: dict, image: str) -> GraderPackage:
    return score_package(config, invocation=CALL, timeout=60, image=image)


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
        allowed_agents=("mcqa_simple_agent",),
        package=grader_package,
        answer_type=AnswerType.TEXT,
    )


def reference_answer(contract: dict) -> str | None:
    """Construct a witness only for the original standard letter modes."""
    if "output_regex" in (contract.get("template_metadata") or {}):
        return None
    gold = str(contract.get("expected_answer") or "").strip().upper()
    allowed = {
        key.upper()
        for option in contract.get("options") or []
        for key, value in option.items()
        if isinstance(key, str) and len(key) == 1 and key.isalpha() and value is not None
    }
    if gold not in allowed:
        return None
    mode = contract.get("grading_mode", "strict_single_letter_boxed")
    if mode in {"strict_single_letter_boxed", "lenient_boxed"}:
        return "\\boxed{" + gold + "}"
    if mode == "lenient_answer_colon":
        return "Answer: " + gold
    if mode == "lenient_answer_colon_md":
        return "**Answer**: " + gold
    return None


async def isolated_checks(
    task: TaskSpec,
    *,
    factory: MachineFactory,
    machine_spec: MachineSpec,
    timeout: float,
) -> VerificationReport:
    """Check empty rejection and a positive witness where the original mode permits one."""
    checks = []
    reference = reference_answer(grader_config(task)["contract"])
    controls = [("empty", "", 0.0)]
    if reference is not None:
        controls.append(("reference", reference, 1.0))
    for name, answer, reward in controls:
        result = await grade_final_message(
            task, TextMessage(role="assistant", content=answer), factory, machine_spec, timeout
        )
        checks.append(control_result(result, name, reward))
    if reference is None:
        checks.append(
            CheckResult(
                check="positive_witness",
                status=CheckStatus.SKIPPED,
                detail="Custom regex, unsupported mode, or missing allowed gold provides no sound constructed witness",
            )
        )
    return VerificationReport(checks)


def verification_report(
    task: TaskSpec,
    *,
    factory: MachineFactory,
    machine_spec: MachineSpec,
    timeout: float,
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
    """Bind original MCQA controls without changing the public request or source contract."""
    machine = machine_spec_identity(machine_spec)
    suite = CheckSuite(
        id="original-ultra-mcqa-controls",
        revision="1",
        parameters={
            "verifier_revision": VERIFIER_REVISION,
            "runner_sha256": hashlib.sha256(invocation_bytes(CALL)).hexdigest(),
            "image": image,
            "backend": factory.backend.value,
            "machine": machine,
            "worker_image": worker_image,
            "timeout": timeout,
        },
        run=partial(
            verification_report,
            factory=factory,
            machine_spec=machine_spec,
            timeout=timeout,
        ),
    )
    return replace(
        recipe,
        policy=replace(
            recipe.policy,
            normalize=partial(
                normalize_isolated,
                image=image,
                normalize_task=recipe.policy.normalize,
            ),
            check_suite=suite,
        ),
    )
