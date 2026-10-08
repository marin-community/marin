# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the locked MarinSkyRL code and seeded SQLite scorers to private TaskSpec graders."""

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

from shellbox.machine import MachineFactory, MachineSpec
from taskcompendium.datasets.code_contracts import has_code_block, validate_code_cases
from taskcompendium.datasets.gretel_text_to_sql import validate_seeded_reference
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
from taskcompendium.pipeline.verification import control_result
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.shell import machine_spec_identity

from experiments.post_training.task_curation.datasets.shared import grade_final_message
from experiments.post_training.task_curation.datasets.skyrl.code_sql import grade_apps, grade_lcb, grade_sql

UPSTREAM_REVISION = "544d5d6f14116a06bde0209352585903133bd618"
APPS_UPSTREAM_REVISION = "b45c0ed78517a3a6492eb77b21cffbb79b1096f1"
APPS_TESTING_UTIL_SHA256 = "9a4e58ff2634ef606c42597457c0733910862e0588ea750d901265bbfe65d36f"
APPS_TESTING_UTIL_PATH = "/opt/apps/eval/testing_util.py"
ANSWER_PATH = "/app/answer.txt"
CONTROL_TIMEOUT = 360.0
# The original LCB child permits 4 GiB; the guest also needs room for its runtime.
CONTROL_MEMORY_MB = 5120
CONTROL_EXECUTION_REVISION = "bounded-original-witness-v1"
APPS_CONTROL_EXECUTION_REVISION = "bounded-apps-original-witness-v1"
APPS_WITNESS_SELECTION_REVISION = "apps-original-first-middle-last-v1"
MAX_APPS_POSITIVE_WITNESSES = 3
THREAD_ENVIRONMENT = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
CODE_INSTRUCTION = "\nReturn the complete Python solution in this format:\n```python\n# solution\n```"
SQL_INSTRUCTION = (
    "\nTarget dialect is SQLite. Write exactly one SELECT statement (a leading WITH is allowed) that "
    "answers the question. Return only the query, inside <solution></solution>."
)
SOURCE_PINS = {
    "apps": ("codeparrot/apps", "21e74ddf8de1a21436da12e3e653065c5213e9d1"),
    "eurus2_code": ("PRIME-RL/Eurus-2-RL-Data", "9776b13264b5aaa0b16495fcf086a0a8d86fd655"),
    "verifiable_code": ("open-r1/verifiable-coding-problems-python", "b761a24a95fa03289a231d2d31c183636ffb9833"),
    "gretel_text_to_sql": ("gretelai/synthetic_text_to_sql", "740ab236e64503fba51be1101df7a1be83bf455d"),
}


def original_package(config: dict, image: str) -> GraderPackage:
    """Bind the source scorer installed in the selected immutable image."""
    if config["evaluator"] == "apps":
        config = {
            **config,
            "apps_source_sha256": APPS_TESTING_UTIL_SHA256,
            "input_output": config["contract"]["input_output"],
        }
        adapter = Path(grade_apps.__file__).read_bytes()
        return GraderPackage(
            ScriptGrader(
                argv=(
                    "python3",
                    "/tests/grade_apps.py",
                    APPS_TESTING_UTIL_PATH,
                    "/tests/config.json",
                    ANSWER_PATH,
                    "/logs/verifier/reward.json",
                ),
                cwd="/",
                environment=grading_environment(image),
                answer_path=ANSWER_PATH,
                reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.json", format=RewardFileFormat.JSON),)),
                timeout=330,
            ),
            (
                inline_resource("config.json", json.dumps(config, allow_nan=False).encode()),
                inline_resource("grade_apps.py", adapter),
            ),
        )
    if config["evaluator"] == "gretel_text_to_sql":
        scorer = grade_sql
        config = {
            **config,
            "reference_sql": config["contract"]["sql"],
            "context_sql": config["contract"]["sql_context"],
        }
    else:
        scorer = grade_lcb
        contract = config["contract"]
        ground_truth = (
            contract["reward_model"]["ground_truth"]
            if config["evaluator"] == "eurus2_code"
            else contract["verification_info"]
        )
        config = {**config, "test_cases": ground_truth}
    script = Path(scorer.__file__)
    return GraderPackage(
        ScriptGrader(
            argv=(
                "python3",
                "/tests/" + script.name,
                "/tests/config.json",
                ANSWER_PATH,
                "/logs/verifier/score.json",
            ),
            cwd="/",
            environment=grading_environment(image),
            answer_path=ANSWER_PATH,
            reward=FileReward(files=(RewardFile(path="/logs/verifier/score.json", format=RewardFileFormat.JSON),)),
            timeout=330,
        ),
        (
            inline_resource("config.json", json.dumps(config, allow_nan=False).encode()),
            inline_resource(script.name, script.read_bytes()),
        ),
    )


def normalize_isolated(
    row: RawRow, *, normalize_task: Callable[[RawRow], TaskSpec | ImportRejection], image: str
) -> TaskSpec | ImportRejection:
    task = normalize_task(row)
    if isinstance(task, ImportRejection):
        return task
    config = grader_config(task)
    evaluator = config["evaluator"]
    if (row.source.dataset, row.source.revision) != SOURCE_PINS[evaluator]:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="unsupported_native_source_revision",
            detail=f"Only the pinned {evaluator} source is bound",
        )
    if evaluator not in ("apps", "gretel_text_to_sql"):
        contract = config["contract"]
        if evaluator == "eurus2_code":
            ground_truth = contract["reward_model"]["ground_truth"]
        else:
            ground_truth = contract["verification_info"]
        try:
            validate_code_cases(ground_truth)
        except (ValueError, TypeError) as error:
            return ImportRejection(
                kind=ImportFailureKind.UNSUPPORTED,
                reason="original_lcb_contract_unsupported",
                detail=str(error),
            )
    elif evaluator == "gretel_text_to_sql":
        contract = config["contract"]
        try:
            validate_seeded_reference(contract["sql_context"], contract["sql"])
        except ValueError as error:
            return ImportRejection(
                kind=ImportFailureKind.UNSUPPORTED,
                reason="original_gretel_preparation_unsupported",
                detail=str(error),
            )
    package = original_package(config, image)
    instruction = SQL_INSTRUCTION if evaluator == "gretel_text_to_sql" else CODE_INSTRUCTION
    events = list(task.context.events)
    last_user = max(
        index for index, event in enumerate(events) if isinstance(event, TextMessage) and event.role == "user"
    )
    event = events[last_user]
    assert isinstance(event, TextMessage)
    events[last_user] = event.model_copy(update={"content": event.content + instruction})
    return task.model_copy(
        update={
            "context": task.context.model_copy(update={"events": tuple(events)}),
            "grader": package.grader,
            "resources": task.resources.model_copy(update={"verifier": package.resources}),
        }
    )


@dataclass(frozen=True)
class SourceWitness:
    original_index: int | None
    source_solution_sha256: str
    candidate_sha256: str
    answer: str


def _source_witness(solution: str, answer: str, original_index: int | None) -> SourceWitness:
    return SourceWitness(
        original_index,
        hashlib.sha256(solution.encode()).hexdigest(),
        hashlib.sha256(answer.encode()).hexdigest(),
        answer,
    )


def positive_witnesses(config: dict) -> tuple[SourceWitness, ...]:
    """Select bounded unchanged source solutions; Eurus supplies no golden response."""
    contract = config["contract"]
    if config["evaluator"] == "gretel_text_to_sql":
        solution = contract["sql"]
        return (_source_witness(solution, f"<solution>{solution}</solution>", None),)
    if config["evaluator"] == "apps":
        solutions = contract.get("solutions")
        if isinstance(solutions, str):
            try:
                solutions = json.loads(solutions)
            except json.JSONDecodeError:
                return ()
        if not isinstance(solutions, list):
            return ()
        distinct = {}
        for index, solution in enumerate(solutions):
            if isinstance(solution, str) and solution.strip():
                distinct.setdefault(solution, index)
        candidates = list(distinct.items())
        if not candidates:
            return ()
        positions = dict.fromkeys((0, len(candidates) // 2, len(candidates) - 1))
        selected = [
            (candidates[position][1], candidates[position][0])
            for position in list(positions)[:MAX_APPS_POSITIVE_WITNESSES]
        ]
    else:
        solution = contract.get("gold_standard_solution")
        if not isinstance(solution, str) or not solution.strip():
            return ()
        selected = [(None, solution)]
    return tuple(
        _source_witness(
            solution,
            (solution if has_code_block(solution) else f"```python\n{solution}\n```"),
            index,
        )
        for index, solution in selected
    )


async def isolated_checks(
    task: TaskSpec, *, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
) -> VerificationReport:
    config = grader_config(task)
    is_sql = config["evaluator"] == "gretel_text_to_sql"
    diagnostic = "SELECT FROM" if is_sql else '```python\nraise RuntimeError("negative control")\n```'
    controls = [("empty", "", 0.0), ("native_runtime", diagnostic, 0.0)]
    checks = []
    for name, answer, expected in controls:
        result = await grade_final_message(
            task, TextMessage(role="assistant", content=answer), factory, machine_spec, timeout
        )
        check = control_result(result, name, expected)
        if name == "native_runtime" and result.detail and result.detail.get("execution_error"):
            check = check.model_copy(update={"status": CheckStatus.INFRA_ERROR, "detail": str(result.detail)})
        if result.error:
            check = check.model_copy(update={"detail": check.detail + "; " + result.error})
        checks.append(check)
        if check.status == CheckStatus.INFRA_ERROR or result.status == Outcome.INVALID_TASK:
            return VerificationReport(checks)
    witnesses = positive_witnesses(config)
    if not witnesses:
        checks.append(
            CheckResult(
                check="positive_witness", status=CheckStatus.SKIPPED, detail="No source-provided golden solution"
            )
        )
        return VerificationReport(checks)
    attempts = []
    result = None
    for witness in witnesses:
        result = await grade_final_message(
            task, TextMessage(role="assistant", content=witness.answer), factory, machine_spec, timeout
        )
        attempts.append(
            {
                "original_solution_index": witness.original_index,
                "source_solution_sha256": witness.source_solution_sha256,
                "candidate_sha256": witness.candidate_sha256,
                "status": result.status,
                "reward": result.reward,
                "error": result.error,
                "detail": result.detail,
            }
        )
        if result.status in (Outcome.INFRA_ERROR, Outcome.INVALID_TASK) or result.reward == 1.0:
            break
    assert result is not None
    check = control_result(result, "positive_witness", 1.0)
    checks.append(
        check.model_copy(
            update={
                "detail": json.dumps(
                    {
                        "selection_policy": (
                            APPS_WITNESS_SELECTION_REVISION if config["evaluator"] == "apps" else "source-single"
                        ),
                        "attempts": attempts,
                    }
                )
            }
        )
    )
    return VerificationReport(checks)


def isolated_verification(
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
    """Bind the TaskSpec verifier and fresh controls; runtime images remain caller-owned."""
    machine_spec = replace(machine_spec, env={**machine_spec.env, **THREAD_ENVIRONMENT})
    machine = machine_spec_identity(machine_spec)
    scorer = (
        grade_apps
        if recipe.source.dataset == "codeparrot/apps"
        else grade_sql if recipe.source.dataset == "gretelai/synthetic_text_to_sql" else grade_lcb
    )
    suite = CheckSuite(
        id="original-marinskyrl-code-sql-controls",
        revision=APPS_UPSTREAM_REVISION if recipe.source.dataset == "codeparrot/apps" else UPSTREAM_REVISION,
        parameters={
            "image": image,
            "backend": factory.backend.value,
            "machine": machine,
            "worker_image": worker_image,
            "timeout": timeout,
            "runner_sha256": hashlib.sha256(Path(scorer.__file__).read_bytes()).hexdigest(),
            "source_evaluator_revision": (
                APPS_UPSTREAM_REVISION if recipe.source.dataset == "codeparrot/apps" else UPSTREAM_REVISION
            ),
            "control_execution_revision": (
                APPS_CONTROL_EXECUTION_REVISION
                if recipe.source.dataset == "codeparrot/apps"
                else CONTROL_EXECUTION_REVISION
            ),
            "positive_witness_selection": (
                APPS_WITNESS_SELECTION_REVISION if recipe.source.dataset == "codeparrot/apps" else "source-single"
            ),
            "max_positive_witnesses": MAX_APPS_POSITIVE_WITNESSES if recipe.source.dataset == "codeparrot/apps" else 1,
        },
        run=partial(isolated_verification, factory=factory, machine_spec=machine_spec, timeout=timeout),
    )
    return replace(
        recipe,
        policy=replace(
            recipe.policy,
            normalize=partial(normalize_isolated, normalize_task=recipe.policy.normalize, image=image),
            check_suite=suite,
        ),
    )
