# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""External Harbor receipt fixtures for curation admission tests."""

from copy import deepcopy

from finestore.rl.curation import ContentEvidenceReference

from taskcompendium.grading_environment import GradingEnvironmentMode
from taskcompendium.harbor.negative_control import NegativeControlSetupKind, negative_control_setup_kind
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.oracle_filter import (
    ExecutabilityRunReceipt,
    NegativeControlApplicability,
    NegativeControlPolicy,
    OracleExclusionReason,
    OracleFilterReceipt,
    OracleNegativeControlReceipt,
    OracleTaskReceipt,
    OracleTrialReceipt,
    ResourceCalibrationReceipt,
    VerifierResourceProfile,
    negative_control_applicability,
)
from taskcompendium.runtime.resources import has_oracle_solution


def negative_control_policy() -> NegativeControlPolicy:
    return NegativeControlPolicy(
        version=1,
        scope="global_workspace",
        max_identical_budget_repeats=1,
        answer_graded="not_applicable",
        source_deadline="non_passing_annotation",
        source_error="non_passing_annotation",
        decision=ContentEvidenceReference(uri="s3://fixtures/global-nop-policy.json", sha256="e" * 64),
    )


def oracle_receipt(
    tasks: list[TaskSpec],
    recipe_identity: str | None = None,
    *,
    failed: frozenset[str] = frozenset(),
    executability: bool = True,
):
    entries = {}
    for task in tasks:
        oracle = has_oracle_solution(task)
        reward = 0 if task.id in failed else 1
        checksum = canonical_sha256({"native_task": task.id})
        result = {
            "agent_info": {"name": "oracle"},
            "config": {
                "agent": {"name": "oracle"},
                "environment": {"type": "local"},
                "resolved_timeouts": {
                    "agent": task.execution_requirements.agent_timeout_sec,
                    "verifier": task.execution_requirements.verifier_timeout_sec,
                },
            },
            "task_name": task.id,
            "task_checksum": checksum,
            "exception_info": None,
            "finished_at": "2026-11-12T13:00:00Z",
            "completion": {"trajectory_present": False},
            "verifier_result": {"rewards": {"reward": reward}},
        }
        if not oracle and executability:
            result["agent_info"] = {"name": "task-spec-chat"}
            result["config"]["agent"] = {
                "name": "task-spec-chat",
                "model_name": "openai/glm-5.3",
                "kwargs": {"max_turns": 3, "request_parameters": {"temperature": 0.7}},
            }
            if task.id in failed:
                result["exception_info"] = {"exception_type": "SourceContractError"}
        trial = OracleTrialReceipt(
            path=f"jobs/{task.id}/result.json",
            result=result,
            verdict={"status": "scored", "reward": reward},
            oracle_exit_code=0,
            result_sha256=canonical_sha256(result),
            execution_requirements=task.execution_requirements,
        )
        applicability = negative_control_applicability(task) if oracle else None
        negative = None
        if applicability is NegativeControlApplicability.WORKSPACE:
            nop = deepcopy(result)
            nop["agent_info"] = {"name": "nop"}
            nop["config"]["agent"] = {"name": "nop"}
            nop["exception_info"] = None
            setup_kind = negative_control_setup_kind(task)
            converted = setup_kind is NegativeControlSetupKind.TASK_SPEC_PREPARATION
            if converted:
                nop["config"]["task"] = {}
                nop["environment_setup"] = {"started_at": "2026-11-12T12:59:00Z", "finished_at": "2026-11-12T12:59:01Z"}
            nop["verifier_result"] = {"rewards": {"reward": 0}, "exit_code": 0}
            evidence = ContentEvidenceReference(uri="s3://fixtures/nop.tar.zst", sha256="e" * 64)
            negative = OracleNegativeControlReceipt(
                source_task_sha256=canonical_sha256(task.model_dump(mode="json")),
                source_task_checksum=checksum,
                runtime_sha256=None,
                consumed_task_sha256=canonical_sha256(task.model_dump(mode="json")),
                consumed_task_checksum=checksum,
                archive=evidence,
                control=evidence,
                result_artifact=evidence,
                setup_artifact=evidence,
                setup_kind=setup_kind,
                preparation_artifact=evidence if converted else None,
                verdict_artifact=evidence,
                trial=OracleTrialReceipt(
                    path=f"nop/{task.id}/result.json",
                    result=nop,
                    verdict={"status": "scored", "reward": 0},
                    oracle_exit_code=None,
                    result_sha256=canonical_sha256(nop),
                    execution_requirements=task.execution_requirements,
                ),
                score=0,
                verifier_exit_code=0,
                setup_exit_code=None if converted else 0,
                original_public_setup=True,
            )
        entries[task.id] = OracleTaskReceipt(
            task_sha256=canonical_sha256(task.model_dump(mode="json")),
            harbor_task_checksum=checksum,
            source_task_sha256=canonical_sha256(task.model_dump(mode="json")),
            source_task_checksum=checksum,
            negative_control_applicability=applicability,
            negative_control=negative,
            has_oracle=has_oracle_solution(task),
            trials=(trial,) if has_oracle_solution(task) else (),
            calibration=ResourceCalibrationReceipt(
                declared_requirements=task.execution_requirements,
                verifier=VerifierResourceProfile(mode=GradingEnvironmentMode.SHARED),
                oracle_kind="original" if has_oracle_solution(task) else "calibration_only",
                attempts=(trial,),
            ),
            executability_run=(
                ExecutabilityRunReceipt(
                    actor_turns=1,
                    trial=trial,
                    model_evidence=(ContentEvidenceReference(uri="s3://fixtures/model-evidence.json", sha256="b" * 64),),
                )
                if not oracle and executability
                else None
            ),
            exclusion_reason=(
                OracleExclusionReason.ORACLE_FAILED
                if reward == 0
                else (OracleExclusionReason.EXECUTABILITY_PROOF_MISSING if not oracle and not executability else None)
            ),
        )
    return OracleFilterReceipt(
        negative_control_policy=negative_control_policy(),
        recipe_identity=recipe_identity,
        input_cohort_sha256="a" * 64,
        input_rows=len(tasks),
        conversion_rejections=0,
        tasks=entries,
    )
