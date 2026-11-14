# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""External Harbor receipt fixtures for curation admission tests."""

from finestore.rl.curation import ContentEvidenceReference

from taskcompendium.grading_environment import GradingEnvironmentMode
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.oracle_filter import (
    ExecutabilityRunReceipt,
    OracleExclusionReason,
    OracleFilterReceipt,
    OracleTaskReceipt,
    OracleTrialReceipt,
    ResourceCalibrationReceipt,
    VerifierResourceProfile,
)
from taskcompendium.runtime.resources import has_oracle_solution


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
        entries[task.id] = OracleTaskReceipt(
            task_sha256=canonical_sha256(task.model_dump(mode="json")),
            harbor_task_checksum=checksum,
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
        recipe_identity=recipe_identity,
        input_cohort_sha256="a" * 64,
        input_rows=len(tasks),
        conversion_rejections=0,
        tasks=entries,
    )
