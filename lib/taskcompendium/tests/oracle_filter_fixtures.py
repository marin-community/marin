# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""External Harbor receipt fixtures for curation admission tests."""

from taskcompendium.grading_environment import GradingEnvironmentMode
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.oracle_filter import (
    NoOracleLivenessReceipt,
    OracleFilterReceipt,
    OracleTaskReceipt,
    OracleTrialReceipt,
    ResourceCalibrationReceipt,
    VerifierResourceProfile,
    has_oracle_solution,
)


def oracle_receipt(
    tasks: list[TaskSpec],
    recipe_identity: str | None = None,
    *,
    failed: frozenset[str] = frozenset(),
    no_oracle_liveness: bool = True,
):
    entries = {}
    for task in tasks:
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
            no_oracle_liveness=(
                NoOracleLivenessReceipt(
                    task_id=task.id,
                    task_sha256=canonical_sha256(task.model_dump(mode="json")),
                    runtime_sha256=None,
                    submission_kind="empty",
                    outcome="negative_verdict",
                    exit_code=0,
                    reward=0,
                    tests_executed=2,
                )
                if not has_oracle_solution(task) and no_oracle_liveness
                else None
            ),
            exclusion_reason=(
                "oracle_failed"
                if reward == 0
                else "no_oracle_liveness_missing" if not has_oracle_solution(task) and not no_oracle_liveness else None
            ),
        )
    return OracleFilterReceipt(
        recipe_identity=recipe_identity,
        input_cohort_sha256="a" * 64,
        input_rows=len(tasks),
        conversion_rejections=0,
        tasks=entries,
    )
