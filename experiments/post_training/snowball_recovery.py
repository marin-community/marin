# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover preserved step-50 exports or saved-answer grades through a new artifact version.

No model inference is performed. Original training failures and reports remain
unchanged. Regrading requires independently pinned original-grader admission proof.
"""

import json
from dataclasses import dataclass
from enum import StrEnum

import click
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import IRIS_HUB_CLUSTER_CONFIG, IrisSkyRLExecution, run_skyrl
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.snowball_inventory_evaluate import (
    EvaluationOutcome,
    PanelTask,
    launch_from_document,
    runtime_operation,
    summarize_panel,
)
from experiments.post_training.snowball_inventory_smoke import CLUSTER, CONTAINER_PROFILE
from experiments.post_training.snowball_recovery_records import encoded, join_receipts, pinned_bytes

WORKER = "experiments.post_training.snowball_recovery_runtime"


class Stage(StrEnum):
    EXPORT = "export"
    REGRADE = "regrade"


class RecoveredResult(Artifact):
    report_uri: str
    recovery_kind: str


@dataclass(frozen=True)
class RecoveryConfig:
    request_uri: str
    request_sha256: str
    arm: str
    stage: Stage
    output_path: str


def write_exclusive(uri: str, value: dict) -> None:
    conditional_object(uri).write(encoded(value), expected_version=None)


def summarize_rows(rows: list[dict]) -> dict:
    tasks = [PanelTask(row["task_id"], row["evaluation_family"], row["contract_id"]) for row in rows]
    outcomes = [
        EvaluationOutcome(
            row["task_id"], row["score"], row["stop_reason"], row["verification_status"], row["failure_type"]
        )
        for row in rows
    ]
    return summarize_panel(tasks, outcomes)


def run_recovery(config: RecoveryConfig) -> RecoveredResult:
    request = json.loads(pinned_bytes(StoragePath(config.request_uri).read_bytes(), config.request_sha256))
    if request["arm"] != config.arm:
        raise ValueError("Recovery request belongs to another initializer arm")
    output = config.output_path.rstrip("/") + "/recovery"
    worker_input = {"request": request, "output_path": output}
    if config.stage is Stage.EXPORT:
        prepared = runtime_operation("prepare-export", worker_input, worker_module=WORKER)
        write_exclusive(output + "/preflight.json", prepared)
        run_skyrl(launch_from_document(prepared["launch_config_yaml"]))
        verification = runtime_operation(
            "verify-model", {"model_uri": prepared["policy_export_uri"]}, worker_module=WORKER
        )
        report = {**prepared, "verification": verification, "original_training_job_success": False}
    else:
        prepared = runtime_operation("prepare-regrade", worker_input, worker_module=WORKER)
        write_exclusive(output + "/prepared.json", prepared)
        graded = runtime_operation(
            "execute-regrade", {**worker_input, "prepared": prepared}, worker_module=WORKER, timeout=7200
        )
        original = json.loads(
            pinned_bytes(StoragePath(request["report"]["uri"]).read_bytes(), request["report"]["sha256"])
        )
        evidence = {
            row["execution_evidence_sha256"]: (
                StoragePath(output + "/" + row["execution_evidence_sha256"] + ".json").read_bytes()
            )
            for row in graded["receipts"]
        }
        report = join_receipts(original, prepared, graded["receipts"], evidence, summarize_rows)
        report.update(
            input_request_sha256=config.request_sha256,
            original_report_sha256=request["report"]["sha256"],
            grading_receipts=graded["receipts"],
        )
    report_uri = output + "/report.json"
    write_exclusive(report_uri, report)
    return RecoveredResult(path=config.output_path, report_uri=report_uri, recovery_kind=config.stage.value)


@click.command(help=__doc__)
@click.option("--arm", type=click.Choice(["sft", "drdoom"]), required=True)
@click.option("--stage", type=click.Choice([stage.value for stage in Stage]), required=True)
@click.option("--request-uri", required=True)
@click.option("--request-sha256", required=True)
@rl_build_options
def main(arm: str, stage: str, request_uri: str, request_sha256: str) -> ArtifactStep[RecoveredResult]:
    name = user_owned_name(f"recovery/snowball/{arm}/{stage}/step50")
    version = resolve_version(name, None)
    request = ArtifactStep.adopt(f"documents/snowball-recovery/{arm}", version, request_uri)
    execution = IrisSkyRLExecution(
        cluster=CLUSTER,
        cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
        cpu=4,
        memory="32GB",
        disk="64GB",
        priority="interactive",
        max_retries=0,
        target_cluster=CLUSTER,
        parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
        coordinator_timeout_hours=6,
        container_profile=CONTAINER_PROFILE,
    )
    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=RecoveredResult,
        run=run_recovery,
        build_config=lambda ctx: RecoveryConfig(
            ctx.artifact_path(request), request_sha256, arm, Stage(stage), ctx.output_path
        ),
        deps=(request,),
        runtime_args={"skyrl_execution": execution},
    )


if __name__ == "__main__":
    main()
