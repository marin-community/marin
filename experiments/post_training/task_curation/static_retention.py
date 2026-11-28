# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write canonical wire5 evidence for unchanged, complete no-oracle populations."""

import argparse
import json
from bisect import bisect_left
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import ContentEvidenceReference, NoOracleStaticBlocker, NoOracleStaticRetentionEvidence
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import task_spec_sha256
from taskcompendium.models import DockerBuildExecution
from taskcompendium.pipeline.oracle_filter import MechanicalOracleTaskReceipt, OracleExclusionReason, OracleFilterReceipt
from taskcompendium.pipeline.static_retention import NoOracleSourceContext, StaticRetainedTaskReceipt
from taskcompendium.runtime.resources import has_oracle_solution

from experiments.post_training.task_curation.calibration_inventory import task_blockers
from experiments.post_training.task_curation.harbor_inspection import native_task_checksums
from experiments.post_training.task_curation.oracle_filter import cohort_sha256, row_runtime, row_runtime_hash, row_task

STATIC_REASONS = {
    "task_spec_production_backend_required": OracleExclusionReason.BACKEND_CONTRACT_UNSUPPORTED,
    "task_spec_shared_executor_required": OracleExclusionReason.SHARED_EXECUTOR_REQUIRED,
    "grading_environment_mismatch": OracleExclusionReason.SOURCE_ENV_INCOMPATIBLE_WITH_GRADER,
    "grading_environment_unavailable": OracleExclusionReason.SOURCE_ENV_INCOMPATIBLE_WITH_GRADER,
    "no_executable_task_spec_grader": OracleExclusionReason.SOURCE_ENV_INCOMPATIBLE_WITH_GRADER,
    "task_spec_tool_provider_required": OracleExclusionReason.TOOL_PROVIDER_REQUIRED,
    "task_spec_state_capture_required": OracleExclusionReason.BACKEND_CONTRACT_UNSUPPORTED,
    "task_spec_provider_contract_invalid": OracleExclusionReason.BACKEND_CONTRACT_UNSUPPORTED,
    "task_spec_unresolved_build": OracleExclusionReason.ENVIRONMENT_CONTRACT_UNRESOLVED,
    "mutable_image": OracleExclusionReason.UNPINNED_ENVIRONMENT,
    "mutable_docker_base": OracleExclusionReason.UNPINNED_ENVIRONMENT,
    "missing_environment": OracleExclusionReason.ENVIRONMENT_CONTRACT_UNRESOLVED,
    "missing_docker_base": OracleExclusionReason.ENVIRONMENT_CONTRACT_UNRESOLVED,
    "missing_packages_lock": OracleExclusionReason.ENVIRONMENT_CONTRACT_UNRESOLVED,
}
PARQUET_BATCH_ROWS = 128


def collect_mixed_retention(
    cohort: Path,
    context: NoOracleSourceContext,
    cohort_evidence: ContentEvidenceReference,
    collected_filter: Path,
    calibrated_tasks: Path,
    output: Path,
) -> OracleFilterReceipt:
    """Combine original-oracle outcomes with unchanged static no-oracle rows."""
    digest = cohort_sha256(cohort)
    original = OracleFilterReceipt.model_validate_json(collected_filter.read_bytes())
    if (
        digest != context.input_cohort_sha256
        or digest != cohort_evidence.sha256
        or original.input_cohort_sha256 != digest
        or original.input_rows != context.current_input.measured_tasks
        or original.version != 4
        or original.conversion_rejections
    ):
        raise ValueError("Mixed retention differs from its complete frozen source and oracle collection")
    projected = {row_task(row).id: row for row in pq.read_table(calibrated_tasks).to_pylist()}
    table = pq.read_table(cohort)
    rows = table.to_pylist()
    entries = {}
    kept, excluded = [], []
    diagnostics = {}
    for row in rows:
        task = row_task(row)
        if task is None or task.id in entries or task.id not in original.tasks:
            raise ValueError("Mixed retention requires every original task exactly once")
        previous = original.tasks[task.id]
        oracle = has_oracle_solution(task)
        source_hash = (
            previous.task_sha256 if isinstance(previous, MechanicalOracleTaskReceipt) else previous.source_task_sha256
        )
        if (
            previous.has_oracle != oracle
            or source_hash != task_spec_sha256(task)
            or previous.runtime_sha256 != row_runtime_hash(row)
        ):
            raise ValueError(f"Mixed retention changed the original task or oracle presence: {task.id}")
        blockers = task_blockers(task) - {"requires_calibration_solution_and_empty_liveness"}
        if blockers - STATIC_REASONS.keys():
            raise ValueError(f"Unclassified static blockers for {task.id}: {sorted(blockers)}")
        identity = {"task_sha256": task_spec_sha256(task), "runtime_sha256": row_runtime_hash(row)}
        if blockers:
            primary = next(reason for reason in STATIC_REASONS if reason in blockers)
            reason = STATIC_REASONS[primary]
            entries[task.id] = MechanicalOracleTaskReceipt(
                **identity, has_oracle=oracle, exclusion_reason=reason, evidence=(cohort_evidence,)
            )
            diagnostics[task.id] = {"reason": reason.value, "blockers": sorted(blockers)}
            excluded.append(row)
            continue
        if oracle:
            entries[task.id] = previous
            if previous.passed:
                consumed = projected[task.id]
                if not original.admits(row_task(consumed), row_runtime(consumed)):
                    raise ValueError(f"Mixed retention lacks its unchanged original oracle proof: {task.id}")
                kept.append(consumed)
            else:
                excluded.append(row)
            continue
        entries[task.id] = StaticRetainedTaskReceipt(
            **identity,
            native_task_checksum=native_task_checksums((task,)).get(task.id),
            input_cohort_sha256=digest,
            policy_sha256=context.policy.sha256,
        )
        kept.append(row)
    if entries.keys() != original.tasks.keys():
        raise ValueError("Mixed retention omits tasks from its original oracle collection")
    receipt = OracleFilterReceipt(
        version=6,
        negative_control_policy=original.negative_control_policy,
        no_oracle_executability_policy=context.policy,
        no_oracle_source=context,
        recipe_identity=original.recipe_identity,
        input_cohort_sha256=digest,
        input_rows=len(entries),
        conversion_rejections=0,
        tasks=entries,
    )
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist(kept, schema=table.schema), output / "tasks.parquet")
    pq.write_table(pa.Table.from_pylist(excluded, schema=table.schema), output / "excluded-tasks.parquet")
    (output / "oracle-filter.json").write_text(receipt.model_dump_json(indent=2) + "\n")
    (output / "static-exclusions.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    summary = {
        "input_rows": len(entries),
        "output_rows": len(kept),
        "static_retained_count": sum(isinstance(entry, StaticRetainedTaskReceipt) for entry in entries.values()),
        "oracle_retained_count": sum(entry.has_oracle and entry.passed for entry in entries.values()),
        "exclusions_by_reason": dict(
            Counter(entry.exclusion_reason.value for entry in entries.values() if entry.exclusion_reason is not None)
        ),
        "static_executability": "unmeasured",
        "original_oracle_filter_sha256": file_sha256(StoragePath(str(collected_filter))),
    }
    (output / "static-retention.json").write_text(json.dumps(summary, indent=2) + "\n")
    return receipt


def collect_static_retention(
    cohort: Path,
    context: NoOracleSourceContext,
    cohort_evidence: ContentEvidenceReference,
    output: Path,
    *,
    recipe_identity: str | None = None,
    docker_build: DockerBuildExecution = DockerBuildExecution.REJECT,
) -> OracleFilterReceipt:
    """Retain original task/runtime rows without calibration, liveness, or model execution."""
    digest = cohort_sha256(cohort)
    if digest != context.input_cohort_sha256 or cohort_evidence.sha256 != digest:
        raise ValueError("Static retention input differs from the independently frozen cohort")
    identifiers = set()
    blocked = {}
    entries = {}
    kept = []
    with pq.ParquetFile(cohort) as parquet:
        schema = parquet.schema_arrow
        offset = 0
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_ROWS):
            rows = batch.to_pylist()
            tasks = []
            for row in rows:
                task = row_task(row)
                if task is None or task.id in identifiers or has_oracle_solution(task):
                    raise ValueError("Static retention requires a complete, unique, known no-oracle population")
                identifiers.add(task.id)
                tasks.append(task)
                blockers = task_blockers(task, docker_build=docker_build) - {
                    "requires_calibration_solution_and_empty_liveness"
                }
                if blockers - STATIC_REASONS.keys():
                    raise ValueError(f"Unclassified static blockers for {task.id}: {sorted(blockers)}")
                if blockers:
                    # Preserve all diagnostics; one deterministic primary reason partitions exclusions.
                    primary = next(reason for reason in STATIC_REASONS if reason in blockers)
                    blocked[task.id] = {"reason": STATIC_REASONS[primary], "blockers": sorted(blockers)}
            checksums = native_task_checksums(task for task in tasks if task.id not in blocked)
            for index, (row, task) in enumerate(zip(rows, tasks, strict=True)):
                identity = {
                    "task_sha256": task_spec_sha256(task),
                    "runtime_sha256": row_runtime_hash(row),
                }
                entries[task.id] = (
                    MechanicalOracleTaskReceipt(
                        **identity,
                        has_oracle=False,
                        exclusion_reason=blocked[task.id]["reason"],
                        evidence=(cohort_evidence,),
                    )
                    if task.id in blocked
                    else StaticRetainedTaskReceipt(
                        **identity,
                        native_task_checksum=checksums.get(task.id),
                        input_cohort_sha256=digest,
                        policy_sha256=context.policy.sha256,
                    )
                )
                if task.id not in blocked:
                    kept.append(offset + index)
            offset += batch.num_rows
    if len(identifiers) != context.current_input.measured_tasks:
        raise ValueError("Static retention input does not cover the complete frozen source population")
    receipt = OracleFilterReceipt(
        version=5,
        no_oracle_executability_policy=context.policy,
        no_oracle_source=context,
        recipe_identity=recipe_identity,
        input_cohort_sha256=digest,
        input_rows=len(identifiers),
        conversion_rejections=0,
        tasks=entries,
    )
    output.mkdir(parents=True, exist_ok=False)
    # A whole-table take combines string chunks and can exceed Arrow's 2 GB offset limit.
    offset = 0
    with (
        pq.ParquetFile(cohort) as parquet,
        pq.ParquetWriter(output / "tasks.parquet", schema, compression="zstd") as writer,
    ):
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_ROWS):
            end = offset + batch.num_rows
            indices = [index - offset for index in kept[bisect_left(kept, offset) : bisect_left(kept, end)]]
            if indices:
                writer.write_batch(batch.take(pa.array(indices, type=pa.int64())))
            offset = end
    (output / "oracle-filter.json").write_text(receipt.model_dump_json(indent=2) + "\n")
    (output / "static-exclusions.json").write_text(json.dumps(blocked, sort_keys=True, indent=2) + "\n")
    exclusion_counts = dict(Counter(NoOracleStaticBlocker(item["reason"].value) for item in blocked.values()))
    filter_hash = file_sha256(StoragePath(str(output / "oracle-filter.json")))
    summary = {
        "source": context.model_dump(mode="json"),
        "input_rows": len(identifiers),
        "static_retained_count": len(kept),
        "unique_static_excluded_count": len(blocked),
        "exclusion_counts": exclusion_counts,
        "executability": "unmeasured",
        "docker_build_execution": docker_build.value,
        "source_commands": 0,
        "provider_creations": 0,
        "model_requests": 0,
        "tasks_sha256": file_sha256(StoragePath(str(output / "tasks.parquet"))),
        "oracle_filter_sha256": filter_hash,
    }
    evidence = NoOracleStaticRetentionEvidence(
        source_id=context.source_id,
        source_release=context.source_release,
        current_input=context.current_input,
        intended_use=context.intended_use,
        retained_cohort=cohort_evidence,
        retained_count=len(identifiers),
        static_retained_count=len(kept),
        exclusion_counts=exclusion_counts,
        oracle_filter=ContentEvidenceReference(uri=str((output / "oracle-filter.json").resolve()), sha256=filter_hash),
        policy_sha256=context.policy.sha256,
    )
    (output / "static-retention-evidence.json").write_text(evidence.model_dump_json(indent=2) + "\n")
    (output / "static-retention.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--cohort-evidence", required=True, type=Path)
    parser.add_argument("--source-context", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--recipe-identity")
    parser.add_argument(
        "--docker-build-execution",
        type=DockerBuildExecution,
        choices=list(DockerBuildExecution),
        default=DockerBuildExecution.REJECT,
    )
    args = parser.parse_args()
    collect_static_retention(
        args.cohort,
        NoOracleSourceContext.model_validate_json(args.source_context.read_text()),
        ContentEvidenceReference.model_validate_json(args.cohort_evidence.read_text()),
        args.output_dir,
        recipe_identity=args.recipe_identity,
        docker_build=args.docker_build_execution,
    )


if __name__ == "__main__":
    main()
