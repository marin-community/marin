# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write canonical wire5 evidence for unchanged, complete no-oracle populations."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import ContentEvidenceReference, NoOracleStaticBlocker, NoOracleStaticRetentionEvidence
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.oracle_filter import MechanicalOracleTaskReceipt, OracleExclusionReason, OracleFilterReceipt
from taskcompendium.pipeline.static_retention import NoOracleSourceContext, StaticRetainedTaskReceipt
from taskcompendium.runtime.resources import has_oracle_solution

from experiments.post_training.task_curation.calibration_inventory import task_blockers
from experiments.post_training.task_curation.harbor_inspection import native_task_checksums
from experiments.post_training.task_curation.oracle_filter import cohort_sha256, row_runtime_hash, row_task

STATIC_REASONS = {
    "grading_environment_mismatch": OracleExclusionReason.SOURCE_ENV_INCOMPATIBLE_WITH_GRADER,
    "grading_environment_unavailable": OracleExclusionReason.SOURCE_ENV_INCOMPATIBLE_WITH_GRADER,
    "no_executable_task_spec_grader": OracleExclusionReason.SOURCE_ENV_INCOMPATIBLE_WITH_GRADER,
    "task_spec_tool_provider_required": OracleExclusionReason.TOOL_PROVIDER_REQUIRED,
    "task_spec_shared_executor_required": OracleExclusionReason.SHARED_EXECUTOR_REQUIRED,
    "task_spec_state_capture_required": OracleExclusionReason.BACKEND_CONTRACT_UNSUPPORTED,
    "task_spec_provider_contract_invalid": OracleExclusionReason.BACKEND_CONTRACT_UNSUPPORTED,
    "task_spec_unresolved_build": OracleExclusionReason.ENVIRONMENT_CONTRACT_UNRESOLVED,
    "mutable_image": OracleExclusionReason.UNPINNED_ENVIRONMENT,
    "mutable_docker_base": OracleExclusionReason.UNPINNED_ENVIRONMENT,
    "missing_environment": OracleExclusionReason.ENVIRONMENT_CONTRACT_UNRESOLVED,
    "missing_docker_base": OracleExclusionReason.ENVIRONMENT_CONTRACT_UNRESOLVED,
    "missing_packages_lock": OracleExclusionReason.ENVIRONMENT_CONTRACT_UNRESOLVED,
}


def collect_static_retention(
    cohort: Path,
    context: NoOracleSourceContext,
    cohort_evidence: ContentEvidenceReference,
    output: Path,
    *,
    recipe_identity: str | None = None,
) -> OracleFilterReceipt:
    """Retain original task/runtime rows without calibration, liveness, or model execution."""
    digest = cohort_sha256(cohort)
    if digest != context.input_cohort_sha256 or cohort_evidence.sha256 != digest:
        raise ValueError("Static retention input differs from the independently frozen cohort")
    table = pq.read_table(cohort)
    tasks = {}
    rows = table.to_pylist()
    blocked = {}
    for row in rows:
        task = row_task(row)
        if task is None or task.id in tasks or has_oracle_solution(task):
            raise ValueError("Static retention requires a complete, unique, known no-oracle population")
        tasks[task.id] = task
        blockers = task_blockers(task) - {"requires_calibration_solution_and_empty_liveness"}
        if blockers - STATIC_REASONS.keys():
            raise ValueError(f"Unclassified static blockers for {task.id}: {sorted(blockers)}")
        if blockers:
            # The persisted diagnostic preserves every reason. One deterministic
            # primary reason partitions unique exclusions without double counting.
            primary = next(reason for reason in STATIC_REASONS if reason in blockers)
            blocked[task.id] = {"reason": STATIC_REASONS[primary], "blockers": sorted(blockers)}
    if len(tasks) != context.current_input.measured_tasks:
        raise ValueError("Static retention input does not cover the complete frozen source population")
    checksums = native_task_checksums(task for task_id, task in tasks.items() if task_id not in blocked)
    entries = {}
    for row in rows:
        task = row_task(row)
        assert task is not None
        identity = {
            "task_sha256": canonical_sha256(task.model_dump(mode="json")),
            "runtime_sha256": row_runtime_hash(row),
        }
        entries[task.id] = (
            MechanicalOracleTaskReceipt(
                **identity, has_oracle=False, exclusion_reason=blocked[task.id]["reason"], evidence=(cohort_evidence,)
            )
            if task.id in blocked
            else StaticRetainedTaskReceipt(
                **identity,
                native_task_checksum=checksums.get(task.id),
                input_cohort_sha256=digest,
                policy_sha256=context.policy.sha256,
            )
        )
    receipt = OracleFilterReceipt(
        version=5,
        no_oracle_executability_policy=context.policy,
        no_oracle_source=context,
        recipe_identity=recipe_identity,
        input_cohort_sha256=digest,
        input_rows=len(tasks),
        conversion_rejections=0,
        tasks=entries,
    )
    kept = [index for index, task_id in enumerate(tasks) if task_id not in blocked]
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(table.take(pa.array(kept, type=pa.int64())), output / "tasks.parquet", compression="zstd")
    (output / "oracle-filter.json").write_text(receipt.model_dump_json(indent=2) + "\n")
    (output / "static-exclusions.json").write_text(json.dumps(blocked, sort_keys=True, indent=2) + "\n")
    exclusion_counts = dict(Counter(NoOracleStaticBlocker(item["reason"].value) for item in blocked.values()))
    filter_hash = file_sha256(StoragePath(str(output / "oracle-filter.json")))
    summary = {
        "source": context.model_dump(mode="json"),
        "input_rows": len(tasks),
        "static_retained_count": len(kept),
        "unique_static_excluded_count": len(blocked),
        "exclusion_counts": exclusion_counts,
        "executability": "unmeasured",
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
        retained_count=len(tasks),
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
    args = parser.parse_args()
    collect_static_retention(
        args.cohort,
        NoOracleSourceContext.model_validate_json(args.source_context.read_text()),
        ContentEvidenceReference.model_validate_json(args.cohort_evidence.read_text()),
        args.output_dir,
        recipe_identity=args.recipe_identity,
    )


if __name__ == "__main__":
    main()
