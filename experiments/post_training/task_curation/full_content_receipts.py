# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify whole-source review records before staging a final binding transport."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from finestore.rl.curation import ActorCensusInput, ContentEvidenceReference, QualityResultFinalization
from marin.execution.artifact import RECORD_FILENAME
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import ReviewRecord, ReviewStatus
from taskcompendium.pipeline.reviewer_spec import ReviewerSpec

from experiments.post_training.task_curation.census_binding import receipt_from_census
from experiments.post_training.task_curation.full_content_results import (
    FullContentResultBinding,
)
from experiments.post_training.task_curation.full_qc import REVIEW_RUBRIC, ReviewerVersion, local_reviewer
from experiments.post_training.task_curation.full_qc_artifacts import FullQCReview
from experiments.post_training.task_curation.full_qc_inputs import task_contract
from experiments.post_training.task_curation.full_qc_repair import (
    REPAIR_MAX_TOKENS,
    REPAIR_RETRY_MAX_TOKENS,
    primary_records,
    repair_identity,
)
from experiments.post_training.task_curation.full_qc_review import quality_result
from experiments.post_training.task_curation.lexical_artifacts import ActorCensus, SourceTextPopulation
from experiments.post_training.task_curation.published_census import population_documents


def full_content_result_binding(
    review_receipt: ContentEvidenceReference,
    finalization: QualityResultFinalization,
    waiver: ContentEvidenceReference | None,
    reviewer: ReviewerSpec,
    actor_input: ActorCensusInput,
    population: SourceTextPopulation | None = None,
) -> FullContentResultBinding:
    """Recompute a final Result from SHA-checked observations and exact retained contracts."""
    receipt_path = StoragePath(review_receipt.uri)
    if receipt_path.name != RECORD_FILENAME or file_sha256(receipt_path) != review_receipt.sha256:
        raise ValueError("Whole-source review receipt changed")
    review = FullQCReview.raw_load(str(receipt_path.parent))
    primary_identity = {"mode": "chat", **reviewer.identity}
    repair = replace(reviewer, max_tokens=REPAIR_MAX_TOKENS, retry_max_tokens=REPAIR_RETRY_MAX_TOKENS)
    identities = (primary_identity, repair_identity(primary_identity, {"mode": "chat", **repair.identity}))
    if canonical_sha256(review.reviewer) not in {canonical_sha256(identity) for identity in identities}:
        raise ValueError("Whole-source Result uses another pinned reviewer")
    records = primary_records(review, REVIEW_RUBRIC)
    observations = [ReviewRecord.model_validate_json(row["review"]) for row in records.values()]
    if (
        quality_result(observations) != review.result
        or review.attempted_count != len(observations)
        or review.coverage_complete != all(row.status is ReviewStatus.REVIEWED for row in observations)
    ):
        raise ValueError("Whole-source Result differs from its task-level observations")
    population = population or review.inputs.population
    if population.source_id != review.source_id or population.task_count != review.inputs.retained_count:
        raise ValueError("Final binding requires an exact retained population census")
    if (
        actor_input.dataset_id,
        actor_input.dataset_revision,
        actor_input.converter_version,
        actor_input.converter_revision,
        actor_input.measured_tasks,
        actor_input.task_specs_sha256,
    ) != (
        population.dataset_id,
        population.dataset_revision,
        population.source_version,
        population.converter_revision,
        population.task_count,
        population.task_specs_sha256,
    ):
        raise ValueError("Actor census differs from the reviewed retained population")
    seen = set()
    for document, runtime in population_documents(population):
        task = TaskSpec.model_validate(document)
        contract = task_contract(task, runtime)
        if task.id in seen or task.id not in records or contract != {key: records[task.id][key] for key in contract}:
            raise ValueError("Final binding population differs from reviewed task/runtime contracts")
        seen.add(task.id)
    if seen != set(records):
        raise ValueError("Final binding population does not cover every retained review")
    if waiver is not None:
        path = StoragePath(waiver.uri)
        if file_sha256(path) != waiver.sha256:
            raise ValueError("Whole-source finalization waiver changed")
        evidence = json.loads(path.read_text())
        if (
            evidence["source_id"],
            evidence["retained_count"],
            evidence["contract_index_sha256"],
            evidence["review_artifact_sha256"],
            evidence["records_sha256"],
        ) != (
            review.source_id,
            len(records),
            review.inputs.index_sha256,
            review_receipt.sha256,
            review.records_sha256,
        ):
            raise ValueError("Finalization waiver belongs to another whole-source review")
        unreviewed = {row.task_id for row in observations if row.status is not ReviewStatus.REVIEWED}
        waived = evidence["qc_unreviewed"]
        if (
            len(waived) != len(unreviewed)
            or {row["task_id"] for row in waived} != unreviewed
            or any(not row["reason"].strip() for row in waived)
        ):
            raise ValueError("Finalization waiver must enumerate every unreviewed task with its reason")
        approved_result = evidence["result"]
        if any(
            approved_result[name] != getattr(review.result, name)
            for name in ("reviewed", "issue_count", "uncertain_count", "unavailable_count")
        ):
            raise ValueError("Finalization waiver differs from the recorded Result and uncertainty counts")
    return FullContentResultBinding(
        source_id=review.source_id,
        input=actor_input,
        contract_index=ContentEvidenceReference(uri=review.inputs.index_path, sha256=review.inputs.index_sha256),
        review_receipt=review_receipt,
        records=ContentEvidenceReference(uri=review.records_path, sha256=review.records_sha256),
        result=review.result.model_copy(update={"issues": []}),
        coverage_complete=review.coverage_complete,
        finalization=finalization,
        waiver=waiver,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-receipt", required=True)
    parser.add_argument("--review-receipt-sha256", required=True)
    parser.add_argument("--actor-census", required=True)
    parser.add_argument("--reviewer-version", type=ReviewerVersion, choices=list(ReviewerVersion), required=True)
    parser.add_argument("--prompt-budget", type=int, required=True)
    parser.add_argument(
        "--finalization", type=QualityResultFinalization, choices=list(QualityResultFinalization), required=True
    )
    parser.add_argument("--waiver")
    parser.add_argument("--waiver-sha256")
    parser.add_argument("--population", help="Exact retained census when the original review population included drops")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve the staged binding and choose a new output: {args.output}")
    configure_coreweave_s3()
    census = ActorCensus.raw_load(args.actor_census)
    binding = full_content_result_binding(
        ContentEvidenceReference(uri=args.review_receipt, sha256=args.review_receipt_sha256),
        args.finalization,
        ContentEvidenceReference(uri=args.waiver, sha256=args.waiver_sha256) if args.waiver is not None else None,
        replace(
            local_reviewer(args.reviewer_version),
            prompt_budget=args.prompt_budget,
            retry_prompt_budget=args.prompt_budget,
        ),
        receipt_from_census(census.model_dump(mode="json"), None).input,
        SourceTextPopulation.raw_load(args.population) if args.population is not None else None,
    )
    if census.source_id != binding.source_id:
        raise ValueError("Actor census belongs to another source")
    args.output.write_text(binding.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
