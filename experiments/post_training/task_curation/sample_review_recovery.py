# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover every SAMPLE source's carried prior reviews at its current pin.

The runner surveys the private registry's receipt history for superseded
SAMPLE panels, then recovers each source in one in-region job:

- For every source with prior evidence, each prior receipt's panel reviews
  carry onto the current pin exactly as the typed carry-forward defines:
  conclusive reviews whose task-contract identity matches the retained
  population, under a per-pair absent-versus-null normalization when one
  verifies.
- The source's Result stays computed from its fixed random panel alone; the
  carried set's own rate is reported beside the coverage. The level promotes
  to FULL only when the merged observations cover every retained task under
  the existing completeness and waiver rules.
- No new reviews are launched: the recovery reports what already exists.

Prior FULL-run reviews ride the FULL campaign's own grandfathered receipts and
are not re-derived here.
"""

import argparse
import json
import logging
from dataclasses import dataclass

from finestore.rl.curation import ContentEvidenceReference
from huggingface_hub import HfApi, hf_hub_download
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.registry import RegistryReference, read_registry_document
from experiments.post_training.task_curation.review_carry_forward import (
    ContractNormalization,
    NormalizationRefused,
    SampleRecovery,
    build_normalization_receipt,
    earlier_panel_evidence,
    panel_observations,
    plan_review_carry_forward,
    population_identities,
    sample_recovery_outcome,
)
from experiments.post_training.task_curation.sample_qc_binding import SampleQCReceipt
from infra.marina.apps.rl_data_catalog.server.source_identity import SourceId

logger = logging.getLogger(__name__)

REGISTRY_REPO = "open-athena/marin-rl-data-atlas-registry"
RECEIPT_PARTS_DIRECTORY = "experiments/post_training/task_curation/datasets/sample_qc_receipts"
DEFAULT_NORMALIZATION_FIELDS = ("execution_requirements",)

ReceiptReference = RegistryReference
"""One registry revision's exact receipt document, reusing the registry's pinned model."""


class RecoveryTarget(BaseModel):
    """One source's current receipt and its superseded prior receipts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    current: ReceiptReference
    prior: tuple[ReceiptReference, ...] = ()
    current_population: str


class RecoveryQueue(BaseModel):
    targets: tuple[RecoveryTarget, ...]


def document_sha256(document: dict) -> str:
    """Digest one parsed receipt document with the shared canonical form."""
    return canonical_sha256(document)


def read_receipt_reference(reference: ReceiptReference, source_id: str) -> SampleQCReceipt:
    """Download and verify one receipt document, then validate it."""
    document = read_registry_document(reference)
    if document.get("source_id") != source_id:
        raise ValueError(f"Receipt belongs to another source: {document.get('source_id')}")
    return SampleQCReceipt.model_validate(document)


def collect_receipt_history(documents_by_revision: dict[str, dict[str, dict]]) -> dict[str, list[ReceiptReference]]:
    """Group a registry's receipt documents per source, newest first.

    ``documents_by_revision`` maps revision → path → parsed receipt document,
    newest revision first. A source's prior evidence is every distinct document
    whose panel pin differs from its newest revision's panel pin.
    """
    revisions = list(documents_by_revision)
    newest = documents_by_revision[revisions[0]]
    current_pin = {
        document["source_id"]: document["review"]["panel"]["dataset_revision"] for document in newest.values()
    }
    history: dict[str, dict[str, ReceiptReference]] = {}
    for revision in revisions:
        for path, document in documents_by_revision[revision].items():
            source_id = document["source_id"]
            if document["review"]["panel"]["dataset_revision"] == current_pin.get(source_id):
                continue
            sha256 = document_sha256(document)
            entry = history.setdefault(source_id, {})
            entry.setdefault(sha256, ReceiptReference(repo=REGISTRY_REPO, revision=revision, path=path, sha256=sha256))
    return {source_id: list(references.values()) for source_id, references in sorted(history.items()) if references}


def survey_queue() -> RecoveryQueue:
    """Walk the registry's receipt history and build the recovery queue."""
    api = HfApi()
    revisions = [commit.commit_id for commit in api.list_repo_commits(REGISTRY_REPO, repo_type="dataset")]
    documents_by_revision: dict[str, dict[str, dict]] = {}
    for revision in revisions:
        parts = [
            name
            for name in api.list_repo_files(REGISTRY_REPO, repo_type="dataset", revision=revision)
            if name.startswith(f"{RECEIPT_PARTS_DIRECTORY}/") and name.endswith(".json")
        ]
        if not parts:
            continue
        documents_by_revision[revision] = {
            name: json.loads(StoragePath(downloaded).read_text())
            for name in parts
            for downloaded in (hf_hub_download(REGISTRY_REPO, repo_type="dataset", revision=revision, filename=name),)
        }
    prior = collect_receipt_history(documents_by_revision)
    newest = documents_by_revision[revisions[0]]
    current = {
        document["source_id"]: ReceiptReference(
            repo=REGISTRY_REPO,
            revision=revisions[0],
            path=path,
            sha256=document_sha256(document),
        )
        for path, document in newest.items()
    }
    targets = [
        RecoveryTarget(
            source_id=source_id,
            current=current[source_id],
            prior=tuple(prior[source_id]),
            current_population=newest[current[source_id].path]["review"]["panel"]["input_artifact"],
        )
        for source_id in sorted(prior)
        if source_id in current
    ]
    return RecoveryQueue(targets=targets)


@dataclass(frozen=True)
class PriorOutcome:
    reference: ReceiptReference
    reviewer_sha256: str
    carried: int
    normalization: ContractNormalization | None


def recover_target(
    target: RecoveryTarget,
    output_root: str,
    normalization_fields: tuple[str, ...],
) -> tuple[SampleRecovery, list[PriorOutcome]]:
    """Recover one source's carried reviews and its augmented SAMPLE outcome."""
    current_receipt = read_receipt_reference(target.current, target.source_id)
    current_population = SourceTextPopulation.raw_load(target.current_population)
    panel_reviews = panel_observations(current_receipt)
    identity_cache: dict[frozenset[str], dict[str, str]] = {}

    def current_identities(normalized_fields: frozenset[str]) -> dict[str, str]:
        if normalized_fields not in identity_cache:
            identity_cache[normalized_fields] = population_identities(
                current_population, normalized_fields=normalized_fields
            )
        return identity_cache[normalized_fields]

    carried: dict[str, object] = {}
    priors: list[PriorOutcome] = []
    reviewers: list[dict] = []
    for reference in target.prior:
        prior_receipt = read_receipt_reference(reference, target.source_id)
        if prior_receipt.review.reviewer not in reviewers:
            reviewers.append(prior_receipt.review.reviewer)
        prior_population = SourceTextPopulation.raw_load(prior_receipt.review.panel.input_artifact)
        normalization = None
        normalized_fields: frozenset[str] = frozenset()
        for field in normalization_fields:
            try:
                normalization = build_normalization_receipt(
                    target.source_id,
                    prior_population.path,
                    current_population.path,
                    field,
                    _normalization_evidence_url(output_root, target, reference, field),
                )
            except NormalizationRefused as error:
                logger.info("Normalization refused for %s from %s: %s", target.source_id, reference.revision[:12], error)
                continue
            normalized_fields = frozenset({field})
            break
        evidence = earlier_panel_evidence(
            prior_receipt, _evidence_reference(reference), normalized_fields=normalized_fields
        )
        plan = plan_review_carry_forward(
            target.source_id,
            evidence,
            prior_population,
            current_population,
            population_identities(prior_population, normalized_fields=normalized_fields),
            current_identities(normalized_fields),
            normalization=normalization,
        )
        for review in plan.carried_reviews:
            carried.setdefault(review.task_identity_sha256, review)
        priors.append(
            PriorOutcome(
                reference=reference,
                reviewer_sha256=canonical_sha256(evidence.reviewer),
                carried=len(plan.carried_reviews),
                normalization=normalization,
            )
        )
    recovery = sample_recovery_outcome(
        target.source_id,
        current_population.task_count,
        panel_reviews,
        tuple(carried.values()),  # type: ignore[arg-type]
        reviewers=tuple(reviewers) or (current_receipt.review.reviewer,),
    )
    return recovery, priors


def _evidence_reference(reference: ReceiptReference) -> ContentEvidenceReference:
    return ContentEvidenceReference(
        uri=f"registry:{reference.repo}@{reference.revision}:{reference.path}", sha256=reference.sha256
    )


def _normalization_evidence_url(
    output_root: str, target: RecoveryTarget, reference: ReceiptReference, field: str
) -> str:
    key = canonical_sha256({"source": target.source_id, "revision": reference.revision, "field": field})[:20]
    return prefix_join(output_root, f"normalization/{key}.parquet")


def recover_queue(queue: RecoveryQueue, output_root: str, normalization_fields: tuple[str, ...]) -> list[dict]:
    """Recover every queued source and write its record beside the summary."""
    records = []
    for target in queue.targets:
        recovery, priors = recover_target(target, output_root, normalization_fields)
        record = {
            "source_id": recovery.source_id,
            "retained": recovery.retained,
            "panel_conclusive": recovery.panel_conclusive,
            "carried_reviews": recovery.carried_reviews,
            "carried_issue_count": recovery.carried_issue_count,
            "carried_issue_rate": (
                recovery.carried_issue_count / recovery.carried_reviews if recovery.carried_reviews else None
            ),
            "coverage_conclusive": recovery.coverage_conclusive,
            "reviewed_fraction": recovery.reviewed_fraction,
            "level": recovery.level.value,
            "became_full": recovery.full,
            "reviewer_sha256s": [canonical_sha256(reviewer) for reviewer in recovery.reviewers],
            "priors": [
                {
                    "revision": outcome.reference.revision,
                    "receipt_sha256": outcome.reference.sha256,
                    "reviewer_sha256": outcome.reviewer_sha256,
                    "carried": outcome.carried,
                    "normalization_field": outcome.normalization.field if outcome.normalization else None,
                }
                for outcome in priors
            ],
        }
        records.append(record)
        logger.info(
            "%s: retained %d, carried %d, coverage %.1f%%, full=%s",
            recovery.source_id,
            recovery.retained,
            recovery.carried_reviews,
            100 * recovery.reviewed_fraction,
            recovery.full,
        )
        key = canonical_sha256(recovery.source_id)[:20]
        StoragePath(prefix_join(output_root, f"records/{key}.json")).write_text(json.dumps(record, indent=2) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    survey = subparsers.add_parser("survey", help="Walk the registry history and write the recovery queue")
    survey.add_argument("--output", required=True)
    recover = subparsers.add_parser("recover", help="Recover every queued source in-region")
    recover.add_argument("--queue", required=True)
    recover.add_argument("--queue-sha256", required=True)
    recover.add_argument("--output-root", required=True)
    recover.add_argument("--normalization-field", action="append", default=[])
    recover.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    if args.command == "survey":
        queue = survey_queue()
        StoragePath(args.output).parent.mkdirs(exist_ok=True)
        StoragePath(args.output).write_text(json.dumps(queue.model_dump(mode="json"), indent=2) + "\n")
        print(json.dumps({"targets": len(queue.targets), "output": args.output}))
        return
    queue_path = StoragePath(args.queue)
    if file_sha256(queue_path) != args.queue_sha256:
        raise ValueError("Recovery queue changed")
    queue = RecoveryQueue.model_validate_json(queue_path.read_text())
    if not args.run:
        print(json.dumps({"targets": len(queue.targets)}))
        return
    fields = tuple(args.normalization_field) or DEFAULT_NORMALIZATION_FIELDS
    records = recover_queue(queue, args.output_root, fields)
    summary = {
        "targets": len(queue.targets),
        "sources": [
            {
                "source_id": record["source_id"],
                "retained": record["retained"],
                "carried_reviews": record["carried_reviews"],
                "coverage_percent": round(100 * record["reviewed_fraction"], 2),
                "became_full": record["became_full"],
            }
            for record in records
        ],
    }
    StoragePath(prefix_join(args.output_root, "summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["sources"], indent=2))


if __name__ == "__main__":
    main()
