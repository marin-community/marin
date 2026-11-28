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

The ``completion`` subcommand builds the fresh-review completion queue the
``sample_review_completion`` runner consumes: sources whose current panel's
conclusive coverage is incomplete get the exact top-up task IDs their receipt's
missing observations enumerate, beside the derivation record.
"""

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

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
    EquivalenceAllowance,
    GraderEnvironmentReceipt,
    GraderEnvironmentRefused,
    NormalizationRefused,
    SampleRecovery,
    build_grader_environment_receipt,
    build_normalization_receipt,
    earlier_panel_evidence,
    panel_observations,
    plan_review_carry_forward,
    population_identities,
    sample_recovery_outcome,
    sample_topup_ids,
)
from experiments.post_training.task_curation.sample_qc_binding import SampleQCReceipt
from experiments.post_training.task_curation.sample_review_completion import CompletionQueue, CompletionTarget
from infra.marina.apps.rl_data_catalog.server.source_identity import SourceId

logger = logging.getLogger(__name__)

REGISTRY_REPO = "open-athena/marin-rl-data-atlas-registry"
RECEIPT_PARTS_DIRECTORY = "experiments/post_training/task_curation/datasets/sample_qc_receipts"
DEFAULT_NORMALIZATION_FIELDS = ("execution_requirements",)

ReceiptReference = RegistryReference
"""One registry revision's exact PART file reference, reusing the registry's
pinned model; the source's receipt document is an entry of that part."""


class RecoveryTarget(BaseModel):
    """One source's current receipt and its superseded prior receipts.

    Receipt references pin the registry PART file that holds the source's
    receipt document; ``read_receipt_reference`` extracts the source's entry
    after verifying the part's bytes.
    """

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
    """Verify one receipt part's bytes and extract the source's receipt."""
    part = read_registry_document(reference)
    document = part.get(source_id)
    if document is None:
        raise ValueError(f"Receipt part holds no entry for this source: {reference.path}")
    if document.get("source_id") != source_id:
        raise ValueError(f"Receipt belongs to another source: {document.get('source_id')}")
    return SampleQCReceipt.model_validate(document)


def collect_receipt_history(
    documents_by_revision: dict[str, dict[str, tuple[ReceiptReference, dict]]],
) -> dict[str, list[ReceiptReference]]:
    """Group a registry's receipt documents per source, newest first.
    ``documents_by_revision`` maps revision → source ID → (part reference,
    parsed receipt document), newest revision first, where each part reference
    pins the part file's exact bytes. Blank shard placeholders carry nothing.
    A source's prior evidence is every distinct document, deduplicated by its
    canonical digest, whose panel pin differs from its newest revision's panel
    pin.
    """
    revisions = list(documents_by_revision)
    newest = documents_by_revision[revisions[0]]
    current_pin = {
        source_id: document["review"]["panel"]["dataset_revision"]
        for source_id, (_, document) in newest.items()
        if document.get("review")
    }
    history: dict[str, dict[str, ReceiptReference]] = {}
    for revision in revisions:
        for source_id, (reference, document) in documents_by_revision[revision].items():
            if not document.get("review"):
                continue
            if document["review"]["panel"]["dataset_revision"] == current_pin.get(source_id):
                continue
            sha256 = document_sha256(document)
            entry = history.setdefault(source_id, {})
            entry.setdefault(sha256, reference)
    return {source_id: list(references.values()) for source_id, references in sorted(history.items()) if references}


def survey_queue() -> RecoveryQueue:
    """Walk the registry's receipt history and build the recovery queue."""
    api = HfApi()
    revisions = [commit.commit_id for commit in api.list_repo_commits(REGISTRY_REPO, repo_type="dataset")]
    documents_by_revision: dict[str, dict[str, tuple[ReceiptReference, dict]]] = {}
    for revision in revisions:
        parts = [
            name
            for name in api.list_repo_files(REGISTRY_REPO, repo_type="dataset", revision=revision)
            if name.startswith(f"{RECEIPT_PARTS_DIRECTORY}/") and name.endswith(".json")
        ]
        if not parts:
            continue
        documents: dict[str, tuple[ReceiptReference, dict]] = {}
        for name in parts:
            downloaded = hf_hub_download(REGISTRY_REPO, repo_type="dataset", revision=revision, filename=name)
            part_bytes = Path(downloaded).read_bytes()
            reference = ReceiptReference(
                repo=REGISTRY_REPO,
                revision=revision,
                path=name,
                sha256=hashlib.sha256(part_bytes).hexdigest(),
            )
            for source_id, document in json.loads(part_bytes).items():
                documents[source_id] = (reference, document)
        documents_by_revision[revision] = documents
    prior = collect_receipt_history(documents_by_revision)
    newest = documents_by_revision[revisions[0]]
    current = {source_id: reference for source_id, (reference, document) in newest.items() if document.get("review")}
    targets = [
        RecoveryTarget(
            source_id=source_id,
            current=current[source_id],
            prior=tuple(prior[source_id]),
            current_population=newest[source_id][1]["review"]["panel"]["input_artifact"],
        )
        for source_id in sorted(prior)
        if source_id in current
    ]
    return RecoveryQueue(targets=targets)


def _newest_sample_receipts() -> dict[str, tuple[ReceiptReference, SampleQCReceipt]]:
    """Verify and return the newest revision's sample receipts, keyed by source.

    The newest revision holding sample receipt parts is the current selection:
    the family appends only, so its parts carry every source's current receipt.
    """
    api = HfApi()
    for commit in api.list_repo_commits(REGISTRY_REPO, repo_type="dataset"):
        parts = [
            name
            for name in api.list_repo_files(REGISTRY_REPO, repo_type="dataset", revision=commit.commit_id)
            if name.startswith(f"{RECEIPT_PARTS_DIRECTORY}/") and name.endswith(".json")
        ]
        if not parts:
            continue
        receipts: dict[str, tuple[ReceiptReference, SampleQCReceipt]] = {}
        for name in parts:
            downloaded = hf_hub_download(REGISTRY_REPO, repo_type="dataset", revision=commit.commit_id, filename=name)
            part_bytes = Path(downloaded).read_bytes()
            reference = ReceiptReference(
                repo=REGISTRY_REPO,
                revision=commit.commit_id,
                path=name,
                sha256=hashlib.sha256(part_bytes).hexdigest(),
            )
            document = json.loads(part_bytes)
            for source_id, row in document.items():
                if not isinstance(row, dict) or not row.get("review"):
                    continue
                receipts[source_id] = (reference, read_receipt_reference(reference, source_id))
        return receipts
    raise ValueError(f"The registry holds no sample receipt parts: {REGISTRY_REPO}")


def completion_queue(output_root: str, source_ids: tuple[str, ...]) -> CompletionQueue:
    """Build the fresh-review completion queue the sample completion runner consumes.

    A source's own receipt enumerates its missing observations — the fail-closed
    qualification refuses anything else — and each missing task's fresh review
    completes the panel's conclusive coverage. The derivation record written
    beside the queue carries the verified receipt reference and the count math,
    so every target's evidence URL resolves to how its tasks were derived.
    """
    configure_coreweave_s3()
    receipts = _newest_sample_receipts()
    unknown = sorted(set(source_ids) - set(receipts))
    if unknown:
        raise ValueError(f"No current sample receipt for: {', '.join(unknown)}")
    targets = []
    for source_id in source_ids:
        reference, receipt = receipts[source_id]
        missing = receipt.missing_observations
        if not missing:
            continue
        review = receipt.review
        task_ids = tuple(row.review.task_id for row in missing)
        derivation = {
            "kind": "sample-completion-derivation",
            "source_id": source_id,
            "receipt": {
                "repo": reference.repo,
                "revision": reference.revision,
                "path": reference.path,
                "sha256": reference.sha256,
            },
            "panel": {
                "dataset_id": review.panel.dataset_id,
                "dataset_revision": review.panel.dataset_revision,
                "sample_size": review.panel.sample_size,
                "input_artifact": review.panel.input_artifact,
            },
            "reviewed": review.result.reviewed,
            "attempted": review.attempted_count,
            "missing_task_ids": list(task_ids),
        }
        key = canonical_sha256({"kind": "completion-derivation", "source": source_id})[:20]
        record_path = prefix_join(output_root, f"completion-derivation/{key}.json")
        StoragePath(record_path).parent.mkdirs(exist_ok=True)
        StoragePath(record_path).write_text(json.dumps(derivation, indent=2, sort_keys=True) + "\n")
        targets.append(
            CompletionTarget(
                source_id=source_id,
                population=review.panel.input_artifact,
                task_ids=task_ids,
                topup_evidence=record_path,
            )
        )
    return CompletionQueue(targets=tuple(targets))


@dataclass(frozen=True)
class PriorOutcome:
    reference: ReceiptReference
    reviewer_sha256: str
    carried: int
    normalization: ContractNormalization | None
    grader_environment: GraderEnvironmentReceipt | None = None
    carried_reviews: tuple = ()
    reviewer: dict | None = None
    refusal: str | None = None


def recover_target(
    target: RecoveryTarget,
    output_root: str,
    normalization_fields: tuple[str, ...],
    allowances: tuple[EquivalenceAllowance, ...] = (),
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
        try:
            outcome = _recover_prior(
                target, output_root, normalization_fields, allowances, reference, current_population, current_identities
            )
        except ValueError as error:
            logger.warning("Prior receipt refused for %s from %s: %s", target.source_id, reference.revision[:12], error)
            priors.append(
                PriorOutcome(reference=reference, reviewer_sha256="", carried=0, normalization=None, refusal=str(error))
            )
            continue
        if outcome.reviewer not in reviewers:
            reviewers.append(outcome.reviewer)
        for review in outcome.carried_reviews:
            carried.setdefault(review.task_identity_sha256, review)
        priors.append(outcome)
    recovery = sample_recovery_outcome(
        target.source_id,
        current_population.task_count,
        panel_reviews,
        tuple(carried.values()),  # type: ignore[arg-type]
        reviewers=tuple(reviewers) or (current_receipt.review.reviewer,),
    )
    topup = sample_topup_ids(
        current_population.task_count,
        panel_reviews,
        tuple(carried.values()),  # type: ignore[arg-type]
        current_identities(frozenset()).keys(),
    )
    return recovery.model_copy(update={"topup_ids": topup}), priors


def _recover_prior(
    target: RecoveryTarget,
    output_root: str,
    normalization_fields: tuple[str, ...],
    allowances: tuple[EquivalenceAllowance, ...],
    reference: ReceiptReference,
    current_population: SourceTextPopulation,
    current_identities,
) -> PriorOutcome:
    """Carry one prior receipt's conclusive reviews onto the current pin.

    Tries the absent-versus-null normalization for each requested field, then
    the grading-environment-only equivalence; the first verified record binds
    the pair and the identity maps follow its terms.
    """
    prior_receipt = read_receipt_reference(reference, target.source_id)
    prior_population = SourceTextPopulation.raw_load(prior_receipt.review.panel.input_artifact)
    normalization = None
    grader_environment = None
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
    if normalization is None:
        try:
            grader_environment = build_grader_environment_receipt(
                target.source_id,
                prior_population.path,
                current_population.path,
                _grader_environment_evidence_url(output_root, target, reference),
                allowances=allowances,
            )
        except GraderEnvironmentRefused as error:
            logger.info(
                "Grading-environment equivalence refused for %s from %s: %s",
                target.source_id,
                reference.revision[:12],
                error,
            )
    evidence = earlier_panel_evidence(prior_receipt, _evidence_reference(reference), normalized_fields=normalized_fields)
    plan = plan_review_carry_forward(
        target.source_id,
        evidence,
        prior_population,
        current_population,
        population_identities(prior_population, normalized_fields=normalized_fields),
        current_identities(normalized_fields),
        normalization=normalization,
        grader_environment=grader_environment,
    )
    return PriorOutcome(
        reference=reference,
        reviewer_sha256=canonical_sha256(evidence.reviewer),
        carried=len(plan.carried_reviews),
        normalization=normalization,
        grader_environment=grader_environment,
        carried_reviews=plan.carried_reviews,
        reviewer=evidence.reviewer,
    )


def _evidence_reference(reference: ReceiptReference) -> ContentEvidenceReference:
    return ContentEvidenceReference(
        uri=f"registry:{reference.repo}@{reference.revision}:{reference.path}", sha256=reference.sha256
    )


def _normalization_evidence_url(
    output_root: str, target: RecoveryTarget, reference: ReceiptReference, field: str
) -> str:
    key = canonical_sha256({"source": target.source_id, "revision": reference.revision, "field": field})[:20]
    return prefix_join(output_root, f"normalization/{key}.parquet")


def _grader_environment_evidence_url(output_root: str, target: RecoveryTarget, reference: ReceiptReference) -> str:
    seed = {"kind": "grader-environment", "revision": reference.revision, "source": target.source_id}
    key = canonical_sha256(seed)[:20]
    return prefix_join(output_root, f"grader-environment/{key}.parquet")


def _equivalence_allowances(requested: tuple[str, ...]) -> tuple[EquivalenceAllowance, ...]:
    """The typed allowances a recovery run requests, by name."""
    known = {allowance.value: allowance for allowance in EquivalenceAllowance}
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ValueError(f"Unknown equivalence allowances: {unknown}")
    return tuple(known[name] for name in sorted(set(requested)))


def recover_queue(
    queue: RecoveryQueue, output_root: str, normalization_fields: tuple[str, ...], allowances=()
) -> list[dict]:
    """Recover every queued source and write its record beside the summary."""
    records = []
    for target in queue.targets:
        recovery, priors = recover_target(target, output_root, normalization_fields, allowances)
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
            "topup_count": len(recovery.topup_ids),
            "topup_ids": list(recovery.topup_ids),
            "reviewer_sha256s": [canonical_sha256(reviewer) for reviewer in recovery.reviewers],
            "priors": [
                {
                    "revision": outcome.reference.revision,
                    "receipt_sha256": outcome.reference.sha256,
                    "reviewer_sha256": outcome.reviewer_sha256,
                    "carried": outcome.carried,
                    "normalization_field": outcome.normalization.field if outcome.normalization else None,
                    "grader_environment": outcome.grader_environment is not None,
                    "refusal": outcome.refusal,
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
    recover.add_argument(
        "--allowance",
        action="append",
        default=[],
        help="Typed equivalence allowance to attempt (calibration_completion, actor_setup_pinning, lock_resources)",
    )
    recover.add_argument("--run", action="store_true")
    completion = subparsers.add_parser(
        "completion", help="Write the fresh-review completion queue for sources with incomplete panel coverage"
    )
    completion.add_argument("--source-id", action="append", required=True, help="The source to complete; repeatable.")
    completion.add_argument("--output", required=True, help="Where the queue document is written.")
    completion.add_argument("--output-root", required=True, help="Where the derivation records are written.")
    args = parser.parse_args()
    configure_coreweave_s3()
    if args.command == "survey":
        queue = survey_queue()
        StoragePath(args.output).parent.mkdirs(exist_ok=True)
        StoragePath(args.output).write_text(json.dumps(queue.model_dump(mode="json"), indent=2) + "\n")
        print(json.dumps({"targets": len(queue.targets), "output": args.output}))
        return
    if args.command == "completion":
        queue = completion_queue(args.output_root, tuple(args.source_id))
        output = StoragePath(args.output)
        output.parent.mkdirs(exist_ok=True)
        output.write_text(json.dumps(queue.model_dump(mode="json"), indent=2) + "\n")
        print(
            json.dumps(
                {
                    "targets": len(queue.targets),
                    "tasks": sum(len(target.task_ids) for target in queue.targets),
                    "queue_sha256": file_sha256(output),
                    "output": args.output,
                }
            )
        )
        return
    queue_path = StoragePath(args.queue)
    if file_sha256(queue_path) != args.queue_sha256:
        raise ValueError("Recovery queue changed")
    queue = RecoveryQueue.model_validate_json(queue_path.read_text())
    if not args.run:
        print(json.dumps({"targets": len(queue.targets)}))
        return
    fields = tuple(args.normalization_field) or DEFAULT_NORMALIZATION_FIELDS
    allowances = _equivalence_allowances(tuple(args.allowance))
    records = recover_queue(queue, args.output_root, fields, allowances)
    summary = {
        "targets": len(queue.targets),
        "sources": [
            {
                "source_id": record["source_id"],
                "retained": record["retained"],
                "carried_reviews": record["carried_reviews"],
                "coverage_percent": round(100 * record["reviewed_fraction"], 2),
                "became_full": record["became_full"],
                "topup_count": record["topup_count"],
            }
            for record in records
        ],
    }
    StoragePath(prefix_join(args.output_root, "summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["sources"], indent=2))


if __name__ == "__main__":
    main()
