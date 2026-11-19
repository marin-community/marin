# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Carry QC reviews across re-pinned populations on exact task-contract identity.

This module is the standard carry-forward for QC reviews when a source's pin
moves. Its contract:

- Reviews are keyed on each task's contract identity
  (``taskcompendium.identity.task_identity``), not on the artifact-level
  population digest or the task ID.
- A review from an earlier pin carries forward only when the identity matches
  exactly, and only when the earlier receipt recorded it as REVIEWED. The
  merged Result counts low-confidence and unknown-quality observations as
  uncertain, per the shared Result rules.
- Carried reviews keep the earlier receipt's recorded reviewer identity
  verbatim. Relabeling them as another reviewer's output is refused, matching
  the typed ``GrandfatheredReview`` marker's recorded decision evidence.
- The delta plan reports carried, changed, new and dropped counts and the exact
  task IDs that need review. SAMPLE panels keep their unchanged members and
  redraw only the replaced ones with the fixed seed.
- Oversight derivation uses carried plus fresh reviews: FULL when every
  retained task has an observation and none are unavailable under the existing
  completeness and waiver rules, SAMPLE otherwise.
"""

import argparse
import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import (
    FULL_WAIVER_MAX_UNREVIEWED_FRACTION,
    ContentEvidenceReference,
    OversightLevel,
    QualityResultFinalization,
    SourceQualityResult,
)
from marin.execution.artifact import RECORD_FILENAME
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import TASK_IDENTITY_EXCLUDED_FIELDS, canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import ReviewRecord, ReviewStatus
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample

from experiments.post_training.task_curation.content_identity import PopulationDigest
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.review_quality import is_conclusive, quality_result
from experiments.post_training.task_curation.sample_qc_artifacts import SAMPLE_SEED, SAMPLE_SIZE, SamplePanel
from experiments.post_training.task_curation.sample_qc_binding import SampleQCReceipt
from infra.marina.apps.rl_data_catalog.server.source_identity import SourceId

logger = logging.getLogger(__name__)

NORMALIZATION_EVIDENCE_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("original_sha256", pa.string()),
        ("reissued_sha256", pa.string()),
        ("normalized_sha256", pa.string()),
    ]
)


def contract_identity(document: dict, *, normalized_fields: frozenset[str] = frozenset()) -> str:
    """Hash a stored TaskSpec document exactly like ``task_identity``.

    The spec shards store each task's ``model_dump(mode="json")`` document, so
    dropping the same provenance keys and hashing the canonical JSON gives the
    task's contract identity without re-validating the model. A verified
    ``ContractNormalization`` adds its absent-versus-null field to the dropped
    keys on both sides of the comparison.
    """
    excluded = TASK_IDENTITY_EXCLUDED_FIELDS | normalized_fields
    payload = {key: value for key, value in document.items() if key not in excluded}
    return canonical_sha256(payload)


WAIVER_BOUND_SCALE = round(1 / FULL_WAIVER_MAX_UNREVIEWED_FRACTION)
"""Integer multiplier that mirrors ``FullContentResultBinding``'s exact
``unavailable * 100 > sample_count`` waiver test; 100 for the 1% bound."""


def _population_documents(population: SourceTextPopulation) -> Iterator[dict]:
    """Stream every retained task document in the population's stable shard order."""
    for path in sorted(StoragePath(prefix_join(population.specs_output_dir, "*.parquet")).glob(), key=str):
        with path.open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=512, columns=["task_spec"]):
                for encoded in batch.column(0).to_pylist():
                    yield json.loads(encoded)


def population_identities(
    population: SourceTextPopulation, *, normalized_fields: frozenset[str] = frozenset()
) -> dict[str, str]:
    """Map every retained task ID to its contract identity, streaming the spec shards.

    Unlike ``content_identity.population_task_digests`` (which hashes the exact
    stored bytes), this hashes each parsed contract minus its provenance fields,
    so identities match across re-serializations that keep the contract whole.
    Pass a verified normalization's field to compare absent-versus-null
    re-serializations as equal.
    """
    identities: dict[str, str] = {}
    for document in _population_documents(population):
        task_id = document["id"]
        if task_id in identities:
            raise ValueError(f"Duplicate task ID in {population.source_id}: {task_id}")
        identities[task_id] = contract_identity(document, normalized_fields=normalized_fields)
    if len(identities) != population.task_count:
        raise ValueError(
            f"{population.source_id}: read {len(identities)} task identities for {population.task_count} declared tasks"
        )
    return identities


class NormalizationRefused(ValueError):
    """The absent-versus-null normalization does not hold for this population pair."""


class ContractNormalization(BaseModel):
    """Typed absent-versus-explicit-null equivalence between two contract pins.

    A TaskSpec field whose documented default is null-or-absent may be absent
    in one serialization and explicitly null in the other. This record names
    that field, binds the two population digests, and records the verified
    evidence that every shared task differs in nothing else. The carry-forward
    identity treats the pair as equal; ``task_identity`` itself is never
    changed, and an explicit non-null value always refuses the record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    field: str
    original: PopulationDigest
    reissued: PopulationDigest
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Digest of the shared tasks' normalized identity map the builder verified."""
    evidence_url: str

    def binds(self, source_id: str, original: PopulationDigest, reissued: PopulationDigest) -> bool:
        """Whether this record binds exactly this source's population pair."""
        if source_id != self.source_id:
            return False
        return (original.task_specs_sha256, reissued.task_specs_sha256) == (
            self.original.task_specs_sha256,
            self.reissued.task_specs_sha256,
        )


def build_normalization_receipt(
    source_id: str,
    original_artifact: str,
    reissued_artifact: str,
    field: str,
    evidence_url: str,
) -> ContractNormalization:
    """Verify absent-versus-null is the only contract difference and record it.

    Requires ``field`` to be a TaskSpec field whose documented default is None.
    Every task shared by the two populations must differ in exactly that field:
    absent on one side, explicitly null on the other, with byte-equal payloads
    once the field is dropped. An explicit non-null value, or any other
    difference, refuses the record. The evidence parquet records each shared
    task's normalized identity; ``identity_sha256`` digests that map, so a plan
    can recompute it from its own identity maps and bind them to this record.
    """
    spec_field = TaskSpec.model_fields.get(field)
    if spec_field is None or spec_field.default is not None:
        raise NormalizationRefused(f"Normalization requires a TaskSpec field whose documented default is None: {field}")
    original = SourceTextPopulation.raw_load(original_artifact)
    reissued = SourceTextPopulation.raw_load(reissued_artifact)
    original_documents = {document["id"]: document for document in _population_documents(original)}
    reissued_documents = {document["id"]: document for document in _population_documents(reissued)}
    if len(original_documents) != original.task_count or len(reissued_documents) != reissued.task_count:
        raise ValueError(f"Normalization populations do not match their declared counts: {source_id}")
    shared = sorted(set(original_documents) & set(reissued_documents))
    normalized_map: dict[str, str] = {}
    evidence_rows = []
    for task_id in shared:
        earlier_document, current_document = original_documents[task_id], reissued_documents[task_id]
        earlier_has = field in earlier_document
        current_has = field in current_document
        if earlier_has == current_has:
            raise NormalizationRefused(
                f"Normalization refused for {task_id}: {field} must be absent on one side and present on the other"
            )
        present = earlier_document if earlier_has else current_document
        if present[field] is not None:
            raise NormalizationRefused(
                f"Normalization refused for {task_id}: {field} is explicitly set to a non-null value"
            )
        if contract_identity(earlier_document, normalized_fields={field}) != contract_identity(
            current_document, normalized_fields={field}
        ):
            raise NormalizationRefused(f"Normalization refused for {task_id}: the contracts differ beyond {field}")
        normalized_identity = contract_identity(earlier_document, normalized_fields={field})
        normalized_map[task_id] = normalized_identity
        evidence_rows.append(
            {
                "task_id": task_id,
                "original_sha256": contract_identity(earlier_document),
                "reissued_sha256": contract_identity(current_document),
                "normalized_sha256": normalized_identity,
            }
        )
    if not shared:
        raise NormalizationRefused(f"Normalization refused: the populations share no tasks: {source_id}")
    table = pa.Table.from_pylist(evidence_rows, schema=NORMALIZATION_EVIDENCE_SCHEMA)
    with StoragePath(evidence_url).open("wb", auto_mkdir=True) as stream:
        pq.write_table(table, stream, compression="zstd")
    logger.info(
        "Normalization %s: %d shared tasks equal modulo %s across %s and %s",
        source_id,
        len(shared),
        field,
        original.task_specs_sha256[:12],
        reissued.task_specs_sha256[:12],
    )
    return ContractNormalization(
        source_id=source_id,
        field=field,
        original=PopulationDigest(
            artifact=original_artifact, task_specs_sha256=original.task_specs_sha256, task_count=original.task_count
        ),
        reissued=PopulationDigest(
            artifact=reissued_artifact, task_specs_sha256=reissued.task_specs_sha256, task_count=reissued.task_count
        ),
        identity_sha256=canonical_sha256(normalized_map),
        evidence_url=evidence_url,
    )


class CarriedReview(BaseModel):
    """One earlier-pin review that carries forward on an exact identity match."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_task_id: str
    current_task_id: str
    review: ReviewRecord


class EarlierPanelEvidence(BaseModel):
    """An earlier pin's SAMPLE reviews, keyed by contract identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    panel: SamplePanel
    member_identities: dict[str, str]
    """Panel member task ID at the earlier pin → contract identity."""
    conclusive: dict[str, CarriedReview]
    """Contract identity → the carried conclusive review."""
    reviewer: dict
    """The earlier receipt's reviewer identity, recorded verbatim."""
    receipt: ContentEvidenceReference


def earlier_panel_evidence(
    receipt: SampleQCReceipt,
    receipt_reference: ContentEvidenceReference,
    *,
    normalized_fields: frozenset[str] = frozenset(),
) -> EarlierPanelEvidence:
    """Extract an earlier receipt's conclusive reviews, keyed by task-contract identity.

    Verifies the frozen panel and observation bytes, then keys every REVIEWED
    row by the contract identity computed from the panel's own task bytes.
    Unavailable or invalid rows never carry; their members stay on the panel as
    tasks that need review. Pass a verified normalization's field to key the
    identities modulo its absent-versus-null difference.
    """
    panel = receipt.review.panel
    panel_bytes = StoragePath(panel.panel_path)
    if file_sha256(panel_bytes) != panel.panel_sha256:
        raise ValueError(f"Sample panel changed: {receipt.source_id}")
    records = StoragePath(receipt.review.records_path)
    if file_sha256(records) != receipt.review.records_sha256:
        raise ValueError(f"Sample observations changed after review: {receipt.source_id}")
    with panel_bytes.open("rb") as stream:
        rows = pq.read_table(stream).to_pylist()
    member_identities: dict[str, str] = {}
    for row in rows:
        if row["task_id"] != json.loads(row["task_spec"])["id"]:
            raise ValueError(f"Sample panel IDs differ from their task contracts: {receipt.source_id}")
        member_identities[row["task_id"]] = contract_identity(
            json.loads(row["task_spec"]), normalized_fields=normalized_fields
        )
    if len(member_identities) != panel.sample_size:
        raise ValueError(f"Sample panel does not cover its selected members exactly once: {receipt.source_id}")
    with records.open("rb") as stream:
        observations = pq.read_table(stream).to_pylist()
    if len(observations) != panel.sample_size or len({row["task_id"] for row in observations}) != panel.sample_size:
        raise ValueError(f"Sample observations do not cover the panel exactly once: {receipt.source_id}")
    conclusive: dict[str, CarriedReview] = {}
    for row in observations:
        identity = member_identities.get(row["task_id"])
        if identity is None:
            raise ValueError(f"Sample observation belongs to another panel: {receipt.source_id}")
        review = ReviewRecord.model_validate_json(row["review"])
        if review.task_id != row["task_id"]:
            raise ValueError(f"Sample observation belongs to another task: {receipt.source_id}")
        if review.status is not ReviewStatus.REVIEWED:
            continue
        conclusive[identity] = CarriedReview(
            task_identity_sha256=identity,
            original_task_id=row["task_id"],
            current_task_id="",
            review=review,
        )
    return EarlierPanelEvidence(
        source_id=receipt.source_id,
        panel=panel,
        member_identities=member_identities,
        conclusive=conclusive,
        reviewer=receipt.review.reviewer,
        receipt=receipt_reference,
    )


class PopulationDelta(BaseModel):
    """Retained-population counts across the re-pin, by contract identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    carried: int = Field(ge=0)
    """Retained tasks whose contract identity existed at the earlier pin."""
    changed: int = Field(ge=0)
    """Retained tasks whose task ID existed earlier with a different identity."""
    new: int = Field(ge=0)
    """Retained tasks whose task ID did not exist at the earlier pin."""
    dropped: int = Field(ge=0)
    """Earlier task IDs no longer present in the retained population."""


class ReviewCarryForwardPlan(BaseModel):
    """The delta plan for carrying an earlier pin's reviews into a retained population."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    earlier_dataset_id: str
    earlier_dataset_revision: str
    earlier_task_count: int = Field(gt=0)
    current_dataset_id: str
    current_dataset_revision: str
    current_task_count: int = Field(gt=0)
    evidence: ContentEvidenceReference
    reviewer: dict
    """The earlier receipt's reviewer identity, recorded verbatim."""
    population: PopulationDelta
    carried_reviews: tuple[CarriedReview, ...]
    kept_members: tuple[str, ...]
    """Current task IDs that keep an earlier panel member's slot."""
    replaced_members: int = Field(ge=0)
    """Panel slots redrawn because their member's identity left the population."""
    needs_review: tuple[str, ...]
    """Exact retained task IDs that need a fresh review."""
    replacement_seed: int


def _verify_carry_forward_inputs(
    source_id: str,
    earlier: EarlierPanelEvidence,
    earlier_population: SourceTextPopulation,
    current_population: SourceTextPopulation,
    earlier_identities: dict[str, str],
    current_identities: dict[str, str],
    reviewer: dict | None,
    normalization: ContractNormalization | None,
) -> None:
    """Refuse evidence that does not describe exactly these two populations."""
    if earlier.source_id != source_id or current_population.source_id != source_id:
        raise ValueError(f"Carry-forward sources differ: {source_id}")
    if earlier_population.source_id != source_id:
        raise ValueError(f"Earlier population belongs to another source: {earlier_population.source_id}")
    panel = earlier.panel
    if (
        panel.dataset_id,
        panel.converter_version,
        panel.retained_tasks,
        panel.task_specs_sha256,
    ) != (
        earlier_population.dataset_id,
        earlier_population.source_version,
        earlier_population.task_count,
        earlier_population.task_specs_sha256,
    ):
        raise ValueError(f"Earlier panel belongs to another earlier population: {source_id}")
    if len(earlier_identities) != earlier_population.task_count:
        raise ValueError(f"Earlier identity map does not cover the earlier population: {source_id}")
    if len(current_identities) != current_population.task_count:
        raise ValueError(f"Current identity map does not cover the retained population: {source_id}")
    if reviewer is not None and canonical_sha256(reviewer) != canonical_sha256(earlier.reviewer):
        raise ValueError(
            "Carried reviews keep their original reviewer identity; refusing to relabel "
            f"{canonical_sha256(earlier.reviewer)[:12]} as {canonical_sha256(reviewer)[:12]}: {source_id}"
        )
    if normalization is not None:
        if not normalization.binds(
            source_id,
            PopulationDigest(
                artifact=earlier_population.path,
                task_specs_sha256=earlier_population.task_specs_sha256,
                task_count=earlier_population.task_count,
            ),
            PopulationDigest(
                artifact=current_population.path,
                task_specs_sha256=current_population.task_specs_sha256,
                task_count=current_population.task_count,
            ),
        ):
            raise ValueError(f"Normalization binds another population pair: {source_id}")
        shared = sorted(set(earlier_identities) & set(current_identities))
        verified = canonical_sha256({task_id: earlier_identities[task_id] for task_id in shared})
        if verified != normalization.identity_sha256:
            raise ValueError(
                f"Identity maps are not the normalized ones the receipt verified: {source_id}; build them with "
                f"population_identities(normalized_fields={{{normalization.field!r}}})"
            )


@dataclass(frozen=True)
class PopulationClassification:
    """Retained tasks classified against the earlier population's identities."""

    delta: PopulationDelta
    kept_members: list[str]
    needs_review: list[str]
    identity_to_current_id: dict[str, str]


def _classify_population(
    earlier: EarlierPanelEvidence,
    earlier_identities: dict[str, str],
    current_identities: dict[str, str],
) -> PopulationClassification:
    """Split the retained population into carried, changed and new tasks.

    A task is carried when its contract identity existed at the earlier pin.
    Panel members keep their slot; a member whose earlier observation was not
    conclusive keeps the slot but needs a fresh review. Changed and new tasks
    always need review.
    """
    member_identity_values = set(earlier.member_identities.values())
    earlier_identity_values = set(earlier_identities.values())
    identity_to_current_id: dict[str, str] = {}
    for task_id, identity in current_identities.items():
        identity_to_current_id.setdefault(identity, task_id)

    carried = changed = new = 0
    kept_members: list[str] = []
    needs_review: list[str] = []
    for task_id, identity in current_identities.items():
        if identity in earlier_identity_values:
            carried += 1
            if identity in member_identity_values:
                kept_members.append(task_id)
                if identity not in earlier.conclusive:
                    # The member is unchanged but its earlier observation was
                    # not conclusive: keep the slot and review the same task
                    # fresh.
                    needs_review.append(task_id)
            continue
        if task_id in earlier_identities:
            changed += 1
        else:
            new += 1
        needs_review.append(task_id)
    dropped = len(earlier_identities) - len(set(earlier_identities) & set(current_identities))
    return PopulationClassification(
        delta=PopulationDelta(carried=carried, changed=changed, new=new, dropped=dropped),
        kept_members=kept_members,
        needs_review=needs_review,
        identity_to_current_id=identity_to_current_id,
    )


def _replacement_draw(
    current_identities: dict[str, str], kept_members: list[str], *, seed: int, max_panel_size: int, source_id: str
) -> tuple[int, list[str]]:
    """Redraw only the replaced panel slots with the fixed seed."""
    panel_size = min(max_panel_size, len(current_identities))
    replaced = panel_size - len(kept_members)
    kept = set(kept_members)
    pool = [task_id for task_id in current_identities if task_id not in kept]
    _, replacements = seeded_sample(pool, size=replaced, key=lambda task_id: seeded_order(task_id, seed))
    if len(replacements) != replaced:
        raise ValueError(f"Replacement draw does not match the retained population: {source_id}")
    return replaced, replacements


def plan_review_carry_forward(
    source_id: str,
    earlier: EarlierPanelEvidence,
    earlier_population: SourceTextPopulation,
    current_population: SourceTextPopulation,
    earlier_identities: dict[str, str],
    current_identities: dict[str, str],
    *,
    reviewer: dict | None = None,
    seed: int = SAMPLE_SEED,
    max_panel_size: int = SAMPLE_SIZE,
    normalization: ContractNormalization | None = None,
) -> ReviewCarryForwardPlan:
    """Compute the typed carry-forward delta plan for a re-pinned population.

    ``earlier_identities`` and ``current_identities`` map task ID → contract
    identity for the earlier and current populations; build them with
    ``population_identities``. ``reviewer``, when supplied, must be the earlier
    receipt's recorded reviewer identity: carried reviews are never relabeled.
    ``normalization``, when supplied, must be the verified absent-versus-null
    record for this population pair, and the identity maps must be built with
    its field excluded. Kept panel members are the unchanged ones; replaced
    slots are redrawn with the fixed seed, and members without a conclusive
    review keep their slot but need a fresh review.
    """
    _verify_carry_forward_inputs(
        source_id,
        earlier,
        earlier_population,
        current_population,
        earlier_identities,
        current_identities,
        reviewer,
        normalization,
    )
    classification = _classify_population(earlier, earlier_identities, current_identities)
    replaced, replacements = _replacement_draw(
        current_identities, classification.kept_members, seed=seed, max_panel_size=max_panel_size, source_id=source_id
    )
    needs_review = [*classification.needs_review, *replacements]
    carried_reviews = tuple(
        CarriedReview(
            task_identity_sha256=identity,
            original_task_id=earlier.conclusive[identity].original_task_id,
            current_task_id=classification.identity_to_current_id[identity],
            review=earlier.conclusive[identity].review,
        )
        for identity in sorted(classification.identity_to_current_id.keys() & earlier.conclusive.keys())
    )
    plan = ReviewCarryForwardPlan(
        source_id=source_id,
        earlier_dataset_id=earlier_population.dataset_id,
        earlier_dataset_revision=earlier_population.dataset_revision,
        earlier_task_count=earlier_population.task_count,
        current_dataset_id=current_population.dataset_id,
        current_dataset_revision=current_population.dataset_revision,
        current_task_count=len(current_identities),
        evidence=earlier.receipt,
        reviewer=earlier.reviewer,
        population=classification.delta,
        carried_reviews=carried_reviews,
        kept_members=tuple(sorted(classification.kept_members)),
        replaced_members=replaced,
        needs_review=tuple(sorted(set(needs_review))),
        replacement_seed=seed,
    )
    logger.info(
        "%s carry-forward: %d carried, %d changed, %d new, %d dropped; %d panel members kept, "
        "%d replaced, %d reviews needed",
        source_id,
        plan.population.carried,
        plan.population.changed,
        plan.population.new,
        plan.population.dropped,
        len(plan.kept_members),
        plan.replaced_members,
        len(plan.needs_review),
    )
    return plan


class MergedReviewResult(BaseModel):
    """The oversight outcome of carried plus fresh reviews over one retained population."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: OversightLevel
    result: SourceQualityResult
    coverage_complete: bool
    finalization: QualityResultFinalization | None = None


def merged_review_result(
    retained: int,
    carried: tuple[CarriedReview, ...],
    fresh: tuple[ReviewRecord, ...],
    *,
    waiver: ContentEvidenceReference | None = None,
) -> MergedReviewResult:
    """Derive the oversight level from carried plus fresh reviews under the existing rules.

    Every retained task with a carried or fresh observation counts as attempted.
    FULL requires complete coverage with no unavailable observations — the
    ``covers_whole_population`` rule — or an explicitly recorded waiver at or
    below the 1% unreviewed bound. Uncertain observations still count as
    covered. Anything less stays SAMPLE. Carried reviews keep their original
    reviewer identity; the caller records that marker beside the merged result.
    """
    if retained <= 0:
        raise ValueError("Merged oversight requires a nonempty retained population")
    observations = [entry.review for entry in carried] + list(fresh)
    observed_ids = [record.task_id for record in observations]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("A task has both a carried and a fresh review")
    result = quality_result(observations)
    unavailable = result.unavailable_count + (retained - len(observations))
    merged = SourceQualityResult(
        reviewed=result.reviewed,
        issue_count=result.issue_count,
        issues=result.issues,
        repair_recommendation=result.repair_recommendation,
        unavailable_count=unavailable,
        uncertain_count=result.uncertain_count,
        sample_count=retained,
    )
    coverage_complete = unavailable == 0
    if coverage_complete:
        if waiver is not None:
            raise ValueError("A complete merged review cannot carry a waiver")
        return MergedReviewResult(
            level=OversightLevel.FULL,
            result=merged,
            coverage_complete=True,
            finalization=QualityResultFinalization.COMPLETE,
        )
    if waiver is None:
        return MergedReviewResult(level=OversightLevel.SAMPLE, result=merged, coverage_complete=False)
    if unavailable * WAIVER_BOUND_SCALE > retained:
        raise ValueError(
            f"Accepted incomplete finalization waives {unavailable} of {retained} retained tasks, above the "
            f"{FULL_WAIVER_MAX_UNREVIEWED_FRACTION:.0%} unreviewed bound"
        )
    return MergedReviewResult(
        level=OversightLevel.FULL,
        result=merged,
        coverage_complete=False,
        finalization=QualityResultFinalization.ACCEPTED_INCOMPLETE,
    )


def panel_observations(receipt: SampleQCReceipt) -> tuple[ReviewRecord, ...]:
    """Read a receipt's fixed-panel observations with their frozen bytes verified."""
    panel = receipt.review.panel
    if file_sha256(StoragePath(panel.panel_path)) != panel.panel_sha256:
        raise ValueError(f"Sample panel changed: {receipt.source_id}")
    records = StoragePath(receipt.review.records_path)
    if file_sha256(records) != receipt.review.records_sha256:
        raise ValueError(f"Sample observations changed after review: {receipt.source_id}")
    with records.open("rb") as stream:
        rows = pq.read_table(stream).to_pylist()
    reviews = []
    for row in rows:
        review = ReviewRecord.model_validate_json(row["review"])
        if review.task_id != row["task_id"]:
            raise ValueError(f"Sample observation belongs to another task: {receipt.source_id}")
        reviews.append(review)
    if len(reviews) != panel.sample_size or len({review.task_id for review in reviews}) != panel.sample_size:
        raise ValueError(f"Sample observations do not cover the panel exactly once: {receipt.source_id}")
    return tuple(reviews)


class SampleRecovery(BaseModel):
    """One source's recovered evidence under the augmented SAMPLE oversight rules.

    The Result comes from the fixed random panel alone, so carried reviews
    cannot bias its issue rate. The carried set's own rate is reported beside
    the coverage, and the level promotes to FULL only under the merged
    completeness and waiver rules.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    retained: int = Field(gt=0)
    panel_conclusive: int = Field(ge=0)
    carried_reviews: int = Field(ge=0)
    carried_issue_count: int = Field(ge=0)
    coverage_conclusive: int = Field(ge=0)
    reviewed_fraction: float = Field(ge=0, le=1)
    level: OversightLevel
    full: bool
    panel_result: SourceQualityResult
    reviewers: tuple[dict, ...]
    """Every carried review's original reviewer identity, recorded verbatim and
    never relabeled; the current receipt's reviewer stands when nothing carries."""
    topup_ids: tuple[str, ...] = ()
    """Task IDs whose fresh review completes the panel's conclusive coverage."""


def sample_recovery_outcome(
    source_id: str,
    retained: int,
    panel_reviews: tuple[ReviewRecord, ...],
    carried: tuple[CarriedReview, ...],
    *,
    reviewers: tuple[dict, ...],
) -> SampleRecovery:
    """Recover one source's SAMPLE evidence with carried prior reviews joined in.

    The Result stays computed from the fixed panel alone. The carried set's
    conclusive reviews join the reviewed set for coverage and for the merged
    FULL determination, never into the panel's issue rate. A task the panel
    also observed counts once in the coverage, and the panel's own observation
    governs the merged determination for it.
    """
    if retained <= 0:
        raise ValueError("Recovery requires a nonempty retained population")
    panel_ids = {record.task_id for record in panel_reviews}
    panel_result = quality_result(panel_reviews)
    carried_result = quality_result([entry.review for entry in carried])
    covered = {record.task_id for record in panel_reviews if is_conclusive(record)}
    covered |= {entry.current_task_id for entry in carried if is_conclusive(entry.review)}
    coverage_conclusive = len(covered)
    merged = merged_review_result(
        retained,
        tuple(entry for entry in carried if entry.current_task_id not in panel_ids),
        panel_reviews,
    )
    return SampleRecovery(
        source_id=source_id,
        retained=retained,
        panel_conclusive=panel_result.reviewed,
        carried_reviews=carried_result.reviewed,
        carried_issue_count=carried_result.issue_count,
        coverage_conclusive=coverage_conclusive,
        reviewed_fraction=coverage_conclusive / retained,
        level=merged.level,
        full=merged.level is OversightLevel.FULL,
        panel_result=panel_result,
        reviewers=reviewers,
    )


def sample_topup_ids(
    retained: int,
    panel_reviews: tuple[ReviewRecord, ...],
    carried: tuple[CarriedReview, ...],
    current_ids: Iterable[str],
    *,
    seed: int = SAMPLE_SEED,
    max_panel_size: int = SAMPLE_SIZE,
) -> tuple[str, ...]:
    """The exact fresh reviews that complete the panel's conclusive coverage.

    The reviewed set is the panel plus the carried reviews. First complete the
    panel: its members without a conclusive observation go first. Then draw
    seeded replacements from the retained population until
    ``min(max_panel_size, retained)`` distinct tasks have a conclusive review.
    """
    panel_ids = {record.task_id for record in panel_reviews}
    covered = {record.task_id for record in panel_reviews if is_conclusive(record)}
    covered |= {entry.current_task_id for entry in carried if is_conclusive(entry.review)}
    target = min(max_panel_size, retained)
    topup = sorted(task_id for task_id in panel_ids if task_id not in covered)
    remaining = target - len(covered) - len(topup)
    if remaining > 0:
        pool = [task_id for task_id in current_ids if task_id not in panel_ids and task_id not in covered]
        _, replacements = seeded_sample(pool, size=remaining, key=lambda task_id: seeded_order(task_id, seed))
        topup.extend(replacements)
    return tuple(topup)


def load_earlier_receipt(document_path: str, source_id: str) -> SampleQCReceipt:
    """Load and verify an earlier receipt from its registry or artifact document."""
    reference = StoragePath(document_path)
    document = json.loads(reference.read_text())
    if document.get("source_id") != source_id:
        raise ValueError(f"Evidence receipt belongs to another source: {document.get('source_id')}")
    receipt = SampleQCReceipt.model_validate(document)
    record = StoragePath(str(StoragePath(receipt.review.path) / RECORD_FILENAME))
    if file_sha256(record) != receipt.artifact_sha256:
        raise ValueError(f"Evidence receipt record changed: {receipt.review.path}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--earlier-receipt", required=True, help="Earlier SampleQCReceipt JSON document")
    parser.add_argument("--earlier-population", required=True, help="Earlier SourceTextPopulation artifact directory")
    parser.add_argument("--current-population", required=True, help="Current SourceTextPopulation artifact directory")
    parser.add_argument("--output", required=True, help="Delta plan report output path")
    parser.add_argument(
        "--normalization-field",
        help="Build and apply the typed absent-versus-null normalization for this TaskSpec field",
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    receipt = load_earlier_receipt(args.earlier_receipt, args.source_id)
    if not args.run:
        return
    reference = ContentEvidenceReference(uri=args.earlier_receipt, sha256=file_sha256(StoragePath(args.earlier_receipt)))
    earlier_population = SourceTextPopulation.raw_load(args.earlier_population)
    current_population = SourceTextPopulation.raw_load(args.current_population)
    normalization = None
    normalized_fields: frozenset[str] = frozenset()
    if args.normalization_field is not None:
        evidence_url = str(StoragePath(args.output).parent / "normalization-evidence.parquet")
        normalization = build_normalization_receipt(
            args.source_id,
            args.earlier_population,
            args.current_population,
            args.normalization_field,
            evidence_url,
        )
        normalized_fields = frozenset({normalization.field})
    earlier = earlier_panel_evidence(receipt, reference, normalized_fields=normalized_fields)
    earlier_identities = population_identities(earlier_population, normalized_fields=normalized_fields)
    logger.info("Earlier population: %d task identities", len(earlier_identities))
    current_identities = population_identities(current_population, normalized_fields=normalized_fields)
    logger.info("Current population: %d task identities", len(current_identities))
    plan = plan_review_carry_forward(
        args.source_id,
        earlier,
        earlier_population,
        current_population,
        earlier_identities,
        current_identities,
        normalization=normalization,
    )
    report = {
        "source_id": plan.source_id,
        "earlier": {
            "dataset_id": plan.earlier_dataset_id,
            "dataset_revision": plan.earlier_dataset_revision,
            "task_count": plan.earlier_task_count,
        },
        "current": {
            "dataset_id": plan.current_dataset_id,
            "dataset_revision": plan.current_dataset_revision,
            "task_count": plan.current_task_count,
        },
        "evidence": plan.evidence.model_dump(mode="json"),
        "normalization": normalization.model_dump(mode="json") if normalization is not None else None,
        "reviewer_sha256": canonical_sha256(plan.reviewer),
        "population": plan.population.model_dump(mode="json"),
        "carried_reviews": len(plan.carried_reviews),
        "kept_members": len(plan.kept_members),
        "replaced_members": plan.replaced_members,
        "replacement_seed": plan.replacement_seed,
        "needs_review": list(plan.needs_review),
    }
    output = StoragePath(args.output)
    output.parent.mkdirs(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "source_id": plan.source_id,
                "carried": plan.population.carried,
                "changed": plan.population.changed,
                "new": plan.population.new,
                "dropped": plan.population.dropped,
                "carried_reviews": len(plan.carried_reviews),
                "replaced_members": plan.replaced_members,
                "needs_review": len(plan.needs_review),
                "normalized_field": normalization.field if normalization is not None else None,
                "output": args.output,
            }
        )
    )


if __name__ == "__main__":
    main()
