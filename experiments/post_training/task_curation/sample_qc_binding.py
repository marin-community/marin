# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind exact-input sampled content Results without changing execution qualification."""

from dataclasses import replace

from finestore.rl.curation import ActorCensusInput, ContentEvidenceReference, IndependentQC, SamplePanelCurrency
from pydantic import BaseModel, ConfigDict, model_validator
from taskcompendium.pipeline.models import ReviewRecord

from experiments.post_training.task_curation.content_identity import ContentIdentityReceipt
from experiments.post_training.task_curation.full_content_binding import (
    RetainedLineageProof,
    clear_stale_full_content_result,
    lineage_note,
)
from experiments.post_training.task_curation.sample_qc_artifacts import SampleQCReview
from experiments.post_training.task_curation.source import CatalogConfig, RlDataSource
from infra.marina.apps.rl_data_catalog.server.quality import SAMPLE_MAX_MISSING_FRACTION
from infra.marina.apps.rl_data_catalog.server.source_identity import SourceId

SAMPLE_BINDING_DRIFTS: dict[str, str] = {}
"""The sample content panels whose source's census went pending at its pin, recorded by
the population build for the declaration audit (ruling 2026-11-19): the audit reports
each as a waivable served-catalog gap instead of crashing the build."""


def sample_matches_lineage(
    receipt: "SampleQCReceipt",
    lineage: RetainedLineageProof,
    content_identity: ContentIdentityReceipt | None = None,
) -> bool:
    """Whether an archived panel is exactly the retained release's byte-identical original panel.

    A recorded content-identity receipt binds an archived panel whose artifact
    digest differs from the lineage input's only through re-serialization:
    every TaskSpec byte verified identical across the two conversions. Any
    differing task refuses, and without a receipt the digests must match exactly.
    """
    panel = receipt.review.panel
    if receipt.source_id != lineage.source_id or (
        panel.dataset_id,
        panel.dataset_revision,
        panel.converter_version,
        panel.retained_tasks,
    ) != (
        lineage.original_dataset_id,
        lineage.original_dataset_revision,
        lineage.original_input.converter_version,
        lineage.original_input.measured_tasks,
    ):
        return False
    if panel.task_specs_sha256 == lineage.original_input.task_specs_sha256:
        return True
    return (
        content_identity is not None
        and content_identity.dataset_id == lineage.original_dataset_id
        and content_identity.dataset_revision == lineage.original_dataset_revision
        and content_identity.original.task_count == panel.retained_tasks
        and content_identity.binds(receipt.source_id, panel.task_specs_sha256, lineage.original_input.task_specs_sha256)
    )


def panel_matches_current_census(receipt: "SampleQCReceipt", actor: ActorCensusInput | None) -> bool:
    """Whether the panel reviewed exactly the served population's bytes at the current census.

    A panel whose recorded population digest and task count equal the current actor
    census reviewed exactly the served bytes, whatever release carried them before:
    the content-identity carry-forward across a byte-identical re-pin.
    """
    panel = receipt.review.panel
    return (
        actor is not None
        and panel.task_specs_sha256 == actor.task_specs_sha256
        and panel.retained_tasks == actor.measured_tasks
    )


class SampleMissingObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    review: ReviewRecord
    batch_receipt: ContentEvidenceReference


class SampleQCReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    review: SampleQCReview
    artifact_sha256: str
    missing_observations: tuple[SampleMissingObservation, ...] = ()

    @model_validator(mode="after")
    def _source_identity(self) -> "SampleQCReceipt":
        if self.source_id != self.review.panel.source_id:
            raise ValueError("Sample Result belongs to another source")
        return self


def require_sample_qualification(receipt: SampleQCReceipt) -> None:
    """Require at least 95% conclusive reviews of the unchanged attempted panel."""
    review = receipt.review
    result = review.result
    missing = result.unavailable_count + result.uncertain_count
    if (
        review.attempted_count != review.panel.sample_size
        or result.reviewed + missing != review.attempted_count
        or missing > SAMPLE_MAX_MISSING_FRACTION * review.attempted_count
    ):
        raise ValueError("SAMPLE requires at least 95% reviewed rows in its fixed panel (at most 5% missing)")
    ids = [row.review.task_id for row in receipt.missing_observations]
    if len(ids) != missing or len(set(ids)) != missing:
        raise ValueError("SAMPLE receipt must enumerate every missing task and its batch evidence")


def panel_matches_identity(
    receipt: "SampleQCReceipt",
    actor: ActorCensusInput,
    content_identity: ContentIdentityReceipt,
) -> bool:
    """Whether a pin-matching panel's digest differs only by artifact re-serialization.

    The panel was drawn at this exact pin; a content-identity receipt binding
    the panel's recorded digest and the current census input's digest proves
    every TaskSpec byte identical across the two conversions of the pin.
    """
    panel = receipt.review.panel
    return (
        content_identity.dataset_id == panel.dataset_id
        and content_identity.dataset_revision == panel.dataset_revision
        and content_identity.original.task_count == panel.retained_tasks == actor.measured_tasks
        and content_identity.binds(receipt.source_id, panel.task_specs_sha256, actor.task_specs_sha256)
    )


def bind_sample_result[ConfigT: CatalogConfig](
    source: RlDataSource[ConfigT],
    receipt: SampleQCReceipt | None,
    *,
    retained_lineage: RetainedLineageProof | None = None,
    content_identity: ContentIdentityReceipt | None = None,
) -> RlDataSource[ConfigT]:
    """Attach exact-input content evidence while preserving execution qualification."""
    if receipt is None:
        return clear_stale_full_content_result(source)
    panel = receipt.review.panel
    dataset, actor = source.dataset, source.info.actor_census_input
    if dataset is None or actor is None:
        # The census went pending at this source's pin (ruling 2026-11-19): keep the
        # configured declaration without the sample evidence and record the drift, so
        # the audit reports a waivable served-catalog gap instead of crashing the
        # population build.
        SAMPLE_BINDING_DRIFTS[source.info.id] = panel.dataset_revision
        note = "Sample content panel pending: the census is unbound at this pin."
        return replace(source, info=replace(source.info, notes=f"{source.info.notes} {note}".strip()))
    full_match = (
        source.info.id,
        dataset.name,
        dataset.revision,
        source.version,
        actor.task_specs_sha256,
        actor.measured_tasks,
    ) == (
        receipt.source_id,
        panel.dataset_id,
        panel.dataset_revision,
        panel.converter_version,
        panel.task_specs_sha256,
        panel.retained_tasks,
    )
    if not full_match:
        # Byte identity at the current pin outranks the lineage label: a panel that
        # matches the current census reviewed exactly the served population's bytes,
        # whatever release carried them before.
        if not panel_matches_current_census(receipt, actor):
            identity_current = (
                content_identity is not None
                and (
                    source.info.id,
                    dataset.name,
                    dataset.revision,
                    source.version,
                )
                == (receipt.source_id, panel.dataset_id, panel.dataset_revision, panel.converter_version)
                and panel_matches_identity(receipt, actor, content_identity)
            )
            if retained_lineage is not None and sample_matches_lineage(receipt, retained_lineage, content_identity):
                result = receipt.review.result
                return replace(
                    source,
                    info=replace(
                        source.info,
                        quality_result=result,
                        independent_qc=IndependentQC(
                            verdict=result.verdict,
                            method=lineage_note("", retained_lineage, "Archived SAMPLE content panel"),
                            reviewed=result.reviewed,
                            evidence_url=receipt.review.path,
                            panel_currency=SamplePanelCurrency.ARCHIVED_LINEAGE,
                        ),
                    ),
                )
            if not identity_current:
                raise ValueError(f"Sample Result belongs to another current population: {source.info.id}")
    source = clear_stale_full_content_result(source)
    if source.info.full_content_result_binding is not None:
        return source
    result = receipt.review.result
    if actor.task_specs_sha256 == panel.task_specs_sha256:
        cross_pin = not full_match
        method = (
            f"SAMPLE content panel: seed {panel.sample_seed}, {panel.sample_size} selected / "
            f"{panel.retained_tasks} current tasks; pinned reviewer; execution qualification pending"
            + (
                f"; the panel's recorded population digest equals the current pin's actor census "
                f"digest ({actor.task_specs_sha256[:12]}), so the reviewed bytes are the served "
                f"population's across the {panel.dataset_revision[:12]} re-pin"
                if cross_pin
                else ""
            )
        )
    else:
        # The panel was drawn at this pin before a byte-identical re-serialization;
        # the content-identity receipt is what keeps it current.
        method = (
            f"SAMPLE content panel: seed {panel.sample_seed}, {panel.sample_size} selected / "
            f"{panel.retained_tasks} current tasks; pinned GLM-5.3 v3; execution qualification pending; "
            f"content-identity receipt {content_identity.identity_sha256[:12]} binds the panel digest "
            f"{panel.task_specs_sha256[:12]} byte-identical to the re-serialized population "
            f"{actor.task_specs_sha256[:12]}"
        )
    return replace(
        source,
        info=replace(
            source.info,
            quality_result=result,
            independent_qc=IndependentQC(
                verdict=result.verdict,
                method=method,
                reviewed=result.reviewed,
                evidence_url=receipt.review.path,
                panel_currency=SamplePanelCurrency.CURRENT_PIN,
            ),
        ),
    )
