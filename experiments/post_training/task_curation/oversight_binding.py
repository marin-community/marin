# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive each source's oversight level from its actual QA evidence.

The oversight level is never hand-declared: it is read from the same single
source of truth the Atlas main table's Quality column uses — the current-pin
FULL binding, then the current-pin SAMPLE receipt, then conversion-only QUICK.
A SAMPLE receipt whose panel attempted every retained task with complete
coverage is a whole-source review and derives FULL, mirroring the table's
aggregate FULL policy. Recipe-QA prose survives only as clearly labeled
history, which can never assert a level.
"""

from dataclasses import replace

from finestore.rl.curation import OversightEvidence, OversightLevel

from experiments.post_training.task_curation.full_content_binding import (
    RetainedLineageProof,
    full_content_matches_lineage,
    matches_full_content_input,
)
from experiments.post_training.task_curation.full_content_results import FullContentResultBinding
from experiments.post_training.task_curation.sample_qc_binding import SampleQCReceipt
from experiments.post_training.task_curation.source import CatalogConfig, RlDataSource


def covers_whole_population(sample: SampleQCReceipt) -> bool:
    """Whether the panel attempted every retained task with complete coverage.

    This mirrors the table's aggregate FULL policy: the recorded QC evidence
    covers the retained population and no observation is unavailable, so the
    panel is a whole-source review rather than a sample.
    """
    panel, result = sample.review.panel, sample.review.result
    return (
        sample.review.attempted_count == panel.retained_tasks
        and result.reviewed + result.uncertain_count == panel.retained_tasks
        and result.unavailable_count == 0
    )


def derived_oversight_level(
    source: RlDataSource,
    sample: SampleQCReceipt | None,
    full_content: FullContentResultBinding | None,
    retained_lineage: RetainedLineageProof | None = None,
) -> OversightLevel:
    """The oversight level the current-pin evidence supports, strongest first.

    A byte-preserving retained lineage keeps its original-population review
    valid: an archived whole-source Result that matches the lineage's original
    pin derives FULL for the retained source, exactly as a current-pin Result
    would.
    """
    if full_content is not None and (
        matches_full_content_input(source, full_content)
        or (retained_lineage is not None and full_content_matches_lineage(full_content, retained_lineage))
    ):
        return OversightLevel.FULL
    if sample is not None:
        if covers_whole_population(sample):
            return OversightLevel.FULL
        return OversightLevel.SAMPLE
    return OversightLevel.QUICK


def bind_derived_oversight[ConfigT: CatalogConfig](
    source: RlDataSource[ConfigT],
    sample: SampleQCReceipt | None,
    full_content: FullContentResultBinding | None,
    retained_lineage: RetainedLineageProof | None = None,
) -> RlDataSource[ConfigT]:
    """Attach the evidence-derived oversight level, refusing a contradicting declaration.

    A declared level that survives alongside the derived one (none of the dataset
    declarations set one any more, but a binding or a test fixture may) must agree
    with the evidence, or ingestion fails with a clear message.
    """
    level = derived_oversight_level(source, sample, full_content, retained_lineage)
    declared = source.info.oversight
    if declared is not None and declared.level is not OversightLevel.NONE and declared.level is not level:
        raise ValueError(
            f"Declared oversight level {declared.level.value} contradicts the evidence-derived "
            f"level {level.value}: {source.info.id}"
        )
    receipt = None
    reviewed = None
    basis = "current-pin conversion only"
    if level is OversightLevel.FULL and full_content is not None:
        receipt = full_content.review_receipt.uri
        reviewed = full_content.result.reviewed
        basis = "current-pin FULL content Result binding"
    elif sample is not None:
        receipt = sample.review.path
        if level is OversightLevel.FULL:
            reviewed = sample.review.result.reviewed
            basis = "current-pin complete-population content panel"
        else:
            basis = "current-pin SAMPLE content Result receipt"
    evidence = OversightEvidence(
        verification=f"Evidence-derived from the {basis}; identical to the Atlas main-table Quality.",
        level=level,
        reviewed=reviewed,
        receipt_uri=receipt,
    )
    return replace(source, info=replace(source.info, oversight=evidence))
