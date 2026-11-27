# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Admit oracle-filtration-flow declarations through their base-pinned conversion releases.

The taskspec re-admission assembly (0c2df8b7d6) admits ledger-declared cohorts: their
ledger entries carry the conversion release, so the admission is a ledger edit. The
mechanically-excluded compendium populations are oracle-filtration-flow declarations —
their zero-task oracle-filter pins bind through the oracle-release receipts, and no
ledger entry carries their conversion. This module is their typed input path: a
``ConversionAdmissionReceipt`` records the qualification evidence sealed at a
published conversion release (the passing acceptance run, the staged census measured
at the staged population artifact per option A, the clean lexical measurement, the
grading census, and the SAMPLE receipt once its panel reviews), and
``bind_conversion_admission`` re-binds the configured original declaration to the
conversion release the same way ``bind_oracle_admitted_release`` re-binds an oracle
subset. Fail-closed throughout: a receipt that does not belong to the configured
original population, a non-clean lexical measurement, a census or grading census
that does not cover the staged population, or an unqualified SAMPLE refuses the
binding. A receipt whose SAMPLE is still pending binds as a staged declaration —
the declaration carries the release pin and its receipts while the SAMPLE panel
reviews, and admits when the SAMPLE receipt registers.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from finestore.rl.curation import (
    DatasetRelease,
    LexicalContaminationEvidence,
    SourceUse,
    TaskFormat,
    WithinSourceDedupEvidence,
)
from pydantic import BaseModel, ConfigDict, model_validator
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.census_binding import ActorCensusReceipt, bind_actor_census
from experiments.post_training.task_curation.datasets.lexical_measurements import lexical_measurements
from experiments.post_training.task_curation.datasets.native_releases import TASKSPEC_CONVERSION_VERSION
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort, import_taskcompendium_prebuilt
from experiments.post_training.task_curation.receipt_shards import read_receipt_shards
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.sample_qc_binding import (
    SampleQCReceipt,
    bind_sample_result,
    require_sample_qualification,
)
from experiments.post_training.task_curation.source import RlDataSource
from infra.marina.apps.rl_data_catalog.server.source_identity import SourceId


class ConversionAdmissionReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    intended_use: SourceUse
    original_release: DatasetRelease
    """The zero-task oracle-filter pin the configured declaration carries."""
    original_pipeline_revision: str
    conversion_release: DatasetRelease
    """The published base-pinned conversion release (taskspec-20261126b)."""
    acceptance: str
    """The passing acceptance Iris job id; the run executed the declared build contexts."""
    actor: ActorCensusReceipt
    """The staged census measured at the staged population artifact (option A)."""
    lexical_measurement_uri: str
    """The registered lexical measurement receipt's registry document (0 contaminated)."""
    grading_evidence_uri: str
    """The grading census receipt at the staged population (100% matched)."""
    population_record: str
    """The staged population artifact the census, lexical and grading receipts share."""
    sample: SampleQCReceipt | None = None
    """The seed-zero SAMPLE receipt; None while the panel reviews (staged, not admitted)."""

    @model_validator(mode="after")
    def _one_population(self) -> "ConversionAdmissionReceipt":
        actor, census = self.actor.input, self.actor.measurement
        release_pin = (self.conversion_release.repo, self.conversion_release.revision)
        if release_pin != (census.dataset_id, census.dataset_revision) or release_pin != (
            actor.dataset_id,
            actor.dataset_revision,
        ):
            raise ValueError("The census receipt binds another release pin")
        if self.intended_use not in (SourceUse.TRAIN, SourceUse.EVAL):
            raise ValueError(f"Untyped intended use on the conversion admission: {self.source_id}")
        return self


def conversion_lexical_evidence(receipt: ConversionAdmissionReceipt) -> LexicalContaminationEvidence:
    """The lexical evidence from the registered measurement at the conversion release.

    The registered measurement binds the release pin and carries its own receipt:
    a TRAIN source's clean release receipt (the taskspec construction) or an
    EVAL source's benchmark-measured receipt — both fail closed on any
    contaminated task.
    """
    measurement_receipt = lexical_measurements().get(receipt.source_id)
    if measurement_receipt is None:
        raise ValueError(f"The conversion release has no registered lexical measurement: {receipt.source_id}")
    if measurement_receipt.input.dataset_id != receipt.conversion_release.repo:
        raise ValueError(
            f"The registered lexical measurement binds another release: {receipt.source_id} "
            f"({measurement_receipt.input.dataset_id})"
        )
    measurement = measurement_receipt.measurement
    if measurement.contaminated_tasks:
        raise ValueError(f"The conversion release's lexical measurement is not clean: {receipt.source_id}")
    return measurement


def bind_conversion_admission(source: RlDataSource, receipt: ConversionAdmissionReceipt) -> RlDataSource[PrebuiltCohort]:
    """Re-bind the configured original declaration to its conversion release.

    Mirrors ``bind_oracle_admitted_release``: the identity checks pin the receipt to
    the configured original population, the declaration re-binds to the conversion
    release as an executable TaskSpec population, and the census, sample, lexical
    and grading bindings attach. The staged population record is the alignment
    anchor: the census, lexical and grading receipts must bind the same recorded
    population, whose within-source dedup becomes the declaration's. An original
    declaration that is not bound to the receipt's recorded oracle-filter pin
    refuses.
    """
    if (
        source.info.id != receipt.source_id
        or source.current_release != receipt.original_release
        or source.intended_use is not receipt.intended_use
    ):
        raise ValueError("Conversion admission receipt belongs to another configured original population")
    # The receipt is immutable but a staged edit (model_copy) bypasses the model
    # validators, so the binding re-verifies the pin consistency itself.
    release_pin = (receipt.conversion_release.repo, receipt.conversion_release.revision)
    actor = receipt.actor.input
    census = receipt.actor.measurement
    if release_pin != (actor.dataset_id, actor.dataset_revision) or release_pin != (
        census.dataset_id,
        census.dataset_revision,
    ):
        raise ValueError("The census receipt binds another release pin")
    population = SourceTextPopulation.raw_load(receipt.population_record)
    if (
        population.dataset_id,
        population.dataset_revision,
        population.source_version,
        population.converter_revision,
        population.task_count,
        population.task_specs_sha256,
    ) != (
        actor.dataset_id,
        actor.dataset_revision,
        actor.converter_version,
        actor.converter_revision,
        actor.measured_tasks,
        actor.task_specs_sha256,
    ):
        raise ValueError(f"The staged population differs from the census binding: {receipt.source_id}")
    lexical = conversion_lexical_evidence(receipt)
    count = actor.measured_tasks
    dedup = WithinSourceDedupEvidence(
        raw_count=population.within_source_dedup["raw_count"],
        unique_count=population.within_source_dedup["unique_count"],
        within_source_duplicates=population.within_source_dedup["within_source_duplicates"],
        source_dataset_id=receipt.conversion_release.repo,
        source_revision=receipt.conversion_release.revision,
        converter_revision=actor.converter_revision,
        converter_version=actor.converter_version,
        drop_manifest_uri=population.within_source_dedup.get("drop_manifest_uri"),
        evidence_url=f"{receipt.population_record}/.artifact.json",
    )
    if dedup.unique_count != count:
        raise ValueError(f"The conversion population's dedup differs from the census: {receipt.source_id}")
    config = PrebuiltCohort(
        name=source.name,
        version=TASKSPEC_CONVERSION_VERSION,
        repo=receipt.conversion_release.repo,
        revision=receipt.conversion_release.revision,
        expected_rows=count,
        intended_use=source.config.intended_use,
    )
    history = source.info.recipe_qa_history
    info = replace(
        source.info,
        current_release=receipt.conversion_release,
        dataset=config.dataset,
        count=count,
        task_format=TaskFormat.TASKSPEC,
        lexical_contamination=lexical,
        grading_environment=receipt_actor_grading(receipt),
        within_source_dedup=dedup,
        recipe_qa_history=(
            history.model_copy(
                update={
                    "campaign": receipt.acceptance,
                    "recipe_revision": actor.converter_revision,
                    "verification": (
                        "Base-pinned conversion admission: every FROM base digest-pinned; passing acceptance "
                        f"run {receipt.acceptance}; staged census, lexical (0 contaminated of {count}) and "
                        f"grading census bound at the staged population {receipt.population_record}."
                    ),
                }
            )
            if history is not None
            else None
        ),
        daytona_run_evidence=receipt.acceptance,
        tags=tuple(tag for tag in source.info.tags if tag not in {"excluded", "deprecated"}),
        notes=(
            f"Base-pinned conversion admitted {count} executable TaskSpecs at "
            f"{receipt.conversion_release.repo}@{receipt.conversion_release.revision}; the zero-task "
            f"oracle-filter pin {receipt.original_release.repo}@{receipt.original_release.revision} and "
            "its filtration evidence remain archived. " + source.info.notes
        ),
    )
    staged = RlDataSource(info=info, config=config, pipeline=import_taskcompendium_prebuilt)
    bound = bind_actor_census(staged, receipt.actor)
    bound = bind_sample_result(bound, receipt.sample)
    return bound


def receipt_actor_grading(receipt: ConversionAdmissionReceipt):
    """The grading census evidence bound at the staged population, loaded from its receipt."""
    document = json.loads(StoragePath(receipt.grading_evidence_uri).read_bytes())
    for row in document if isinstance(document, list) else [document]:
        if row.get("source_id") == receipt.source_id:
            return row["evidence"]
    raise ValueError(f"The grading census receipt does not cover {receipt.source_id}: {receipt.grading_evidence_uri}")


def sample_qualification(receipt: ConversionAdmissionReceipt) -> None:
    """Fail closed unless the receipt's SAMPLE panel qualifies (admission-time check)."""
    if receipt.sample is None:
        raise ValueError(f"The conversion admission has no SAMPLE receipt yet: {receipt.source_id}")
    require_sample_qualification(receipt.sample)


def conversion_admission_receipts() -> dict[str, ConversionAdmissionReceipt]:
    """The registered conversion-admission receipts from the committed pointer family."""
    return read_receipt_shards(
        Path(__file__).with_name("datasets") / "conversion_admission_receipts",
        ConversionAdmissionReceipt,
        "conversion admission",
        load_json=read_registry_json,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True, help="The receipt documents to shard.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipts = {}
    for path in args.receipts:
        for raw in json.loads(path.read_text()):
            receipt = ConversionAdmissionReceipt.model_validate(raw)
            if receipt.source_id in receipts:
                raise ValueError(f"Duplicate conversion admission receipt: {receipt.source_id}")
            receipts[receipt.source_id] = receipt.model_dump(mode="json")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from experiments.post_training.task_curation.receipt_shards import write_receipt_shards  # noqa: PLC0415

    write_receipt_shards(args.output_dir, receipts)


if __name__ == "__main__":
    main()
