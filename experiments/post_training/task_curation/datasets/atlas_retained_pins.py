# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind Atlas-retained release pins onto main's configured declarations.

The Atlas seals every retained release with a receipt recording the retained pin,
the release bytes it read back, and the censuses over the original and retained
populations. The receipts' pin-level content (never dataset bytes) lives in the
public dataset ``open-athena/atlas-source-pins`` at the pinned revision below;
this module reads it and binds each entry onto main's raw declaration at
``all_sources()`` time, so main's durable reference is the Atlas admitted version
and selection runs the retained reader. The table is published by the Atlas
release step on ``atlas-env-audit``.
"""

import json
from dataclasses import replace
from functools import cache

from finestore.rl.curation import (
    DatasetRelease,
    LexicalContaminationEvidence,
    LexicalUse,
    PipelineReference,
    SourceUse,
    huggingface_dataset_url,
    require_lexical_qualification,
)
from finestore.rl.licenses import LicenseReceipt
from finestore.rl.pinned_file import PinnedHfFile, read_pinned_hf_file
from pydantic import BaseModel, ConfigDict, Field, model_validator
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.retained_release import (
    RETAINED_IMPORT_VERSION,
    RetainedCohort,
    import_retained,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceReference

RECEIPTS_REPO = "open-athena/marin-rl-data-atlas-registry"
PINS_REPO = "open-athena/atlas-source-pins"
PINS_REVISION = "5ab081a1db11de1f5a3fa07563b8944b86d15541"
PINS_SHA256 = "baabdcd8ef3904f6fb910f120920ba50e4f5a76f147f71de5456ddfe755ee013"
PINS_BYTES = 1539862
PINS_FILENAME = "retained_pins.json"


# Runs inside the audit checkout, where the sealed receipt models live: every
# retained-release shard reference is downloaded, digest-verified and validated,
# then reduced to the pin-level content the reader binds. The lexical evidence is
# the publication's own release proof when it admits the retained pin, else the
# frozen lexical measurement measured at that pin.
class RetainedCensus(BaseModel):
    """The population a census measured, pinned to its conversion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    dataset_revision: str
    converter_version: str
    measured_tasks: int = Field(gt=0)
    task_specs_sha256: str | None = None
    snapshot_count: int | None = None
    count_basis: str | None = None
    execution_failures: int | None = None


class RetainedPin(BaseModel):
    """The pin-level content of one sealed retained-release receipt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    original_repo: str
    original_revision: str
    converter_version: str
    original_pipeline: PipelineReference
    original_census: RetainedCensus
    retained_census: RetainedCensus
    release_repo: str
    release_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    retained_count: int = Field(gt=0)
    retained_dedup_unique_count: int
    retained_dedup_duplicates: int
    tasks_parquet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lexical_use: LexicalUse
    lexical: LexicalContaminationEvidence
    license: LicenseReceipt
    evidence_url: str
    registry_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    registry_path: str
    registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    notes: str = ""

    @property
    def intended_use(self) -> IntendedUse:
        return IntendedUse.TRAIN if self.lexical_use is LexicalUse.TRAIN else IntendedUse.EVAL

    @property
    def release(self) -> DatasetRelease:
        return DatasetRelease(
            repo=self.release_repo,
            revision=self.release_revision,
            url=huggingface_dataset_url(self.release_repo, self.release_revision),
        )

    @model_validator(mode="after")
    def _sealed_lineage(self) -> "RetainedPin":
        if self.original_census.dataset_id != self.original_repo or (
            self.original_census.dataset_revision != self.original_revision
        ):
            raise ValueError(f"Original census covers another pin: {self.source_id}")
        if self.retained_census.dataset_id != self.release_repo or (
            self.retained_census.dataset_revision != self.release_revision
        ):
            raise ValueError(f"Retained census covers another pin: {self.source_id}")
        if self.retained_census.converter_version != RETAINED_IMPORT_VERSION:
            raise ValueError(f"Retained census was not measured at the retained importer: {self.source_id}")
        if self.retained_census.measured_tasks != self.retained_count or (
            self.retained_dedup_unique_count != self.retained_count
        ):
            raise ValueError(f"Retained census does not cover the retained population: {self.source_id}")
        if self.retained_dedup_duplicates != 0:
            raise ValueError(f"The retained population is not the deduplicated population: {self.source_id}")
        if not self.release_repo.startswith("open-athena/"):
            raise ValueError(f"Retained release is not a private open-athena publication: {self.source_id}")
        if (self.license.dataset_id, self.license.revision) != (self.release_repo, self.release_revision):
            raise ValueError(f"License receipt certifies another pin: {self.source_id}")
        require_lexical_qualification(
            self.lexical, self.release_repo, self.release_revision, SourceUse(self.intended_use.value)
        )
        return self

    def cohort(self, *, name: str) -> RetainedCohort:
        """The retained-reader configuration for one bound declaration."""
        return RetainedCohort(
            name=name + "-retained",
            version=RETAINED_IMPORT_VERSION,
            release=self.release,
            original_dataset=SourceReference(
                self.original_repo,
                self.original_revision,
                huggingface_dataset_url(self.original_repo, self.original_revision),
            ),
            expected_rows=self.retained_count,
            tasks_parquet_sha256=self.tasks_parquet_sha256,
            intended_use=self.intended_use,
            evidence_url=self.evidence_url,
        )


def _pin_table_reference() -> PinnedHfFile:
    return PinnedHfFile(
        repo=PINS_REPO,
        revision=PINS_REVISION,
        filename=PINS_FILENAME,
        sha256=PINS_SHA256,
        byte_size=PINS_BYTES,
    )


@cache
def atlas_retained_pins() -> dict[str, RetainedPin]:
    """The pinned pin table, validated entry by entry at first read."""
    document = json.loads(read_pinned_hf_file(_pin_table_reference()))
    pins = {source_id: RetainedPin.model_validate(entry) for source_id, entry in document["pins"].items()}
    if len(pins) != len(document["pins"]):
        raise ValueError("Retained-pin table repeats a source id")
    return pins


def bind_retained_pins(sources: list[RlDataSource]) -> list[RlDataSource]:
    """Swap each pinned declaration to its Atlas-retained pin and the retained reader.

    A declaration whose lineage does not match its entry keeps the raw pin with a
    recorded note; the parity test fails on the drifted pin, so the drift stays
    visible instead of blocking every other binding.
    """
    pins = atlas_retained_pins()
    if pins is None:
        return sources
    bound = []
    for source in sources:
        pin = pins.get(source.info.id)
        if pin is None:
            bound.append(source)
            continue
        try:
            bound.append(_bind(source, pin))
        except ValueError as error:
            bound.append(source_with_note(source, f"Retained pin drift: {error}"))
    return bound


def _configured_reference(source: RlDataSource, pin: RetainedPin) -> str | None:
    """Check the declaration's pipeline against the sealed one; return a version drift note.

    The dataset pin, the pipeline name and the consumed files are the lineage
    identity: a mismatch fails the binding. The converter version is a label
    main's module may have bumped since the seal; a drift binds anyway and the
    note records it, because the reader consumes the published bytes and
    re-validates every task.
    """
    dataset = source.dataset
    configured = None
    if source.pipeline is not None:
        configured = (
            source.name,
            dataset.name if dataset else None,
            dataset.revision if dataset else None,
            tuple(source.files),
        )
    sealed = pin.original_pipeline
    observed = (sealed.name, sealed.source, sealed.revision, tuple(sealed.files))
    if configured != observed:
        raise ValueError(f"retention lineage belongs to another pipeline: {source.info.id}")
    if source.version != sealed.version:
        return (
            f"retention sealed at converter version {sealed.version}; the declaration labels its "
            f"converter {source.version} — the sealed release predates that change and its re-seal is pending"
        )
    return None


def _bind(source: RlDataSource, pin: RetainedPin) -> RlDataSource:
    """Re-validate the sealed lineage and select the retained pin and reader."""
    dataset = source.dataset
    if dataset is None:
        raise ValueError(f"retention binds to an unpinned declaration: {source.info.id}")
    expected = (source.info.id, dataset.name, dataset.revision)
    observed = (pin.source_id, pin.original_repo, pin.original_revision)
    if expected != observed:
        raise ValueError(f"retention belongs to another conversion input: expected {expected}, sealed {observed}")
    version_note = _configured_reference(source, pin)
    require_lexical_qualification(pin.lexical, pin.release_repo, pin.release_revision, SourceUse(pin.intended_use.value))

    config = pin.cohort(name=source.name)
    release = pin.release
    inherited = None
    try:
        declared = source.current_release
        if source.declaration.status == "Available":
            inherited = declared
    except ValueError:
        inherited = None
    drift = "" if version_note is None else f"{version_note}. "
    info = replace(
        source.info,
        current_release=release,
        upstream_url=source.info.upstream_url or dataset.url,
        inherited_admission=inherited,
        dataset=config.dataset,
        count=pin.retained_count,
        lexical_contamination=pin.lexical,
        license_receipt=pin.license,
        daytona_snapshot_count=pin.retained_census.snapshot_count,
        daytona_count_basis=pin.retained_census.count_basis,
        daytona_execution_failures=pin.retained_census.execution_failures,
        notes=(
            drift + "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; "
            f"original oversight and quality evidence apply through this subset lineage: {pin.evidence_url}. "
            + source.info.notes
        ),
    )
    bound = RlDataSource(info=info, config=config, pipeline=import_retained)
    bound.require_license()
    return bound


def source_with_note(source: RlDataSource, note: str) -> RlDataSource:
    return replace(source, info=replace(source.info, notes=f"{source.info.notes} {note}".strip()))
