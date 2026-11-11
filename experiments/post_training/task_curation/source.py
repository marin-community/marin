# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source declarations and their curation pipeline interfaces."""

from dataclasses import dataclass
from typing import Protocol

from finestore.rl.curation import (
    DAYTONA_SAFE_SNAPSHOT_LIMIT,
    IndependentQC,
    OracleValidation,
    OversightEvidence,
    OversightLevel,
    PipelineReference,
    SourceDeclaration,
    daytona_verdict,
)
from marin.execution.lazy import ArtifactStep

from experiments.post_training.task_curation.campaign import CampaignArtifact
from experiments.post_training.task_curation.config import PipelineOptions


class PinnedReference(Protocol):
    """Catalog identity exposed by a dataset's existing pinned input."""

    @property
    def name(self) -> str: ...

    @property
    def revision(self) -> str | None: ...

    @property
    def url(self) -> str: ...


class CatalogConfig(Protocol):
    """Catalog facts; each dataset supplies its own execution configuration."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def dataset(self) -> PinnedReference | None: ...

    @property
    def files(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class SourceReference:
    """A pinned dataset reference."""

    name: str
    revision: str | None
    url: str


@dataclass(frozen=True, kw_only=True)
class SourceInfo:
    """Identity and discovery information for one source population.

    ``count`` is the number of selected input rows at the dataset revision,
    before conversion or curation. Leave it unknown unless the pinned payload
    or a complete manifest establishes it. A sample size is not a source count.

    ``dataset`` identifies inventory entries without a configuration. Runnable
    sources expose their configured pinned input directly. Tags describe task type,
    interaction, benchmark status, and source-specific search terms.
    """

    id: str
    title: str
    origin: str
    family: str = ""
    tags: tuple[str, ...] = ()
    count: int | None = None
    notes: str = ""
    dataset: SourceReference | None = None
    independent_qc: IndependentQC | None = None
    daytona_snapshot_count: int | None = None
    """Distinct environments the source's own task specs require, as measured.

    Each unique agent or grader image a run provisions is one Daytona snapshot.
    ``None`` means the count was never taken; it is never inferred from a declaration.
    """

    daytona_count_basis: str | None = None
    """Identity rule and payload source behind ``daytona_snapshot_count``."""

    daytona_execution_failures: int | None = None
    """Tasks the Daytona backend refuses on their declared execution requirements."""

    daytona_run_evidence: str | None = None
    """A campaign run that executed this source on the Daytona RL org, when one exists."""

    oracle_validation: OracleValidation | None = None

    oversight: OversightEvidence | None = None
    """The review this source's admission rests on, with its evidence pointers."""

    """How execution was qualified when the source ships no oracle solutions.

    ``executed`` runs the shipped oracle solutions; ``verifier_executed`` runs the
    grader against a solved and an empty submission when no oracle exists; ``skipped``
    certifies nothing. The Atlas shows the value beside the catalog.
    """


@dataclass(frozen=True)
class RlDataSource[ConfigT: CatalogConfig]:
    """One source population, its dataset-owned configuration and callable.

    Stable IDs preserve Atlas review history even when the callable changes.
    Sources without a pipeline remain discoverable in the inventory.
    Configurations expose catalog facts independently of execution. Inventory
    entries without a configuration or callable retain their authored metadata.
    """

    info: SourceInfo
    config: ConfigT | None = None
    pipeline: "CurationPipeline[ConfigT] | None" = None

    @property
    def name(self) -> str:
        return self.config.name if self.config is not None else self.info.id.partition(":")[2]

    @property
    def version(self) -> str:
        return self.config.version if self.config is not None else "1"

    @property
    def files(self) -> tuple[str, ...]:
        return self.config.files if self.config is not None else ()

    @property
    def dataset(self) -> PinnedReference | None:
        return self.config.dataset if self.config is not None else self.info.dataset

    @property
    def declaration(self) -> SourceDeclaration:
        """Project a source declaration into catalog metadata without executing it."""
        info, pipeline = self.info, self.pipeline
        dataset = self.dataset
        invocation = None
        if pipeline is not None:
            invocation = PipelineReference(
                name=self.name,
                version=self.version,
                source=dataset.name if dataset else None,
                revision=dataset.revision if dataset else None,
                files=list(self.files),
            )
        tags = info.tags
        safety = daytona_verdict(
            snapshot_count=info.daytona_snapshot_count,
            execution_failures=info.daytona_execution_failures,
            agentic="agentic" in tags,
        )
        return SourceDeclaration(
            id=info.id,
            name=self.name,
            display_name=info.title,
            origin=info.origin,
            family=info.family,
            tags=list(tags),
            task_count=info.count,
            notes=info.notes,
            dataset_id=dataset.name if dataset else None,
            dataset_revision=dataset.revision if dataset else None,
            url=dataset.url if dataset else None,
            type=next(
                (
                    label
                    for tag, label in (
                        ("eval", "eval"),
                        ("rlvr", "RLVR"),
                        ("agentic", "Agentic"),
                        ("alignment", "Alignment"),
                    )
                    if tag in tags
                ),
                "",
            ),
            turns=next(
                (label for tag, label in (("single-turn", "Single-turn"), ("multi-turn", "Multi-turn")) if tag in tags),
                "",
            ),
            is_benchmark="benchmark" in tags,
            kind="Generator" if "generator" in tags else "Dataset",
            status=next(
                (label for tag, label in (("deprecated", "Deprecated"), ("excluded", "Excluded")) if tag in tags),
                "Available",
            ),
            pipeline=invocation,
            independent_qc=info.independent_qc,
            daytona_safety=safety,
            daytona_snapshot_count=info.daytona_snapshot_count,
            daytona_count_basis=info.daytona_count_basis,
            oracle_validation=info.oracle_validation,
            oversight=info.oversight.level if info.oversight else OversightLevel.NONE,
            oversight_evidence=info.oversight,
            daytona_snapshot_threshold=(
                DAYTONA_SAFE_SNAPSHOT_LIMIT if info.daytona_snapshot_count is not None else None
            ),
        )


class CurationPipeline[ConfigT: CatalogConfig](Protocol):
    def __call__(self, source: RlDataSource[ConfigT], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]: ...
