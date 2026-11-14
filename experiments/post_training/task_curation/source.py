# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source declarations and their curation pipeline interfaces."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Protocol

from finestore.rl.curation import (
    DAYTONA_SAFE_SNAPSHOT_LIMIT,
    ActorCensusInput,
    ContentEvidenceReference,
    ConvertedExamples,
    DatasetRelease,
    DaytonaMeasurement,
    DaytonaUnsafeReason,
    GradingEnvironmentEvidence,
    IndependentQC,
    IntegrityContractSummary,
    LexicalContaminationEvidence,
    OracleFiltrationEvidence,
    OracleValidation,
    OversightEvidence,
    OversightLevel,
    PipelineReference,
    QualityResultFinalization,
    RoleExecutionEvidence,
    SourceDeclaration,
    SourceOrigin,
    SourceQualityResult,
    SourceUse,
    WithinSourceDedupEvidence,
    source_identifier,
    source_title,
)
from marin.execution.lazy import ArtifactStep
from pydantic import Field
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.admission_binding import population_admission_blocker
from experiments.post_training.task_curation.campaign import CampaignArtifact
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.decontamination import require_private_release
from experiments.post_training.task_curation.full_content_results import FullContentResultBinding
from experiments.post_training.task_curation.source_policy import (
    GRANDFATHERED_DAYTONA_EXEMPTION,
    GRANDFATHERED_MARINSKYRL_SOURCES,
)
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id


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

    @property
    def intended_use(self) -> IntendedUse: ...


class TaskFormat(StrEnum):
    """The ingested task layout of a source's pinned payload.

    Only task-directory layouts are accepted for agentic sources: a task is a
    directory with ``task.toml``, its environment and its verifier, in either
    TaskCompendium's or Harbor's variant. ``MARINSKYRL`` marks a Marin-internal
    blend whose rows are not task directories; it is accepted only for the
    sources listed in ``GRANDFATHERED_MARINSKYRL_SOURCES``.
    """

    TASKCOMPENDIUM = "taskcompendium"
    HARBOR = "harbor"
    MARINSKYRL = "marinskyrl"


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

    id: str = ""
    origin: SourceOrigin
    upstream_dataset: str
    upstream_name: str
    build_path: str
    subset: str | None = None
    variant: str | None = None
    title: str | None = None
    current_release: DatasetRelease | None = None
    inherited_admission: DatasetRelease | None = None
    upstream_url: str | None = None
    quality_result: SourceQualityResult | None = None
    quality_result_finalization: QualityResultFinalization | None = None
    quality_result_waiver: ContentEvidenceReference | None = None
    full_content_result_binding: FullContentResultBinding | None = None
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

    task_format: TaskFormat | None = None
    """The ingested task layout of this source's pinned payload.

    Required for every runnable agentic source. ``MARINSKYRL`` is accepted only for
    the sources in ``GRANDFATHERED_MARINSKYRL_SOURCES``; a new agentic source must
    ingest TaskCompendium or Harbor task directories.
    """

    daytona_execution_failures: int | None = None
    """Tasks the Daytona backend refuses on their declared execution requirements."""

    daytona_run_evidence: str | None = None
    """A campaign run that executed this source on the Daytona RL org, when one exists."""

    oracle_validation: OracleValidation | None = None
    """How the source's grader was qualified.

    ``executed`` runs the shipped oracle solutions; ``verifier_executed`` records
    a negative verdict on an untouched submission without certifying an oracle;
    ``skipped`` certifies neither. The Atlas shows this value beside the catalog.
    """

    oversight: OversightEvidence | None = None
    """The review this source's admission rests on, with its evidence pointers."""

    grading_environment: GradingEnvironmentEvidence | None = None
    integrity_contracts: IntegrityContractSummary | None = None
    actor_census_input: ActorCensusInput | None = None
    converted_examples: ConvertedExamples | None = None
    oracle_filtration: Annotated[OracleFiltrationEvidence | None, Field(exclude_if=lambda value: value is None)] = None
    within_source_dedup: WithinSourceDedupEvidence | None = None
    lexical_contamination: LexicalContaminationEvidence | None = None
    daytona_machine_slots: int | None = None
    daytona_roles: dict[str, dict[str, RoleExecutionEvidence]] | None = None
    daytona_unsafe_reasons: tuple[DaytonaUnsafeReason, ...] = ()
    daytona_unsafe_details: dict[str, str] | None = None
    daytona_unsafe_counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.origin, SourceOrigin):
            raise ValueError("origin must be an upstream SourceOrigin")
        expected_id = source_identifier(self.origin, self.upstream_dataset, self.subset, self.variant)
        if self.id and canonical_source_id(self.id) != expected_id:
            raise ValueError("Source ID differs from generated identity")
        object.__setattr__(self, "id", expected_id)
        title = source_title(self.upstream_name, self.subset, self.variant)
        if self.title is not None and self.title != title:
            raise ValueError("Source title differs from generated title")
        object.__setattr__(self, "title", title)


@dataclass(frozen=True)
class RlDataSource[ConfigT: CatalogConfig]:
    """One source population, its dataset-owned configuration and callable.

    Stable IDs preserve Atlas review history even when the callable changes.
    A catalog declaration requires a registered conversion pipeline. QUICK
    records conversion without review; it does not make tasks selectable.
    """

    info: SourceInfo
    config: ConfigT | None = None
    pipeline: "CurationPipeline[ConfigT] | None" = None

    @property
    def intended_use(self) -> SourceUse:
        """Resolve population policy from the registered converter."""
        if self.config is None:
            raise ValueError(f"No registered conversion Use: {self.info.id}")
        return SourceUse(self.config.intended_use.value)

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
    def pipeline_reference(self) -> PipelineReference:
        if self.pipeline is None or self.config is None:
            raise ValueError(f"Source {self.info.id} has no registered conversion pipeline")
        dataset = self.dataset
        return PipelineReference(
            name=self.name,
            version=self.version,
            source=dataset.name if dataset else None,
            revision=dataset.revision if dataset else None,
            files=list(self.files),
        )

    @property
    def has_actor_census(self) -> bool:
        info = self.info
        return (
            info.daytona_snapshot_count is not None
            and info.daytona_machine_slots is not None
            and info.daytona_execution_failures is not None
            and bool((info.daytona_count_basis or "").strip())
        )

    @property
    def is_grandfathered_blend(self) -> bool:
        return self.info.task_format is TaskFormat.MARINSKYRL and self.info.id in GRANDFATHERED_MARINSKYRL_SOURCES

    @property
    def catalog_declaration(self) -> SourceDeclaration | None:
        """Hold metadata pending while QUICK produces the actor census."""
        filtered_to_zero = self.info.oracle_filtration is not None and self.info.oracle_filtration.admitted_count == 0
        if not self.is_grandfathered_blend and not self.has_actor_census and not filtered_to_zero:
            return None
        return self.declaration

    @property
    def current_release(self) -> DatasetRelease:
        dataset = self.dataset
        if dataset is None or not dataset.revision:
            raise ValueError(f"Source {self.info.id} requires an exact current release pin")
        release = DatasetRelease(repo=dataset.name, revision=dataset.revision, url=dataset.url)
        declared = self.info.current_release
        if declared is not None and declared != release:
            raise ValueError("Current release must match the configured dataset pin")
        return release

    @property
    def admission_blockers(self) -> tuple[str, ...]:
        """Evidence missing before a source can enter admitted selection."""
        info = self.info
        inherited_legacy_admission = info.id in GRANDFATHERED_MARINSKYRL_SOURCES and info.inherited_admission is not None
        blockers = []
        if info.oracle_filtration is not None and info.oracle_filtration.admitted_count == 0:
            blockers.append(
                "oracle filtration retained no tasks: " + ", ".join(sorted(info.oracle_filtration.exclusion_counts))
            )
        dataset = self.dataset
        actor = info.actor_census_input
        if actor is not None and (
            dataset is None
            or (actor.dataset_id, actor.dataset_revision, actor.converter_version)
            != (dataset.name, dataset.revision, self.version)
        ):
            blockers.append("actor census does not cover the current configured input")
        if blocker := population_admission_blocker(
            actor,
            info.within_source_dedup.unique_count if info.within_source_dedup else 0,
            info.grading_environment,
        ):
            blockers.append(blocker)
        if self.is_grandfathered_blend:
            return tuple(blockers)
        try:
            require_private_release(
                info.lexical_contamination, dataset.name if dataset else None, dataset.revision if dataset else None
            )
        except ValueError as error:
            blockers.append(str(error))
        if not self.has_actor_census:
            blockers.append("actor census evidence is missing")
        elif inherited_legacy_admission:
            actor = info.actor_census_input
            if (
                actor is None
                or dataset is None
                or (actor.dataset_id, actor.dataset_revision, actor.converter_version)
                != (dataset.name, dataset.revision, self.version)
            ):
                blockers.append("actor census does not cover the current retained release")
        if not inherited_legacy_admission and (
            info.oversight is None
            or info.oversight.level not in {OversightLevel.SAMPLE, OversightLevel.FULL}
            or not info.oversight.verification.strip()
        ):
            blockers.append("SAMPLE or FULL oversight with verification evidence is missing")
        if "agentic" not in info.tags:
            return tuple(blockers)
        if info.task_format is None or (
            info.task_format is TaskFormat.MARINSKYRL and info.id not in GRANDFATHERED_MARINSKYRL_SOURCES
        ):
            blockers.append("accepted agentic task format is missing")
        if info.daytona_snapshot_count is None or not (info.daytona_count_basis or "").strip():
            blockers.append("production snapshot identity measurement is missing")
        if info.daytona_execution_failures is None:
            blockers.append("execution requirement measurement is missing")
        return tuple(blockers)

    @property
    def declaration(self) -> SourceDeclaration:
        """Project a source declaration into catalog metadata without executing it."""
        info, pipeline = self.info, self.pipeline
        if pipeline is None or self.config is None:
            raise ValueError(f"Source {info.id} has no registered conversion pipeline")
        dataset = self.dataset
        invocation = self.pipeline_reference
        dedup = info.within_source_dedup
        if dedup is None:
            raise ValueError(f"Source {info.id} requires a complete within-source deduplication census")
        tags = info.tags
        if self.is_grandfathered_blend or (
            info.oracle_filtration is not None and info.oracle_filtration.admitted_count == 0
        ):
            safety = None
        else:
            if (
                info.daytona_snapshot_count is None
                or info.daytona_machine_slots is None
                or info.daytona_execution_failures is None
                or not info.daytona_count_basis
                or not info.daytona_count_basis.strip()
            ):
                raise ValueError(f"Source {info.id} requires an actor safety measurement before catalog declaration")
            release = self.current_release
            measurement = DaytonaMeasurement(
                dataset_id=release.repo,
                dataset_revision=release.revision,
                daytona_snapshot_count=info.daytona_snapshot_count,
                daytona_machine_slots=info.daytona_machine_slots,
                daytona_execution_failures=info.daytona_execution_failures,
                daytona_count_basis=info.daytona_count_basis,
                daytona_unsafe_reasons=list(info.daytona_unsafe_reasons),
                daytona_unsafe_details=info.daytona_unsafe_details or {},
                daytona_unsafe_counts=info.daytona_unsafe_counts or {},
                daytona_roles=info.daytona_roles,
            )
            safety = measurement.safety
        return SourceDeclaration(
            id=info.id,
            name=self.name,
            display_name=source_title(info.upstream_name, info.subset, info.variant),
            origin=info.origin,
            build_path=info.build_path,
            upstream_dataset=info.upstream_dataset,
            upstream_name=info.upstream_name,
            subset=info.subset,
            variant=info.variant,
            current_release=self.current_release,
            quality_result=info.quality_result,
            quality_result_finalization=info.quality_result_finalization,
            quality_result_waiver=info.quality_result_waiver,
            family=info.family,
            tags=list(tags),
            task_count=dedup.unique_count,
            within_source_dedup=dedup,
            notes=" ".join(
                (info.notes, *(f"Admission blocked: {reason}." for reason in self.admission_blockers))
            ).strip(),
            dataset_id=dataset.name if dataset else None,
            dataset_revision=dataset.revision if dataset else None,
            url=info.upstream_url or (dataset.url if dataset else None),
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
            intended_use=self.intended_use,
            kind="Generator" if "generator" in tags else "Dataset",
            status=next(
                (label for tag, label in (("deprecated", "Deprecated"), ("excluded", "Excluded")) if tag in tags),
                "Not selectable" if self.admission_blockers else "Available",
            ),
            pipeline=invocation,
            independent_qc=info.independent_qc,
            daytona_safety=safety,
            daytona_exemption=GRANDFATHERED_DAYTONA_EXEMPTION if self.is_grandfathered_blend else None,
            daytona_snapshot_count=info.daytona_snapshot_count,
            daytona_count_basis=info.daytona_count_basis,
            oracle_validation=info.oracle_validation,
            oversight=info.oversight.level if info.oversight else OversightLevel.NONE,
            oversight_evidence=info.oversight,
            grading_environment=info.grading_environment,
            integrity_contracts=info.integrity_contracts,
            actor_census_input=info.actor_census_input,
            converted_examples=info.converted_examples,
            oracle_filtration=info.oracle_filtration,
            lexical_contamination=info.lexical_contamination,
            daytona_machine_slots=info.daytona_machine_slots,
            daytona_roles=info.daytona_roles,
            daytona_execution_failures=info.daytona_execution_failures,
            daytona_unsafe_reasons=list(info.daytona_unsafe_reasons),
            daytona_unsafe_details=info.daytona_unsafe_details or {},
            daytona_unsafe_counts=info.daytona_unsafe_counts or {},
            daytona_snapshot_threshold=(
                DAYTONA_SAFE_SNAPSHOT_LIMIT if info.daytona_snapshot_count is not None else None
            ),
        )


class CurationPipeline[ConfigT: CatalogConfig](Protocol):
    def __call__(self, source: RlDataSource[ConfigT], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]: ...
