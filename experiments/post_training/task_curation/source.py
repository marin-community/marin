# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source declarations and their curation pipeline interfaces."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from finestore.rl.curation import (
    DAYTONA_SAFE_SNAPSHOT_LIMIT,
    ConvertedExamples,
    DaytonaUnsafeReason,
    GradingEnvironmentEvidence,
    IndependentQC,
    LexicalContaminationEvidence,
    OracleValidation,
    OversightEvidence,
    OversightLevel,
    PipelineReference,
    RoleExecutionEvidence,
    SourceDeclaration,
    daytona_verdict,
)
from marin.execution.lazy import ArtifactStep

from experiments.post_training.task_curation.campaign import CampaignArtifact
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.decontamination import require_private_release


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


GRANDFATHERED_MARINSKYRL_SOURCES: frozenset[str] = frozenset(
    {
        # Nemotron Ultra mix components: JSONL rows produced by Marin's own RL
        # pipeline, predating the task-directory format.
        "MarinSkyRL:nemotron_ultra_mopd/agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/SWE-Gym/SWE-Gym",
        "MarinSkyRL:nemotron_ultra_mopd/agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/nebius/SWE-rebench-V2",
        "MarinSkyRL:nemotron_ultra_mopd/makeshn_ultra_v3_ipi_train",
        "MarinSkyRL:nemotron_ultra_mopd/swe_pivot_len40k/SWE-Gym/SWE-Gym",
        "MarinSkyRL:nemotron_ultra_mopd/swe_pivot_len40k/nebius/SWE-rebench-V2",
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_lean",
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_math_tir",
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_rdkit",
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_toolcall_schema",
        "MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/SWE-Gym/SWE-Gym",
        "MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/nebius/SWE-rebench-V2",
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_lean",
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_math_tir",
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2",
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_tau_pivot",
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_toolcall_schema",
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_lean",
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_math_tir",
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2",
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_tau_pivot",
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_toolcall_schema",
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_rdkit",
    }
)
"""Marin-internal blend sources admitted before the task-directory requirement."""


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
    converted_examples: ConvertedExamples | None = None
    lexical_contamination: LexicalContaminationEvidence | None = None
    daytona_machine_slots: int | None = None
    daytona_roles: dict[str, dict[str, RoleExecutionEvidence]] | None = None
    daytona_unsafe_reasons: tuple[DaytonaUnsafeReason, ...] = ()
    daytona_unsafe_details: dict[str, str] | None = None


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
    def admission_blockers(self) -> tuple[str, ...]:
        """Evidence missing before a source can enter admitted selection."""
        if self.info.id in GRANDFATHERED_MARINSKYRL_SOURCES:
            return ()
        info = self.info
        blockers = []
        dataset = self.dataset
        try:
            require_private_release(
                info.lexical_contamination, dataset.name if dataset else None, dataset.revision if dataset else None
            )
        except ValueError as error:
            blockers.append(str(error))
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
        if (
            info.oversight is None
            or info.oversight.level not in {OversightLevel.SAMPLE, OversightLevel.FULL}
            or not info.oversight.verification.strip()
        ):
            blockers.append("SAMPLE or FULL oversight with verification evidence is missing")
        return tuple(blockers)

    @property
    def declaration(self) -> SourceDeclaration:
        """Project a source declaration into catalog metadata without executing it."""
        info, pipeline = self.info, self.pipeline
        if pipeline is None or self.config is None:
            raise ValueError(f"Source {info.id} has no registered conversion pipeline")
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
            notes=" ".join(
                (info.notes, *(f"Admission blocked: {reason}." for reason in self.admission_blockers))
            ).strip(),
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
                "Not selectable" if self.admission_blockers else "Available",
            ),
            pipeline=invocation,
            independent_qc=info.independent_qc,
            daytona_safety=safety,
            daytona_snapshot_count=info.daytona_snapshot_count,
            daytona_count_basis=info.daytona_count_basis,
            oracle_validation=info.oracle_validation,
            oversight=info.oversight.level if info.oversight else OversightLevel.NONE,
            oversight_evidence=info.oversight,
            grading_environment=info.grading_environment,
            converted_examples=info.converted_examples,
            lexical_contamination=info.lexical_contamination,
            daytona_machine_slots=info.daytona_machine_slots,
            daytona_roles=info.daytona_roles,
            daytona_execution_failures=info.daytona_execution_failures,
            daytona_unsafe_reasons=list(info.daytona_unsafe_reasons),
            daytona_unsafe_details=info.daytona_unsafe_details or {},
            daytona_snapshot_threshold=(
                DAYTONA_SAFE_SNAPSHOT_LIMIT if info.daytona_snapshot_count is not None else None
            ),
        )


class CurationPipeline[ConfigT: CatalogConfig](Protocol):
    def __call__(self, source: RlDataSource[ConfigT], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]: ...
