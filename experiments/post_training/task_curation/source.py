# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source declarations and their curation pipeline interfaces."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from finestore.rl.curation import (
    DAYTONA_SAFE_SNAPSHOT_LIMIT,
    DatasetRelease,
    IndependentQC,
    OracleValidation,
    OversightEvidence,
    OversightLevel,
    PipelineReference,
    SourceDeclaration,
    SourceOrigin,
    SourceUse,
    daytona_verdict,
    source_identifier,
    source_title,
)
from finestore.rl.licenses import LicenseReceipt, require_license
from finestore.rl.status_taxonomy import DataRestriction
from marin.execution.lazy import ArtifactStep
from taskcompendium.pipeline.models import IntendedUse

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


GRANDFATHERED_MARINSKYRL_SOURCES: frozenset[str] = frozenset(
    {
        # Nemotron Ultra mix components: JSONL rows produced by Marin's own RL
        # pipeline, predating the task-directory format.
        "Nemotron:nemotron_ultra_mopd/agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/SWE-Gym/SWE-Gym",
        "Nemotron:nemotron_ultra_mopd/agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/nebius/SWE-rebench-V2",
        "Nemotron:nemotron_ultra_mopd/makeshn_ultra_v3_ipi_train",
        "Nemotron:nemotron_ultra_mopd/swe_pivot_len40k/SWE-Gym/SWE-Gym",
        "Nemotron:nemotron_ultra_mopd/swe_pivot_len40k/nebius/SWE-rebench-V2",
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_lean",
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_math_tir",
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_rdkit",
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_toolcall_schema",
        "Nemotron:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/SWE-Gym/SWE-Gym",
        "Nemotron:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/nebius/SWE-rebench-V2",
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_lean",
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_math_tir",
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2",
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_tau_pivot",
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_toolcall_schema",
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_lean",
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_math_tir",
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2",
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_tau_pivot",
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_toolcall_schema",
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_rdkit",
    }
)
"""Marin-internal blend sources admitted before the task-directory requirement."""


@dataclass(frozen=True)
class SourceReference:
    """A pinned dataset reference."""

    name: str
    revision: str | None
    url: str


EXCLUDED_TAG = "excluded"
"""The source tag that keeps a declared source out of admission."""


@dataclass(frozen=True, kw_only=True)
class SourceInfo:
    """Identity and discovery information for one source population.

    ``id`` and ``title`` are generated from the upstream identity keys:
    ``<origin>:<upstream_dataset>[/<subset>][@<variant>]`` and
    ``<upstream_name> · <subset> · <variant>``. Stored records written under
    historical IDs resolve through the alias table where they are read, never
    at authoring. The source declares no row count. Catalog publication may
    use a COMPLETE attempt's count or carry the prior row's count at the same pin and recipe.

    ``dataset`` identifies inventory entries without a configuration. Runnable
    sources expose their configured pinned input directly. Tags describe task type,
    interaction, benchmark status, and source-specific search terms.
    """

    origin: SourceOrigin
    upstream_dataset: str
    upstream_name: str
    build_path: str
    """Internal campaign or pipeline, independently of the upstream origin."""
    subset: str | None = None
    variant: str | None = None
    data_restriction: DataRestriction
    """The source's typed restriction tier, set explicitly on every source.

    ``PRIVATE_ALWAYS`` keeps the source's content in its private store no matter
    what else changes. The tier travels with the qualified declaration the
    source's ingestion publishes; this legacy catalog row does not carry it.
    """
    current_release: DatasetRelease | None = None
    """The exact release consumed for training, separately from upstream provenance."""
    intended_use: IntendedUse | None = None
    """The declared use of a source recorded without its registered converter.

    Configured sources resolve their use from the converter instead; a source
    that declares it in both places is refused.
    """
    upstream_url: str | None = None
    """Upstream provenance URL, including for repaired releases."""
    license_receipt: LicenseReceipt | None = None
    """The stage-0 license resolution this source's ingest and admission rest on."""
    family: str = ""
    tags: tuple[str, ...] = ()
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

    oracle_validation: OracleValidation | None = None
    """How the source's grader was qualified.

    ``executed`` runs the shipped oracle solutions; ``verifier_executed`` records
    a negative verdict on an untouched submission without certifying an oracle;
    ``skipped`` certifies neither. The Atlas shows this value beside the catalog.
    """

    oversight: OversightEvidence | None = None
    """The review this source's admission rests on, with its evidence pointers."""

    @property
    def id(self) -> str:
        return source_identifier(self.origin, self.upstream_dataset, self.subset, self.variant)

    @property
    def title(self) -> str:
        return source_title(self.upstream_name, self.subset, self.variant)


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

    def require_license(self) -> None:
        """Require stage-0 permission at the current pin before constructing any work."""
        dataset = self.dataset
        require_license(
            self.info.license_receipt,
            self.info.id,
            dataset.name if dataset else None,
            dataset.revision if dataset else None,
            self.intended_use.value,
            ban=None,
        )

    @property
    def intended_use(self) -> SourceUse:
        """Resolve population policy from the registered converter, or the record's declared use."""
        if self.config is not None and self.info.intended_use is not None:
            raise ValueError(f"Source {self.info.id} declares its use twice")
        if self.config is not None:
            return SourceUse(self.config.intended_use.value)
        if self.info.intended_use is None:
            raise ValueError(f"No registered conversion Use: {self.info.id}")
        return SourceUse(self.info.intended_use.value)

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
            notes=info.notes,
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
            kind="Generator" if "generator" in tags else "Dataset",
            status=next(
                (label for tag, label in (("deprecated", "Deprecated"), (EXCLUDED_TAG, "Excluded")) if tag in tags),
                "Inventory" if pipeline is None else "Available",
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
