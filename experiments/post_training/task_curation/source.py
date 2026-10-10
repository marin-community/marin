# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source identity and assessments, independent of the Atlas presentation."""

from dataclasses import dataclass, field
from typing import Literal, Protocol

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


MARINSKYRL_GRADING_REVISION = "e44c4bfcb62c489286a1264094e6d9c883aaf0d2"
HARBOR_GRADING_REVISION = "8abc63e3bdb37af1d345fcac123ef7d2122598f3"


@dataclass(frozen=True)
class GradingSelection:
    """Select the native verifier behavior an assessment covers."""

    mode: Literal["verifyit", "legacy", "harbor"]
    agents: tuple[str, ...]
    marinskyrl_revision: str
    harbor_revision: str


@dataclass(frozen=True)
class SourceReference:
    """A dataset or verifier at the revision an assessment covers."""

    name: str
    revision: str | None
    url: str
    grading: GradingSelection | None = None


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
    verifier: SourceReference | None = None


@dataclass(frozen=True)
class DataSourceReview:
    """An authored assessment and the dataset/verifier revisions it covers.

    Executed reviews and difficulty measurements remain in the Atlas database.
    Declaring a conversion recipe does not imply a favorable quality review.
    """

    grade: Literal["good", "some_issues", "bad"] | None = None
    evidence_url: str | None = None
    reviewed_at: str | None = None
    dataset_revision: str | None = None
    verifier_revision: str | None = None


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
    review: DataSourceReview = field(default_factory=DataSourceReview)

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


class CurationPipeline[ConfigT: CatalogConfig](Protocol):
    def __call__(self, source: RlDataSource[ConfigT], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]: ...
