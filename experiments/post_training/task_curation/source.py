# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source identity and assessments, independent of the Atlas presentation."""

from dataclasses import dataclass, field
from typing import Literal

from experiments.post_training.task_curation.pipeline import RlDataPipeline


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

    Runnable sources take dataset identity from their pipeline. ``dataset``
    identifies inventory entries without a pipeline. Tags describe task type,
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
class RlDataSource:
    """One source population, its assessment, and an optional conversion recipe.

    Stable IDs preserve Atlas review history even when the recipe changes.
    Sources without a pipeline remain discoverable in the inventory.
    """

    info: SourceInfo
    pipeline: RlDataPipeline | None = None
    review: DataSourceReview = field(default_factory=DataSourceReview)

    def __post_init__(self) -> None:
        if self.pipeline is not None and self.info.dataset is not None:
            raise ValueError("Runnable sources define their dataset only in pipeline.source")

    @property
    def name(self) -> str:
        return self.pipeline.name if self.pipeline is not None else self.info.id.partition(":")[2]
