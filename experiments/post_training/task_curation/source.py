# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source identity and assessments, independent of the Atlas presentation."""

from dataclasses import dataclass, field
from typing import Literal

from experiments.post_training.task_curation.invocation import CurationPipeline


@dataclass(frozen=True)
class SourceReference:
    """A dataset or verifier at the revision an assessment covers."""

    name: str
    revision: str | None
    url: str


@dataclass(frozen=True, kw_only=True)
class SourceInfo:
    """Identity and discovery information for one source population.

    ``count`` is the number of selected input rows at the dataset revision,
    before conversion or curation. Leave it unknown unless the pinned payload
    or a complete manifest establishes it. A sample size is not a source count.

    ``dataset`` identifies runnable sources and inventory entries. Recipe
    registration derives it from the pinned input. Tags describe task type,
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
class RlDataSource[PipelineT: CurationPipeline]:
    """One source population, its assessment, and an optional curation callable.

    Stable IDs preserve Atlas review history even when the callable changes.
    Sources without a pipeline remain discoverable in the inventory.
    Recipe registration derives immutable catalog fields from its declaration.
    Other datasets specify those fields independently.
    """

    info: SourceInfo
    pipeline: PipelineT | None = None
    review: DataSourceReview = field(default_factory=DataSourceReview)
    version: str = "1"
    files: tuple[str, ...] = ()
    name: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.info.id.partition(":")[2])

    @property
    def dataset(self) -> SourceReference | None:
        return self.info.dataset
