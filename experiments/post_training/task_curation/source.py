# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source identity and assessments, independent of the Atlas presentation."""

from dataclasses import dataclass, field
from typing import Literal

from experiments.post_training.task_curation.invocation import CurationPipeline
from experiments.post_training.task_curation.pipeline import HfSource, RlDataPipeline


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

    ``dataset`` identifies custom runnable sources and inventory entries. The
    standard declaration adapter derives it from the pinned input. Tags describe task type,
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
    """One source population, its assessment, and an optional curation callable.

    Stable IDs preserve Atlas review history even when the callable changes.
    Sources without a pipeline remain discoverable in the inventory.
    Standard declarations derive immutable catalog fields from their recipe;
    custom declarations specify their dataset, version and files independently.
    """

    info: SourceInfo
    pipeline: CurationPipeline | None = None
    review: DataSourceReview = field(default_factory=DataSourceReview)
    version: str = "1"
    files: tuple[str, ...] = ()
    dataset: SourceReference | None = field(init=False)
    name: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", self.info.id.partition(":")[2])
        object.__setattr__(self, "dataset", self.info.dataset)
        if not isinstance(self.pipeline, RlDataPipeline):
            return
        if self.info.dataset is not None:
            raise ValueError("Standard sources derive their dataset from pipeline.source")
        # Derive standard catalog identity at construction, including dataclasses.replace.
        # Consumers never need to inspect an execution callable for metadata.
        upstream = self.pipeline.source
        if isinstance(upstream, HfSource):
            dataset = SourceReference(
                upstream.repo,
                upstream.revision,
                f"https://huggingface.co/datasets/{upstream.repo}/tree/{upstream.revision}",
            )
            files = upstream.files
        else:
            dataset = SourceReference(upstream.filename, upstream.sha256, upstream.url)
            files = (upstream.filename,)
        object.__setattr__(self, "name", self.pipeline.name)
        object.__setattr__(self, "version", self.pipeline.version)
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "files", files)
