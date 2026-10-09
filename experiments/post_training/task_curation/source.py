# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source declarations combine conversion recipes with Atlas metadata."""

from dataclasses import dataclass, field
from typing import Literal

from experiments.post_training.task_curation.pipeline import RlDataPipeline


@dataclass(frozen=True)
class DataSourceReview:
    """An authored source assessment, with the evidence and revisions it covers.

    Executed reviews and difficulty measurements remain in the Atlas database.
    Declaring a conversion recipe does not imply a favorable quality review.
    """

    grade: Literal["good", "some_issues", "bad"] | None = None
    evidence_url: str | None = None
    reviewed_at: str | None = None
    dataset_revision: str | None = None
    verifier_revision: str | None = None


@dataclass(frozen=True, kw_only=True)
class DataSourceMetadata:
    """Source inventory and provenance; counts describe the identified population."""

    id: str
    name: str
    origin: str
    display_name: str = ""
    url: str = ""
    dataset_id: str = ""
    revision: str = ""
    revised_at: str = ""
    dataset_revision: str = ""
    verifier_revision: str | None = None
    family: str = ""
    environment: str = ""
    type: str = ""
    turns: str = ""
    task_count: int | None = None
    count_basis: str = ""
    count_precision: str = "unknown"
    count_url: str = ""
    recorded_at: str = ""
    status: str = "Available"
    kind: str = "Dataset"
    notes: str = ""
    split: str = "train"
    is_benchmark: bool = False
    benchmark_basis: str = ""
    family_basis: str = ""
    family_url: str = ""
    classification_basis: str = ""
    canonical_source: str = ""
    canonical_url: str = ""
    provenance_url: str = ""
    license: tuple[str, ...] = ()
    verification: str = ""
    snapshot_safe: bool = False
    snapshot_safety_basis: str = ""
    upstream_repository: str = ""
    upstream_configuration: str | None = None
    upstream_url: str = ""
    upstream_link_basis: str = ""
    input_count: int | None = None
    languages: tuple[str, ...] = ()
    modes: tuple[str, ...] = ()
    gym_alias: str = ""
    gym_url: str = ""
    gym_entrypoint: str = ""
    canonical_id: str = ""
    registry_name: str = ""
    component_name: str = ""
    component_selector: str = ""
    component_file_sha256: str = ""
    canonical_task_count: int | None = None
    component_ratio: str = ""
    dataset_revised_at: str = ""
    registry_revised_at: str = ""
    verifier_url: str = ""
    verifier_revised_at: str = ""
    revision_basis: str = ""
    grading_revision: str = ""


@dataclass(frozen=True)
class RlDataSource:
    """One source population and its optional TaskSpec conversion recipe.

    A missing pipeline keeps excluded or unfinished sources visible in the
    inventory without pretending they can run. The stable metadata ID preserves
    existing Atlas review links across changes to conversion recipes.
    """

    metadata: DataSourceMetadata
    pipeline: RlDataPipeline | None = None
    review: DataSourceReview = field(default_factory=DataSourceReview)

    @property
    def name(self) -> str:
        return self.pipeline.name if self.pipeline is not None else self.metadata.name
