# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned handles for task curriculum catalogs and experiments."""

import json

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.task_curriculum.models import CurriculumCatalog, SubjectInventory

CATALOG_ARTIFACT_NAME = "post-training/task-curriculum/catalog"


class TaskCurriculumCatalogArtifact(Artifact):
    """Typed access to an immutable task curriculum catalog bundle."""

    @property
    def catalog_uri(self) -> str:
        return prefix_join(self.path, "curriculum.json")

    def read_catalog(self) -> CurriculumCatalog:
        """Load and validate the catalog from the artifact."""
        return CurriculumCatalog.model_validate_json(StoragePath(self.catalog_uri).read_bytes())


class TaskCurriculumSubjectInventoryArtifact(Artifact):
    """Typed access to the subject inventory used for catalog generation."""

    @property
    def inventory_uri(self) -> str:
        return prefix_join(self.path, "subject_inventory.json")

    def read_inventory(self) -> SubjectInventory:
        """Load and validate the subject inventory from the artifact."""
        return SubjectInventory.model_validate(json.loads(StoragePath(self.inventory_uri).read_bytes()))


TASK_CURRICULUM = ArtifactStep.adopt(
    CATALOG_ARTIFACT_NAME,
    "2026.09.20.1",
    source="s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.20-subject-local-v3-0ce32038771d",
    kind=TaskCurriculumCatalogArtifact,
)

TASK_CURRICULUM_V2 = ArtifactStep.adopt(
    CATALOG_ARTIFACT_NAME,
    "2026.09.19.2",
    source="s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.19-json-1faada4eeda8/publish",
    kind=TaskCurriculumCatalogArtifact,
)

TASK_CURRICULUM_V1 = ArtifactStep.adopt(
    CATALOG_ARTIFACT_NAME,
    "2026.09.18.2",
    source="s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.18-72a763b98f9e",
    kind=Artifact,
)

TASK_CURRICULUM_YAML_V2 = ArtifactStep.adopt(
    CATALOG_ARTIFACT_NAME,
    "2026.09.19.1",
    source="s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.19-2578589fb0de",
    kind=Artifact,
)

TASK_CURRICULUM_SUBJECT_INVENTORY = ArtifactStep.adopt(
    "post-training/task-curriculum/subject-inventory",
    "2026.09.18.1",
    source="s3://marin-us-east-02a/marin/task-curriculum/inventories/2026.09.18-cross-domain-v1-ef10281c71ad",
    kind=TaskCurriculumSubjectInventoryArtifact,
)

TASK_CURRICULUM_SOURCE_SURVEY_V2 = ArtifactStep.adopt(
    "post-training/task-curriculum/source-survey-v2",
    "2026.09.19.1",
    source="s3://marin-us-east-02a/marin/task-curriculum/experiments/2026.09.19-source-survey-v2-148820c9ecae",
    kind=Artifact,
)

TASK_CURRICULUM_SOURCE_AUDIT_V3 = ArtifactStep.adopt(
    "post-training/task-curriculum/source-audit-v3",
    "2026.09.19.1",
    source="s3://marin-us-east-02a/marin/task-curriculum/experiments/2026.09.19-bounded-source-audit-v3-f0edd8ce5141",
    kind=Artifact,
)
