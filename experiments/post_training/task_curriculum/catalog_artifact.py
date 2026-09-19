# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned handle for the canonical task curriculum catalog."""

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep

CATALOG_FILENAME = "curriculum.yaml"
CATALOG_SHA256 = "72a763b98f9ecf7f8f598b788c4f59e7ace213c01a30b403768a8f8f16f55382"
CATALOG_URI = "s3://marin-us-east-02a/marin/task-curriculum/catalogs/" "2026.09.18-72a763b98f9e/curriculum.yaml"
CATALOG_ROOT_URI = CATALOG_URI.removesuffix(f"/{CATALOG_FILENAME}")

TASK_CURRICULUM = ArtifactStep.adopt(
    "post-training/task-curriculum/catalog",
    "2026.09.18.2",
    source=CATALOG_ROOT_URI,
    kind=Artifact,
    config={
        "catalog_version": "2026.09.18-cross-domain-v1",
        "filename": CATALOG_FILENAME,
        "sha256": CATALOG_SHA256,
    },
)
