# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned handle for the canonical task curriculum catalog."""

import posixpath

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep

CATALOG_FILENAME = "curriculum.yaml"
CATALOG_SHA256 = "72a763b98f9ecf7f8f598b788c4f59e7ace213c01a30b403768a8f8f16f55382"
CATALOG_ROOT_URI = "s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.18-72a763b98f9e"
CATALOG_URI = posixpath.join(CATALOG_ROOT_URI, CATALOG_FILENAME)
SOURCE_SURVEY_ROOT_URI = (
    "s3://marin-us-east-02a/marin/task-curriculum/experiments/2026.09.19-source-survey-v2-148820c9ecae"
)
SOURCE_SURVEY_COMPARISON_SHA256 = "148820c9ecaee60bd7051788b17d8e2b93110d5b380564e3afa86fd592519c59"
SOURCE_SURVEY_EVIDENCE_SHA256 = "973c97ebe42ca84eae06982527bb32cb7f5406ec2ba9188124b2e3aae339b78f"
SOURCE_AUDIT_V3_ROOT_URI = posixpath.join(
    "s3://marin-us-east-02a/marin/task-curriculum/experiments",
    "2026.09.19-bounded-source-audit-v3-f0edd8ce5141",
)
SOURCE_AUDIT_V3_CATALOG_SHA256 = "faa15da8458147bb5415b9213fc76f95963215a25ad889d3b9215687cae9993b"
SOURCE_AUDIT_V3_COMPARISON_SHA256 = "f0edd8ce5141a4dc01dbeb26dbd55fdd0b438db9fd10d74e3b7ab2e4c8916730"
SOURCE_AUDIT_V3_EVIDENCE_SHA256 = "25ff4b75fb7b08e6e69fa4b8fd29f6f371b84a435b35af8253ed3cb49e0482c2"

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

TASK_CURRICULUM_SOURCE_SURVEY_V2 = ArtifactStep.adopt(
    "post-training/task-curriculum/source-survey-v2",
    "2026.09.19.1",
    source=SOURCE_SURVEY_ROOT_URI,
    kind=Artifact,
    config={
        "baseline_catalog_sha256": CATALOG_SHA256,
        "comparison_filename": "comparison.json",
        "comparison_sha256": SOURCE_SURVEY_COMPARISON_SHA256,
        "evidence_filename": "evidence.tar.gz",
        "evidence_sha256": SOURCE_SURVEY_EVIDENCE_SHA256,
        "subjects": ["D01", "D08", "D17", "D31", "D41"],
    },
)

TASK_CURRICULUM_SOURCE_AUDIT_V3 = ArtifactStep.adopt(
    "post-training/task-curriculum/source-audit-v3",
    "2026.09.19.1",
    source=SOURCE_AUDIT_V3_ROOT_URI,
    kind=Artifact,
    config={
        "baseline_catalog_sha256": CATALOG_SHA256,
        "catalog_filename": CATALOG_FILENAME,
        "catalog_sha256": SOURCE_AUDIT_V3_CATALOG_SHA256,
        "comparison_filename": "comparison.json",
        "comparison_sha256": SOURCE_AUDIT_V3_COMPARISON_SHA256,
        "evidence_filename": "evidence.tar.gz",
        "evidence_sha256": SOURCE_AUDIT_V3_EVIDENCE_SHA256,
        "experimental": True,
        "subjects_replaced": ["D01", "D02", "D08", "D17", "D31", "D41"],
    },
)
