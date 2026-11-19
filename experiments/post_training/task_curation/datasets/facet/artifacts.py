# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable records for FACET pristine image preservation."""

from marin.execution.artifact import Artifact


class PreservedImage(Artifact):
    bundle_id: str
    source_repo: str
    source_revision: str
    source_filename: str
    sif_sha256: str
    sif_bytes: int
    partition_sha256: str
    manifest_digest: str
    config_digest: str
    diff_id: str
    tree_sha256: str
    tree_entries: int
    provenance_uri: str
    provenance_sha256: str
