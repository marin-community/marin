# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable bindings for FineEnvs grouped public-input images."""

from finestore.rl.status_taxonomy import DataRestriction
from marin.execution.artifact import Artifact


class GroupedInputImage(Artifact):
    data_restriction: DataRestriction
    group: int
    base_image: str
    input_signatures: tuple[str, ...]
    manifest_digest: str
    tree_sha256: str
    provenance_uri: str
    provenance_sha256: str
