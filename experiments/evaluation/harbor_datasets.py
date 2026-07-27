# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable, region-local Harbor dataset artifacts."""

import re
from dataclasses import dataclass

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import hf_download
from rigging.filesystem import marin_temp_bucket

_CACHE_TTL_DAYS = 7
_CATALOG_VERSION = "2026.07.27"
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class HuggingFaceHarborDataset:
    """One Harbor task repository pinned to an immutable Hugging Face commit."""

    repository: str
    commit: str

    def __post_init__(self) -> None:
        if self.repository.count("/") != 1:
            raise ValueError(f"expected an org/repository Hugging Face id, got {self.repository!r}")
        if not _COMMIT_PATTERN.fullmatch(self.commit):
            raise ValueError(f"expected a full immutable Hugging Face commit, got {self.commit!r}")

    @property
    def slug(self) -> str:
        return self.repository.replace("/", "--")

    def mirror_uri(self, placement_prefix: str) -> str:
        """Return the cache path colocated with ``placement_prefix``."""
        return marin_temp_bucket(
            ttl_days=_CACHE_TTL_DAYS,
            prefix=f"evaluation/harbor-datasets/{self.slug}/{self.commit}",
            source_prefix=placement_prefix,
        )

    def artifact_for(self, placement_prefix: str) -> ArtifactStep[Artifact]:
        """Return the lazy Hugging Face-to-regional-cache transfer."""
        return hf_download(
            name=f"evaluation/harbor-datasets/{self.slug}",
            hf_id=self.repository,
            revision=self.commit,
            version=_CATALOG_VERSION,
            pin=self.mirror_uri(placement_prefix),
        )
