# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable Harbor dataset artifacts."""

import re
from dataclasses import dataclass

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import hf_download

_MIRROR_FORMAT_VERSION = "2026.07.27"
_HARBOR_DATASET_NAMESPACE = "evaluation/harbor-datasets"
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

    def artifact(self) -> ArtifactStep[Artifact]:
        """Return the lazy Hugging Face download at the evaluator's artifact prefix."""
        return hf_download(
            name=f"{_HARBOR_DATASET_NAMESPACE}/{self.slug}/{self.commit}",
            hf_id=self.repository,
            revision=self.commit,
            version=_MIRROR_FORMAT_VERSION,
        )
