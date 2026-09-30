# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove archive data and provenance."""

from dataclasses import dataclass

from taskcompendium.models import Source, TaskSpec

IMPORTER_REVISION = "taskcompendium-tasktrove-v0.3"


@dataclass(frozen=True)
class TaskArchive:
    """A bounded archive with caller-supplied release provenance."""

    upstream_subset: str
    archive_path: str
    release_uri: str
    release_revision: str
    archive_sha256: str
    files: dict[str, bytes]

    @property
    def source(self) -> Source:
        return Source(
            dataset=self.release_uri,
            revision=self.release_revision,
            row=f"{self.upstream_subset}:{self.archive_path}",
            importer_revision=IMPORTER_REVISION,
        )


@dataclass(frozen=True)
class TaskTroveSourceEvidence:
    """The clean-release row metadata needed to audit an imported task."""

    source: str
    path: str
    family: str
    converter: str
    template_id: str
    mode: str
    archive_sha256: str


@dataclass(frozen=True)
class TaskTroveImportResult:
    """An imported private task with its ordered release tags and source evidence."""

    specification: TaskSpec
    tags: tuple[str, ...]
    source_evidence: TaskTroveSourceEvidence
