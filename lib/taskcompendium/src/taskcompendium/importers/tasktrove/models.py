# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove archive data and provenance."""

from dataclasses import dataclass

from taskcompendium.models import Source

IMPORTER_REVISION = "taskcompendium-tasktrove-v0.2"


@dataclass(frozen=True)
class TaskArchive:
    """A bounded archive with caller-supplied release provenance."""

    upstream_subset: str
    archive_path: str
    release_uri: str
    release_revision: str
    files: dict[str, bytes]

    @property
    def source(self) -> Source:
        return Source(
            dataset=self.release_uri,
            revision=self.release_revision,
            row=f"{self.upstream_subset}:{self.archive_path}",
            importer_revision=IMPORTER_REVISION,
        )
