# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RL data curation: dataset declarations, their artifacts and the campaign driver."""

from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object
from rigging.filesystem.storage_path import StoragePath

CANONICAL_ARTIFACT_ROOT = "s3://marin-us-east-02a/marin"


def publish_once(output: StoragePath, files: dict[str, bytes]) -> None:
    """Create files or verify identical existing bytes."""
    output.mkdirs()
    for name, contents in files.items():
        target = conditional_object(str(output / name))
        try:
            target.write(contents, expected_version=None)
        except ConditionalWriteError as error:
            existing = target.read()
            if existing is None or existing.data != contents:
                raise FileExistsError(f"Different bytes are already published at {target.path}") from error
