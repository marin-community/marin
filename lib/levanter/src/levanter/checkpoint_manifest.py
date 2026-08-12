# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""``manifest.json``: the storage format and every array in a checkpoint.

Written by process 0 when a save starts. This contains a mirror of the information in the
OCDBT database but can be read without reading all chunks. Older checkpoints have no
manifest and fall back to listing.
"""

import logging
from typing import Sequence

from pydantic import BaseModel, ConfigDict

from rigging.filesystem import StoragePath, prefix_join

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"

CHECKPOINT_FORMAT_VERSION = 1
"""Bump when a change makes an older reader unable to load a newer checkpoint."""


class CheckpointArray(BaseModel):
    """One serialized array, keyed by its dotted leaf path with dots replaced by slashes."""

    model_config = ConfigDict(frozen=True)

    path: str
    shape: tuple[int, ...]
    dtype: str
    chunk_shape: tuple[int, ...]
    """The zarr3 write-chunk grid. Divides each writer's slice, so writers never share a chunk."""


class CheckpointManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    format_version: int
    array_driver: str
    kvstore_driver: str
    arrays: tuple[CheckpointArray, ...]

    @property
    def array_paths(self) -> frozenset[str]:
        return frozenset(array.path for array in self.arrays)


def build_manifest(arrays: Sequence[CheckpointArray], *, array_driver: str, kvstore_driver: str) -> CheckpointManifest:
    return CheckpointManifest(
        format_version=CHECKPOINT_FORMAT_VERSION,
        array_driver=array_driver,
        kvstore_driver=kvstore_driver,
        arrays=tuple(arrays),
    )


def manifest_path(checkpoint_root: str) -> str:
    return prefix_join(checkpoint_root, MANIFEST_FILENAME)


def write_manifest(checkpoint_root: str, manifest: CheckpointManifest) -> None:
    StoragePath(manifest_path(checkpoint_root)).write_text(manifest.model_dump_json(indent=2))


def read_manifest(checkpoint_root: str) -> CheckpointManifest | None:
    """Return the manifest, or None for a checkpoint written before manifests existed."""
    path = StoragePath(manifest_path(checkpoint_root))
    if not path.exists():
        return None

    manifest = CheckpointManifest.model_validate_json(path.read_text())
    if manifest.format_version > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint {checkpoint_root} is format version {manifest.format_version}, but this "
            f"build of levanter understands at most {CHECKPOINT_FORMAT_VERSION}. Upgrade levanter."
        )
    return manifest
