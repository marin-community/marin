# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read exact evidence members without extracting a production trial archive."""

import hashlib
import io
import tarfile
from pathlib import PurePosixPath

import zstandard
from finestore.rl.curation import ContentEvidenceReference
from rigging.filesystem.storage_path import StoragePath


def verified_archive_members(
    archive: ContentEvidenceReference, names: set[str], *, optional_names: frozenset[str] = frozenset()
) -> dict[str, bytes]:
    """Verify immutable archive bytes and return each requested regular member once."""
    requested = names | optional_names
    for name in requested:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name:
            raise ValueError("Evidence requires exact relative archive members")
    payload = StoragePath(archive.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != archive.sha256:
        raise ValueError("Production evidence archive bytes changed")
    selected = {}
    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(payload)) as raw:
        with tarfile.open(fileobj=raw, mode="r|") as source:
            for item in source:
                if item.name not in requested:
                    continue
                if item.name in selected or not item.isfile():
                    raise ValueError("Production archive contains ambiguous evidence members")
                stream = source.extractfile(item)
                assert stream is not None
                selected[item.name] = stream.read()
    if not names <= selected.keys():
        raise ValueError("Production archive omits requested evidence members")
    return selected
