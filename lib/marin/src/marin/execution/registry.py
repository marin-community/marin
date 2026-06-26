# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Artifact registry: the build-once-immutability guard and run provenance.

An artifact is addressed by an explicit ``name@version`` whose output path is
``{prefix}/{name}/{version}`` — no content hash. To keep that address honest, the
registry records *how* the artifact was built (its recipe ``fingerprint``) plus
*who/when/which-commit* (provenance) in a small ``.artifact_record.json`` file next
to the output, and enforces one rule:

    a ``name@version`` is built once. Rebuilding it from a changed recipe (a
    different fingerprint) is an error — bump the version.

This is what replaces content-addressing: instead of a new hash silently forking a
new output path on every code change, the version is explicit and the guard makes
a stale-recipe rebuild loud. The escape hatch is a *mutable* version (``dev`` or a
``-dev`` suffix), which skips the guard and always rebuilds — for iteration.

The fingerprint and version travel on a ``StepSpec`` via ``hash_attrs`` (keys
:data:`FINGERPRINT_KEY` / :data:`VERSION_KEY`), so the runner can apply the guard
before serving a cached output.
"""

import json
from dataclasses import asdict, dataclass

from rigging.filesystem import open_url, url_to_fs

from marin.execution.step_spec import StepSpec
from marin.utilities.json_encoder import CustomJsonEncoder

# A distinct name from the output-payload sidecar (``.artifact.json``, written by
# marin.execution.artifact) so a build record never shadows a payload read.
RECORD_FILENAME = ".artifact_record.json"

# Keys under ``StepSpec.hash_attrs`` that carry the artifact's identity, so the
# runner can apply the immutability guard without knowing about the lazy layer.
FINGERPRINT_KEY = "fingerprint"
VERSION_KEY = "version"


class ImmutableArtifactError(Exception):
    """Raised when a fixed ``name@version`` is rebuilt from a changed recipe."""


@dataclass(frozen=True)
class ArtifactRecord:
    """What produced an artifact at a given output path."""

    name: str
    version: str
    fingerprint: str
    output_path: str
    git_commit: str | None
    user: str | None
    created_at: str
    deps: list[str]
    """Dependency identities as ``name@version`` strings."""
    source: str | None = None
    """For an *adopted* artifact, the pre-existing data location this ``name@version``
    aliases (where consumers resolve). ``None`` for a computed artifact, whose data
    lives at ``output_path``."""


def is_mutable_version(version: str) -> bool:
    """A ``dev`` version is mutable: the guard is skipped and it always rebuilds."""
    return version == "dev" or version.endswith("-dev")


def _record_path(output_path: str) -> str:
    return f"{output_path.rstrip('/')}/{RECORD_FILENAME}"


def read_record(output_path: str) -> ArtifactRecord | None:
    """The recorded build of the artifact at ``output_path``, or ``None`` if absent."""
    path = _record_path(output_path)
    fs = url_to_fs(path, use_listings_cache=False)[0]
    if not fs.exists(path):
        return None
    with open_url(path, "r") as f:
        return ArtifactRecord(**json.load(f))


def write_record(record: ArtifactRecord) -> None:
    """Persist ``record`` next to its output path."""
    path = _record_path(record.output_path)
    with open_url(path, "w") as f:
        f.write(json.dumps(asdict(record), indent=2, cls=CustomJsonEncoder))


def guard_immutable(output_path: str, name: str, version: str, fingerprint: str) -> None:
    """Enforce build-once immutability for a fixed ``name@version``.

    No-op when the artifact is unbuilt, when the recorded fingerprint matches, or
    when the version is mutable. Raises :class:`ImmutableArtifactError` when an
    existing record was built from a different recipe.
    """
    if is_mutable_version(version):
        return
    existing = read_record(output_path)
    if existing is None or existing.fingerprint == fingerprint:
        return
    raise ImmutableArtifactError(
        f"{name}@{version} was already built from a different recipe "
        f"(recorded fingerprint {existing.fingerprint}, now {fingerprint}). "
        f"Bump the version to build the new recipe, or delete {output_path} to rebuild in place."
    )


def enforce_immutability(step: StepSpec) -> bool:
    """Apply the immutability guard for a lazy-artifact ``step`` and report whether
    it is mutable.

    Steps without a fingerprint (plain ``StepSpec``s, not lazy artifacts) are
    unguarded and immutable (``False``). A mutable (``dev``) version returns
    ``True`` so the caller rebuilds instead of serving a cached output.
    """
    fingerprint = step.hash_attrs.get(FINGERPRINT_KEY)
    if fingerprint is None:
        return False
    version = step.hash_attrs.get(VERSION_KEY, "")
    if is_mutable_version(version):
        return True
    guard_immutable(step.output_path, step.name, version, fingerprint)
    return False
