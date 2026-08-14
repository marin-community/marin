# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit, ordered upgrades between FineStore on-disk format versions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from rigging.filesystem import StoragePath

from finestore.commit import read_snapshot
from finestore.layout import FORMAT_VERSION, ArchiveMetadata, CommitToken, FineStoreLayout, FormatVersionError

from . import m0001_manifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _MigrationRevision:
    """One registered transition between archive format versions."""

    migration_id: str
    from_version: int
    to_version: int
    write_open_safe: bool
    apply: Callable[[str], CommitToken]


_MIGRATIONS = (
    _MigrationRevision(
        migration_id=m0001_manifest.MIGRATION_ID,
        from_version=m0001_manifest.FROM_VERSION,
        to_version=m0001_manifest.TO_VERSION,
        write_open_safe=m0001_manifest.WRITE_OPEN_SAFE,
        apply=m0001_manifest.migrate,
    ),
)
_MIGRATION_BY_VERSION = {migration.from_version: migration for migration in _MIGRATIONS}


@dataclass(frozen=True)
class MigrationResult:
    """The revisions applied to one archive and its resulting commit token."""

    initial_version: int
    current_version: int
    applied: tuple[str, ...]
    token: CommitToken | None


def archive_format_version(root: str) -> int:
    """Read the format marker without opening the archive for reads or writes."""
    layout = FineStoreLayout(root)
    try:
        data = StoragePath(layout.archive_path).read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"FineStore archive at {root!r} has no format marker") from exc
    return ArchiveMetadata.model_validate_json(data).format_version


def _migration_plan(root: str, initial_version: int) -> tuple[_MigrationRevision, ...]:
    if initial_version > FORMAT_VERSION:
        raise FormatVersionError(root, found=initial_version)

    current_version = initial_version
    revisions = []
    while current_version < FORMAT_VERSION:
        revision = _MIGRATION_BY_VERSION.get(current_version)
        if revision is None:
            raise ValueError(
                f"FineStore has no migration from format v{current_version} to v{FORMAT_VERSION} for {root!r}"
            )
        revisions.append(revision)
        current_version = revision.to_version
    return tuple(revisions)


def _apply_migrations(root: str, initial_version: int, revisions: tuple[_MigrationRevision, ...]) -> MigrationResult:
    current_version = initial_version
    applied = []
    for revision in revisions:
        logger.info(
            "Applying FineStore migration %s to %s: v%d -> v%d",
            revision.migration_id,
            root,
            revision.from_version,
            revision.to_version,
        )
        revision.apply(root)
        observed_version = archive_format_version(root)
        if observed_version != revision.to_version:
            raise RuntimeError(
                f"FineStore migration {revision.migration_id} left {root!r} at format v{observed_version}; "
                f"expected v{revision.to_version}"
            )
        applied.append(revision.migration_id)
        current_version = observed_version

    token = read_snapshot(FineStoreLayout(root)).token
    return MigrationResult(
        initial_version=initial_version,
        current_version=current_version,
        applied=tuple(applied),
        token=token,
    )


def migrate(root: str) -> MigrationResult:
    """Upgrade ``root`` through every registered revision to this build's format.

    Callers must satisfy each revision's operational preconditions. Every revision is safe
    to retry.
    """
    initial_version = archive_format_version(root)
    return _apply_migrations(root, initial_version, _migration_plan(root, initial_version))


def migrate_for_write(root: str) -> MigrationResult | None:
    """Apply revisions that are safe during writable open.

    ``None`` means no archive migration is needed. If any required revision needs explicit
    operator coordination, the archive remains unopened and a format error directs the caller
    to :func:`migrate`.
    """
    try:
        initial_version = archive_format_version(root)
    except FileNotFoundError:
        return None
    revisions = _migration_plan(root, initial_version)
    if not revisions:
        return None
    if any(not revision.write_open_safe for revision in revisions):
        raise FormatVersionError(root, found=initial_version)
    return _apply_migrations(root, initial_version, revisions)
