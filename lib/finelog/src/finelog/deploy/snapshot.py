# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy a bounded slice of a deployed finelog's local store.

What a shadow boot rehearses lives in the local store directory, not in the
archive: ``Store::new`` opens the catalog, adopts the local Parquet, and
rehydrates every namespace's registered schema, while the remote reconcile is
deliberately backgrounded and never blocks the bind. So a snapshot is the
catalog plus the newest few segments per namespace and their ``.fidx`` sidecars
— never a pull from ``gs://``/``s3://``, which would be a cross-region read of
someone else's bucket.

The catalog records each segment's **absolute** path and boot adoption matches
rows to files by exact path, which is why [`STORE_DIR`] is fixed: a snapshot is
only a faithful rehearsal when it is mounted back where it was taken from. A
segment left behind is not a problem — a `LOCAL` row whose file is gone is
dropped at boot and a `BOTH` row collapses to `REMOTE`, so the store prunes
itself down to what was actually copied.
"""

import shlex
import sqlite3
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from finelog.deploy.bootstrap import CACHE_DIR

# Where both deploy backends mount the store: the GCE bootstrap bind-mounts
# CACHE_DIR and the k8s Deployment mounts its PVC at the same path.
STORE_DIR = CACHE_DIR

CATALOG_FILENAME = "_finelog_catalog.sqlite"

# The catalog sqlite plus any rollback journal beside it, and the adoption
# sentinel whose presence keeps a boot on the ordinary fast path instead of
# re-running the rebuild-from-disk scan.
CATALOG_PATTERNS = (f"{CATALOG_FILENAME}*", ".finelog-rust-catalog")

# Bounded by default. Enough recent segments per namespace to exercise catalog
# adoption, index-bundle reads, and projection substitution, without copying a
# production store to a dev box.
DEFAULT_SEGMENTS_PER_NAMESPACE = 8
DEFAULT_MAX_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class CatalogSegment:
    """One row of the catalog's ``segments`` table, as far as planning cares."""

    namespace: str
    path: str
    max_seq: int
    byte_size: int


@dataclass(frozen=True)
class SnapshotPlan:
    """Which store files to copy, and what the bound left behind."""

    # Store-relative paths and glob patterns, in copy order.
    patterns: tuple[str, ...]
    selected_bytes: int
    selected_per_namespace: dict[str, int]
    skipped_per_namespace: dict[str, int]

    def describe(self) -> str:
        lines = [f"{self.selected_bytes / 1024**2:.0f} MiB of segments selected"]
        for namespace in sorted(self.selected_per_namespace | self.skipped_per_namespace):
            selected = self.selected_per_namespace.get(namespace, 0)
            skipped = self.skipped_per_namespace.get(namespace, 0)
            lines.append(f"  {namespace}: {selected} segments, {skipped} left behind")
        return "\n".join(lines)


def read_catalog_segments(catalog: Path) -> list[CatalogSegment]:
    """Read the local segment rows from a copied catalog.

    ``REMOTE`` rows are skipped: their local Parquet has already been unlinked,
    so there is nothing on the host to copy.
    """
    connection = sqlite3.connect(f"file:{catalog}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT namespace, path, max_seq, byte_size FROM segments WHERE location != 'REMOTE'"
        ).fetchall()
    finally:
        connection.close()
    return [CatalogSegment(namespace=row[0], path=row[1], max_seq=row[2], byte_size=row[3]) for row in rows]


def store_relative(path: str) -> str:
    """Map an absolute catalog path to its store-relative form."""
    try:
        return str(PurePosixPath(path).relative_to(STORE_DIR))
    except ValueError as exc:
        raise ValueError(
            f"catalog path {path!r} is not under {STORE_DIR}; this deployment mounts its store "
            "somewhere else, and a snapshot taken from it would not adopt when mounted back"
        ) from exc


def plan_snapshot(
    segments: Sequence[CatalogSegment],
    *,
    segments_per_namespace: int = DEFAULT_SEGMENTS_PER_NAMESPACE,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> SnapshotPlan:
    """Choose the newest segments to copy, newest-first and round-robin.

    Round-robin across namespaces rather than newest-first globally: a byte
    budget spent entirely on the busiest namespace would leave the others with
    a catalog entry and no data, which rehearses less than it appears to.
    """
    by_namespace: dict[str, list[CatalogSegment]] = {}
    for segment in segments:
        by_namespace.setdefault(segment.namespace, []).append(segment)
    for candidates in by_namespace.values():
        candidates.sort(key=lambda segment: segment.max_seq, reverse=True)

    selected: list[CatalogSegment] = []
    selected_bytes = 0
    taken = {namespace: 0 for namespace in by_namespace}
    for rank in range(segments_per_namespace):
        for namespace, candidates in sorted(by_namespace.items()):
            if rank >= len(candidates):
                continue
            candidate = candidates[rank]
            if selected and selected_bytes + candidate.byte_size > max_bytes:
                continue
            selected.append(candidate)
            selected_bytes += candidate.byte_size
            taken[namespace] += 1

    patterns = list(CATALOG_PATTERNS)
    for segment in selected:
        relative = store_relative(segment.path)
        patterns.append(relative)
        # Sidecars share the segment's filename as a prefix: `<segment>.fidx`
        # for the index bundle and `<segment>.fidx.<name>.parquet` for each
        # covering projection.
        patterns.append(f"{relative}.fidx*")
    return SnapshotPlan(
        patterns=tuple(patterns),
        selected_bytes=selected_bytes,
        selected_per_namespace={namespace: count for namespace, count in taken.items() if count},
        skipped_per_namespace={
            namespace: len(candidates) - taken[namespace]
            for namespace, candidates in by_namespace.items()
            if len(candidates) - taken[namespace] > 0
        },
    )


def tar_command(store_dir: str, patterns: Sequence[str]) -> str:
    """The shell command that streams ``patterns`` under ``store_dir`` as a tar archive.

    Paths are relative to ``store_dir`` so the archive unpacks under whichever
    directory a snapshot lands in. ``find`` does the matching, not the shell:
    each pattern reaches it quoted, so a namespace with a shell metacharacter in
    its name cannot expand into something else, and a pattern that matches
    nothing — a segment with no index bundle, a store with no adoption sentinel —
    contributes nothing instead of failing the archive.
    """
    if not patterns:
        raise ValueError("a snapshot needs at least one path to copy")
    clauses = " -o ".join(f"-path {shlex.quote(f'./{pattern}')}" for pattern in patterns)
    return f"cd {shlex.quote(store_dir)} && find . \\( {clauses} \\) -print0 | tar -cf - --null -T -"


def extract_tar(archive: Path, destination: Path) -> None:
    """Unpack a store archive into ``destination``.

    The ``data`` filter refuses absolute paths, escapes, and special files: this
    archive was built on another host, so it is not trusted to name where it
    lands.
    """
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")


def namespaces_in_catalog(catalog: Path) -> set[str]:
    """Every namespace the catalog has registered."""
    connection = sqlite3.connect(f"file:{catalog}?mode=ro", uri=True)
    try:
        return {row[0] for row in connection.execute("SELECT namespace FROM namespaces")}
    finally:
        connection.close()
