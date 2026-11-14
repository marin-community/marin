# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit metadata operations for archive migrations."""

from __future__ import annotations

import argparse
import hashlib
import json
from uuid import uuid4

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from finestore.commit import ArchiveSnapshot, CommitCoordinator, CommitDelta, initialize_archive, write_schema
from finestore.layout import CommitToken, FineStoreLayout, HeadMetadata, Manifest, ManifestTable, TableMetadata


def set_table_metadata(root: str, table: str, metadata: TableMetadata) -> CommitToken:
    """Replace one table's logical metadata without changing its visible shards."""
    layout = FineStoreLayout(root)
    coordinator = CommitCoordinator(layout)
    metadata_path = write_schema(layout, metadata)
    return coordinator.commit(CommitDelta(metadata_updates={table: metadata_path}))


def drop_table(root: str, table: str) -> CommitToken:
    """Remove a table from the next manifest while retaining its immutable objects."""
    layout = FineStoreLayout(root)
    return CommitCoordinator(layout).commit(CommitDelta(removals=frozenset({table})))


def fork_relocated_archive(
    original_root: str, copied_root: str, output_root: str, *, reference_roots: tuple[str, ...] = ()
) -> dict:
    """Fork copied immutable shards into a writable archive with regional references."""
    output_path = StoragePath(output_root)
    copied_path = StoragePath(copied_root)
    reference_paths = tuple(StoragePath(root) for root in (original_root, copied_root, *reference_roots))
    output = FineStoreLayout(str(output_path))
    if StoragePath(output.head_path).exists():
        raise ValueError("Preserve existing archive forks; use a new output root")

    def relocated(path: str) -> str:
        location = StoragePath(path)
        for root in reference_paths:
            try:
                relative = location.relative_to(root)
            except ValueError:
                continue
            return str(copied_path / relative)
        raise ValueError(f"Archive reference is outside the copied roots: {path}")

    copied = FineStoreLayout(str(copied_path))
    head = HeadMetadata.model_validate_json(StoragePath(copied.head_path).read_bytes())
    manifest_bytes = StoragePath(relocated(head.manifest_path)).read_bytes()
    manifest = Manifest.model_validate_json(manifest_bytes)
    tables = {}
    for name, table in manifest.tables.items():
        metadata = TableMetadata.model_validate_json(StoragePath(relocated(table.metadata_path)).read_bytes())
        tables[name] = ManifestTable(
            metadata_path=write_schema(output, metadata),
            shards=tuple(shard.model_copy(update={"path": relocated(shard.path)}) for shard in table.shards),
        )
    initialize_archive(output)
    # Preserve row-commit bounds and sequence numbers when the immutable shards move.
    seed = manifest.model_copy(
        update={"commit_id": uuid4().hex, "parent_commit_id": None, "tables": tables, "maintainer_epoch": 0}
    )
    StoragePath(output.manifest_path(seed.commit_id)).write_text(seed.model_dump_json(indent=2))
    token = CommitCoordinator(output).commit(CommitDelta(), base=ArchiveSnapshot(manifest=seed, token=None))
    return {
        "original_root": original_root,
        "copied_root": copied_root,
        "reference_roots": list(reference_roots),
        "output_root": output_root,
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "commit_id": token.commit_id,
        "sequence": token.sequence,
        "tables": len(tables),
        "shards": sum(len(table.shards) for table in tables.values()),
        "rows": sum(shard.rows for table in tables.values() for shard in table.shards),
        "all_metadata_paths_regional": all(
            StoragePath(table.metadata_path).relative_to(output_path) for table in tables.values()
        ),
        "all_shard_paths_regional": all(
            StoragePath(shard.path).relative_to(copied_path) for table in tables.values() for shard in table.shards
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    fork = subcommands.add_parser("fork-relocated")
    fork.add_argument("--original-root", required=True)
    fork.add_argument("--copied-root", required=True)
    fork.add_argument("--output-root", required=True)
    fork.add_argument("--reference-root", action="append", default=[])
    args = parser.parse_args()
    configure_coreweave_s3()
    print(
        json.dumps(
            fork_relocated_archive(
                args.original_root, args.copied_root, args.output_root, reference_roots=tuple(args.reference_root)
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
