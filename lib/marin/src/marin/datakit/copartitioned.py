# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Co-partitioned input and output shard layout for Datakit stages."""

import dataclasses
import json
import os
from collections.abc import Mapping, Sequence

from rigging.filesystem import StoragePath, prefix_join

SOURCE_MANIFEST_FILENAME = ".source_manifest.json"
SOURCE_MANIFEST_VERSION = "v1"


@dataclasses.dataclass(frozen=True)
class CopartitionedSource:
    """One source directory and its stable identity."""

    source_key: str
    input_dir: str


@dataclasses.dataclass(frozen=True)
class CopartitionedShard:
    """One input shard and its matching attribute output shard."""

    file_idx: int
    input_path: str
    source_key: str
    source_tag: str
    basename: str
    output_path: str


@dataclasses.dataclass(frozen=True)
class CopartitionedSourceManifestEntry:
    """One source mapping in a co-partitioned output manifest."""

    source_tag: str
    source_key: str
    attribute_dir: str


@dataclasses.dataclass(frozen=True)
class CopartitionedSourceManifest:
    """Source mappings for one co-partitioned output."""

    version: str
    sources: list[CopartitionedSourceManifestEntry]


def build_copartitioned_shards(
    *,
    sources: Sequence[CopartitionedSource],
    output_path: str,
) -> tuple[list[CopartitionedShard], dict[str, str]]:
    """Build a deterministic, co-partitioned shard layout.

    Source sequence order controls global file priority. Source tags use
    sorted source keys, so output directories do not depend on input order.
    """
    source_keys = [source.source_key for source in sources]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("sources contains duplicate source keys")

    source_tags = {source_key: f"source_{source_rank:03d}" for source_rank, source_key in enumerate(sorted(source_keys))}
    attr_dirs = {
        source_key: prefix_join(output_path, f"outputs/{source_tag}") for source_key, source_tag in source_tags.items()
    }

    shards: list[CopartitionedShard] = []
    for source in sources:
        source_key = source.source_key
        input_dir = source.input_dir
        input_paths = sorted(StoragePath(prefix_join(input_dir, "*.parquet")).glob(), key=str)
        if not input_paths:
            raise FileNotFoundError(f"No Parquet files found under {input_dir}")

        for input_path in input_paths:
            basename = os.path.basename(str(input_path))
            shards.append(
                CopartitionedShard(
                    file_idx=len(shards),
                    input_path=str(input_path),
                    source_key=source_key,
                    source_tag=source_tags[source_key],
                    basename=basename,
                    output_path=prefix_join(attr_dirs[source_key], basename),
                )
            )

    return shards, attr_dirs


def write_copartitioned_source_manifest(*, output_path: str, attr_dirs: Mapping[str, str]) -> None:
    """Write the source-tag mapping for a completed co-partitioned output."""
    sources = [
        CopartitionedSourceManifestEntry(
            source_tag=os.path.basename(attr_dir.rstrip("/")),
            source_key=source_key,
            attribute_dir=f"outputs/{os.path.basename(attr_dir.rstrip('/'))}",
        )
        for source_key, attr_dir in attr_dirs.items()
    ]
    sources.sort(key=lambda source: source.source_tag)
    manifest = CopartitionedSourceManifest(version=SOURCE_MANIFEST_VERSION, sources=sources)
    manifest_path = StoragePath(prefix_join(output_path, SOURCE_MANIFEST_FILENAME))
    manifest_path.parent.mkdirs()
    manifest_path.write_text(json.dumps(dataclasses.asdict(manifest), indent=2) + "\n")
