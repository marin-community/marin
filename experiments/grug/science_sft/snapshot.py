# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze committed science chat Parquet files for a reproducible SFT run."""

import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pyarrow.parquet as pq
from marin.datakit.chat_normalize import CHAT_SCHEMA
from rigging.filesystem.buckets import filesystem_for


logger = logging.getLogger(__name__)


def _etag(info: dict) -> str | None:
    return info.get("ETag") or info.get("etag")


def _source_slug(filename: str, expected: set[str]) -> str:
    matches = [slug for slug in expected if filename.startswith(f"{slug}__")]
    if len(matches) != 1:
        raise ValueError(f"Unrecognized or ambiguous source for {filename}")
    return matches[0]


def snapshot(source_url: str, destination_url: str, source_names: list[str], workers: int) -> dict:
    """Copy one stable listing within a bucket, then verify every copied Parquet footer."""
    source_fs, source = filesystem_for(source_url)
    destination_fs, destination = filesystem_for(destination_url)
    if source_fs is not destination_fs and source_fs.protocol != destination_fs.protocol:
        raise ValueError("Snapshot requires same-store server-side copies")
    if source.split("/", 1)[0] != destination.split("/", 1)[0] or source == destination:
        raise ValueError("Snapshot source and destination must be distinct paths in one bucket")
    if source.startswith(destination + "/") or destination.startswith(source + "/"):
        raise ValueError("Snapshot paths must not nest")
    if destination_fs.exists(f"{destination}/_READY"):
        raise FileExistsError(f"Snapshot is already sealed: {destination_url}")

    listings = source_fs.ls(source, detail=True)
    files = sorted((info for info in listings if info["name"].endswith(".parquet")), key=lambda item: item["name"])
    if not files:
        raise FileNotFoundError(f"No committed Parquet files under {source_url}")
    expected = {name.replace("/", "__") for name in source_names}
    found = {_source_slug(info["name"].rsplit("/", 1)[-1], expected) for info in files}
    if found != expected:
        raise ValueError(f"Missing science sources: {sorted(expected - found)}")
    if any(info["size"] <= 0 for info in files):
        raise ValueError("Committed Parquet listing contains an empty file")
    logger.info("Freezing %d committed files from %d sources", len(files), len(found))
    target_dir = f"{destination}/outputs/main"
    destination_fs.makedirs(target_dir, exist_ok=True)

    def copy_one(info: dict) -> None:
        name = info["name"].rsplit("/", 1)[-1]
        target = f"{destination}/outputs/main/{name}"
        if destination_fs.exists(target):
            if destination_fs.info(target)["size"] != info["size"]:
                raise ValueError(f"Existing snapshot file differs in size: {name}")
            return
        # s3fs.copy uses the object store's same-bucket CopyObject path here.
        source_fs.copy(info["name"], target)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(copy_one, files))

    destination_fs.invalidate_cache(target_dir)
    copies = {
        info["name"].rsplit("/", 1)[-1]: info
        for info in destination_fs.ls(target_dir, detail=True)
        if info["name"].endswith(".parquet")
    }
    names = {info["name"].rsplit("/", 1)[-1] for info in files}
    if copies.keys() != names:
        raise ValueError(f"Snapshot file set differs: missing={len(names - copies.keys())}, extra={len(copies.keys() - names)}")
    if any(copies[info["name"].rsplit("/", 1)[-1]]["size"] != info["size"] for info in files):
        raise ValueError("Snapshot file size differs from the frozen listing")

    def inspect_one(info: dict) -> tuple[str, int]:
        name = info["name"].rsplit("/", 1)[-1]
        with destination_fs.open(f"{target_dir}/{name}", "rb") as stream:
            metadata = pq.ParquetFile(stream).metadata
        if metadata.num_rows <= 0 or not metadata.schema.to_arrow_schema().equals(CHAT_SCHEMA, check_metadata=False):
            raise ValueError(f"Bad SFT-ready chat schema or empty footer: {name}")
        return _source_slug(name, expected), metadata.num_rows

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(inspect_one, files))
    counts = Counter()
    for slug, count in rows:
        counts[slug] += count
    manifest = {
        "source": source_url,
        "destination": destination_url,
        "captured_utc": datetime.now(UTC).isoformat(),
        "files": [
            {
                "name": info["name"].rsplit("/", 1)[-1],
                "bytes": info["size"],
                "source_etag": _etag(info),
                "snapshot_etag": _etag(copies[info["name"].rsplit("/", 1)[-1]]),
            }
            for info in files
        ],
        "file_count": len(files),
        "bytes": sum(info["size"] for info in files),
        "conversations": sum(counts.values()),
        "rows_by_source": dict(sorted(counts.items())),
        "schema": str(CHAT_SCHEMA),
    }
    destination_fs.pipe(f"{destination}/snapshot-manifest.json", json.dumps(manifest, indent=2).encode())
    destination_fs.pipe(f"{destination}/_READY", b"verified\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--sources-json", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    with open(args.sources_json) as stream:
        source_names = [source["name"] for source in json.load(stream)["sources"]]
    logging.basicConfig(level=logging.INFO)
    result = snapshot(args.source, args.destination, source_names, args.workers)
    logger.info("Snapshot sealed: %d conversations in %d files", result["conversations"], result["file_count"])


if __name__ == "__main__":
    main()
