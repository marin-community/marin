# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read bounded metadata from a pinned TaskTrove Clean release."""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

METADATA_COLUMNS = ("source", "path", "family", "template_id", "converter", "mode", "tags")
SPLITS = ("tasks", "sft")
BATCH_SIZE = 65_536


def inventory(release_uri: str) -> dict:
    """Count source metadata without reading any archive bytes."""
    configure_coreweave_s3()
    release = StoragePath(release_uri)
    manifest_path = release / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    revision = release.segments[-1]
    counts: dict[str, Counter[str]] = {
        dimension: Counter() for dimension in ("split", "route", "source", "family", "template_id", "converter", "mode")
    }
    files = []
    total_rows = 0
    for split in SPLITS:
        paths = sorted((release / split / "part-*.parquet").glob(), key=str)
        for path in paths:
            opened = path.open("rb")
            object_info = opened.fs.info(opened.path)
            with opened as stream:
                parquet = pq.ParquetFile(stream)
                available_columns = set(parquet.schema_arrow.names)
                columns = [name for name in METADATA_COLUMNS if name in available_columns]
                if "route" in available_columns:
                    columns.append("route")
                file_rows = 0
                for batch in parquet.iter_batches(columns=columns, batch_size=BATCH_SIZE):
                    for row in batch.to_pylist():
                        file_rows += 1
                        total_rows += 1
                        counts["split"][split] += 1
                        counts["route"][str(row.get("route") or split)] += 1
                        for dimension in ("source", "family", "template_id", "converter", "mode"):
                            value = row.get(dimension)
                            counts[dimension][str(value) if value is not None else "<missing>"] += 1
                files.append(
                    {
                        "path": str(path),
                        "size_bytes": object_info.get("size"),
                        "etag": object_info.get("ETag"),
                        "version_id": object_info.get("VersionId"),
                        "last_modified": (
                            str(object_info.get("LastModified")) if object_info.get("LastModified") is not None else None
                        ),
                        "rows": file_rows,
                        "row_groups": parquet.metadata.num_row_groups,
                        "schema": str(parquet.schema_arrow),
                    }
                )
    return {
        "release_uri": str(release),
        "release_revision": revision,
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_manifest": manifest,
        "input_rows": total_rows,
        "files": files,
        "counts": {dimension: dict(sorted(counter.items())) for dimension, counter in counts.items()},
        "archive_bytes_read": 0,
    }


def main() -> None:
    release_uri = os.environ.get("TASKTROVE_RELEASE_URI", "s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.18.3")
    result = inventory(release_uri)
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "tasktrove-metadata-inventory.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "release_uri": result["release_uri"],
                "input_rows": result["input_rows"],
                "source_manifest_sha256": result["source_manifest_sha256"],
                "inventory_path": str(output / "tasktrove-metadata-inventory.json"),
                "archive_bytes_read": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
