# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish bounded metadata samples only when every selected row has upstream coordinates."""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.buckets import filesystem_for

from experiments.datakit.scripts.metadata_upstream_locators import SourceManifest, validate_locator_mapping
from experiments.datakit.store.metadata import METADATA_SCHEMA, ShardManifest, export_metadata_shard

SOURCE_ROW_TYPE = pa.struct(
    [
        pa.field("source_ref", pa.string(), nullable=False),
        pa.field("row_offset", pa.int64(), nullable=False),
        pa.field("file", pa.string(), nullable=False),
        pa.field("file_row_offset", pa.int64(), nullable=False),
    ]
)
RELEASE_SCHEMA = pa.schema(
    [field for field in METADATA_SCHEMA if field.name not in {"normalized_shard", "normalized_row", "store_task_id"}]
    + [pa.field("source_rows", pa.list_(pa.field("item", SOURCE_ROW_TYPE, nullable=False)), nullable=False)]
)


def export_release_metadata_shard(
    manifest: ShardManifest,
    upstream_rows: str,
    source_manifests: dict[str, SourceManifest],
    output_dir: str,
    *,
    cluster_view: int,
    max_rows: int,
) -> dict:
    """Join a normalized-shard provenance sidecar and reject incomplete row mappings.

    The sidecar contains ``id``, ``normalized_row`` and ``source_rows`` for the
    normalized shard in ``manifest``. Locator lists preserve transform input
    order. This validates recorded coordinates against pinned file manifests;
    it does not prove that a transform supplied every necessary input row.
    """
    destination_fs, destination = filesystem_for(output_dir)
    if destination_fs.exists(destination):
        raise FileExistsError(f"Use a fresh release output directory: {output_dir}")
    for key, source_manifest in source_manifests.items():
        if key != source_manifest.sha256:
            raise ValueError("Source catalog key does not match its pinned file manifest")
    with tempfile.TemporaryDirectory(prefix="metadata-release-") as temporary:
        result = export_metadata_shard(manifest, temporary, cluster_view=cluster_view, max_rows=max_rows)
        buckets = []
        selected = {}
        for output in result["outputs"]:
            rows = pq.ParquetFile(output["path"]).read().to_pylist()
            buckets.append((output, rows))
            for row in rows:
                selected[(row["id"], row["normalized_row"])] = row
        matched = set()
        fs, path = filesystem_for(upstream_rows)
        with fs.open(path, "rb") as stream:
            parquet = pq.ParquetFile(stream)
            for batch in parquet.iter_batches(columns=["id", "normalized_row", "source_rows"]):
                for mapping in batch.to_pylist():
                    key = (mapping["id"], mapping["normalized_row"])
                    if key not in selected:
                        continue
                    if key in matched:
                        raise ValueError(f"Ambiguous upstream mapping for selected row {key}")
                    locators = mapping["source_rows"]
                    if not locators:
                        raise ValueError(f"No upstream rows for selected row {key}")
                    for locator in locators:
                        if not locator or locator.get("source_ref") not in source_manifests:
                            raise ValueError(f"Missing or unknown upstream source reference for {key}")
                        validate_locator_mapping(source_manifests[locator["source_ref"]], locator)
                    selected[key]["source_rows"] = locators
                    matched.add(key)
        if missing := selected.keys() - matched:
            raise ValueError(f"Missing upstream coordinates for {len(missing)} selected occurrences")
        # No public files are written until the whole selected sample passes.
        outputs = []
        for original, rows in buckets:
            relative = str(Path(original["path"]).relative_to(temporary))
            target = destination.rstrip("/") + "/" + relative
            destination_fs.makedirs(str(Path(target).parent), exist_ok=True)
            with destination_fs.open(target, "wb") as stream:
                pq.write_table(pa.Table.from_pylist(rows, schema=RELEASE_SCHEMA), stream, compression="zstd")
            outputs.append({**original, "path": output_dir.rstrip("/") + "/" + relative})
        report = {
            "status": "passed",
            "selected_occurrences": len(selected),
            "outputs": outputs,
            "source_manifests": sorted(source_manifests),
            "source_catalog": output_dir.rstrip("/") + "/sources.json",
            "validation": "coordinates checked; independent upstream transform replay required",
        }
        destination_fs.makedirs(destination, exist_ok=True)
        with destination_fs.open(destination.rstrip("/") + "/sources.json", "w") as stream:
            json.dump({key: asdict(value) for key, value in source_manifests.items()}, stream, indent=2)
        with destination_fs.open(destination.rstrip("/") + "/_complete.json", "w") as stream:
            json.dump(report, stream, indent=2)
        return report
