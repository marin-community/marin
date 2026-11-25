# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the paired RL pool in region, excluding the frozen 128 SWESmith tasks."""

import argparse
import hashlib
import io
import json
import logging
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from finestore.rl.strata import read_selection_manifest
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy

from experiments.evaluation.benchmarks.harbor import image_lock
from experiments.post_training.mochi_loop_guards.panel import SWESMITH_SOURCE_URI, _row_hash
from experiments.post_training.mochi_loop_guards.presets import VERSION
from experiments.post_training.task_curation.swesmith_training import (
    SWESMITH_DATASET,
    SWESMITH_MACHINE,
    _derive_selection,
)

PARENT = "s3://marin-us-east-02a/marin/rl-data/task-selection/snowball-100k-agentic/2026.11.19.2"
NATIVE = "s3://marin-us-east-08a/tmp/ttl=3d/mochi-loop-guards/native-parent/" + VERSION
OUTPUT = (
    "s3://marin-us-east-08a/marin/rl-data/task-selection/users/benfeuer/mochi-loop-guards/"
    "snowball-100k-agentic-ho-excluded/" + VERSION
)
logger = logging.getLogger(__name__)


def publish_pool(manifest: dict, output: str) -> dict:
    """Lower the declared parent with its existing converter and remove held-out IDs."""
    entries = manifest["cohorts"]["swesmith-ho"]["rows"]
    excluded = {entry["task_id"] for entry in entries}
    raw = pq.read_table(io.BytesIO(StoragePath(SWESMITH_SOURCE_URI).read_bytes())).to_pylist()
    for entry in entries:
        if _row_hash(raw[entry["index"]]) != entry["row_hash"]:
            raise ValueError("Held-out row hash differs from the frozen panel")
    del raw
    if not (StoragePath(NATIVE) / "summary.json").exists():
        _derive_selection(
            PARENT,
            NATIVE,
            "mochi-loop-guards-native-parent",
            VERSION,
            SWESMITH_MACHINE,
            image_lock(SWESMITH_DATASET),
        )
    source, destination = StoragePath(NATIVE), StoragePath(output)
    summary = json.loads((source / "summary.json").read_text())
    destination.mkdirs()
    removed = set()
    families: Counter[str] = Counter()
    with TemporaryDirectory(prefix="mochi-guards-pool-") as directory:
        train = Path(directory) / "train.parquet"
        with (source / "train.parquet").open("rb") as stream:
            parquet = pq.ParquetFile(stream)
            with pq.ParquetWriter(train, parquet.schema_arrow) as writer:
                for batch in parquet.iter_batches(batch_size=256):
                    rows = batch.to_pylist()
                    kept = []
                    for row in rows:
                        task = json.loads(row["task_spec"])
                        if task["id"] in excluded:
                            removed.add(task["id"])
                            continue
                        kept.append(row)
                        families[row["family"]] += 1
                    writer.write_table(pa.Table.from_pylist(kept, schema=parquet.schema_arrow))
        if removed != excluded:
            raise ValueError(f"Expected all {len(excluded)} held-out tasks in the parent; removed {len(removed)}")
        copy(str(train), str(destination / "train.parquet"))
        with train.open("rb") as stream:
            train_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    with (source / "task_manifest.parquet").open("rb") as stream:
        table = pq.read_table(stream)
    table = table.filter(pc.invert(pc.is_in(table["task_id"], value_set=pa.array(sorted(excluded)))))
    with (destination / "task_manifest.parquet").open("wb") as stream:
        pq.write_table(table, stream)
    for name in ("validation.parquet", "full_validation.parquet"):
        copy(str(source / name), str(destination / name))
    definition = summary["selection_strata"]
    definition.update(
        parent={"root": PARENT, "name": "snowball-100k-agentic", "version": "2026.11.19.2"},
        name="snowball-100k-agentic-ho-excluded",
        version=VERSION,
    )
    summary["training_rows"] -= len(removed)
    summary["training_family_counts"] = dict(families)
    summary["grader_kind_counts"]["train"]["image"] -= len(removed)
    counts = summary["source_counts"][SWESMITH_SOURCE_URI]
    counts["selected"] -= len(removed)
    counts["heldout_overlap"] += len(removed)
    receipt = {
        "parent": PARENT,
        "frozen_manifest_sha256": manifest["manifest_sha256"],
        "excluded_rows": entries,
        "removed_task_ids": sorted(removed),
        "training_rows": summary["training_rows"],
        "training_family_counts": dict(families),
        "train_sha256": train_sha256,
    }
    (destination / "exclusions.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    read_selection_manifest(output)
    logger.info("Published pool: %d rows; excluded %d held-out tasks", summary["training_rows"], len(removed))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="cohorts/manifest.json")
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    result = publish_pool(json.loads(StoragePath(args.manifest).read_text()), args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "excluded_rows"}, indent=2))
