# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Census frozen TaskSpec populations and prepare combined mechanical drop manifests."""

import argparse
import json
import logging
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from enum import StrEnum
from itertools import batched
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import LexicalUse
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256, task_identity
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.deduplication import DUPLICATE_WITHIN_SOURCE

from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.source_identity import (
    canonical_source_id,
    canonical_source_map,
    original_source_id,
)

logger = logging.getLogger(__name__)
LEXICAL_CONTAMINATION = "LEXICAL_CONTAMINATION"
DROP_SCHEMA = pa.schema([("task_id", pa.string()), ("kept_id", pa.string()), ("reasons", pa.list_(pa.string()))])


class DropManifestKind(StrEnum):
    DUPLICATES = "duplicates"
    COMBINED = "combined"


def parquet_rows(path: str, columns: list[str]) -> Iterator[dict]:
    for file in sorted(StoragePath(path).glob(), key=str):
        with file.open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=1, columns=columns):
                yield from batch.to_pylist()


def write_drops(database: sqlite3.Connection, path: str, *, kind: DropManifestKind, lexical_use: LexicalUse) -> None:
    exclude_lexical = kind is DropManifestKind.COMBINED and lexical_use is LexicalUse.TRAIN
    condition = "t.id != g.kept OR g.flagged" if exclude_lexical else "t.id != g.kept"
    rows = database.execute(
        f"SELECT t.id, g.kept, g.flagged FROM tasks t JOIN groups g ON t.contract = g.contract "
        f"WHERE {condition} ORDER BY t.id"
    )
    with StoragePath(path).open("wb", auto_mkdir=True) as stream:
        with pq.ParquetWriter(stream, DROP_SCHEMA, compression="zstd") as writer:
            for batch in batched(rows, 1000):
                writer.write_table(
                    pa.Table.from_pylist(
                        [
                            {
                                "task_id": identifier,
                                "kept_id": kept,
                                "reasons": (
                                    ([DUPLICATE_WITHIN_SOURCE] if identifier != kept else [])
                                    + ([LEXICAL_CONTAMINATION] if flagged and exclude_lexical else [])
                                ),
                            }
                            for identifier, kept, flagged in batch
                        ],
                        schema=DROP_SCHEMA,
                    )
                )


def census_source(
    population: SourceTextPopulation, measurement: dict, output_root: str, *, lexical_use: LexicalUse
) -> dict:
    """Count exact contracts on disk and retain every dropped ID with its canonical ID."""
    key = canonical_sha256(original_source_id(population.source_id))[:20]
    output = StoragePath(output_root) / key
    if output.exists():
        raise FileExistsError(f"Existing census output requires a new artifact version: {output}")
    with TemporaryDirectory(prefix="within-source-") as scratch:
        with closing(sqlite3.connect(str(Path(scratch) / "contracts.sqlite"))) as database:
            database.execute("PRAGMA journal_mode = OFF")
            database.execute(
                "CREATE TABLE tasks (id TEXT PRIMARY KEY, contract TEXT NOT NULL, flagged INTEGER NOT NULL)"
            )
            flagged = set()
            if measurement["contaminated_tasks"]:
                drop = StoragePath(measurement["evidence_url"]).parent / "drop_manifest.parquet"
                with drop.open("rb") as stream:
                    for batch in pq.ParquetFile(stream).iter_batches(columns=["id"]):
                        flagged.update(batch.column(0).to_pylist())
                if len(flagged) != measurement["contaminated_tasks"]:
                    raise ValueError(f"Lexical manifest count mismatch: {population.source_id}")
            count = 0
            entries = []
            for row in parquet_rows(str(StoragePath(population.specs_output_dir) / "*.parquet"), ["id", "task_spec"]):
                task = TaskSpec.model_validate_json(row["task_spec"])
                if row["id"] != task.id:
                    raise ValueError(f"Task ID differs from spec: {row['id']}")
                entries.append((task.id, task_identity(task), int(task.id in flagged)))
                count += 1
                if len(entries) >= 1000:
                    database.executemany("INSERT INTO tasks VALUES (?, ?, ?)", entries)
                    entries.clear()
                if count % 100000 == 0:
                    logger.info("%s: %d tasks indexed", population.source_id, count)
            if entries:
                database.executemany("INSERT INTO tasks VALUES (?, ?, ?)", entries)
            if count != population.task_count or count != measurement["measured_tasks"]:
                raise ValueError(f"Frozen population count mismatch: {population.source_id}: {count}")
            matched = database.execute("SELECT SUM(flagged) FROM tasks").fetchone()[0] or 0
            if matched != len(flagged):
                raise ValueError(f"Lexical manifest includes unknown task IDs: {population.source_id}")
            database.execute("CREATE INDEX contracts ON tasks(contract)")
            database.execute(
                "CREATE TABLE groups AS SELECT contract, MIN(id) AS kept, COUNT(*) AS copies, "
                "MAX(flagged) AS flagged FROM tasks GROUP BY contract"
            )
            database.execute("CREATE UNIQUE INDEX group_contracts ON groups(contract)")
            unique, lexical_unique = database.execute("SELECT COUNT(*), SUM(flagged) FROM groups").fetchone()
            duplicates = count - unique
            duplicate_manifest = str(output / "duplicates.parquet")
            combined_manifest = str(output / "combined-drops.parquet")
            write_drops(database, duplicate_manifest, kind=DropManifestKind.DUPLICATES, lexical_use=lexical_use)
            write_drops(database, combined_manifest, kind=DropManifestKind.COMBINED, lexical_use=lexical_use)
            multiplicities = dict(database.execute("SELECT copies, COUNT(*) FROM groups GROUP BY copies"))
    result = {
        "source_id": population.source_id,
        "source_dataset_id": population.dataset_id,
        "source_revision": population.dataset_revision,
        "converter_revision": population.converter_revision,
        "converter_version": population.source_version,
        "input_artifact": population.path,
        "task_specs_sha256": population.task_specs_sha256,
        "raw_count": count,
        "unique_count": unique,
        "within_source_duplicates": duplicates,
        "raw_lexical_count": len(flagged),
        "unique_lexical_count": lexical_unique or 0,
        "lexical_use": lexical_use.value,
        "combined_drop_count": duplicates + ((lexical_unique or 0) if lexical_use is LexicalUse.TRAIN else 0),
        "retained_count": unique - ((lexical_unique or 0) if lexical_use is LexicalUse.TRAIN else 0),
        "multiplicities": multiplicities,
        "drop_manifest_uri": duplicate_manifest,
        "combined_drop_manifest_uri": combined_manifest,
        "evidence_url": str(output / "measurement.json"),
    }
    (output / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
    logger.info("%s: %d raw, %d unique, %d duplicates", population.source_id, count, unique, duplicates)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--populations", type=Path, required=True)
    parser.add_argument("--lexical-outcomes", type=Path, required=True)
    parser.add_argument("--source-uses", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    populations = [SourceTextPopulation.model_validate(row) for row in json.loads(args.populations.read_text())]
    measurements = canonical_source_map(json.loads(args.lexical_outcomes.read_text())["measurements"])
    source_uses = canonical_source_map(
        {key: LexicalUse(value) for key, value in json.loads(args.source_uses.read_text()).items()}
    )
    if len({p.source_id for p in populations}) != len(populations) or not {p.source_id for p in populations}.issubset(
        measurements
    ):
        raise ValueError("Frozen source populations must have unique, measured source IDs")
    populations.sort(key=lambda p: ("dapo" not in p.source_id, -p.task_count))
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn"), initializer=configure_coreweave_s3
    ) as pool:
        futures = [
            pool.submit(
                census_source,
                p,
                measurements[canonical_source_id(p.source_id)],
                args.output_root,
                lexical_use=source_uses[p.source_id],
            )
            for p in populations
        ]
        results = [future.result() for future in futures]
    results.sort(key=lambda row: row["source_id"])
    encoded = json.dumps(results, indent=2) + "\n"
    (StoragePath(args.output_root) / "all-sources.json").write_text(encoded)
    args.output.write_text(encoded)


if __name__ == "__main__":
    main()
