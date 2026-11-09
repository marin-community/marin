# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a snapshot written by ``sample.py`` into the SFT Mix Viewer and select it."""

import io
import json
import uuid
from collections.abc import Iterable
from typing import Any

import click
import pyarrow.parquet as pq
from marina.client import DEFAULT_MARINA_URL, AppletQuery, applet_query
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

MAX_BATCH_BYTES = 1_000_000
SOURCES_FILE = "sources.json"
SAMPLES_FILE = "samples.parquet"
TABLE_COLUMNS = {
    "mix_sources": "name TEXT, kind TEXT, family TEXT, share DOUBLE PRECISION, documents BIGINT, cache_path TEXT",
    "mix_samples": "source TEXT, doc_index BIGINT, tokens BIGINT, chars BIGINT, text TEXT",
}


def batches(rows: Iterable[dict[str, Any]]) -> Iterable[str]:
    """Group rows into JSON arrays whose encoded SQL request stays under the batch budget."""
    batch: list[str] = []
    size = 2
    for row in rows:
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        # The SQL API carries the row array as a JSON string, adding another escape layer.
        length = len(json.dumps(encoded).encode()) + 1
        if length + 2 > MAX_BATCH_BYTES:
            raise ValueError("One row exceeds the SQL batch budget")
        if size + length > MAX_BATCH_BYTES:
            yield "[" + ",".join(batch) + "]"
            batch, size = [], 2
        batch.append(encoded)
        size += length
    if batch:
        yield "[" + ",".join(batch) + "]"


def insert_rows(query: AppletQuery, table: str, snapshot: str, rows: Iterable[dict[str, Any]]) -> int:
    columns = TABLE_COLUMNS[table]
    names = ", ".join(column.split()[0] for column in columns.split(", "))
    count = 0
    for body in batches(rows):
        result = query(
            f"INSERT INTO {table} (snapshot, {names}) SELECT :snapshot, {names} "
            f"FROM jsonb_to_recordset(CAST(:rows AS JSONB)) AS row({columns})",
            {"snapshot": snapshot, "rows": body},
        )
        count += result["row_count"]
    return count


def load_snapshot(sources: list[dict[str, Any]], samples: list[dict[str, Any]], query: AppletQuery) -> dict[str, Any]:
    """Stage the rows under a new snapshot id, select it, then drop the snapshots it replaces."""
    sampled = {row["source"] for row in samples}
    missing = [row["name"] for row in sources if row["name"] not in sampled]
    if missing:
        raise ValueError(f"Sources without samples: {missing[:5]}")
    snapshot = uuid.uuid4().hex
    counts = {
        "mix_sources": insert_rows(query, "mix_sources", snapshot, sources),
        "mix_samples": insert_rows(query, "mix_samples", snapshot, samples),
    }
    if counts != {"mix_sources": len(sources), "mix_samples": len(samples)}:
        raise RuntimeError(f"Staged row counts differ from the snapshot: {counts}")
    query(
        "INSERT INTO mix_current (name, snapshot) VALUES ('latest', :snapshot) "
        "ON CONFLICT (name) DO UPDATE SET snapshot = EXCLUDED.snapshot",
        {"snapshot": snapshot},
    )
    for table in TABLE_COLUMNS:
        query(f"DELETE FROM {table} WHERE snapshot NOT IN (SELECT snapshot FROM mix_current)", {})
    return {"snapshot": snapshot, **counts}


@click.command(help=__doc__)
@click.option("--input", "input_root", required=True, help="Directory or object-storage prefix sample.py wrote.")
@click.option("--app-id", "applet_id", required=True)
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
def main(input_root: str, applet_id: str, service_url: str) -> None:
    configure_coreweave_s3()
    root = StoragePath.parse(input_root)
    sources = json.loads((root / SOURCES_FILE).read_text())["sources"]
    samples = pq.read_table(io.BytesIO((root / SAMPLES_FILE).read_bytes())).to_pylist()
    click.echo(json.dumps(load_snapshot(sources, samples, applet_query(service_url, applet_id))))


if __name__ == "__main__":
    main()
