# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a snapshot written by ``sample.py`` into the SFT Mix Viewer and select it."""

import json
import uuid
from dataclasses import asdict
from typing import Any

import click
from marina.client import DEFAULT_MARINA_URL, AppletQuery, applet_query
from marina.table_load import insert_load_rows
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from infra.marina.apps.sft_mix_viewer.snapshot import SampledSource, read_snapshot

TABLE_COLUMNS = {
    "mix_sources": "name TEXT, kind TEXT, family TEXT, share DOUBLE PRECISION, cache_path TEXT, documents BIGINT",
    "mix_samples": "source TEXT, doc_index BIGINT, tokens BIGINT, chars BIGINT, text TEXT",
}


def load_snapshot(sampled: list[SampledSource], query: AppletQuery) -> dict[str, Any]:
    """Stage the rows under a new load id, select it, then drop the loads it replaces."""
    empty = [entry.source.name for entry in sampled if not entry.samples]
    if empty:
        raise ValueError(f"Sources without samples: {empty[:5]}")
    rows = {
        "mix_sources": [{**asdict(entry.source), "documents": entry.documents} for entry in sampled],
        "mix_samples": [
            {"source": entry.source.name, **asdict(sample)} for entry in sampled for sample in entry.samples
        ],
    }
    load_id = uuid.uuid4().hex
    counts = {
        table: insert_load_rows(query, table, TABLE_COLUMNS[table], load_id, table_rows)
        for table, table_rows in rows.items()
    }
    if counts != {table: len(table_rows) for table, table_rows in rows.items()}:
        raise RuntimeError(f"Staged row counts differ from the snapshot: {counts}")
    query(
        "INSERT INTO mix_current (load_id) VALUES (:load_id) "
        "ON CONFLICT (singleton) DO UPDATE SET load_id = EXCLUDED.load_id",
        {"load_id": load_id},
    )
    for table in TABLE_COLUMNS:
        query(f"DELETE FROM {table} WHERE load_id NOT IN (SELECT load_id FROM mix_current)", {})
    return {"load_id": load_id, **counts}


@click.command(help=__doc__)
@click.option("--input", "input_root", required=True, help="Directory or object-storage prefix sample.py wrote.")
@click.option("--app-id", "applet_id", required=True)
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
def main(input_root: str, applet_id: str, service_url: str) -> None:
    configure_coreweave_s3()
    sampled = read_snapshot(StoragePath.parse(input_root))
    click.echo(json.dumps(load_snapshot(sampled, applet_query(service_url, applet_id))))


if __name__ == "__main__":
    main()
