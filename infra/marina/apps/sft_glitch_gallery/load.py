# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load example files into the SFT Glitch Gallery and select them.

Each ``--input`` is a Parquet file of examples: ``detect.py`` writes one, and loop examples fetched from the
token caches arrive as another with the same columns. Categories come from ``categories.py``.
"""

import io
import json
import uuid
from dataclasses import asdict
from typing import Any

import click
import pyarrow.parquet as pq
from marina.client import DEFAULT_MARINA_URL, AppletQuery, applet_query
from marina.table_load import insert_load_rows
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from infra.marina.apps.sft_glitch_gallery.categories import CATEGORIES, Category, Example

TABLE_COLUMNS = {
    "gallery_categories": "key TEXT, title TEXT, severity TEXT, description TEXT, prevalence TEXT, position BIGINT",
    "gallery_examples": (
        "category TEXT, source TEXT, doc_index BIGINT, tokens BIGINT, note TEXT, text TEXT, "
        "match_start BIGINT, match_length BIGINT, position BIGINT"
    ),
}


def load_gallery(categories: tuple[Category, ...], examples: list[Example], query: AppletQuery) -> dict[str, Any]:
    """Stage the rows under a new load id, select it, then drop the loads it replaces."""
    known = {category.key for category in categories}
    unknown = sorted({example.category for example in examples} - known)
    if unknown:
        raise ValueError(f"Examples name categories that are not defined: {unknown}")
    rows: dict[str, list[dict[str, Any]]] = {
        "gallery_categories": [
            {**asdict(category), "position": position} for position, category in enumerate(categories)
        ],
        # Examples keep the order of their input files, which put the clearest first.
        "gallery_examples": [{**asdict(example), "position": position} for position, example in enumerate(examples)],
    }
    load_id = uuid.uuid4().hex
    counts = {
        table: insert_load_rows(query, table, TABLE_COLUMNS[table], load_id, table_rows)
        for table, table_rows in rows.items()
    }
    if counts != {table: len(table_rows) for table, table_rows in rows.items()}:
        raise RuntimeError(f"Staged row counts differ from the input: {counts}")
    query(
        "INSERT INTO gallery_current (load_id) VALUES (:load_id) "
        "ON CONFLICT (singleton) DO UPDATE SET load_id = EXCLUDED.load_id",
        {"load_id": load_id},
    )
    for table in TABLE_COLUMNS:
        query(f"DELETE FROM {table} WHERE load_id NOT IN (SELECT load_id FROM gallery_current)", {})
    return {"load_id": load_id, **counts}


def read_examples(path: StoragePath) -> list[Example]:
    return [Example(**row) for row in pq.read_table(io.BytesIO(path.read_bytes())).to_pylist()]


@click.command(help=__doc__)
@click.option("--input", "inputs", required=True, multiple=True, help="Parquet file of examples; repeat for several.")
@click.option("--app-id", "applet_id", required=True)
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
def main(inputs: tuple[str, ...], applet_id: str, service_url: str) -> None:
    configure_coreweave_s3()
    examples = [example for path in inputs for example in read_examples(StoragePath.parse(path))]
    click.echo(json.dumps(load_gallery(CATEGORIES, examples, applet_query(service_url, applet_id))))


if __name__ == "__main__":
    main()
