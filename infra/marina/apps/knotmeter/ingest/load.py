# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load saved collector bundles and corpus structures into an explicitly selected Marina."""

import argparse
import json
import subprocess
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import requests
from marina.client import AppletQuery, applet_query

from .adapter import ImportRun, prepare_bundle, read_json

BATCH_BYTES = 750_000


def batches(rows: Iterable[dict[str, Any]]) -> Iterator[str]:
    """Bound encoded SQL parameters, including their JSON transport escape layer."""
    batch: list[str] = []
    size = 2
    for row in rows:
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        length = len(json.dumps(encoded).encode()) + 1
        if length > BATCH_BYTES:
            raise ValueError("One record exceeds the Marina import batch budget")
        if size + length > BATCH_BYTES:
            yield "[" + ",".join(batch) + "]"
            batch, size = [], 2
        batch.append(encoded)
        size += length
    if batch:
        yield "[" + ",".join(batch) + "]"


def load_corpus(path: Path, query: AppletQuery) -> int:
    """Load immutable conversation structures keyed by dataset revision and original ID."""
    with path.open() as stream:
        rows = (json.loads(line) for line in stream if line.strip())
        count = 0
        for body in batches(rows):
            result = query(
                "INSERT INTO knotmeter_corpus (dataset, revision, conversation_id, structure) "
                "SELECT dataset, revision, conversation_id, structure FROM jsonb_to_recordset(CAST(:rows AS JSONB)) "
                "AS row(dataset TEXT, revision TEXT, conversation_id TEXT, structure JSONB) "
                "ON CONFLICT (dataset, revision, conversation_id) DO NOTHING",
                {"rows": body},
            )
            count += result["row_count"]
    return count


def load_run(prepared: ImportRun, query: AppletQuery) -> dict[str, Any]:
    """Stage request batches, then atomically expose one complete run without losing concurrent imports."""
    current = query(
        "SELECT c.import_id, i.content_sha256 FROM knotmeter_current c "
        "JOIN knotmeter_imports i ON i.id = c.import_id WHERE c.run_key = :key",
        {"key": prepared.key},
    )["rows"]
    if current and current[0]["content_sha256"] == prepared.digest:
        return {"key": prepared.key, "changed": False, "requests": len(prepared.requests)}
    previous = current[0]["import_id"] if current else None
    import_id = uuid.uuid4().hex
    query(
        "INSERT INTO knotmeter_imports (id, run_key, content_sha256, run, request_count) "
        "VALUES (:id, :key, :digest, CAST(:run AS JSONB), :count)",
        {
            "id": import_id,
            "key": prepared.key,
            "digest": prepared.digest,
            "run": json.dumps(prepared.run, allow_nan=False),
            "count": len(prepared.requests),
        },
    )
    rows = ({"ordinal": i, "payload": row} for i, row in enumerate(prepared.requests))
    for body in batches(rows):
        query(
            "INSERT INTO knotmeter_requests (import_id, ordinal, payload) "
            "SELECT :id, ordinal, payload FROM jsonb_to_recordset(CAST(:rows AS JSONB)) "
            "AS row(ordinal INTEGER, payload JSONB)",
            {"id": import_id, "rows": body},
        )
    selected = query(
        "INSERT INTO knotmeter_current AS current (run_key, import_id) "
        "SELECT run_key, id FROM knotmeter_imports WHERE id = :id "
        "AND request_count = (SELECT COUNT(*) FROM knotmeter_requests WHERE import_id = :id) "
        "ON CONFLICT (run_key) DO UPDATE SET import_id = excluded.import_id "
        "WHERE current.import_id = :previous",
        {"id": import_id, "previous": previous},
    )
    if selected["row_count"] != 1:
        raise RuntimeError("Import was incomplete or another loader selected this run; retry after inspection")
    if previous is not None:
        query(
            "DELETE FROM knotmeter_requests WHERE import_id = :id "
            "AND NOT EXISTS (SELECT 1 FROM knotmeter_current WHERE import_id = :id)",
            {"id": previous},
        )
        query(
            "DELETE FROM knotmeter_imports WHERE id = :id "
            "AND NOT EXISTS (SELECT 1 FROM knotmeter_current WHERE import_id = :id)",
            {"id": previous},
        )
    return {"key": prepared.key, "changed": True, "requests": len(prepared.requests)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Explicit Marina origin; use the printed loopback origin locally")
    parser.add_argument("--app-id", required=True)
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--corpus", type=Path, help="JSONL of dataset, revision, conversation_id and structure")
    parser.add_argument(
        "--source-attempts", type=Path, help="Saved GitHub attempt responses keyed by run ID; otherwise use gh"
    )
    args = parser.parse_args()
    attempts = read_json(args.source_attempts) if args.source_attempts else {}
    prepared = []
    for bundle in args.bundle:
        source = read_json(bundle / "agentx-source.json")["source"]
        actual = attempts.get(str(source["run_id"]))
        if actual is None:
            actual = json.loads(
                subprocess.check_output(
                    [
                        "gh",
                        "api",
                        f"repos/{source['repository']}/actions/runs/{source['run_id']}/attempts/{source['run_attempt']}",
                    ]
                )
            )
        prepared.extend(prepare_bundle(bundle, actual))
    with requests.Session() as session:
        query = applet_query(args.url, args.app_id, session=session)
        corpus = load_corpus(args.corpus, query) if args.corpus else 0
        report = {"corpus_inserted": corpus, "runs": [load_run(run, query) for run in prepared]}
    print(json.dumps(report))


if __name__ == "__main__":
    main()
