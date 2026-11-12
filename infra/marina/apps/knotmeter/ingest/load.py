# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load saved collector bundles and corpus structures into an explicitly selected Marina."""

import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, TypedDict

import requests
from marina.client import AppletQuery, applet_query

from .adapter import BUNDLE_MANIFEST, ImportRun, prepare_bundle, read_json
from .calibration import prepare_calibrations
from .corpus import conversation_summary, dataset_summary

BATCH_BYTES = 750_000
SERIES_CHUNK_CHARS = 200_000


class ImportReport(TypedDict):
    key: str
    changed: bool
    requests: int


def attach_series(prepared: ImportRun, bundle: Path, cache: Path) -> ImportRun:
    """Reduce the saved server export with a pinned parser and retain its source hash."""
    profile = next(name for name in prepared.run["source_files"] if name.endswith("profile_export.jsonl"))
    source = bundle / Path(profile).with_name("server_metrics_export.json")
    if not source.is_file():
        source = source.with_suffix(".json.zst")
        if not source.is_file():
            return prepared
    with source.open("rb") as stream:
        source_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    reducer = Path(__file__).resolve().parent.parent / ".build/reduce-series.mjs"
    reducer_digest = hashlib.sha256(
        reducer.read_bytes() + Path(__file__).with_name("collect_series.py").read_bytes()
    ).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    profile_digest = prepared.run["source_files"][profile]
    cached = cache / f"{source_digest}-{profile_digest}-{reducer_digest}.json"
    if not cached.is_file():
        output = subprocess.check_output(
            [
                "uv",
                "run",
                "--script",
                str(Path(__file__).with_name("collect_series.py")),
                str(source),
                "--framework",
                prepared.run["framework"],
            ]
        )
        series = json.loads(output)
        temporary = cached.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(series, allow_nan=False, separators=(",", ":")))
        temporary.replace(cached)
    series = read_json(cached)
    run = prepared.run | {
        "source_files": prepared.run["source_files"] | {str(source.relative_to(bundle)): source_digest},
        "series_reducer": {"upstream": "315128ce4e49c8a1ec1aa67eafa8f2de26e4c1f9", "sha256": reducer_digest},
        "diagnostic_aggregates": series["aggregates"],
    }
    run.pop("content_sha256", None)
    digest = hashlib.sha256(
        json.dumps([run, prepared.requests, series], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return replace(prepared, run=run | {"content_sha256": digest}, digest=digest, series=series)


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
        rows = [json.loads(line) for line in stream if line.strip()]
        for row in rows:
            row["summary"] = conversation_summary(row["structure"])
        count = 0
        for body in batches(rows):
            result = query(
                "INSERT INTO knotmeter_corpus (dataset, revision, conversation_id, structure, summary) "
                "SELECT dataset, revision, conversation_id, structure, summary "
                "FROM jsonb_to_recordset(CAST(:rows AS JSONB)) "
                "AS row(dataset TEXT, revision TEXT, conversation_id TEXT, structure JSONB, summary JSONB) "
                "ON CONFLICT (dataset, revision, conversation_id) DO UPDATE SET summary=excluded.summary "
                "WHERE knotmeter_corpus.structure=excluded.structure",
                {"rows": body},
            )
            if result["row_count"] != len(json.loads(body)):
                raise ValueError("Corpus revision conflicts with an existing immutable structure")
            count += result["row_count"]
        for dataset, revision in sorted({(row["dataset"], row["revision"]) for row in rows}):
            selected_rows = [r for r in rows if r["dataset"] == dataset and r["revision"] == revision]
            stored_count = query(
                "SELECT count(*) AS count FROM knotmeter_corpus WHERE dataset=:dataset AND revision=:revision",
                {"dataset": dataset, "revision": revision},
            )["rows"][0]["count"]
            if stored_count != len(selected_rows):
                raise ValueError("Load the full immutable corpus revision before selecting its dataset summary")
            summary, distributions = dataset_summary(selected_rows)
            query(
                "INSERT INTO knotmeter_datasets (dataset, revision, summary, distributions) "
                "VALUES (:dataset, :revision, CAST(:summary AS JSONB), CAST(:distributions AS JSONB)) "
                "ON CONFLICT (dataset, revision) DO UPDATE "
                "SET summary=excluded.summary, distributions=excluded.distributions",
                {
                    "dataset": dataset,
                    "revision": revision,
                    "summary": json.dumps(summary),
                    "distributions": json.dumps(distributions),
                },
            )
    return count


def load_run(prepared: ImportRun, query: AppletQuery) -> ImportReport:
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
    serialized = json.dumps(prepared.series, allow_nan=False, separators=(",", ":")) if prepared.series else ""
    chunks = [serialized[i : i + SERIES_CHUNK_CHARS] for i in range(0, len(serialized), SERIES_CHUNK_CHARS)]
    query(
        "INSERT INTO knotmeter_imports (id, run_key, content_sha256, run, request_count, series_count) "
        "VALUES (:id, :key, :digest, CAST(:run AS JSONB), :count, :series_count)",
        {
            "id": import_id,
            "key": prepared.key,
            "digest": prepared.digest,
            "run": json.dumps(prepared.run, allow_nan=False),
            "count": len(prepared.requests),
            "series_count": len(chunks),
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
    for body in batches({"part": i, "payload": chunk} for i, chunk in enumerate(chunks)):
        query(
            "INSERT INTO knotmeter_series (import_id, part, payload) "
            "SELECT :id, part, payload FROM jsonb_to_recordset(CAST(:rows AS JSONB)) "
            "AS row(part INTEGER, payload TEXT)",
            {"id": import_id, "rows": body},
        )
    selected = query(
        "INSERT INTO knotmeter_current AS current (run_key, import_id) "
        "SELECT run_key, id FROM knotmeter_imports WHERE id = :id "
        "AND request_count = (SELECT COUNT(*) FROM knotmeter_requests WHERE import_id = :id) "
        "AND series_count = (SELECT COUNT(*) FROM knotmeter_series WHERE import_id = :id) "
        "ON CONFLICT (run_key) DO UPDATE SET import_id = excluded.import_id "
        "WHERE current.import_id = :previous",
        {"id": import_id, "previous": previous},
    )
    if selected["row_count"] != 1:
        raise RuntimeError("Import was incomplete or another loader selected this run; retry after inspection")
    if previous is not None:
        query(
            "DELETE FROM knotmeter_series WHERE import_id = :id "
            "AND NOT EXISTS (SELECT 1 FROM knotmeter_current WHERE import_id = :id)",
            {"id": previous},
        )
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
    parser.add_argument("--bundle", type=Path, action="append", default=[])
    parser.add_argument("--calibration", type=Path, help="Saved calibration directory containing curve-summary.json")
    parser.add_argument("--calibration-job", help="Import one exact native job from --calibration")
    parser.add_argument("--corpus", type=Path, help="JSONL of dataset, revision, conversation_id and structure")
    parser.add_argument(
        "--series-cache", type=Path, required=True, help="Local cache outside the app for reduced server series"
    )
    parser.add_argument(
        "--source-attempts", type=Path, help="Saved GitHub attempt responses keyed by run ID; otherwise use gh"
    )
    args = parser.parse_args()
    if not args.bundle and not args.calibration:
        parser.error("Select --bundle or --calibration")
    if args.calibration_job and not args.calibration:
        parser.error("--calibration-job requires --calibration")
    attempts = read_json(args.source_attempts) if args.source_attempts else {}
    with requests.Session() as session:
        query = applet_query(args.url, args.app_id, session=session)
        corpus = load_corpus(args.corpus, query) if args.corpus else 0
        loaded = []
        for bundle in args.bundle:
            source = read_json(bundle / BUNDLE_MANIFEST)["source"]
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
            for run in prepare_bundle(bundle, actual):
                loaded.append(load_run(attach_series(run, bundle, args.series_cache), query))
        if args.calibration:
            for run in prepare_calibrations(args.calibration, args.calibration_job):
                print(f"Importing {run.key}", file=sys.stderr, flush=True)
                loaded.append(load_run(attach_series(run, args.calibration, args.series_cache), query))
        report = {"corpus_processed": corpus, "runs": loaded}
    print(json.dumps(report))


if __name__ == "__main__":
    main()
