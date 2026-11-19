# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy archived assessments through independently checkpointed indexed ranges."""

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import pairwise

from marina.client import AppletQuery

from .historical_assessments import RETAIN_COPY_BATCH_ROWS

CHECKPOINT_TABLE = "atlas_history_copy_segments"


@dataclass(frozen=True)
class CopySegment:
    table: str
    pipeline: str
    lower: str | None
    upper: str | None

    @property
    def identity(self) -> str:
        payload = json.dumps([self.table, self.pipeline, self.lower, self.upper], separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class ThreadClient(threading.local):
    query: AppletQuery | None = None


def copy_statement(table: str, columns: str, task_count: int) -> str:
    """Build the bounded copy statement used in the benchmark and the live load."""
    names = [column.strip().split()[0] for column in columns.split(",")]
    identifiers = ", ".join(f":task_id_{index}" for index in range(task_count))
    return (
        f"INSERT INTO atlas_{table} (load_id, {', '.join(names)}) "
        f"SELECT :load_id, {', '.join('previous.' + name for name in names)} "
        f"FROM atlas_{table} previous WHERE previous.load_id=:prior_load_id "
        "AND previous.pipeline_key=:pipeline_key "
        f"AND previous.task_id IN ({identifiers}) "
        "AND NOT EXISTS (SELECT 1 FROM atlas_assessments current_assessment "
        "WHERE current_assessment.load_id=:load_id "
        "AND current_assessment.pipeline_key=previous.pipeline_key) ON CONFLICT DO NOTHING"
    )


def _copy_segment(
    query: AppletQuery,
    columns: str,
    segment: CopySegment,
    prior_load_id: str,
    load_id: str,
    batch_rows: int,
) -> int:
    parameters = {
        "load_id": load_id,
        "prior_load_id": prior_load_id,
        "table_name": segment.table,
        "pipeline_key": segment.pipeline,
        "segment_id": segment.identity,
        "lower_id": segment.lower,
        "upper_id": segment.upper,
    }
    query(
        f"INSERT INTO {CHECKPOINT_TABLE} "
        "(load_id,prior_load_id,table_name,pipeline_key,segment_id,lower_id,upper_id) "
        "VALUES (:load_id,:prior_load_id,:table_name,:pipeline_key,:segment_id,:lower_id,:upper_id) "
        "ON CONFLICT DO NOTHING",
        parameters,
    )
    checkpoint = query(
        f"SELECT prior_load_id,after_task_id,completed FROM {CHECKPOINT_TABLE} "
        "WHERE load_id=:load_id AND segment_id=:segment_id",
        parameters,
    )["rows"][0]
    if checkpoint["prior_load_id"] != prior_load_id:
        raise ValueError("History segment belongs to a different prior load")
    if checkpoint["completed"]:
        return 0
    after = checkpoint["after_task_id"]
    total = 0
    while True:
        restrictions = ""
        if segment.lower is not None:
            restrictions += "AND previous.task_id >= :lower_id "
        if segment.upper is not None:
            restrictions += "AND previous.task_id < :upper_id "
        if after is not None:
            restrictions += "AND previous.task_id > :after_task_id "
        page = query(
            f"SELECT previous.task_id FROM atlas_{segment.table} previous "
            "WHERE previous.load_id=:prior_load_id AND previous.pipeline_key=:pipeline_key "
            + restrictions
            + "ORDER BY previous.task_id LIMIT :batch_rows",
            {**parameters, "after_task_id": after, "batch_rows": batch_rows},
        )["rows"]
        if not page:
            query(
                f"UPDATE {CHECKPOINT_TABLE} SET completed=true " "WHERE load_id=:load_id AND segment_id=:segment_id",
                parameters,
            )
            return total
        # Read the immutable source index directly. An anti-join against a
        # partially copied target can choose a quadratic plan before ANALYZE;
        # the unique key already makes replayed pages harmless.
        result = query(
            copy_statement(segment.table, columns, len(page)),
            {**parameters, **{f"task_id_{index}": row["task_id"] for index, row in enumerate(page)}},
        )
        total += result["row_count"]
        after = page[-1]["task_id"]
        # A checkpoint follows the committed copy; a crash between them only
        # replays ON CONFLICT DO NOTHING, never skips an uncommitted page.
        query(
            f"UPDATE {CHECKPOINT_TABLE} SET after_task_id=:after_task_id "
            "WHERE load_id=:load_id AND segment_id=:segment_id",
            {**parameters, "after_task_id": after},
        )


def retain_prior_assessments_parallel(
    query: AppletQuery,
    query_factory: Callable[[], AppletQuery],
    columns: Mapping[str, str],
    prior_load_id: str,
    load_id: str,
    *,
    workers: int,
    item_batch_rows: int,
    stop: threading.Event,
) -> dict[str, int]:
    """Keep headers last while separate clients copy disjoint, resumable ID ranges."""
    if workers < 1 or item_batch_rows < 1:
        raise ValueError("History copy requires positive workers and batch size")
    query(
        f"CREATE TABLE IF NOT EXISTS {CHECKPOINT_TABLE} ("
        "load_id TEXT NOT NULL,prior_load_id TEXT NOT NULL,table_name TEXT NOT NULL,"
        "pipeline_key TEXT NOT NULL,segment_id TEXT NOT NULL,lower_id TEXT,upper_id TEXT,"
        "after_task_id TEXT,completed BOOLEAN NOT NULL DEFAULT false,"
        "PRIMARY KEY(load_id,segment_id))",
        {},
    )
    parameters = {"load_id": load_id, "prior_load_id": prior_load_id}
    pipelines = query(
        "SELECT pipeline_key FROM (SELECT DISTINCT pipeline_key FROM atlas_cohort_summaries "
        "WHERE load_id=:prior_load_id) candidates WHERE NOT EXISTS "
        "(SELECT 1 FROM atlas_assessments current_assessment WHERE current_assessment.load_id=:load_id "
        "AND current_assessment.pipeline_key=candidates.pipeline_key) ORDER BY pipeline_key",
        parameters,
    )["rows"]
    copied = dict.fromkeys(("populations", "items", "packets", "assessments"), 0)

    def ordinary(table: str) -> int:
        names = [column.strip().split()[0] for column in columns[table].split(",")]
        total = 0
        for pipeline in pipelines:
            total += query(
                f"INSERT INTO atlas_{table} (load_id, {', '.join(names)}) "
                f"SELECT :load_id, {', '.join('previous.' + name for name in names)} "
                f"FROM atlas_{table} previous WHERE previous.load_id=:prior_load_id "
                "AND previous.pipeline_key=:pipeline_key AND NOT EXISTS "
                "(SELECT 1 FROM atlas_assessments current_assessment "
                "WHERE current_assessment.load_id=:load_id "
                "AND current_assessment.pipeline_key=previous.pipeline_key) ON CONFLICT DO NOTHING",
                {**parameters, **pipeline},
            )["row_count"]
        return total

    copied["populations"] = ordinary("populations")
    clients = ThreadClient()

    def execute(segment: CopySegment, size: int) -> int:
        if stop.is_set():
            raise RuntimeError("History copy stopped after another client failed")
        try:
            if clients.query is None:
                clients.query = query_factory()
            client = clients.query

            def bounded_query(sql: str, parameters: dict) -> dict:
                if stop.is_set():
                    raise RuntimeError("History copy stopped after another client failed")
                return client(sql, parameters)

            return _copy_segment(bounded_query, columns[segment.table], segment, prior_load_id, load_id, size)
        except BaseException:
            stop.set()
            raise

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="atlas-history") as executor:
        for table in ("items", "packets"):
            size = item_batch_rows if table == "items" else RETAIN_COPY_BATCH_ROWS[table]
            estimate_sql = (
                "SELECT pipeline_key,SUM(sample_count) AS n FROM atlas_populations "
                if table == "items"
                else "SELECT pipeline_key,COUNT(*) AS n FROM atlas_packets "
            ) + "WHERE load_id=:prior_load_id GROUP BY pipeline_key"
            estimates = {row["pipeline_key"]: row["n"] for row in query(estimate_sql, parameters)["rows"]}
            segments = []
            for pipeline in pipelines:
                key = pipeline["pipeline_key"]
                count = int(estimates.get(key, 0))
                boundaries = []
                if count >= size * workers:
                    for partition in range(1, workers):
                        boundary = query(
                            f"SELECT task_id FROM atlas_{table} WHERE load_id=:prior_load_id "
                            "AND pipeline_key=:pipeline_key ORDER BY task_id LIMIT 1 OFFSET :offset",
                            {**parameters, "pipeline_key": key, "offset": count * partition // workers},
                        )["rows"]
                        if boundary:
                            boundaries.append(boundary[0]["task_id"])
                # Boundaries span the full key space, even if an estimate is stale.
                bounds = [None, *sorted(set(boundaries)), None]
                segments.extend(CopySegment(table, key, lower, upper) for lower, upper in pairwise(bounds))
            futures = [executor.submit(execute, segment, size) for segment in segments]
            failure = None
            for future in as_completed(futures):
                try:
                    copied[table] += future.result()
                except BaseException as error:
                    stop.set()
                    if failure is None:
                        failure = error
            if failure is not None:
                raise failure
    copied["assessments"] = ordinary("assessments")
    return copied
