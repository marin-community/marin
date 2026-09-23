# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay equal scalar or explicit-bucket vLLM histories into a local Finelog.

Run one format per process against an isolated loopback server. This measures
historical ingest and dashboard query cost; the actual SkyRL/Rigging exporter
POST count must be measured separately because this replay batches across polls.
"""

import argparse
import json
import resource
import time
import uuid
from collections.abc import Iterator
from contextlib import nullcontext
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from finelog.client import LogClient
from vllm_observability import (
    VllmIdentityField,
    sql_string,
    vllm_overview_query,
    vllm_overview_table,
    vllm_run_summary_samples_query,
    vllm_run_summary_table,
)

FAMILIES = (
    "request_queue_time_seconds",
    "request_prefill_time_seconds",
    "request_decode_time_seconds",
    "e2e_request_latency_seconds",
    "time_to_first_token_seconds",
    "request_generation_tokens",
    "iteration_tokens_total",
    "request_time_per_output_token_seconds",
)
BOUNDS = tuple((index + 1) / 100 for index in range(19))
FINISH_REASONS = ("stop", "length", "abort", "error", "repetition")
COUNTERS = (
    "num_preemptions_total",
    "prefix_cache_hits_total",
    "prefix_cache_queries_total",
    "generation_tokens_total",
    "prompt_tokens_total",
    "spec_decode_num_drafts_total",
    "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total",
)


def _record(
    timestamp_ms: int,
    name: str,
    attributes: dict[str, str],
    *,
    value: float | None = None,
    body: dict[str, object] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "timestamp_ms": timestamp_ms,
        "kind": "histogram" if body is not None else "gauge",
        "name": name,
        "unit": "s" if name.endswith("seconds") or "seconds_" in name else "1",
        "attributes": attributes,
    }
    if body is None:
        record["value"] = value
    else:
        record["body"] = body
    return record


def _poll_records(mode: str, timestamp_ms: int, poll: int, engines: int) -> Iterator[dict[str, object]]:
    for engine_index in range(engines):
        identity = {"engine": f"engine-{engine_index}", "engine_index": str(engine_index), "metric_source": "vllm"}
        histogram_labels = {**identity, "source_kind": "histogram", "source_temporality": "cumulative_snapshot"}
        for family in FAMILIES:
            count = 95 * (poll + 1)
            total = 10.0 * (poll + 1)
            if mode == "structured":
                yield _record(
                    timestamp_ms,
                    family,
                    histogram_labels,
                    body={
                        "encoding": "explicit_bucket_v1",
                        "aggregation_temporality": "cumulative",
                        "explicit_bounds": BOUNDS,
                        "bucket_counts": [5 * (poll + 1)] * 19 + [0],
                        "count": count,
                        "sum": total,
                        "producer_epoch": identity["engine"],
                        "sequence": poll,
                    },
                )
            else:
                for index, bound in enumerate(BOUNDS):
                    yield _record(
                        timestamp_ms,
                        f"{family}_bucket",
                        {**histogram_labels, "le": str(bound)},
                        value=float(5 * (index + 1) * (poll + 1)),
                    )
                yield _record(
                    timestamp_ms,
                    f"{family}_bucket",
                    {**histogram_labels, "le": "+Inf"},
                    value=float(count),
                )
                yield _record(timestamp_ms, f"{family}_count", histogram_labels, value=float(count))
                yield _record(timestamp_ms, f"{family}_sum", histogram_labels, value=total)
        current = {**identity, "source_kind": "gauge", "source_temporality": "current_snapshot"}
        cumulative = {**identity, "source_kind": "counter", "source_temporality": "cumulative_snapshot"}
        for name in ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"):
            yield _record(timestamp_ms, name, current, value=0.0)
        for reason in ("capacity", "deferred"):
            yield _record(timestamp_ms, "num_requests_waiting_by_reason", {**current, "reason": reason}, value=0.0)
        for name in COUNTERS:
            yield _record(timestamp_ms, name, cumulative, value=float(poll + 1))
        for reason in FINISH_REASONS:
            yield _record(
                timestamp_ms,
                "request_success_total",
                {**cumulative, "finished_reason": reason},
                value=float(poll + 1) if reason == "stop" else 0.0,
            )


def _post_batch(endpoint: str, job_id: str, records: list[dict[str, object]]) -> int:
    batch_id = str(uuid.uuid4())
    payload = json.dumps(
        {
            "version": 1,
            "batch_id": batch_id,
            "resource": {"service": "marinskyrl", "job_id": job_id, "attributes": {}},
            "records": records,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    request = Request(
        endpoint.rstrip("/") + "/v1/telemetry",
        data=payload,
        headers={"Content-Type": "application/json", "Idempotency-Key": batch_id},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        acknowledgement = json.load(response)
    if acknowledgement.get("status") != "accepted" or acknowledgement.get("record_count") != len(records):
        raise RuntimeError(f"Finelog did not accept the full batch: {acknowledgement}")
    return len(payload)


def _server_usage(pid: int | None) -> dict[str, int] | None:
    if pid is None:
        return None
    with open(f"/proc/{pid}/stat", encoding="utf-8") as stream:
        fields = stream.read().rsplit(") ", 1)[1].split()
    with open(f"/proc/{pid}/status", encoding="utf-8") as stream:
        status = stream.read()
    rss_kb = next(int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:"))
    return {"cpu_ticks": int(fields[11]) + int(fields[12]), "rss_kb": rss_kb}


def run(args: argparse.Namespace) -> dict[str, object]:
    if urlsplit(args.endpoint).hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("this replay may write only to a localhost Finelog")
    if args.polls < 2 or args.engines <= 0 or args.interval_ms <= 0:
        raise ValueError("polls must be >=2 and engines/interval_ms must be positive")
    start_usage = _server_usage(args.server_pid)
    rows = posts = request_bytes = 0
    if args.query_only:
        rows = sum(1 for _ in _poll_records(args.mode, args.start_ms, 0, args.engines)) * args.polls
        ingest_seconds = None
    else:
        pending: list[dict[str, object]] = []
        started = time.perf_counter()
        for poll in range(args.polls):
            for record in _poll_records(args.mode, args.start_ms + poll * args.interval_ms, poll, args.engines):
                pending.append(record)
                rows += 1
                if len(pending) == 1_000:
                    request_bytes += _post_batch(args.endpoint, args.job_id, pending)
                    posts += 1
                    pending.clear()
        if pending:
            request_bytes += _post_batch(args.endpoint, args.job_id, pending)
            posts += 1
        ingest_seconds = time.perf_counter() - started
    ingested_usage = _server_usage(args.server_pid)

    client = LogClient.connect(args.endpoint, timeout_ms=120_000)
    count_sql = (
        'SELECT COUNT(*) AS rows FROM "telemetry_v1.marinskyrl" WHERE job_id = '
        + sql_string(args.job_id)
    )
    visibility_started = time.perf_counter()
    while True:
        visible_rows = client.query(count_sql).column("rows")[0].as_py()
        if visible_rows == rows:
            break
        if visible_rows > rows or time.perf_counter() - visibility_started > 120:
            raise RuntimeError(f"Finelog made {visible_rows} of {rows} replay rows visible")
        time.sleep(0.25)
    visibility_seconds = time.perf_counter() - visibility_started
    visible_usage = _server_usage(args.server_pid)

    overview = vllm_overview_query(
        VllmIdentityField.JOB_ID,
        args.job_id,
        args.start_ms,
        args.start_ms + args.polls * args.interval_ms,
        15_000,
    )
    samples_sql = vllm_run_summary_samples_query(overview) if args.summary else overview.samples_sql
    started = time.perf_counter()
    series = client.query(samples_sql, max_rows=50_001)
    query_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if args.summary:
        table = vllm_run_summary_table(overview, series, nullcontext(), max_rows=1_000)
    else:
        table = vllm_overview_table(overview, series, nullcontext(), max_rows=10_000)
    projection_seconds = time.perf_counter() - started
    client.close()
    projected_usage = _server_usage(args.server_pid)
    evidence = [
        {"metric": row["metric"], "stat": row["stat"], "value": row["value"], "samples": row["samples"]}
        for row in table.to_pylist()
        if row["metric"] in ("ttft", "ttft_observations")
        and (row["t"] is None)
    ]
    return {
        "format": args.mode,
        "engines": args.engines,
        "polls": args.polls,
        "duration_ms": args.polls * args.interval_ms,
        "rows": rows,
        "batched_replay_posts": None if args.query_only else posts,
        "request_bytes_uncompressed": None if args.query_only else request_bytes,
        "ingest_seconds": ingest_seconds,
        "visibility_seconds": visibility_seconds,
        "finelog_query_seconds": query_seconds,
        "projection_seconds": projection_seconds,
        "series": series.num_rows,
        "samples": sum(map(len, series["points"].to_pylist())),
        "arrow_bytes": series.nbytes,
        "result_rows": table.num_rows,
        "client_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "server_usage": {
            "before": start_usage,
            "ingested": ingested_usage,
            "visible": visible_usage,
            "projected": projected_usage,
        },
        "evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18873")
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--mode", choices=("scalar", "structured"), required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--start-ms", type=int, required=True)
    parser.add_argument("--engines", type=int, default=4)
    parser.add_argument("--polls", type=int, required=True)
    parser.add_argument("--interval-ms", type=int, default=5_000)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--query-only", action="store_true", help="measure an existing complete replay without reposting it")
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
