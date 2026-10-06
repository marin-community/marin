# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure GPU history through the bridge using live Finelog and recorded metadata.

Metadata replay measures projection and caching, not production RPC latency.
The separately tested signed regional RPC must be deployed before a live
end-to-end metadata benchmark can establish production loading cost.
"""

import argparse
import json
import resource
import time
from dataclasses import replace
from pathlib import Path

import server
from config import BridgeConfig, ClusterTarget
from finelog.deploy.config import load_finelog_config
from finelog.deploy.connect import open_client
from starlette.testclient import TestClient

PRIORITIES = {0: "INHERIT", 1: "PRODUCTION", 2: "INTERACTIVE", 3: "BATCH", 4: "SYSTEM"}
CLUSTERS = ("cw-rno2a", "cw-us-east-02a", "cw-us-east-08a")


class RecordingSource:
    def __init__(self, client):
        self.client = client
        self.target = ClusterTarget("marin", "project", "zone", "finelog", "controller")
        self.calls = []

    def query(self, sql, *, max_rows):
        began = time.monotonic()
        table = self.client.query(sql, max_rows=max_rows)
        self.calls.append(
            {"sql": sql, "seconds": time.monotonic() - began, "rows": table.num_rows, "arrow_bytes": table.nbytes}
        )
        return table


class ReplayRegistry:
    def __init__(self, path):
        self.rows = {cluster: json.loads((path / f"{cluster}-7d-attempts.json").read_text()) for cluster in CLUSTERS}
        self.calls = []

    def gpu_allocation_metadata(self, cluster, start, end, *, max_rows):
        result = []
        for row in self.rows[cluster]:
            if row["created_at_ms"] is not None and row["created_at_ms"] >= end:
                continue
            if row["finished_at_ms"] is not None and row["finished_at_ms"] < start:
                continue
            item = {
                "rootJobId": row["root_job_id"],
                "taskId": row["task_id"],
                "gpuCount": row["gpus"],
                "gpuVariant": row["variant"] or "",
                "requestedPriority": "PRIORITY_BAND_" + PRIORITIES[row["requested"]],
                "currentAppliedPriority": "PRIORITY_BAND_" + PRIORITIES[row["applied"]],
                "currentAttemptId": row["current_attempt_id"],
                "attemptId": row["attempt_id"] if row["attempt_id"] is not None else -1,
            }
            for original, wire in [
                ("created_at_ms", "createdAtMs"),
                ("started_at_ms", "startedAtMs"),
                ("finished_at_ms", "finishedAtMs"),
            ]:
                if row[original] is not None:
                    item[wire] = str(row[original])
            result.append(item)
        self.calls.append(
            {
                "cluster": cluster,
                "from_ms": start,
                "to_ms": end,
                "rows": len(result),
                "json_bytes": len(json.dumps(result)),
            }
        )
        if len(result) > max_rows:
            raise ValueError("recorded metadata exceeds the row budget")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--from-ms", type=int, required=True)
    parser.add_argument("--to-ms", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    registry = ReplayRegistry(args.metadata_dir)
    with open_client(load_finelog_config("marin"), "marin", request_timeout=20) as client:
        source = RecordingSource(client)
        config = replace(BridgeConfig.from_environment(), cw_read_token=None, loom_alerts=None)
        app = server.create_app(config, {"marin": source}, {"marin": registry}, None, None, None)
        params = {"from": args.from_ms, "to": args.to_ms, "clusters": ",".join(CLUSTERS)}
        measurements = []
        with TestClient(app) as bridge:
            for name, offset in [("initial", 0), ("cached", 0), ("refresh", 60_000)]:
                if name == "refresh":
                    # Expire successful request/current-day entries without a wall wait.
                    original = server.time.monotonic
                    server.time.monotonic = lambda original=original: original() + config.cache_ttl + 1
                before = len(source.calls)
                before_metadata = len(registry.calls)
                began = time.perf_counter()
                responses = []
                for model in ["H100", "GB200"]:
                    response = bridge.get(
                        "/finelog/marin/v1/gpu/allocation",
                        params={**params, "from": args.from_ms + offset, "to": args.to_ms + offset, "model": model},
                    )
                    if response.status_code != 200:
                        raise RuntimeError(response.text)
                    responses.extend(response.json())
                measurements.append(
                    {
                        "name": name,
                        "seconds": time.perf_counter() - began,
                        "rows": len(responses),
                        "finelog_queries": len(source.calls) - before,
                        "metadata_replays": len(registry.calls) - before_metadata,
                    }
                )
                (args.output / f"{name}-rows.json").write_text(json.dumps(responses, indent=2))
                if name == "refresh":
                    server.time.monotonic = original
                print(json.dumps(measurements[-1]), flush=True)
        plans = []
        if args.explain:
            for record in source.calls:
                began = time.perf_counter()
                plan = client.query("EXPLAIN ANALYZE " + record["sql"], max_rows=100).to_pylist()
                plans.append({"sql": record["sql"], "seconds": time.perf_counter() - began, "plan": plan})
        result = {
            "mode": "Live Finelog + recorded regional metadata replay; not production metadata RPC latency",
            "finelog_cache_conditions": "Client/bridge initially empty; remote Finelog cache warmth uncontrolled",
            "from_ms": args.from_ms,
            "to_ms": args.to_ms,
            "measurements": measurements,
            "finelog_calls": source.calls,
            "metadata_replays": registry.calls,
            "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "plans": plans,
        }
        (args.output / "cost.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
