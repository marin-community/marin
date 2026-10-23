# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure a frozen request set after warmup, requiring real server batching."""

import argparse
import copy
import hashlib
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fray.iris_backend import convert_constraints, convert_resources
from fray.types import ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient, iris_ctx
from iris.cluster.types import Entrypoint, EnvironmentSpec
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.weight_merging.rsi.levanter_267.protocol import RUN_PROVENANCE_PATH


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _request(base, spec, warmup):
    payload = copy.deepcopy(spec["payload"])
    if warmup:
        payload["max_tokens"] = 8
    start = time.monotonic()
    request = urllib.request.Request(
        base + spec["path"], data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    return dict(id=spec["id"], response=result, wall_seconds=time.monotonic() - start)


def _signature(result):
    choice = result["response"]["choices"][0]
    logprobs = choice["logprobs"]
    tokens = [item["token"] for item in logprobs["content"]] if "content" in logprobs else logprobs["tokens"]
    text = choice.get("text", choice.get("message", {}).get("content"))
    return text, tokens, choice["finish_reason"], result["response"]["usage"]["completion_tokens"]


def measure(endpoint, specs, gate, output):
    base = iris_ctx().client.resolve_endpoint(endpoint).rstrip("/")
    root = StoragePath(output)
    provenance = _get(base + RUN_PROVENANCE_PATH)
    for key in ("run_id", "source_commit", "serve_job", "model_path"):
        if provenance.get(key) != gate.get(key):
            raise ValueError("Throughput gate report belongs to another serving process")
    if provenance.get("max_seqs", 1) <= 1:
        raise ValueError("Owner requires an engine with real batching before this concurrency probe")
    (root / "requests.json").write_text(json.dumps(specs, indent=2))
    phases = []
    baseline = None
    # Include8 to measure the actual historical evaluator's configured limit,
    # alongside the owner's requested1/4/16 comparison.
    for concurrency in (1, 4, 8, 16):
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(lambda spec: _request(base, spec, True), specs))
        before = _get(base + RUN_PROVENANCE_PATH)["batch_counts"]
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda spec: _request(base, spec, False), specs))
        elapsed = time.monotonic() - start
        after = _get(base + RUN_PROVENANCE_PATH)["batch_counts"]
        batches = {
            key: after.get(key, 0) - before.get(key, 0) for key in after if after.get(key, 0) != before.get(key, 0)
        }
        signatures = [_signature(result) for result in results]
        if baseline is None:
            baseline = signatures
        mismatches = [
            specs[idx]["id"] for idx, pair in enumerate(zip(signatures, baseline, strict=True)) if pair[0] != pair[1]
        ]
        count = sum(result["response"]["usage"]["completion_tokens"] for result in results)
        phase = dict(
            concurrency=concurrency,
            requests=len(results),
            completion_tokens=count,
            wall_seconds=elapsed,
            tokens_per_second=count / elapsed,
            median_request_seconds=statistics.median(result["wall_seconds"] for result in results),
            successful_batch_sizes=batches,
            greedy_mismatches=mismatches,
            results=results,
        )
        (root / f"concurrency-{concurrency}.json").write_text(json.dumps(phase, indent=2))
        phases.append({key: value for key, value in phase.items() if key != "results"})
        if mismatches or sum(int(size) * n for size, n in batches.items()) != len(specs):
            raise RuntimeError("Batched greedy results or observed server batch accounting disagree")
        if concurrency > 1 and not any(int(size) > 1 and n for size, n in batches.items()):
            raise RuntimeError("Concurrent callers did not exercise a batched engine")
    summary = dict(server=provenance, phases=phases, prior="Issue #3826: 762 vs 7940 tok/s, different stack/workload")
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="levanter267-throughput")
    args = parser.parse_args()
    gate = json.loads(args.gate_report.read_text())
    if (
        not gate.get("accepted")
        or gate.get("endpoint") != args.endpoint
        or not gate.get("first_token_matches_cached_score")
    ):
        parser.error("Require the accepted cached/production gate for this endpoint")
    specs = json.loads(args.requests.read_text())
    if len(specs) != 32 or len({spec["id"] for spec in specs}) != len(specs):
        parser.error("Use the frozen32-request set with unique IDs")
    if not args.output.startswith("gs://marin-us-central2/"):
        parser.error("Place throughput artifacts alongside the central2 serving slice")
    print(
        json.dumps(
            dict(
                endpoint=args.endpoint,
                requests_sha256=hashlib.sha256(args.requests.read_bytes()).hexdigest(),
                output=args.output,
            )
        )
    )
    if not args.submit:
        return
    resources = ResourceConfig.with_cpu(cpu=2, ram="8g", disk="20g", regions=["us-central2"])
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, workspace=Path.cwd(), credentials=endpoint.credentials) as client:
            job = client.submit(
                name=args.name,
                entrypoint=Entrypoint.from_callable(measure, args.endpoint, specs, gate, args.output),
                resources=convert_resources(resources),
                constraints=convert_constraints(resources),
                environment=EnvironmentSpec(),
                timeout=Duration.from_seconds(2700),
                scheduling_timeout=Duration.from_seconds(14400),
                max_retries_failure=0,
            )
            print(job, flush=True)


if __name__ == "__main__":
    main()
