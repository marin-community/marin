# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the 64K AgentX replay against two native EP4 replicas on one H100 host."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from marin.inference.dashboard_server import bind_serving_socket, serve_app_background
from starlette.background import BackgroundTask

from experiments.evaluation.knotmeter.artifacts import read_record
from experiments.evaluation.knotmeter.processes import run_logged
from experiments.evaluation.knotmeter.recipe import benchmark_command, load_series, workload_environment
from experiments.evaluation.knotmeter.workload import check_native_cache, restore_workload

SERIES = Path(__file__).with_name("snowball-v01-tpu-64k.yaml")
NUM_REPLICAS = 2
GPUS_PER_REPLICA = 4
REPLICA_STARTUP_TIMEOUT = 1200
REPLICA_SHUTDOWN_TIMEOUT = 15

REPLICA_OPTIONS = """
--attention gpu
--decode-layers unrolled_split_cache
--kv-sharding expert
--decode-experts separate
--kv-cache-contexts 16
--prefix-cache recent_prompts
--max-seqs-in-prefill 4
--max-unrolled-prefill-tokens 65536
--prefill-buckets 512 1024 2048 4096 8192 16384 32768
"""


def make_proxy(backends: list[str], request_counts: Counter[int]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=180, trust_env=False, limits=httpx.Limits(max_connections=512)) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def forward(path: str, request: Request) -> Response:
        body = await request.body()
        rank = 0
        if body:
            items = json.loads(body).get("messages", ())
            # Conversations with the same initial user message retain prefix affinity.
            initial = next((m for m in items if m["role"] == "user"), items[:1])
            root = json.dumps(initial, sort_keys=True, separators=(",", ":")).encode()
            rank = hashlib.blake2b(root, digest_size=1).digest()[0] % len(backends)
            request_counts[rank] += 1
        url = f"{backends[rank]}/v1/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        client = app.state.client
        forward_headers = {"content-type": request.headers["content-type"]} if "content-type" in request.headers else {}
        upstream = await client.send(
            client.build_request(request.method, url, headers=forward_headers, content=body), stream=True
        )
        media_type = upstream.headers.get("content-type", "application/json")
        if "text/event-stream" not in media_type:
            content = await upstream.aread()
            await upstream.aclose()
            return Response(content, status_code=upstream.status_code, media_type=media_type)
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            media_type=media_type,
            background=BackgroundTask(upstream.aclose),
        )

    return app


def wait_ready(workers: list[subprocess.Popen[bytes]], outputs: list[Path]) -> list[str]:
    deadline = time.monotonic() + REPLICA_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        for rank, worker in enumerate(workers):
            if worker.poll() is not None:
                raise RuntimeError(f"Replica {rank} stopped with exit {worker.returncode}")
        ready = [folder / "ready" for folder in outputs]
        if all(marker.is_file() and marker.stat().st_size for marker in ready):
            return [marker.read_text() for marker in ready]
        time.sleep(1)
    raise TimeoutError("Snowball replicas did not finish warmup in 20 minutes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-rounds", type=int, default=16)
    args = parser.parse_args()
    if min(args.concurrency, args.max_rounds) < 1:
        parser.error("--concurrency and --max-rounds must be positive")
    packed_tokens = ((args.concurrency + GPUS_PER_REPLICA - 1) // GPUS_PER_REPLICA) * GPUS_PER_REPLICA
    round_tokens = min(32, packed_tokens)
    decode_buckets = [str(size) for size in (4, 8, 16) if size < round_tokens]
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    if len(devices) != NUM_REPLICAS * GPUS_PER_REPLICA:
        raise RuntimeError(f"Expected exactly eight allocated GPUs, got {len(devices)}")
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    outputs = [output / f"replica-{rank}" for rank in range(NUM_REPLICAS)]
    workers: list[subprocess.Popen[bytes]] = []
    series = load_series(SERIES)
    record = read_record(args.workload)
    try:
        for rank, folder in enumerate(outputs):
            folder.mkdir()
            environment = os.environ | {
                "CUDA_VISIBLE_DEVICES": ",".join(devices[rank * GPUS_PER_REPLICA : (rank + 1) * GPUS_PER_REPLICA]),
                "IRIS_OUTPUT_DIR": str(folder),
            }
            argv = [
                sys.executable,
                "-m",
                "experiments.evaluation.knotmeter.native_snowball_probe",
                "--accelerator",
                f"h100x{GPUS_PER_REPLICA}",
                "--weights",
                args.weights,
                "--ready-file",
                str(folder / "ready"),
                "--concurrency",
                str(args.concurrency),
                "--max-rounds",
                str(args.max_rounds),
                "--max-tokens-per-round",
                str(round_tokens),
                *REPLICA_OPTIONS.split(),
            ]
            if decode_buckets:
                argv.extend(("--decode-buckets", *decode_buckets))
            with (folder / "serving.log").open("wb") as log:
                workers.append(subprocess.Popen(argv, env=environment, stdout=log, stderr=subprocess.STDOUT))

        with tempfile.TemporaryDirectory(prefix="agentx-replicas-") as tmp:
            bundle = Path(tmp) / "bundle"
            restore_workload(record, bundle)
            check_native_cache(bundle, SERIES, output / "workload-check.log")
            backends = wait_ready(workers, outputs)
            sock = bind_serving_socket("127.0.0.1", 0)
            root = f"http://127.0.0.1:{sock.getsockname()[1]}"
            command = benchmark_command(series, args.concurrency, root, output / "aiperf_artifacts")
            metrics_arg = command.index("--server-metrics")
            del command[metrics_arg : metrics_arg + 2]
            request_counts = Counter()
            app = make_proxy(backends, request_counts)
            with serve_app_background(app, sock, name="snowball-native-replicas"):
                run_logged(
                    command,
                    workload_environment(bundle, series),
                    output / "agentx.log",
                    timeout=series["warmup_grace_period"] + series["duration"] + 600,
                    sample=output / "client-load.jsonl",
                )
            (output / "replication.json").write_text(
                json.dumps(
                    {
                        "weights": args.weights,
                        "workload_sha256": record["archive_sha256"],
                        "concurrency": args.concurrency,
                        "replicas": NUM_REPLICAS,
                        "expert_shards_per_replica": GPUS_PER_REPLICA,
                        "max_rounds": args.max_rounds,
                        "requests_per_replica": request_counts,
                    },
                    indent=2,
                )
            )
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=REPLICA_SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()


if __name__ == "__main__":
    main()
