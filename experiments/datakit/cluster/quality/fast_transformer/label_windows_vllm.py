# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Label bme grading windows with a brokered pool of self-hosted GLM-5.2 gangs.

The grading contract is :mod:`label_windows_openrouter`'s exactly — one window
per request as ``<document index="0">``, the v2 rubric system prompt, the
bracketed position notice on middle/end windows and the excerpt marker on a
begin window whose document continues, temperature 0, one-JSON-object verdict —
with the serving path swapped from OpenRouter to a self-hosted pool.

The pool follows the brokered-fleet shape of the OCR extraction route rather
than static work partitioning. This driver runs as an Iris job and owns three
things:

* an :class:`~marin.inference.broker.InferenceBroker` actor holding the leased
  request queue;
* ``--num-gangs`` two-node GLM-5.2-FP8 gangs
  (:mod:`experiments.datakit.cluster.quality.glm52_vllm`), each launched in
  pull mode: the gang's head runs an ``InferenceWorker`` that leases requests
  from the broker and forwards them to its local vLLM;
* the :func:`~marin.inference.proxy.serve_inference_proxy` OpenAI endpoint,
  served in-process, that the labeling loop posts windows to.

Because capacity is whoever is pulling, the pool is elastic where a partition
plan is brittle: labeling starts when the *first* gang is ready, gangs admitted
late simply join, a dead gang sheds throughput instead of stranding its share
of the windows, and scaling a tranche is ``--num-gangs``, not a re-shard.
Failed requests are retried on later runs (chunk resume), and a chunk that
mostly fails still aborts the run — with a broker in front, that now means the
whole pool is unhealthy rather than one server.

Checkpointing is unchanged: chunks under ``<out>.chunks/`` resumed by
``(id, window)``, so tranches and reruns never relabel a window, and a run
whose window set is fully checkpointed writes the final parquet without
bringing up any GPU at all.
"""

import argparse
import json
import logging
import math
import os
import random
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import httpx
import pyarrow.parquet as pq
from fray.current_client import current_client
from fray.types import ActorConfig, ResourceConfig
from iris.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.rpc import job_pb2
from marin.inference.broker import InferenceBroker
from marin.inference.proxy import serve_inference_proxy
from openai import OpenAI
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3, configure_fsspec_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.label_windows_openrouter import (
    DEFAULT_LABEL_BATCH,
    _chunk_dir,
    _labeled_keys,
    _read_windows,
    consolidate_chunks,
    label_with_checkpoints,
    window_key,
    window_user_content,
)
from experiments.datakit.cluster.quality.fast_transformer.label_with_glm52 import (
    FLEETS,
    MAX_LOGGED_REJECTS,
    MAX_OUTPUT_TOKENS,
    RAY_ENDPOINT,
    VLLM_ENDPOINT,
    _parse_verdict,
)
from experiments.datakit.cluster.quality.fast_transformer.rubric import SYSTEM_PROMPT
from experiments.datakit.cluster.quality.glm52_vllm import (
    MODEL,
    WEIGHTS_SEED_ENDPOINT,
    BrokerWorkerConfig,
    Glm52LaunchConfig,
    ServerConfig,
    submit_glm52,
)

logger = logging.getLogger(__name__)

# A 512-gemma-token window plus the system prompt stays well under 4k GLM tokens,
# and the 4k reasoning budget sits on top: 8k holds both with headroom while
# giving each sequence half the KV cache of the whole-document labeler's 16k.
DEFAULT_MAX_MODEL_LEN = 8_192
# 192 sequences measured 3.96 docs/s on this fleet at 10.5k-char prompts; windows
# are ~5x shorter, so 192 keeps the queue full without starving KV cache.
DEFAULT_MAX_NUM_SEQS = 192
DEFAULT_NUM_GANGS = 8
# Requests each gang's pull worker keeps in flight against its vLLM. Matching
# max_num_seqs keeps every engine slot fed without stacking a latency queue on
# the server side; the broker holds the real backlog.
DEFAULT_GANG_IN_FLIGHT = DEFAULT_MAX_NUM_SEQS
# Windows per checkpoint. The pool's offered concurrency is gangs x in-flight
# (1,536 at the defaults), so chunks several times that keep the pipeline full
# while a preemption still only costs a few minutes of fleet time.
DEFAULT_CHUNK_SIZE = 8_000

# Bounds on a hung request, not load shedding; far above the tens of seconds a
# window verdict takes. BrokerConfig's ordering rule applies: worker < lease <
# proxy, so a request that dies at any hop re-enters the queue before the hop
# above it gives up on it.
WORKER_REQUEST_TIMEOUT = 900.0
LEASE_TIMEOUT = 1020.0
PROXY_REQUEST_TIMEOUT = 1140.0
# Weight-load budget: a gang streams 756 GB before it can serve, and admission
# of the first gang can queue behind the cluster. Same window the direct-serve
# path allows its endpoint.
POOL_READY_TIMEOUT = 3 * 3600.0
BROKER_READY_TIMEOUT = 900.0
# In-flight payloads are a few KB of window text each, so the broker needs no
# per-fleet memory sizing the way the image-carrying OCR broker does.
BROKER_RESOURCES = ResourceConfig.with_cpu(cpu=2, ram="8g", disk="20g", preemptible=False)
# Ceiling on the driver's sender threads. Past this the engines run below their
# per-gang in-flight budget at full fleet, which costs a little throughput; a
# single Python process holding many thousands of mostly-blocked threads costs
# more. The broker's queue, not the sender count, is what keeps engines fed.
DRIVER_MAX_CONCURRENCY = 4_096
# Gangs per submission wave. Weight streaming measured ~0.5 GB/s per gang alone
# and ~6-7 GB/s aggregate across the shared cross-region path, so a wave of 12
# saturates the link without the all-at-once herd that left a 24-gang fleet
# collectively unloaded for an hour. The next wave launches when half the
# submitted gangs serve, or on timeout — whichever comes first.
DEFAULT_WAVE_SIZE = 12
WAVE_READY_FRACTION = 0.5
WAVE_TIMEOUT = 2_400.0
WAVE_POLL = 60.0


def _expand_globs(patterns: Sequence[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        if "*" not in pattern:
            paths.append(pattern)
            continue
        matches = sorted(str(m) for m in StoragePath(pattern).glob())
        if not matches:
            raise ValueError(f"no parquet files match {pattern}")
        paths.extend(matches)
    return paths


def label_window(client: OpenAI, row: dict, rejects: list[str]) -> dict | None:
    """One window's verdict plus its token usage, or ``None`` if unusable."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": window_user_content(row)},
            ],
        )
    except Exception as failure:  # one bad request must not abort the run
        logger.warning("label_windows_vllm: request failed for %s: %r", window_key(row), failure)
        return None
    choice = response.choices[0]
    content = choice.message.content or ""
    verdict = _parse_verdict(content)
    if verdict is None:
        if len(rejects) < MAX_LOGGED_REJECTS:
            rejects.append(f"finish_reason={choice.finish_reason} reply={content[-400:]!r}")
        return None
    usage = response.usage
    return verdict | {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }


def label_rows(
    client: OpenAI, rows: Sequence[dict], concurrency: int, label_batch: str, rejects: list[str]
) -> list[dict]:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        verdicts = list(pool.map(lambda r: label_window(client, r, rejects), rows))
    return [
        {
            "id": row["id"],
            "source": row["source"],
            "window": row["window"],
            "token_start": row["token_start"],
            "token_end": row["token_end"],
            "doc_tokens": row["doc_tokens"],
            "text": row["text"],
            "content_type": verdict["content_type"],
            "valid": verdict["valid"],
            "quality": verdict["quality"],
            "score_normalized": (verdict["quality"] - 1) / 4.0,
            "why": verdict["why"],
            "label_batch": label_batch,
            "prompt_tokens": verdict["prompt_tokens"],
            "completion_tokens": verdict["completion_tokens"],
        }
        for row, verdict in zip(rows, verdicts, strict=True)
        if verdict is not None
    ]


def _label_through_pool(
    ctx,
    rows: Sequence[dict],
    *,
    out: str,
    server: ServerConfig,
    run_id: str,
    num_gangs: int,
    interactive_gangs: int,
    gang_in_flight: int,
    wave_size: int,
    chunk_size: int,
    label_batch: str,
    fleet_name: str,
    object_store_endpoint: str | None,
):
    """Bring up broker, gangs, and proxy; label ``rows``; tear everything down."""
    job_info = get_job_info()
    assert job_info is not None

    broker_group = current_client().create_actor_group(
        InferenceBroker,
        name=f"glm52-broker-{run_id}",
        count=1,
        request_lease_timeout_seconds=LEASE_TIMEOUT,
        resources=BROKER_RESOURCES,
        actor_config=ActorConfig(max_task_retries=0, priority=job_pb2.PRIORITY_BAND_INTERACTIVE),
    )
    gang_jobs = []
    jobs_lock = threading.Lock()
    stop_waves = threading.Event()
    try:
        broker = broker_group.wait_ready(count=1, timeout=BROKER_READY_TIMEOUT)[0]

        def submit_gang(gang: int) -> None:
            # The first ``interactive_gangs`` gangs guarantee forward progress;
            # the rest ride the batch band, joining the pool whenever the
            # scheduler admits them and rejoining after preemption (gang
            # submission leaves iris's generous preemption retries in place).
            band = job_pb2.PRIORITY_BAND_INTERACTIVE if gang < interactive_gangs else job_pb2.PRIORITY_BAND_BATCH
            launch = Glm52LaunchConfig(
                f"{VLLM_ENDPOINT}-{run_id}-g{gang}",
                f"{RAY_ENDPOINT}-{run_id}-g{gang}",
                server,
                fleet=FLEETS[fleet_name],
                object_store_endpoint=object_store_endpoint,
                priority_band=band,
                broker_worker=BrokerWorkerConfig(
                    broker=broker, max_in_flight=gang_in_flight, request_timeout=WORKER_REQUEST_TIMEOUT
                ),
                # The first wave's simultaneous background fills form the seed
                # swarm (their per-file trades keep the cross-region cost near
                # one snapshot); later waves fill through the swarm regardless.
                fill_weights_cache=gang < wave_size,
            )
            job = submit_glm52(ctx, launch, name=f"vllm-g{gang}", max_retries_failure=5)
            with jobs_lock:
                gang_jobs.append(job)

        def endpoint_count(name: str) -> int:
            try:
                result = ctx.resolver.resolve(name)
            except Exception:
                return 0
            return 0 if result.is_empty else len(result.endpoints)

        def serving_count(submitted: int) -> int:
            return sum(1 for g in range(submitted) if endpoint_count(f"{VLLM_ENDPOINT}-{run_id}-g{g}") > 0)

        def submit_waves() -> None:
            # Wave 1 saturates the shared cross-region path; each further wave
            # launches once the fleet shows capacity to absorb it — half the
            # submitted gangs serving, two complete seeds (which make later
            # bring-ups a datacenter copy), or the timeout, whichever is first.
            submitted = 0
            while submitted < num_gangs and not stop_waves.is_set():
                size = min(wave_size, num_gangs - submitted)
                for gang in range(submitted, submitted + size):
                    submit_gang(gang)
                submitted += size
                logger.info("label_windows_vllm: submitted %d/%d gangs", submitted, num_gangs)
                if submitted >= num_gangs:
                    return
                deadline = time.monotonic() + WAVE_TIMEOUT
                needed = max(1, math.ceil(submitted * WAVE_READY_FRACTION))
                while not stop_waves.is_set() and time.monotonic() < deadline:
                    if serving_count(submitted) >= needed or endpoint_count(WEIGHTS_SEED_ENDPOINT) >= 2:
                        break
                    stop_waves.wait(WAVE_POLL)

        waves = threading.Thread(target=submit_waves, name="gang-waves", daemon=True)
        waves.start()

        concurrency = min(num_gangs * gang_in_flight, DRIVER_MAX_CONCURRENCY)
        with serve_inference_proxy(
            broker=broker,
            model=MODEL,
            host=job_info.advertise_host,
            port=0,
            request_timeout_seconds=PROXY_REQUEST_TIMEOUT,
            readiness_timeout_seconds=POOL_READY_TIMEOUT,
            max_pending_requests=concurrency * 2,
            response_fetch_batch_size=256,
            server_start_timeout_seconds=60.0,
        ) as running_model:
            # Blocks until the first gang registers and serves; later gangs join live.
            readiness = httpx.get(running_model.endpoint.url("models"), timeout=POOL_READY_TIMEOUT)
            readiness.raise_for_status()
            logger.info("label_windows_vllm: pool serving at %s", running_model.endpoint.base_url)
            # httpx defaults to 100 pooled connections, which silently caps
            # in-flight requests below the sender's thread count (the OCR
            # campaign measured engines pinned at ~99 running requests until
            # the pool was sized to the concurrency).
            limits = httpx.Limits(max_connections=concurrency + 64, max_keepalive_connections=concurrency)
            client = OpenAI(
                base_url=running_model.endpoint.base_url,
                api_key="EMPTY",
                timeout=PROXY_REQUEST_TIMEOUT,
                http_client=httpx.Client(limits=limits, timeout=PROXY_REQUEST_TIMEOUT),
            )

            def label_chunk(batch: Sequence[dict], rejects: list[str]) -> list[dict]:
                return label_rows(client, batch, concurrency, label_batch, rejects)

            return label_with_checkpoints(label_chunk, rows, chunk_size=chunk_size, out=out)
    finally:
        # Never let a teardown failure mask the real error, but never leave a
        # GPU gang or the broker running either.
        stop_waves.set()
        with jobs_lock:
            remaining = list(gang_jobs)
        for job in remaining:
            try:
                job.terminate()
            except Exception:
                logger.warning("label_windows_vllm: failed to terminate a GLM-5.2 gang", exc_info=True)
        try:
            broker_group.shutdown()
        except Exception:
            logger.warning("label_windows_vllm: failed to shut down the broker actor", exc_info=True)


def run(
    *,
    windows: list[str],
    out: str,
    server: ServerConfig,
    run_id: str,
    num_gangs: int,
    interactive_gangs: int,
    gang_in_flight: int,
    wave_size: int,
    chunk_size: int,
    label_batch: str,
    limit: int | None,
    fleet_name: str,
    object_store_endpoint: str | None,
) -> None:
    ctx = iris_ctx()
    if ctx is None or ctx.client is None:
        raise RuntimeError("window labeling must run inside an Iris job so it can submit the GLM-5.2 pool")

    rows = _read_windows(_expand_globs(windows))
    # A stable shuffle so --limit draws a representative smoke rather than one
    # shard's head, and so a run's chunks mix sources evenly.
    random.Random(0).shuffle(rows)
    if limit:
        rows = rows[:limit]
    logger.info("label_windows_vllm: %d windows in the set", len(rows))

    # Hoisted: the comprehension's filter runs per row, and _labeled_keys is an
    # S3 glob plus a read of every checkpoint chunk.
    already = _labeled_keys(_chunk_dir(out))
    pending = [r for r in rows if window_key(r) not in already]
    if not pending:
        logger.info("label_windows_vllm: nothing pending, consolidating existing checkpoints")
        table = consolidate_chunks(_chunk_dir(out))
    else:
        logger.info("label_windows_vllm: %d windows pending", len(pending))
        table = _label_through_pool(
            ctx,
            rows,
            out=out,
            server=server,
            run_id=run_id,
            num_gangs=num_gangs,
            interactive_gangs=interactive_gangs,
            gang_in_flight=gang_in_flight,
            wave_size=wave_size,
            chunk_size=chunk_size,
            label_batch=label_batch,
            fleet_name=fleet_name,
            object_store_endpoint=object_store_endpoint,
        )

    with StoragePath(out).open("wb") as fh:
        pq.write_table(table, fh)
    stats = {
        "attempted": len(rows),
        "labeled": table.num_rows,
        "dropped": len(rows) - table.num_rows,
        "prompt_tokens": sum(table.column("prompt_tokens").to_pylist()) if table.num_rows else 0,
        "completion_tokens": sum(table.column("completion_tokens").to_pylist()) if table.num_rows else 0,
    }
    with StoragePath(f"{out.removesuffix('.parquet')}_stats.json").open("w") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("label_windows_vllm: %s", json.dumps(stats))
    logger.info("label_windows_vllm: wrote %d labels to %s", table.num_rows, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True, nargs="+", help="windows parquet path(s) or glob(s)")
    parser.add_argument("--out", required=True, help="labels parquet path; checkpoints live under <out>.chunks/")
    parser.add_argument("--run-id", required=True, help="unique tag for this run's broker and gang endpoints")
    parser.add_argument("--num-gangs", type=int, default=DEFAULT_NUM_GANGS, help="GLM-5.2 gangs in the pool")
    parser.add_argument(
        "--interactive-gangs",
        type=int,
        default=None,
        help=(
            "how many gangs run at interactive priority (the rest ride the batch band and join "
            "as admitted); default: all of them"
        ),
    )
    parser.add_argument(
        "--gang-in-flight",
        type=int,
        default=DEFAULT_GANG_IN_FLIGHT,
        help="requests each gang's pull worker keeps in flight",
    )
    parser.add_argument(
        "--wave-size",
        type=int,
        default=DEFAULT_WAVE_SIZE,
        help="gangs submitted per wave; sized to saturate the shared cross-region path without a herd",
    )
    parser.add_argument("--label-batch", default=DEFAULT_LABEL_BATCH)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="windows per checkpoint")
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    parser.add_argument("--limit", type=int, default=None, help="label only the first N windows (smoke runs)")
    # gb200 is the shape this stack is verified on; the H100 path currently fails
    # in FlashInfer's SM90 JIT link.
    parser.add_argument("--fleet", choices=sorted(FLEETS), default="gb200", help="GPU shape to serve on")
    parser.add_argument(
        "--object-store-endpoint",
        default=None,
        help=(
            "S3 endpoint for both this driver and the serving gangs (e.g. https://cwobject.com). "
            "Required when the windows, labels, or weight cache live in a different region than "
            "the GPUs: the pod's node-local LOTA endpoint cannot read cross-region buckets."
        ),
    )
    args = parser.parse_args()
    configure_logging(logging.INFO)
    if args.object_store_endpoint:
        # The pod arrives with FSSPEC_S3/AWS_ENDPOINT_URL pointing at its
        # node-local LOTA endpoint; rebuild the process-wide config against the
        # external endpoint so cross-region reads and writes work.
        os.environ.pop("FSSPEC_S3", None)
        os.environ.pop("AWS_ENDPOINT_URL", None)
        configure_fsspec_s3(args.object_store_endpoint)
    else:
        configure_coreweave_s3()
    run(
        windows=args.windows,
        out=args.out,
        server=ServerConfig(max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs),
        run_id=args.run_id,
        num_gangs=args.num_gangs,
        interactive_gangs=args.interactive_gangs if args.interactive_gangs is not None else args.num_gangs,
        gang_in_flight=args.gang_in_flight,
        wave_size=args.wave_size,
        chunk_size=args.chunk_size,
        label_batch=args.label_batch,
        limit=args.limit,
        fleet_name=args.fleet,
        object_store_endpoint=args.object_store_endpoint,
    )


if __name__ == "__main__":
    main()
