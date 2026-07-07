# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproducer for marin-community/marin#6983: brokered vLLM serving wedge.

Drives one brokered vLLM system with the workload that wedges it: many client
threads, each simulating an incremental decoder — POST /v1/completions with a
growing token-id prompt, max_tokens=1, logprobs=128, and an aggressive client
timeout so some requests are abandoned (the suspected lease/slot leak trigger).

Healthy behavior: throughput stays roughly constant for hours. Wedged behavior
(observed after 1-3h in the field): success rate decays to zero and never
recovers, while `iris task exec` into the worker shows the vLLM engine idle and
serving direct requests instantly.

Run (from repo root, submits one CPU parent + one v5p-8 worker to Iris):

  uv run python -m marin.inference.vllm_wedge_repro --region us-east5

The parent exits 2 when wedged (5 consecutive minutes with zero successes),
0 if the run survives --duration-minutes.
"""

from __future__ import annotations

import random
import threading
import time
from pathlib import Path

import click
import requests
from fray.types import ResourceConfig
from iris.client import IrisClient
from iris.cluster.composer import provider_bundle
from iris.cluster.config import load_config
from iris.cluster.constraints import preemptible_constraint, region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from rigging.log_setup import configure_logging

from marin.inference.types import RunningModel
from marin.inference.vllm import (
    BrokeredVllmSystemConfig,
    InferenceWorkerConfig,
    VllmProxyConfig,
    VllmServerConfig,
    start_iris_brokered_vllm,
)

MODEL = "marin-community/marin-8b-base"
TOP_LOGPROBS = 128
# Aggressive enough that queue spikes abandon requests, the suspected trigger.
CLIENT_TIMEOUT_SECONDS = 60.0
PROMPT_SEED_TOKENS = 32
MAX_CONTEXT_TOKENS = 3_800
WEDGE_MINUTES = 5


class _Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ok = 0
        self.timeouts = 0
        self.http4xx = 0
        self.http5xx = 0
        self.other = 0
        self.first_bad: str | None = None

    def record_bad(self, detail: str) -> None:
        with self.lock:
            if self.first_bad is None:
                self.first_bad = detail

    def snapshot_and_reset(self) -> tuple[int, int, int, int, int]:
        with self.lock:
            snap = (self.ok, self.timeouts, self.http4xx, self.http5xx, self.other)
            self.ok = self.timeouts = self.http4xx = self.http5xx = self.other = 0
        return snap


def _client_loop(base_url: str, model: str, stats: _Stats, stop: threading.Event, seed: int) -> None:
    rng = random.Random(seed)
    session = requests.Session()
    ids = [rng.randrange(1_000, 100_000) for _ in range(PROMPT_SEED_TOKENS)]
    while not stop.is_set():
        try:
            response = session.post(
                f"{base_url.rstrip('/')}/completions",
                json={
                    "model": model,
                    "prompt": ids,
                    "max_tokens": 1,
                    "logprobs": TOP_LOGPROBS,
                    "return_tokens_as_token_ids": True,
                },
                timeout=CLIENT_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            top = response.json()["choices"][0]["logprobs"]["top_logprobs"][0]
            ids.append(int(next(iter(top)).removeprefix("token_id:")))
            with stats.lock:
                stats.ok += 1
        except requests.Timeout:
            with stats.lock:
                stats.timeouts += 1
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            body = exc.response.text[:200] if exc.response is not None else ""
            stats.record_bad(f"status={status} prompt_len={len(ids)} body={body!r}")
            with stats.lock:
                if 400 <= status < 500:
                    stats.http4xx += 1
                else:
                    stats.http5xx += 1
        except Exception as exc:
            stats.record_bad(f"non-http error prompt_len={len(ids)}: {exc!r:.200}")
            with stats.lock:
                stats.other += 1
            time.sleep(1)
        if len(ids) >= MAX_CONTEXT_TOKENS:
            ids = ids[:PROMPT_SEED_TOKENS]


def drive_load(running_model: RunningModel, *, clients: int, duration_minutes: int) -> bool:
    """Return True if the system wedged (WEDGE_MINUTES minutes with zero successes)."""
    stats = _Stats()
    stop = threading.Event()
    threads = [
        threading.Thread(
            target=_client_loop,
            args=(running_model.endpoint.base_url, running_model.endpoint.model, stats, stop, i),
            daemon=True,
        )
        for i in range(clients)
    ]
    for thread in threads:
        thread.start()

    dead_minutes = 0
    try:
        for minute in range(duration_minutes):
            time.sleep(60)
            ok, timeouts, http4xx, http5xx, other = stats.snapshot_and_reset()
            print(
                f"minute={minute + 1} ok={ok} timeouts={timeouts} 4xx={http4xx} 5xx={http5xx} other={other}",
                flush=True,
            )
            if ok > 0:
                dead_minutes = 0
                continue
            if http4xx > timeouts:
                # Fast application-level rejections are a different bug (e.g.
                # context past the model limit), not brokered-serving death.
                print(f"CONFOUNDED: zero successes but 4xx-dominated; first bad: {stats.first_bad}", flush=True)
                raise SystemExit(3)
            dead_minutes += 1
            if dead_minutes >= WEDGE_MINUTES:
                print(f"WEDGED: {WEDGE_MINUTES} consecutive minutes with zero successes", flush=True)
                print(f"first bad response: {stats.first_bad}", flush=True)
                return True
        print("survived without wedging", flush=True)
        return False
    finally:
        stop.set()


@click.command(help=__doc__, context_settings={"help_option_names": ["-h", "--help"], "show_default": True})
@click.option("--clients", type=click.IntRange(min=1), default=24, help="Concurrent incremental-decoder clients.")
@click.option("--duration-minutes", type=click.IntRange(min=1), default=360, help="Give up after this long.")
@click.option("--tpu-type", default="v5p-8", help="TPU type for the vLLM worker.")
@click.option("--region", default="us-east5", help="Region for parent and worker jobs.")
@click.option("--job-name", default="brokered-vllm-wedge-repro", help="Iris parent job name.")
def main(clients: int, duration_minutes: int, tpu_type: str, region: str, job_name: str) -> None:
    configure_logging()
    config = BrokeredVllmSystemConfig(
        model=MODEL,
        tokenizer=MODEL,
        server=VllmServerConfig(
            timeout_seconds=7200,
            max_model_len=8192,
            max_num_seqs=16,
            max_num_batched_tokens=2048,
            max_logprobs=TOP_LOGPROBS,
            enable_prefix_caching=True,
        ),
        # First boot (slice provisioning + download + TPU compile) can pass an hour.
        proxy=VllmProxyConfig(readiness_timeout_seconds=7200),
        workers=InferenceWorkerConfig(count=1, max_in_flight_per_worker=2 * clients),
        worker_resources=ResourceConfig.with_tpu(tpu_type, ram="96g"),
        worker_env_vars={
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
            "VLLM_TPU_SKIP_PRECOMPILE": "1",
        },
    )

    def run_parent() -> None:
        configure_logging()
        with start_iris_brokered_vllm(config) as running_model:
            if drive_load(running_model, clients=clients, duration_minutes=duration_minutes):
                raise SystemExit(2)

    iris_config = load_config("lib/iris/config/marin.yaml")
    controller = provider_bundle(iris_config).controller
    controller_address = iris_config.controller_address() or controller.discover_controller(iris_config.controller)
    with controller.tunnel(address=controller_address) as controller_url:
        with IrisClient.remote(controller_url, workspace=Path.cwd()) as client:
            job = client.submit(
                entrypoint=Entrypoint.from_callable(run_parent),
                name=job_name,
                resources=ResourceSpec(cpu=1.0, memory="8g", disk="16g"),
                environment=EnvironmentSpec(env_vars={}),
                constraints=[preemptible_constraint(False), region_constraint([region])],
            )
            print(f"Submitted Iris parent job {job.job_id}", flush=True)
            job.wait(timeout=float("inf"))


if __name__ == "__main__":
    main()
