# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe the GLM 5.3 relays reachable from this cluster.

For each relay job: resolve its endpoint, list models, try the vLLM ``/tokenize`` route that
Harbor's Mini-SWE context limiter needs, and send one short chat completion. Prints response
shapes and usage only; the bearer token comes from ``GLM_BULK_TOKEN`` and is never printed.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a --cpu 1 --memory 4GB \
      --job-name glm-probe-207 -e GLM_BULK_TOKEN "$(gcloud secrets versions access latest \
      --secret GLM_BULK_TOKEN --project hai-gcp-models)" \
      -- uv run python experiments/post_training/sft_shootout/glm_probe.py
"""

import json
import os
import sys
import time

import httpx
from iris.client.client import iris_ctx
from iris.cluster.types import JobName

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL

RELAY_JOBS = sys.argv[1:] or [
    "/muchanem/glm53-relay",
    "/muchanem/glm53-relay-e",
    "/muchanem/glm53-relay-02a",
    "/muchanem/glm53-relay-08a",
    "/muchanem/glm53-relay-rno2a",
]


def show(label: str, response: httpx.Response) -> None:
    body = response.text
    print(f"  {label}: HTTP {response.status_code} {body[:1500]}")


def probe(client: httpx.Client, base: str) -> None:
    root = base.removesuffix("/v1")
    show("GET /v1/models", client.get(f"{base}/models"))
    messages = [{"role": "user", "content": "Reply with the single word: ready"}]
    for path in (f"{root}/tokenize", f"{base}/tokenize"):
        show(f"POST {path.removeprefix(root)}", client.post(path, json={"model": GLM_MODEL, "messages": messages}))
    started = time.monotonic()
    response = client.post(
        f"{base}/chat/completions",
        json={"model": GLM_MODEL, "messages": messages, "max_tokens": 2048},
    )
    print(f"  chat latency {time.monotonic() - started:.1f}s")
    show("POST /v1/chat/completions", response)
    print(f"  response headers: {json.dumps({k: v for k, v in response.headers.items() if 'auth' not in k.lower()})}")


def main() -> None:
    token = os.environ[GLM_BULK_TOKEN_ENV]
    iris = iris_ctx().client
    with httpx.Client(timeout=300, headers={"Authorization": f"Bearer {token}"}) as client:
        for job in RELAY_JOBS:
            print(f"== {job}")
            try:
                endpoints = iris.resolver_for_job(JobName.from_string(job)).resolve(GLM_MODEL).endpoints
            except Exception as error:
                print(f"  resolve failed: {type(error).__name__}: {error}")
                continue
            print(f"  endpoints: {[e.url for e in endpoints]}")
            for endpoint in endpoints[:1]:
                base = endpoint.url.rstrip("/")
                base = base if base.endswith("/v1") else f"{base}/v1"
                try:
                    probe(client, base)
                except Exception as error:
                    print(f"  probe failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
