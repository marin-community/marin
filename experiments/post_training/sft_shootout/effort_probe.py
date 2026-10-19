# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check how the GLM 5.3 relay takes a reasoning effort setting.

GLM 5.3's chat template renders ``Reasoning Effort: <Low|High|Max>`` from the template variable
``reasoning_effort`` (``low`` or ``high``; anything else, or nothing, renders ``Max``). For each way a
client could send the setting, this asks the relay's ``/tokenize`` route for the rendered prompt and
prints the effort line it holds, then samples a few chat completions and prints their reasoning length.
Prints shapes and usage only; the bearer token comes from ``GLM_BULK_TOKEN`` and is never printed.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a --cpu 1 --memory 4GB \
      --job-name glm-effort-probe -e GLM_BULK_TOKEN "$(gcloud secrets versions access latest \
      --secret GLM_BULK_TOKEN --project hai-gcp-models)" \
      -- uv run python experiments/post_training/sft_shootout/effort_probe.py
"""

import json
import os
import re
import statistics
from concurrent.futures import ThreadPoolExecutor

import httpx

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_BULK_TOKEN_ENV, GLM_MODEL, resolve_glm_base_url

VARIANTS = {
    "default": {},
    "kwargs-low": {"chat_template_kwargs": {"reasoning_effort": "low"}},
    "kwargs-medium": {"chat_template_kwargs": {"reasoning_effort": "medium"}},
    "kwargs-high": {"chat_template_kwargs": {"reasoning_effort": "high"}},
    "top-low": {"reasoning_effort": "low"},
    "top-medium": {"reasoning_effort": "medium"},
    "top-high": {"reasoning_effort": "high"},
}
PROMPT = [
    {
        "role": "user",
        "content": "How many positive integers n <= 2000 make n^2 + n + 41 composite? Answer with the number only.",
    }
]
SAMPLES = 4
EFFORT_LINE = re.compile(r"Reasoning Effort: *\w+")


def rendered_effort(client: httpx.Client, root: str, extra: dict) -> str:
    response = client.post(
        f"{root}/tokenize", json={"model": GLM_MODEL, "messages": PROMPT, "return_token_strs": True, **extra}
    )
    if response.status_code != 200:
        return f"HTTP {response.status_code} {response.text[:300]}"
    payload = response.json()
    text = "".join(payload.get("token_strs") or [])
    match = EFFORT_LINE.search(text)
    return f"{match.group(0) if match else 'no effort line'} (prompt tokens {payload.get('count')})"


def sample(client: httpx.Client, base: str, extra: dict) -> dict:
    response = client.post(
        f"{base}/chat/completions", json={"model": GLM_MODEL, "messages": PROMPT, "max_tokens": 16384, **extra}
    )
    if response.status_code != 200:
        return {"status": response.status_code, "error": response.text[:300]}
    payload = response.json()
    message = payload["choices"][0]["message"]
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    return {
        "status": 200,
        "completion_tokens": payload["usage"]["completion_tokens"],
        "reasoning_tokens": (payload["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "reasoning_chars": len(reasoning),
        "answer": (message.get("content") or "").strip()[:40],
    }


def main() -> None:
    base = resolve_glm_base_url(DEFAULT_GLM_RELAY_JOB)
    root = base.removesuffix("/v1")
    headers = {"Authorization": f"Bearer {os.environ[GLM_BULK_TOKEN_ENV]}"}
    with httpx.Client(timeout=900, headers=headers) as client:
        for name, extra in VARIANTS.items():
            print(f"{name} sends {json.dumps(extra)}: {rendered_effort(client, root, extra)}", flush=True)
        with ThreadPoolExecutor(len(VARIANTS) * SAMPLES) as pool:
            futures = {
                name: [pool.submit(sample, client, base, extra) for _ in range(SAMPLES)]
                for name, extra in VARIANTS.items()
            }
            for name, runs in futures.items():
                results = [run.result() for run in runs]
                tokens = [r["completion_tokens"] for r in results if r["status"] == 200]
                median = statistics.median(tokens) if tokens else None
                print(
                    f"{name}: completion tokens median {median} {json.dumps(results)}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
