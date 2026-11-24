# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bisect pool 502/500s: concurrent large-prompt requests from a pod, capture failure bodies.

Writes the result matrix to in-region S3 (short-job stdout is not reliably retrievable).
"""

import argparse
import json
import threading

import httpx
from iris.client.client import iris_ctx
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.glm_synthesis_serving import pool

_RESULT_PATH = "s3://marin-us-east-08a/marin/users/benfeuer/glm-synthesis-serving/failure-probe.json"
_UNIT = "tool trace line with json payloads and shell commands "


def _body(seed: int, prompt_tokens: int = 48_000) -> dict:
    needed = prompt_tokens * 4
    filler = ((f"row-{seed} " + _UNIT) * (needed // len(_UNIT) + 1))[:needed]
    return {
        "model": "glm-5.3",
        "cache_salt": "failure-probe",
        "messages": [
            {"role": "system", "content": "You convert tool-harness traces into native tool calls. " * 60},
            {"role": "user", "content": filler},
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "thinking_token_budget": 8192,
        "max_tokens": 256,
        "stream": True,
        "tool_choice": "none",
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": "low", "clear_thinking": True},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-name", default=None, help="Target endpoint (default: the pool's).")
    args = parser.parse_args()
    endpoint = args.endpoint_name or pool.ENDPOINT_NAME
    client = iris_ctx().client
    resolved = client.resolve_endpoint(endpoint).rstrip("/")
    jwt = client.mint_endpoint_token(endpoint).token
    headers = {"Authorization": f"Bearer {jwt}"}
    proxied_name = endpoint.lstrip("/").replace("/", ".")
    urls = {
        "proxied": f"http://iris-controller-svc:10000/proxy/{proxied_name}/v1/chat/completions",
        "direct": resolved + "/v1/chat/completions",
    }
    results: dict[str, dict] = {}
    for name, url in urls.items():
        outcomes: list[dict] = []
        lock = threading.Lock()

        def worker(seed: int, target: str, output: list[dict], output_lock: threading.Lock) -> None:
            outcome: dict = {}
            try:
                with httpx.Client(timeout=600.0) as http:
                    with http.stream("POST", target, json=_body(seed), headers=headers) as response:
                        if response.status_code == 200:
                            chunks = sum(1 for line in response.iter_lines() if line.startswith("data:"))
                            outcome = {"status": 200, "chunks": chunks}
                        else:
                            outcome = {
                                "status": response.status_code,
                                "body": response.read()[:600].decode(errors="replace"),
                            }
            except Exception as error:
                outcome = {"error": f"{type(error).__name__}: {error}"[:240]}
            with output_lock:
                output.append({"seed": seed, **outcome})

        threads = [threading.Thread(target=worker, args=(seed, url, outcomes, lock)) for seed in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        ok = sum(1 for o in outcomes if o.get("status") == 200)
        results[name] = {"ok": ok, "of": len(outcomes), "failures": [o for o in outcomes if o.get("status") != 200][:6]}
    payload = {"resolved": resolved, **results}
    configure_coreweave_s3()

    fs, _ = filesystem_for(_RESULT_PATH)
    with fs.open(_RESULT_PATH, "w") as stream:
        stream.write(json.dumps(payload))
    print(json.dumps({k: {"ok": v["ok"], "of": v["of"]} for k, v in results.items()}))


if __name__ == "__main__":
    main()
