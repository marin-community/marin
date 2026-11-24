# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bisect pool 502s: alternate proxied vs direct URLs from a pod, capture failure bodies.

Writes the result matrix to in-region S3 (short-job stdout is not reliably retrievable).
"""

import json

import httpx

from iris.client.client import iris_ctx

_RESULT_PATH = "s3://marin-us-east-08a/marin/users/benfeuer/glm-synthesis-serving/failure-probe.json"
_BODY = {
    "model": "glm-5.3",
    "cache_salt": "failure-probe",
    "messages": [
        {"role": "system", "content": "You convert tool-harness traces into native tool calls."},
        {"role": "user", "content": "Convert this bash call to a native tool call: echo hello"},
    ],
    "temperature": 1.0,
    "top_p": 0.95,
    "thinking_token_budget": 8192,
    "max_tokens": 64,
    "stream": True,
    "tool_choice": "none",
    "stream_options": {"include_usage": True},
    "chat_template_kwargs": {"reasoning_effort": "low", "clear_thinking": True},
}


def main() -> None:
    client = iris_ctx().client
    resolved = client.resolve_endpoint("/serve/glm-5-3-flash").rstrip("/")
    jwt = client.mint_endpoint_token("/serve/glm-5-3-flash").token
    headers = {"Authorization": f"Bearer {jwt}"}
    urls = {
        "proxied": "http://iris-controller-svc:10000/proxy/serve.glm-5-3-flash/v1/chat/completions",
        "direct": resolved + "/v1/chat/completions",
    }
    results: dict[str, dict] = {}
    for name, url in urls.items():
        outcomes = []
        with httpx.Client(timeout=120.0) as http:
            for attempt in range(10):
                try:
                    with http.stream("POST", url, json=_BODY, headers=headers) as response:
                        if response.status_code == 200:
                            chunks = 0
                            for line in response.iter_lines():
                                if line.startswith("data:"):
                                    chunks += 1
                            outcomes.append({"status": 200, "chunks": chunks})
                        else:
                            body = response.read()[:400].decode(errors="replace")
                            outcomes.append({"status": response.status_code, "body": body})
                except Exception as error:
                    outcomes.append({"error": str(error)[:200]})
        results[name] = {
            "ok": sum(1 for o in outcomes if o.get("status") == 200),
            "outcomes": outcomes,
        }
    payload = {"resolved": resolved, **results}
    from rigging.filesystem.s3_compat import configure_coreweave_s3

    configure_coreweave_s3()
    from rigging.filesystem.buckets import filesystem_for

    fs, _ = filesystem_for(_RESULT_PATH)
    with fs.open(_RESULT_PATH, "w") as stream:
        stream.write(json.dumps(payload))
    print(json.dumps({k: {"ok": v["ok"]} for k, v in results.items()}))


if __name__ == "__main__":
    main()
