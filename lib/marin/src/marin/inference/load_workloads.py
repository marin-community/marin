# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build fixed text workloads and recover prompt prefixes from retained evaluations."""

import random
from collections.abc import Sequence

from marin.inference.load_test import LoadRequest

ROLLOUT_CONTEXT = (
    "Inspect the repository and repair the reported failure. Preserve the public API. "
    "Use the logs and tests as evidence, explain the cause, and provide a minimal patch.\n"
    "def dispatch(task, workers):\n    candidates = [w for w in workers if w.ready]\n"
    "    return min(candidates, key=lambda w: w.queue_depth)\n"
    "Tool result: test_retry_preserves_task failed: expected one completed attempt, got zero.\n"
    "Log: allocated worker is healthy; task preparation succeeded; response stream closed before grading.\n"
)


def synthetic_requests(*, seed: int, context_repetitions: int, output_tokens: int, count: int) -> list[LoadRequest]:
    """Use identical rollout-shaped text across model tokenizers; record actual usage at serving time."""
    rng = random.Random(seed)
    return [
        LoadRequest(
            identity=f"synthetic-{seed}-{index}",
            path="chat/completions",
            body={
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Task {rng.randrange(1_000_000)}.\n"
                            + ROLLOUT_CONTEXT * context_repetitions
                            + "\nContinue the investigation. State the next tool action and its expected evidence."
                        ),
                    }
                ],
                "max_tokens": output_tokens,
                "temperature": 0.7,
                "top_p": 1.0,
                "ignore_eos": True,
            },
        )
        for index in range(count)
    ]


def replay_requests(rows: Sequence[dict], *, output_tokens: int) -> list[LoadRequest]:
    """Recover the final generated turn's prompt; historical assistant messages remain context."""
    requests = []
    for row_index, row in enumerate(rows):
        if "body" in row and "path" in row:
            body = {**row["body"], "max_tokens": output_tokens}
            for name in ("model", "stream", "stream_options", "seed"):
                body.pop(name, None)
            path = row["path"].removeprefix("/v1/").lstrip("/")
            if path not in {"chat/completions", "completions"}:
                raise ValueError(f"Unsupported replay endpoint: {path}")
            requests.append(LoadRequest(str(row.get("id", row_index)), path, body))
            continue
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or not row.get("output_response"):
            continue
        if messages[-1]["role"] != "assistant" or len(messages) == 1:
            continue
        prefix = messages[:-1]
        # Native summaries omit tool declarations. Do not fabricate an agentic request from them.
        needs_tools = any(item["role"] == "tool" or item.get("tool_calls") for item in prefix)
        if needs_tools and not row.get("tools"):
            continue
        body = {"messages": prefix, "max_tokens": output_tokens, "temperature": 0.7, "top_p": 1.0}
        if row.get("tools"):
            body["tools"] = row["tools"]
        source = row.get("data_source", "micro210")
        requests.append(LoadRequest(f"{source}:{row.get('uid', row_index)}", "chat/completions", body))
    if not requests:
        raise ValueError("No replayable model requests; supply retained messages or canonical path/body records")
    identities = [request.identity for request in requests]
    if len(set(identities)) != len(identities):
        raise ValueError("Replay input contains duplicate request identities")
    return requests
