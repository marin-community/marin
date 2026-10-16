# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild each trial's conversation from a run's ``requests.jsonl``, the canonical #207 trace.

Harbor's mini-swe-agent trajectory drops the model's reasoning, so training rows are built from the
proxy log instead: the messages each request added, in order, with every assistant turn taken from
the logged reply (``reasoning`` kept as ``reasoning_content``) and tagged with its author and relay
trigger. The final reply, which no later request echoes, closes the conversation. Each trial is
joined to its Harbor result (reward, exception) through ``agent/trial_correlation_id.txt``.

Checks per trial, reported in ``problems``: the logged message ranges are contiguous, and every
assistant message the harness echoed back matches the logged reply's content.

    uv run python experiments/post_training/sft_shootout/rebuild_traces.py <run prefix> <output.jsonl>
"""

import json
import sys
from collections import defaultdict

from rigging.filesystem.storage_path import StoragePath


def assistant_turn(record: dict) -> dict:
    message = dict((record.get("response_messages") or [{}])[0] or {})
    reasoning = message.pop("reasoning", None) or message.get("reasoning_content")
    turn = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        turn["tool_calls"] = message["tool_calls"]
    if reasoning:
        turn["reasoning_content"] = reasoning
    turn["author"] = record.get("author", "glm")
    turn["trigger"] = record.get("trigger")
    turn["usage"] = record.get("usage")
    return turn


def rebuild(records: list[dict]) -> tuple[list[dict], dict | None, list[str]]:
    """Return the conversation, the tools, and any consistency problems for one trial's records."""
    messages: list[dict] = []
    replies: list[dict] = []
    problems: list[str] = []
    tools = records[0].get("tools")
    for record in records:
        if record["first_message"] != len(messages):
            problems.append(f"gap: expected message {len(messages)}, record starts at {record['first_message']}")
        messages.extend(record["messages"])
        if record.get("status") == 200:
            replies.append(assistant_turn(record))
    conversation: list[dict] = []
    reply_index = 0
    for message in messages:
        if message.get("role") != "assistant":
            conversation.append(message)
            continue
        if reply_index >= len(replies):
            problems.append("an echoed assistant message has no logged reply")
            conversation.append(message)
            continue
        turn = replies[reply_index]
        reply_index += 1
        if (message.get("content") or "").strip() != (turn.get("content") or "").strip():
            problems.append(f"assistant turn {reply_index} differs from its logged reply")
        conversation.append(turn)
    conversation.extend(replies[reply_index:])
    return conversation, tools, problems


def harbor_results(root: StoragePath) -> dict[str, dict]:
    """Map trial correlation id to the Harbor trial's name, reward and exception."""
    results = {}
    for directory, _, names in root.walk():
        if "trial_correlation_id.txt" not in names:
            continue
        trial_id = StoragePath(f"{directory}/trial_correlation_id.txt").read_text().strip()
        attempt = str(directory).removesuffix("/agent")
        result = json.loads(StoragePath(f"{attempt}/result.json").read_text())
        results[trial_id] = {
            "trial_name": result["trial_name"],
            "task_name": result["task_name"],
            "reward": ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
            "exception": (result.get("exception_info") or {}).get("exception_type"),
        }
    return results


def main() -> None:
    prefix, output = sys.argv[1].rstrip("/"), sys.argv[2]
    by_trial: dict[str, list[dict]] = defaultdict(list)
    for line in StoragePath(f"{prefix}/requests.jsonl").read_text().splitlines():
        record = json.loads(line)
        by_trial[record.get("trial") or "unknown"].append(record)
    results = harbor_results(StoragePath(prefix))
    rows = []
    for trial_id, records in sorted(by_trial.items()):
        conversation, tools, problems = rebuild(sorted(records, key=lambda r: r["ts"]))
        if trial_id not in results:
            problems.append("no Harbor trial carries this correlation id")
        rows.append({"trial_id": trial_id, **results.get(trial_id, {}), "messages": conversation, "tools": tools, "problems": problems})
    with StoragePath(output).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"trials": len(rows), "with_problems": sum(1 for r in rows if r["problems"]), "output": output}))


if __name__ == "__main__":
    main()
