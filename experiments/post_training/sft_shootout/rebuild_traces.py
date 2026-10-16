# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild each trial's conversation from a run's ``requests.jsonl``, the canonical #207 trace.

Harbor's mini-swe-agent trajectory drops the model's reasoning, so training rows are built from the
proxy log instead: the messages each request added, in order, with every assistant turn taken from
the logged reply (``reasoning`` kept as ``reasoning_content``) and tagged with its author and relay
trigger. The final reply, which no later request echoes, closes the conversation. Each trial attempt
is joined to its Harbor result through ``agent/trial_correlation_id.txt``.

One row per trial attempt: family, agent, package_revision, relay, run, trial_id, task_id (the task
folder name, as in the pool manifest), trial_name, sample (0-based index among the task's samples),
retry (Harbor's infrastructure retry), reward, passed, exception, triggers, tokens (per author:
turns, prompt, completion, reasoning), messages (assistant turns carry author, trigger, usage,
reasoning_content), tools, problems.

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
    """Map trial correlation id to its Harbor trial attempt: task, sample index, retry, reward, exception.

    ``n_attempts`` samples of a task are separate Harbor trials; ``sample`` numbers them by start time.
    ``attempts/NNN`` inside a trial are Harbor retries after infrastructure errors; ``retry`` is NNN.
    """
    attempts = []
    for directory, _, names in root.walk():
        if "trial_correlation_id.txt" not in names:
            continue
        trial_id = StoragePath(f"{directory}/trial_correlation_id.txt").read_text().strip()
        attempt_dir = str(directory).removesuffix("/agent")
        result = json.loads(StoragePath(f"{attempt_dir}/result.json").read_text())
        attempts.append((trial_id, attempt_dir, result))
    first_start = {}
    for _, _, result in attempts:
        key = (result["task_name"], result["trial_name"])
        first_start[key] = min(first_start.get(key, result.get("started_at") or ""), result.get("started_at") or "")
    samples = {}
    for task, trial in sorted(first_start, key=lambda key: (key[0], first_start[key], key[1])):
        samples[(task, trial)] = sum(1 for other in samples if other[0] == task)
    results = {}
    for trial_id, attempt_dir, result in attempts:
        reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        results[trial_id] = {
            "task_id": result["task_name"].rsplit("/", 1)[-1],
            "trial_name": result["trial_name"],
            "sample": samples[(result["task_name"], result["trial_name"])],
            "retry": int(attempt_dir.rsplit("/", 1)[-1]),
            "reward": reward,
            "passed": reward == 1.0,
            "exception": (result.get("exception_info") or {}).get("exception_type"),
        }
    return results


def token_counts(conversation: list[dict]) -> dict[str, dict[str, int]]:
    """Prompt, completion and reasoning tokens summed per author, as reported by the serving backends."""
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"turns": 0, "prompt": 0, "completion": 0, "reasoning": 0})
    for message in conversation:
        usage = message.get("usage") if message.get("role") == "assistant" else None
        if not usage:
            continue
        total = totals[message["author"]]
        total["turns"] += 1
        total["prompt"] += usage.get("prompt_tokens") or 0
        total["completion"] += usage.get("completion_tokens") or 0
        total["reasoning"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    return dict(totals)


def main() -> None:
    prefix, output = sys.argv[1].rstrip("/"), sys.argv[2]
    provenance = json.loads(StoragePath(f"{prefix}/driver_config.json").read_text())
    run = {
        "family": provenance["args"]["family"],
        "agent": provenance["args"]["agent"],
        "package_revision": provenance["args"]["package_revision"],
        "relay": provenance["relay"] is not None,
        "run": prefix,
    }
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
        rows.append(
            {
                **run,
                "trial_id": trial_id,
                **results.get(trial_id, {}),
                "triggers": [m["trigger"] for m in conversation if m.get("trigger")],
                "tokens": token_counts(conversation),
                "messages": conversation,
                "tools": tools,
                "problems": problems,
            }
        )
    with StoragePath(output).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"trials": len(rows), "with_problems": sum(1 for r in rows if r["problems"]), "output": output}))


if __name__ == "__main__":
    main()
