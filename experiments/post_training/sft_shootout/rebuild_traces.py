# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild each trial's conversations from a ``trace_proxy`` request log.

Harbor's mini-swe-agent trajectory drops the model's reasoning, so training rows come from the proxy
log instead: the messages each request added, in order, with every assistant turn taken from the
logged reply (``reasoning`` kept as ``reasoning_content``) and tagged with its author and relay
trigger. The final reply, which no later request echoes, closes the conversation.

Each output row is one conversation of one trial: trial_id, conversation, n_conversations, the Harbor
trial fields (task_name, trial_name, retry, reward, exception), triggers, tokens (per author: turns,
prompt, completion, reasoning), messages, tools, dropped_replies (replies the harness discarded, with
the message index they preceded) and problems. A trial is joined to its Harbor result through the
``trial_correlation_id.txt`` that the agents in ``harbor_agents/trial_agents.py`` write next to their
logs. ``problems`` reports gaps in the logged message ranges, echoed assistant messages that differ
from the logged reply, assistant messages that no logged reply produced, and trials with no Harbor
result.

    uv run python experiments/post_training/sft_shootout/rebuild_traces.py \
      --log <requests.jsonl> --jobs <Harbor jobs dir> --output <rebuilt.jsonl>
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field

from rigging.filesystem.storage_path import StoragePath

RESULT_FILE = "result.json"
ATTEMPTS_DIR = "attempts"
# Written by harbor_agents/trial_agents.py, which imports Harbor and so cannot be imported here.
ID_FILE = "trial_correlation_id.txt"


def assistant_turn(record: dict) -> dict:
    message = dict((record.get("response_messages") or [{}])[0] or {})
    reasoning = message.pop("reasoning", None) or message.get("reasoning_content")
    turn = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        turn["tool_calls"] = message["tool_calls"]
    if reasoning:
        turn["reasoning_content"] = reasoning
    turn["author"] = record["author"]
    turn["trigger"] = record.get("trigger")
    turn["usage"] = record.get("usage")
    return turn


@dataclass
class Conversation:
    messages: list[dict]
    tools: list | None
    problems: list[str] = field(default_factory=list)
    dropped_replies: list[dict] = field(default_factory=list)


def rebuild(records: list[dict]) -> Conversation:
    """Rebuild one conversation from its records in order.

    Each reply is kept until the next request echoes it as its first new message. A request that adds
    nothing (a retry of the same request) replaces the pending reply. A reply the harness discarded
    (mini-swe-agent drops a reply without a valid tool call and sends "Tool call error:" instead) is
    left out of the messages and kept in ``dropped_replies``.
    """
    conversation: list[dict] = []
    problems: list[str] = []
    dropped: list[dict] = []
    seen = 0
    pending: dict | None = None
    for record in records:
        if record["first_message"] != seen:
            problems.append(f"gap: expected message {seen}, record starts at {record['first_message']}")
        seen = record["n_messages"]
        new = list(record["messages"])
        if pending is not None and new:
            if new[0].get("role") != "assistant":
                dropped.append({**pending, "before_message": record["first_message"]})
            else:
                if (new[0].get("content") or "").strip() != (pending.get("content") or "").strip():
                    problems.append(f"assistant message {len(conversation)} differs from its logged reply")
                conversation.append(pending)
                new = new[1:]
        if any(m.get("role") == "assistant" for m in new):
            problems.append(
                f"record at message {record['first_message']} adds an assistant message with no logged reply"
            )
        conversation.extend(new)
        pending = assistant_turn(record) if record.get("status") == 200 else None
    if pending is not None:
        conversation.append(pending)
    return Conversation(conversation, records[0].get("tools"), problems, dropped)


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


def harbor_results(root: StoragePath) -> dict[str, dict]:
    """Map trial correlation id to its Harbor trial attempt.

    The id file sits in the attempt's ``agent/`` logs directory, next to the attempt's ``result.json``.
    ``retry`` is NNN for an ``attempts/NNN`` directory (Harbor's retry after an infrastructure error) and 0
    for a trial without retries.
    """
    results = {}
    for directory, _, names in root.walk():
        if ID_FILE not in names:
            continue
        trial_id = (directory / ID_FILE).read_text().strip()
        attempt = directory.parent
        result_path = attempt / RESULT_FILE
        if not result_path.exists():
            continue  # Running, interrupted, or superseded attempts have no final result yet.
        result = json.loads(result_path.read_text())
        results[trial_id] = {
            "task_name": result["task_name"],
            "trial_name": result["trial_name"],
            "retry": int(attempt.name) if attempt.parent.name == ATTEMPTS_DIR else 0,
            "reward": ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
            "exception": (result.get("exception_info") or {}).get("exception_type"),
        }
    return results


def rebuilt_rows(records: list[dict], results: dict[str, dict]) -> list[dict]:
    by_conversation: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        by_conversation[(record["trial"] or "", record["conversation"])].append(record)
    n_conversations: dict[str, int] = defaultdict(int)
    for trial_id, _ in by_conversation:
        n_conversations[trial_id] += 1
    rows = []
    for (trial_id, index), group in sorted(by_conversation.items()):
        conversation = rebuild(sorted(group, key=lambda r: r["ts"]))
        if trial_id not in results:
            conversation.problems.append("no Harbor trial carries this correlation id")
        rows.append(
            {
                "trial_id": trial_id,
                "conversation": index,
                "n_conversations": n_conversations[trial_id],
                **results.get(trial_id, {}),
                "triggers": [m["trigger"] for m in conversation.messages if m.get("trigger")],
                "tokens": token_counts(conversation.messages),
                "messages": conversation.messages,
                "tools": conversation.tools,
                "problems": conversation.problems,
                "dropped_replies": conversation.dropped_replies,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", required=True, help="The proxy's requests.jsonl.")
    parser.add_argument("--jobs", required=True, help="Harbor jobs directory holding the trials' result.json.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in StoragePath(args.log).read_text().splitlines()]
    rows = rebuilt_rows(records, harbor_results(StoragePath(args.jobs)))
    StoragePath(args.output).write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(json.dumps({"conversations": len(rows), "with_problems": sum(1 for r in rows if r["problems"])}))


if __name__ == "__main__":
    main()
