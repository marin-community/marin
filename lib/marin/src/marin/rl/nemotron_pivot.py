# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PivotRL candidates from NVIDIA's Nemotron agentic pivot releases, one per expert turn.

Each release row is one pivot: a trajectory's history up to an expert turn, and that turn's
action. NVIDIA kept only the turns its own policy found informative (mixed outcomes at a low pass
rate) and repeats a trajectory once per kept turn, at many depths and often several times at the
same one. Every row of a trajectory is a prefix of its longest row, up to item ids. The
``prepare_*_candidates`` functions keep each trajectory's longest row and lay it out long: one
candidate per expert turn in that history, plus the row's own pivot, so a frozen policy's pass
rates decide which turns are pivots. A SWE instance's two or three trajectories are independent
rollouts that share only the task statement, so each is kept. NVIDIA's pass-rate fields describe
its policy and are dropped.

Whole SWE instances or Terminal tasks are held out for validation by hash bucket, since rows of
one trajectory share almost their whole prompt and trajectories of one task share its statement.
For the same reason a validation curve steadies with more held-out groups, not more rows, so
validation keeps only a few evenly spaced turns of each held-out trajectory (:class:`Holdout`).

Prompts keep the teacher's earlier reasoning: :func:`release_messages` attaches each Responses-API
``reasoning`` item to the next assistant message as ``reasoning_content``, which chat templates
render as a thinking block, as NeMo does (``truncate_history_thinking: false``). A turn's own
reasoning is never in its prompt. Rows use the layout MarinSkyRL's ``nemotron_ultra`` environment
trains on.
"""

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from rigging.filesystem.storage_path import StoragePath
from verifyit.adapters.nemotron_pivot import TERMINUS_2_SCHEMA
from verifyit.modes.grade_json_schema import grade_json_schema_candidate
from zephyr.writers import write_parquet_file

CANDIDATES_FILENAME = "candidates.parquet"
VALIDATION_FILENAME = "validation.parquet"
PARTITION_BUCKETS = 10_000
PARTITION_HASH_PERSON = b"nemo-pivot-v1"

SWE_AGENT = "swe_pivot_single_step_tool_use_with_argument_comparison_agent"
TERMINAL_AGENT = "terminus_judge_string_only_simple_agent"

_TOOL_CALL = pa.struct([("name", pa.string()), ("arguments", pa.string())])
_MESSAGE = pa.struct(
    [
        ("role", pa.string()),
        ("content", pa.string()),
        ("tool_call_id", pa.string()),
        ("tool_calls", pa.list_(pa.struct([("id", pa.string()), ("type", pa.string()), ("function", _TOOL_CALL)]))),
        ("reasoning_content", pa.string()),
    ]
)
CANDIDATE_SCHEMA = pa.schema(
    [
        ("prompt", pa.list_(_MESSAGE)),
        ("env_class", pa.string()),
        ("data_source", pa.string()),
        ("reward_model", pa.struct([("ground_truth", pa.string())])),
        (
            "extra_info",
            pa.struct(
                [
                    ("index", pa.int64()),
                    ("split", pa.string()),
                    ("trajectory_id", pa.string()),
                    ("group", pa.string()),
                    ("source_id", pa.string()),
                    (
                        "nemotron_ultra",
                        pa.struct(
                            [
                                (name, pa.string())
                                for name in ("agent", "blend", "route", "uuid", "request_json", "record_json")
                            ]
                        ),
                    ),
                ]
            ),
        ),
    ]
)


def release_messages(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Responses-API history as chat messages, with earlier reasoning as ``reasoning_content``."""
    messages: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    reasoning: list[str] = []

    def assistant(content: str | None, tool_calls: list[dict[str, Any]] | None) -> dict[str, Any]:
        message = {"role": "assistant", "content": content, "tool_calls": tool_calls}
        message["reasoning_content"] = "\n\n".join(reasoning) if reasoning else None
        reasoning.clear()
        return message

    def flush_calls() -> None:
        if calls:
            messages.append(assistant(None, list(calls)))
            calls.clear()

    for item in items:
        match item.get("type", "message"):
            case "reasoning":
                flush_calls()
                parts = [*(item.get("summary") or ()), *(item.get("content") or ())]
                if text := "\n".join(part["text"] for part in parts if part.get("text")):
                    reasoning.append(text)
            case "function_call":
                calls.append(
                    {
                        "id": item["call_id"],
                        "type": "function",
                        "function": {"name": item["name"], "arguments": item["arguments"]},
                    }
                )
            case "function_call_output":
                flush_calls()
                output = item.get("output", "")
                content = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": content})
            case "message":
                flush_calls()
                text = _message_text(item.get("content"))
                if item["role"] == "assistant":
                    messages.append(assistant(text, None))
                else:
                    messages.append({"role": item["role"], "content": text})
            case unsupported:
                raise ValueError(f"unsupported Responses input item {unsupported!r}")
    flush_calls()
    return messages


def _message_text(content: Any) -> str | None:
    if content is None or isinstance(content, str):
        return content
    return "".join(part["text"] for part in content if part.get("type") in ("input_text", "output_text"))


def longest_rows(lines: Iterable[str], key: Callable[[dict[str, Any]], str]) -> list[dict[str, Any]]:
    """The release row with the longest history for each ``key``, the first such row on ties."""
    longest: dict[str, dict[str, Any]] = {}
    for line in lines:
        row = json.loads(line)
        kept = longest.get(key(row))
        if kept is None or len(_history(row)) > len(_history(kept)):
            longest[key(row)] = row
    return list(longest.values())


def _history(row: dict[str, Any]) -> list[dict[str, Any]]:
    return row["responses_create_params"]["input"]


def swe_turns(row: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any]]]:
    """``(start, expected_action)`` for each expert turn of a SWE row, ending with the row's own pivot.

    A turn is the run of reasoning, narration, and calls between observations; its prompt is the
    history before ``start``, so its own reasoning never leaks. As in the release, a turn's expected
    action is its first call; narration-only turns stay in later prompts but are not candidates.
    """
    start: int | None = None
    calls: list[dict[str, Any]] = []
    for position, item in enumerate(_history(row)):
        kind = item.get("type", "message")
        if kind in ("reasoning", "function_call") or (kind == "message" and item["role"] == "assistant"):
            if start is None:
                start = position
            if kind == "function_call":
                calls.append(item)
            continue
        if start is not None and calls:
            yield start, {key: calls[0][key] for key in ("type", "name", "arguments")}
        start, calls = None, []
    yield len(_history(row)), row["expected_action"]


def terminal_turns(row: dict[str, Any]) -> Iterator[tuple[int, str]]:
    """``(start, expected_answer)`` for each Terminus-2 reply of a Terminal row, ending with the row's own.

    Earlier replies that are not schema-valid Terminus-2 actions stay in later prompts, where the
    harness's parse-error feedback follows them, but are not candidates.
    """
    for position, item in enumerate(_history(row)):
        if item["role"] == "assistant" and is_terminus_action(item["content"]):
            yield position, item["content"]
    yield len(_history(row)), row["expected_answer"]


def is_terminus_action(text: str) -> bool:
    try:
        action = json.loads(text)
    except json.JSONDecodeError:
        return False
    return grade_json_schema_candidate(TERMINUS_2_SCHEMA, action).reward == 1.0


@dataclass(frozen=True)
class Holdout:
    """Which groups validation holds out, and how many turns of each held-out trajectory it keeps."""

    buckets: int
    """How many of the :data:`PARTITION_BUCKETS` hash buckets of groups are held out."""
    turns_per_trajectory: int
    """Evenly spaced turns kept from each held-out trajectory, from its first to its last."""

    def holds_out(self, group: str) -> bool:
        """Whether ``group`` is held out; this depends only on its own key, never on other groups."""
        digest = hashlib.blake2b(group.encode(), digest_size=8, person=PARTITION_HASH_PERSON).digest()
        return int.from_bytes(digest, byteorder="big") % PARTITION_BUCKETS < self.buckets

    def turns(self, count: int, split: str) -> list[int]:
        """The turns of a ``count``-turn trajectory that ``split`` keeps."""
        if split == "train" or count <= self.turns_per_trajectory:
            return list(range(count))
        gaps = max(self.turns_per_trajectory - 1, 1)
        return [index * (count - 1) // gaps for index in range(self.turns_per_trajectory)]


def candidate(
    *, dataset: str, trajectory_id: str, group: str, turn: int, prompt: list[dict[str, Any]], split: str, **ultra: str
) -> dict[str, Any]:
    """One candidate row in the layout MarinSkyRL's ``nemotron_ultra`` environment trains on."""
    source_id = f"{dataset}:{trajectory_id}:{turn}"
    return {
        "prompt": release_messages(prompt),
        "env_class": "nemotron_ultra",
        "data_source": f"pivot_{dataset}",
        "reward_model": {"ground_truth": ultra["agent"]},
        "extra_info": {
            "index": turn,
            "split": split,
            "trajectory_id": trajectory_id,
            "group": group,
            "source_id": source_id,
            "nemotron_ultra": {"blend": f"pivot_{dataset}", "route": "skyrl_gym", "uuid": source_id, **ultra},
        },
    }


def swe_candidates(trajectories: list[dict[str, Any]], holdout: Holdout, split: str) -> Iterator[dict[str, Any]]:
    for row in trajectories:
        instance = row["metadata"]["instance_id"]
        if holdout.holds_out(instance) != (split == "validation"):
            continue
        request = {key: value for key, value in row["responses_create_params"].items() if key != "input"}
        turns = list(swe_turns(row))
        for turn in holdout.turns(len(turns), split):
            start, action = turns[turn]
            record = {
                "trajectory_id": row["trajectory_id"],
                "turn": turn,
                "expected_action": action,
                "metadata": row["metadata"],
                "agent_ref": row["agent_ref"],
            }
            yield candidate(
                dataset="swe",
                trajectory_id=str(row["trajectory_id"]),
                group=instance,
                turn=turn,
                prompt=_history(row)[:start],
                split=split,
                agent=SWE_AGENT,
                request_json=json.dumps(request),
                record_json=json.dumps(record),
            )


def terminal_candidates(trajectories: list[dict[str, Any]], holdout: Holdout, split: str) -> Iterator[dict[str, Any]]:
    for row in trajectories:
        if holdout.holds_out(row["task_name"]) != (split == "validation"):
            continue
        metadata = {key: value for key, value in row["metadata"].items() if key != "pivot_agent_turn_index"}
        turns = list(terminal_turns(row))
        for turn in holdout.turns(len(turns), split):
            start, answer = turns[turn]
            record = {
                "schema_version": row["schema_version"],
                "task_name": row["task_name"],
                "turn": turn,
                "expected_answer": answer,
                "metadata": metadata,
                "agent_ref": row["agent_ref"],
            }
            yield candidate(
                dataset="terminal",
                trajectory_id=row["metadata"]["source_trajectory_uid"],
                group=row["task_name"],
                turn=turn,
                prompt=_history(row)[:start],
                split=split,
                agent=TERMINAL_AGENT,
                request_json="{}",
                record_json=json.dumps(record),
            )


def _write_candidates(
    output_path: str,
    trajectories: list[dict[str, Any]],
    holdout: Holdout,
    candidates: Callable[[list[dict[str, Any]], Holdout, str], Iterator[dict[str, Any]]],
) -> None:
    output = StoragePath(output_path)
    for split, filename in (("train", CANDIDATES_FILENAME), ("validation", VALIDATION_FILENAME)):
        rows = candidates(trajectories, holdout, split)
        write_parquet_file(rows, str(output / filename), schema=CANDIDATE_SCHEMA)


def prepare_swe_candidates(*, release_path: str, release_filename: str, output_path: str, holdout: Holdout) -> None:
    """Write long-format SWE candidates from each trajectory's longest row, holding out whole instances.

    Args:
        release_path: The downloaded ``nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1`` release.
        release_filename: Its JSONL file.
        output_path: Where ``candidates.parquet`` and ``validation.parquet`` go.
        holdout: Which SWE instances validation holds out, and how many turns per trajectory it keeps.
    """
    with (StoragePath(release_path) / release_filename).open("r") as release:
        trajectories = longest_rows(release, lambda row: str(row["trajectory_id"]))
    _write_candidates(output_path, trajectories, holdout, swe_candidates)


def prepare_terminal_candidates(*, release_path: str, release_filename: str, output_path: str, holdout: Holdout) -> None:
    """Write long-format Terminal candidates from each trajectory's longest row, holding out whole tasks.

    Args:
        release_path: The downloaded ``nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1`` release.
        release_filename: Its JSONL file.
        output_path: Where ``candidates.parquet`` and ``validation.parquet`` go.
        holdout: Which tasks validation holds out, and how many turns per trajectory it keeps.
    """
    with (StoragePath(release_path) / release_filename).open("r") as release:
        trajectories = longest_rows(release, lambda row: row["metadata"]["source_trajectory_uid"])
    _write_candidates(output_path, trajectories, holdout, terminal_candidates)
