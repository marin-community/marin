# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PivotRL candidates from nebius/SWE-rebench-openhands-trajectories, one per expert tool call.

The release has one row per OpenHands v0.54 trajectory. :func:`prepare_openhands_candidates` turns
resolved trajectories into one candidate per expert tool call, holding out whole trajectories for
validation. Each candidate carries the conversation before the call and what grading needs, worked
out once here: the repository root, the working directory replayed from earlier bash observations,
the expert call, and what it returned. ``think`` and ``task_tracker`` turns stay in later prompts
but are not candidates.
"""

import hashlib
import json
import re
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from verifyit.adapters.openhands_next_action import TurnState, call_action
from zephyr.writers import write_parquet_file

from marin.rl.nemotron_pivot import CANDIDATES_FILENAME, VALIDATION_FILENAME

TRAJECTORIES_FILENAME = "trajectories.parquet"
TOOLS_FILENAME = "tools.json"
MAX_OBSERVATION_CHARS = 8000
"""What the expert's call returned is judge context only; a long file view is mostly irrelevant past this."""


def prepare_openhands_candidates(
    *,
    dataset_path: str,
    output_path: str,
    trajectories: int,
    validation_trajectories: int,
    seed: int,
    max_prompt_chars: int,
) -> None:
    """Write candidate and validation turns from resolved trajectories chosen by seeded hash order.

    Args:
        dataset_path: The downloaded release, holding ``trajectories.parquet`` and ``tools.json``.
        output_path: Where ``candidates.parquet`` and ``validation.parquet`` go.
        trajectories: Resolved trajectories to turn into candidates.
        validation_trajectories: Further resolved trajectories held out whole for validation.
        seed: Selects which trajectories, independent of file order.
        max_prompt_chars: Later turns are dropped once the conversation before them is longer.
    """
    source = StoragePath(dataset_path)
    tools_json = json.dumps(json.loads((source / TOOLS_FILENAME).read_text()))
    with (source / TRAJECTORIES_FILENAME).open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        resolved = parquet.read(columns=["trajectory_id", "resolved"]).to_pylist()
        chosen = sorted(
            (row["trajectory_id"] for row in resolved if row["resolved"] == 1),
            key=lambda identity: hashlib.sha256(f"{seed}:{identity}".encode()).digest(),
        )[: validation_trajectories + trajectories]
        splits = {identity: "validation" for identity in chosen[:validation_trajectories]}
        splits.update({identity: "train" for identity in chosen[validation_trajectories:]})
        message_type = parquet.schema_arrow.field("trajectory").type

        rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
        for group in range(parquet.num_row_groups):
            columns = ["trajectory_id", "instance_id", "repo", "trajectory"]
            for record in parquet.read_row_group(group, columns=columns).to_pylist():
                split = splits.get(record["trajectory_id"])
                if split is not None:
                    rows[split].extend(candidate_rows(record, split, tools_json, max_prompt_chars))

    schema = candidate_schema(message_type)
    output = StoragePath(output_path)
    for split, filename in (("train", CANDIDATES_FILENAME), ("validation", VALIDATION_FILENAME)):
        table = pa.Table.from_pylist(rows[split], schema=schema)
        write_parquet_file(table.to_batches(), str(output / filename), schema=schema)


def candidate_rows(record: dict[str, Any], split: str, tools_json: str, max_prompt_chars: int) -> list[dict[str, Any]]:
    """One row per expert tool call, with the grading state replayed up to it."""
    trajectory = record["trajectory"]
    root = repository_root(trajectory)
    cwd = root
    prompt_chars = 0
    rows = []
    for index, message in enumerate(trajectory):
        # Bash observations end by reporting the shell's working directory.
        directories = re.findall(r"\[Current working directory: (.*?)\]", message["content"] or "")
        if message["role"] == "tool" and directories:
            cwd = directories[-1]
        if prompt_chars > max_prompt_chars:
            break
        call = _single_call(message)
        # Deliberation turns stay in later prompts but are not pivots: they change nothing to grade.
        if call is not None and call["name"] not in ("think", "task_tracker"):
            following = trajectory[index + 1] if index + 1 < len(trajectory) else None
            observation = following["content"] if following and following["role"] == "tool" else ""
            rows.append(
                {
                    "prompt": trajectory[:index],
                    "extra_info": {
                        "source_id": f"{record['trajectory_id']}:{index}",
                        "trajectory_id": record["trajectory_id"],
                        "index": index,
                        "split": split,
                        "instance_id": record["instance_id"],
                        "repo": record["repo"],
                        "openhands": {
                            "repo_root": root,
                            "cwd": cwd,
                            "expected_call_json": json.dumps(call),
                            "observation": (observation or "")[:MAX_OBSERVATION_CHARS],
                            "operation": str(call_action(call, TurnState(root, cwd)).operation),
                            "tools_json": tools_json,
                        },
                    },
                }
            )
        prompt_chars += len(message["content"] or "") + sum(
            len(tool_call["function"]["arguments"] or "") for tool_call in message["tool_calls"] or ()
        )
    return rows


def repository_root(trajectory: list[dict[str, Any]]) -> str:
    """The checkout path the task statement names in ``<uploaded_files>``."""
    for message in trajectory:
        match = re.search(r"<uploaded_files>\s*(\S+)", message["content"] or "")
        if message["role"] == "user" and match:
            return match.group(1).rstrip("/")
    raise ValueError("trajectory names no <uploaded_files> repository")


def candidate_schema(message_type: pa.DataType) -> pa.Schema:
    openhands = pa.struct(
        [
            (name, pa.string())
            for name in ("repo_root", "cwd", "expected_call_json", "observation", "operation", "tools_json")
        ]
    )
    extra_info = pa.struct(
        [
            ("source_id", pa.string()),
            ("trajectory_id", pa.string()),
            ("index", pa.int64()),
            ("split", pa.string()),
            ("instance_id", pa.string()),
            ("repo", pa.string()),
            ("openhands", openhands),
        ]
    )
    return pa.schema([("prompt", message_type), ("extra_info", extra_info)])


def _single_call(message: dict[str, Any]) -> dict[str, str] | None:
    if message["role"] != "assistant" or len(message["tool_calls"] or ()) != 1:
        return None
    function = message["tool_calls"][0]["function"]
    return {"name": function["name"], "arguments": function["arguments"]}
