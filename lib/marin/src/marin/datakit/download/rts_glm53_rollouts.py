# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""open-athena/recursive-task-synthesis-glm-5.3-rollouts download and transform.

GLM-5.3 Terminus-2 rollouts on gold-validated Recursive-Task-Synthesis tasks. The
repository stores complete Harbor trial directories in uncompressed tar shards under
``payload/``: one directory per saved execution, including gold-solution runs and
infrastructure retries.

Only verified model attempts become conversations: ``rts_execution.json`` must record
``phase == "glm"`` and a non-null reward. A null reward marks an execution error (sandbox
start, agent timeout, ...), which is not a verifier failure; such attempts are superseded
by an infrastructure retry or have no final answer. Both solved and failed attempts are
kept, labelled by ``outcome``; ``task_group_id`` is kept for family-level held-out splits.

The conversation comes from the last episode's ``debug.json``: the exact request the
model saw, with the reasoning of every earlier turn, followed by the final response.
Summarization was disabled, so this request carries the whole trajectory.
``trajectory.json`` is not used because its assistant messages are rewritten as
``Analysis:``/``Plan:`` prose rather than the JSON the model produced.
"""

import json
import re
import tarfile
from collections.abc import Iterator

import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import (
    TRAJECTORY_FAILED_TAG,
    TRAJECTORY_SOLVED_TAG,
    checked_openai_chat_document,
)
from marin.datakit.download.terminus import terminus_protocol_messages
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "open-athena/recursive-task-synthesis-glm-5.3-rollouts"
HF_REVISION = "dd6f34c"
PAYLOAD_GLOB = "payload/*/shards/*.tar"
COUNTER_PREFIX = "rts_glm53_rollouts"

RTS_CHAT_SCHEMA = pa.schema(
    [
        *CHAT_SCHEMA,
        pa.field("task_group_id", pa.string()),
        pa.field("outcome", pa.string()),
    ]
)

EPISODE_DEBUG = re.compile(r"agent/episode-(\d+)/debug\.json")


def _count(name: str) -> None:
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/{name}", 1)


def trial_conversation(debug: dict) -> list[dict] | None:
    """Rebuild the full conversation from the final request and its response.

    Returns ``None`` when the final response is missing or the Terminus protocol cannot
    be parsed. Earlier assistant turns carry ``reasoning_content``; the final response
    records it as ``reasoning``.
    """
    choices = debug.get("response", {}).get("choices") or []
    if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
        return None
    response = choices[0]["message"]
    history = debug.get("request", {}).get("messages") or []
    reasoning = response.get("reasoning_content") or response.get("reasoning")
    recorded = [*history, {"role": "assistant", "content": response.get("content"), "reasoning_content": reasoning}]
    messages = terminus_protocol_messages([{"role": m.get("role"), "content": m.get("content")} for m in recorded])
    if messages is None:
        return None
    # The Terminus parser re-serializes each JSON reply and drops reasoning; restore it
    # from the recorded assistant turns, which it keeps one-to-one and in order.
    # The debug log serializes absent values as the string "None".
    thoughts = [m.get("reasoning_content") for m in recorded if m.get("role") == "assistant"]
    assistants = [m for m in messages if m["role"] == "assistant"]
    if len(thoughts) != len(assistants):
        return None
    for message, thought in zip(assistants, thoughts, strict=True):
        if isinstance(thought, str) and thought.strip() and thought != "None":
            message["reasoning_content"] = thought
    return messages


def trial_to_chat_doc(execution: dict, debug: dict | None) -> list[dict]:
    if execution.get("phase") != "glm":
        _count("oracle_skipped")
        return []
    reward = execution.get("reward")
    if reward is None:
        _count("execution_error_filtered")
        return []
    if debug is None:
        _count("missing_debug_filtered")
        return []
    messages = trial_conversation(debug)
    if messages is None:
        _count("terminus_protocol_filtered")
        return []
    return checked_openai_chat_document(
        messages,
        HF_DATASET_ID,
        counter_prefix=f"{COUNTER_PREFIX}/chat",
        source_id=execution["execution_id"],
        task_group_id=execution["task_group_id"],
        outcome=TRAJECTORY_SOLVED_TAG if reward >= 1.0 else TRAJECTORY_FAILED_TAG,
    )


def _trial_docs(members: dict[str, bytes]) -> list[dict]:
    if "rts_execution.json" not in members:
        _count("missing_execution_record")
        return []
    execution = json.loads(members["rts_execution.json"])
    debug = members.get("debug.json")
    return trial_to_chat_doc(execution, json.loads(debug) if debug is not None else None)


def load_trial_shard(path: str) -> Iterator[dict]:
    """Stream one payload tar, emitting a chat document per verified model trial.

    Members of a trial are contiguous. Only ``rts_execution.json`` and the highest
    episode's ``debug.json`` are buffered, so memory is bounded by one trial.
    """
    seen: set[str] = set()
    trial: str | None = None
    members: dict[str, bytes] = {}
    episode = -1
    with open_url(path, "rb") as stream, tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            if not member.isfile():
                continue
            prefix, _, relative = member.name.partition("/")
            if prefix != trial:
                if trial is not None:
                    yield from _trial_docs(members)
                if prefix in seen:
                    raise ValueError(f"Trial {prefix} is not contiguous in {path}")
                seen.add(prefix)
                trial, members, episode = prefix, {}, -1
            if relative == "rts_execution.json":
                members[relative] = archive.extractfile(member).read()
            elif (match := EPISODE_DEBUG.fullmatch(relative)) and int(match[1]) > episode:
                episode = int(match[1])
                members["debug.json"] = archive.extractfile(member).read()
    if trial is not None:
        yield from _trial_docs(members)


def transform_chat(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(prefix_join(input_path, PAYLOAD_GLOB))
        .flat_map(load_trial_shard)
        .write_parquet(
            prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"),
            schema=RTS_CHAT_SCHEMA,
            skip_existing=True,
        )
    )
    ZephyrContext(
        name="rts-glm53-rollouts-chat-transform",
        resources=ResourceConfig(cpu=1, ram="8g"),
        max_workers=128,
    ).execute(pipeline)


def rts_glm53_rollouts_chat_normalize_steps() -> tuple[StepSpec, ...]:
    download = download_hf_step(
        "raw/rts-glm-5.3-rollouts",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[PAYLOAD_GLOB],
        zephyr_max_parallelism=32,
    )
    processed = StepSpec(
        name="processed-chat/rts-glm-5.3-rollouts",
        deps=[download],
        fn=lambda output_path: transform_chat(download.output_path, output_path),
        hash_attrs={"version": "v1"},
    )
    return processed, normalize_chat_step(
        output_schema=RTS_CHAT_SCHEMA, name="normalized-chat/rts-glm-5.3-rollouts", download=processed
    )
