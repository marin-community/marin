# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 compaction summaries paired with their original AgentTrove histories."""

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterator
from functools import partial

import fsspec
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from openai_harmony import Message, Role
from rigging.filesystem.storage_path import prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.chat_normalize import CHAT_SCHEMA, ChatChannel, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import (
    CHAT_CONTROL_TOKEN,
    REASONING_TOKEN,
    chat_document,
    load_parquet_batched,
)
from marin.execution.step_spec import StepSpec

AGENTTROVE_NAME = "agenttrove-glm53-compactions"
AGENTTROVE_REPO = "open-athena/" + AGENTTROVE_NAME
AGENTTROVE_REVISION = "6daf898de0743872e972c2bfb7c399bd052bd31f"
NUM_SHARDS = 32
SOURCE_REPO = "open-thoughts/AgentTrove"
SOURCE_REVISION = "b395a4307a2bc9950a90dc899438f149e115fc60"
OPENCODE_REVISION = "0033bb35599a359def31b53d73e885eb4c44d815"
# Verbatim SUMMARY_TEMPLATE from packages/core/src/session/compaction.ts at OPENCODE_REVISION.
COMPACTION_TEMPLATE = (
    "Output exactly the Markdown structure shown inside <template> and keep the section order unchanged. Do "
    "not include the <template> tags in your response.\n"
    "<template>\n"
    "## Objective\n"
    "- [one or two brief sentences describing what the user is trying to accomplish]\n"
    "\n"
    "## Important Details\n"
    "- [constraints/preferences, decisions and why, important facts/assumptions, exact context needed to "
    'continue, or "(none)"]\n'
    "\n"
    "## Work State\n"
    "### Completed\n"
    '- [finished work, verified facts, or changes made; otherwise "(none)"]\n'
    "\n"
    "### Active\n"
    '- [current work, partial changes, or investigation state; otherwise "(none)"]\n'
    "\n"
    "### Blocked\n"
    '- [blockers, failing commands, or unknowns; otherwise "(none)"]\n'
    "\n"
    "## Next Move\n"
    '1. [immediate concrete action, or "(none)"]\n'
    '2. [next action if known, or "(none)"]\n'
    "\n"
    "## Relevant Files\n"
    '- [file or directory path: why it matters, or "(none)"]\n'
    "</template>\n"
    "\n"
    "Rules:\n"
    "- Keep every section, even when empty.\n"
    "- Use terse bullets, not prose paragraphs.\n"
    "- Preserve exact file paths, symbols, commands, error strings, URLs, and identifiers when known.\n"
    "- Do not mention the summary process or that context was compacted."
)


def compaction_document(row: dict, source: dict) -> dict:
    """Recover the exact compaction request, keeping the old trajectory inside its user turn."""
    history = source["conversations"]
    if hashlib.sha256(json.dumps(history, ensure_ascii=False).encode("utf-8")).hexdigest() != row["trace_id"]:
        raise ValueError("AgentTrove source history hash mismatch")
    if row["opencode_revision"] != OPENCODE_REVISION:
        raise ValueError("Unexpected OpenCode compaction prompt revision")
    context = "\n\n".join(f"[{m['role'].capitalize()}]: {m['content']}" for m in history)
    prompt = "\n\n".join(
        [
            f"Here is the conversation so far:\n\n<conversation>\n{context}\n</conversation>",
            "Create a new anchored summary from the conversation history in the <conversation> tags above "
            "so another coding agent can continue the work.",
            COMPACTION_TEMPLATE,
        ]
    )
    answer = row["compaction"]
    if not prompt.strip() or not answer.strip():
        raise ValueError("A completion must have a nonempty prompt and answer")
    if any(pattern.search(text) for pattern in (CHAT_CONTROL_TOKEN, REASONING_TOKEN) for text in (prompt, answer)):
        raise ValueError("Completion data contains native chat control tokens")
    return chat_document(
        [
            Message.from_role_and_content(Role.USER, prompt),
            Message.from_role_and_content(Role.ASSISTANT, answer).with_channel(ChatChannel.FINAL),
        ],
        AGENTTROVE_REPO,
        source_id=row["trace_id"],
    )


def resolve_reference_file(filename: str, rows: Iterator[dict], *, source_root: str) -> Iterator[dict]:
    """Yield chat documents after validating their source references."""
    if re.fullmatch(r"data/train-[0-9]{5}-of-[0-9]{5}\.parquet", filename) is None:
        raise ValueError("Invalid source filename")
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if (
            row["source_dataset"] != SOURCE_REPO
            or row["dataset_revision"] != SOURCE_REVISION
            or row["source_file"] != filename
            or row["source_split"] != "train"
        ):
            raise ValueError("Unexpected source dataset, revision, file, or split")
        if row["source_row_group"] < 0 or row["source_row_in_group"] < 0:
            raise ValueError("Negative source row coordinate")
        groups[row["source_row_group"]].append(row)
    with fsspec.open(prefix_join(source_root, filename), "rb", block_size=1024 * 1024) as stream:
        parquet = pq.ParquetFile(stream)
        for group, references in sorted(groups.items()):
            sources = parquet.read_row_group(group, columns=["conversations"]).to_pylist()
            for row in references:
                yield compaction_document(row, sources[row["source_row_in_group"]])


def transform_chat(input_path: str, output_path: str) -> None:
    source_root = f"hf://datasets/{SOURCE_REPO}@{SOURCE_REVISION}"
    pipeline = (
        Dataset.from_files(f"{input_path}/data/train-*.parquet")
        .flat_map(load_parquet_batched)
        .group_by(
            key=lambda row: row["source_file"],
            reducer=partial(resolve_reference_file, source_root=source_root),
            num_output_shards=NUM_SHARDS,
        )
        .write_parquet(
            f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", schema=CHAT_SCHEMA, skip_existing=True
        )
    )
    ZephyrContext(name=AGENTTROVE_NAME, resources=ResourceConfig(cpu=1, ram="8g"), max_workers=NUM_SHARDS).execute(
        pipeline
    )


def glm53_compaction_chat_normalize_steps() -> tuple[StepSpec, ...]:
    name = AGENTTROVE_NAME
    download = download_hf_step(
        f"raw/{name}",
        hf_dataset_id=AGENTTROVE_REPO,
        revision=AGENTTROVE_REVISION,
        hf_urls_glob=["data/train-*.parquet"],
    )
    processed = StepSpec(
        name=f"processed-chat/{name}",
        deps=[download],
        fn=lambda output_path: transform_chat(download.output_path, output_path),
        hash_attrs={
            "version": "2026.09.13",
            "source_revision": SOURCE_REVISION,
            "opencode_revision": OPENCODE_REVISION,
            "split": "train",
        },
    )
    return processed, normalize_chat_step(name=f"normalized-chat/{name}", download=processed)
