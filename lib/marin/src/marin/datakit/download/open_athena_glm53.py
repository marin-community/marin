# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 completions joined to their hash-verified source prompts."""

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial

import fsspec
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from openai_harmony import Message, Role
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

WILDCHAT_NAME = "wildchat-glm53-format-completions"
WILDCHAT_REPO = "open-athena/" + WILDCHAT_NAME
WILDCHAT_REVISION = "c20a530940c3b23c832ac89e7b3dc6f6d38b76a9"
AGENTTROVE_NAME = "agenttrove-glm53-compactions"
AGENTTROVE_REPO = "open-athena/" + AGENTTROVE_NAME
AGENTTROVE_REVISION = "6daf898de0743872e972c2bfb7c399bd052bd31f"
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


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def completion_document(prompt: str, answer: str, source: str, source_id: str) -> dict:
    """Keep quoted history and literal output formats as text, without interpreting tool tags."""
    if not prompt.strip() or not answer.strip():
        raise ValueError("A completion must have a nonempty prompt and answer")
    if any(pattern.search(text) for pattern in (CHAT_CONTROL_TOKEN, REASONING_TOKEN) for text in (prompt, answer)):
        raise ValueError("Completion data contains native chat control tokens")
    return chat_document(
        [
            Message.from_role_and_content(Role.USER, prompt),
            Message.from_role_and_content(Role.ASSISTANT, answer).with_channel(ChatChannel.FINAL),
        ],
        source,
        source_id=source_id,
    )


def wildchat_document(row: dict, source: dict) -> dict:
    """Recover the exact single user turn and verify its pinned WildChat identity."""
    index = row["source_turn_index"]
    if index < 0:
        raise ValueError("Negative WildChat turn index")
    message = source["conversation"][index]
    prompt = message["content"].strip()
    if (
        source["conversation_hash"] != row["conversation_hash"]
        or message["role"] != "user"
        or str(message.get("turn_identifier")) != row["turn_identifier"]
        or _sha256(prompt) != row["prompt_sha256"]
        or _sha256(re.sub(r"\s+", " ", prompt).strip().casefold()) != row["source_id"]
    ):
        raise ValueError("WildChat source reference mismatch")
    return completion_document(prompt + "\n\n" + row["format_instruction"], row["answer"], WILDCHAT_REPO, row["pair_id"])


def compaction_document(row: dict, source: dict) -> dict:
    """Recover the exact compaction request, keeping the old trajectory inside its user turn."""
    history = source["conversations"]
    if _sha256(json.dumps(history, ensure_ascii=False)) != row["trace_id"]:
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
    return completion_document(prompt, row["compaction"], AGENTTROVE_REPO, row["trace_id"])


@dataclass(frozen=True)
class ReferencedCompletion:
    name: str
    revision: str
    source_repo: str
    source_revision: str
    columns: tuple[str, ...]
    document: Callable[[dict, dict], dict]
    splits: tuple[str, ...] = ("train",)


WILDCHAT = ReferencedCompletion(
    WILDCHAT_NAME,
    WILDCHAT_REVISION,
    "allenai/WildChat-4.8M",
    "c827c6df8fcf008219ffaffa4d1dd77491099367",
    ("conversation_hash", "conversation"),
    wildchat_document,
    splits=("train", "validation"),
)
COMPACTIONS = ReferencedCompletion(
    AGENTTROVE_NAME,
    AGENTTROVE_REVISION,
    "open-thoughts/AgentTrove",
    "b395a4307a2bc9950a90dc899438f149e115fc60",
    ("conversations",),
    compaction_document,
)


def resolve_reference_file(
    filename: str, rows: Iterator[dict], *, config: ReferencedCompletion, source_root: str
) -> Iterator[dict]:
    """Read each referenced Parquet row group once, projecting only source prompt columns."""
    if re.fullmatch(r"data/train-[0-9]{5}-of-[0-9]{5}\.parquet", filename) is None:
        raise ValueError("Invalid source filename")
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if (
            row["source_dataset"] != config.source_repo
            or row["dataset_revision"] != config.source_revision
            or row["source_file"] != filename
            or row["source_split"] != "train"
        ):
            raise ValueError("Unexpected source dataset, revision, file, or split")
        if row["source_row_group"] < 0 or row["source_row_in_group"] < 0:
            raise ValueError("Negative source row coordinate")
        groups[row["source_row_group"]].append(row)
    with fsspec.open(f"{source_root}/{filename}", "rb", block_size=1024 * 1024) as stream:
        parquet = pq.ParquetFile(stream)
        for group, references in sorted(groups.items()):
            sources = parquet.read_row_group(group, columns=list(config.columns)).to_pylist()
            for row in references:
                yield config.document(row, sources[row["source_row_in_group"]])


def transform_chat(input_path: str, output_path: str, config: ReferencedCompletion, split: str) -> None:
    source_root = f"hf://datasets/{config.source_repo}@{config.source_revision}"
    pipeline = (
        Dataset.from_files(f"{input_path}/data/{split}-*.parquet")
        .flat_map(load_parquet_batched)
        .group_by(
            key=lambda row: row["source_file"],
            reducer=partial(resolve_reference_file, config=config, source_root=source_root),
            num_output_shards=32,
        )
        .write_parquet(
            f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", schema=CHAT_SCHEMA, skip_existing=True
        )
    )
    ZephyrContext(name=config.name, resources=ResourceConfig(cpu=1, ram="8g"), max_workers=32).execute(pipeline)


def chat_normalize_steps(config: ReferencedCompletion, split: str) -> tuple[StepSpec, ...]:
    if split not in config.splits:
        raise ValueError(f"Unsupported split {split} for {config.name}")
    name = config.name if split == "train" else f"{config.name}/{split}"
    download = download_hf_step(
        f"raw/{name}",
        hf_dataset_id=f"open-athena/{config.name}",
        revision=config.revision,
        hf_urls_glob=[f"data/{split}-*.parquet"],
    )
    processed = StepSpec(
        name=f"processed-chat/{name}",
        deps=[download],
        fn=lambda output_path: transform_chat(download.output_path, output_path, config, split),
        hash_attrs={
            "version": "2026.09.13",
            "source_revision": config.source_revision,
            "opencode_revision": OPENCODE_REVISION,
            "split": split,
        },
    )
    return processed, normalize_chat_step(name=f"normalized-chat/{name}", download=processed)


def wildchat_glm53_chat_normalize_steps() -> tuple[StepSpec, ...]:
    return chat_normalize_steps(WILDCHAT, "train")


def wildchat_glm53_validation_chat_normalize_steps() -> tuple[StepSpec, ...]:
    return chat_normalize_steps(WILDCHAT, "validation")


def agenttrove_glm53_chat_normalize_steps() -> tuple[StepSpec, ...]:
    return chat_normalize_steps(COMPACTIONS, "train")
