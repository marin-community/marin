# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unified Stage 0 data (CL12): the three #8225 datasets in the shared protocol.

Every source is converted at its pinned revision into message lists whose assistant
turns carry explicit ``reasoning_content`` — both source marker styles
(``<|start_think|>...<|end_think|>`` and ``<think>...</think>``) are stripped, never
left inline — and whose tool calls stay structured. Trainers render these rows with
the shared chat template (:mod:`experiments.post_training.chat_template`), so
reasoning trains as reasoning on every architecture.

One row-id decision is shared by every trainer: rows whose max rendered length over
all four family tokenizers exceeds the Stage 0 sequence cap are dropped at
conversion, and the manifest records the dropped ids. Per-tokenizer filtering would
have the architectures train on different rows.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Message = dict[str, Any]

MAX_RENDERED_TOKENS = 32768

THINK_STYLES: tuple[tuple[str, str], ...] = (
    ("<|start_think|>", "<|end_think|>"),
    ("<think>", "</think>"),
)


@dataclass(frozen=True)
class SourceSpec:
    """One Stage 0 source at its pinned revision, with its row adapter."""

    name: str
    repo: str
    revision: str
    config: str | None
    adapt: Callable[[dict], list[Message]]


def strip_think(content: str, open_tag: str, close_tag: str) -> tuple[str, str | None]:
    """Split one think block out of a message body into ``(content, reasoning)``.

    Raises ``ValueError`` on an unclosed block: training on a reasoning span that
    never ends would teach the model not to close its own thinking.
    """
    start = content.find(open_tag)
    if start == -1:
        return content, None
    body_start = start + len(open_tag)
    end = content.find(close_tag, body_start)
    if end == -1:
        raise ValueError(f"unclosed think block: {content[:80]!r}")
    reasoning = content[body_start:end].strip()
    remainder = (content[:start] + content[end + len(close_tag) :]).strip()
    return remainder, reasoning or None


def _to_protocol_message(turn: dict) -> Message:
    """One source turn as a protocol message: think markers become reasoning_content."""
    message: Message = {"role": turn["role"], "content": turn.get("content") or ""}
    if turn.get("role") == "assistant":
        for open_tag, close_tag in THINK_STYLES:
            content, reasoning = strip_think(message["content"], open_tag, close_tag)
            message["content"] = content
            if reasoning is not None:
                message["reasoning_content"] = reasoning
                break
    if turn.get("tool_calls"):
        message["tool_calls"] = turn["tool_calls"]
    return message


def adapt_wildchat_rewild(row: dict) -> list[Message]:
    """``conversation`` turns carry plain user/assistant content."""
    return [_to_protocol_message(turn) for turn in row["conversation"]]


def adapt_nemotron_science(row: dict) -> list[Message]:
    """Science reasoning: assistant bodies embed ``<|start_think|>...<|end_think|>``."""
    return [_to_protocol_message(turn) for turn in row["messages"]]


def adapt_nemotron_terminal(row: dict) -> list[Message]:
    """Terminal trajectories: assistant turns embed ``<think>...</think>``; the
    environment's command outputs stay in the user turns verbatim."""
    return [_to_protocol_message(turn) for turn in row["conversations"]]


# The three #8225 Stage 0 sources at their pinned revisions (DESIGN, Stage 0 data).
STAGE0_SOURCES = (
    SourceSpec(
        name="wildchat-rewild",
        repo="nyu-dice-lab/wildchat50m-rewild-sft-385700",
        revision="46a5bb5",
        config=None,
        adapt=adapt_wildchat_rewild,
    ),
    SourceSpec(
        name="nemotron-science-reasoning",
        repo="laion/llama-nemotron-science-reasoning-on-canonical-think-full",
        revision="bae881d",
        config=None,
        adapt=adapt_nemotron_science,
    ),
    SourceSpec(
        name="nemotron-terminal",
        repo="nvidia/Nemotron-Terminal-Corpus",
        revision="a1667c4",
        config="dataset_adapters",
        adapt=adapt_nemotron_terminal,
    ),
)

# Stage order is #8225's curriculum: chat, then thinking, then terminal protocol.
STAGE_ORDER = ("wildchat-rewild", "nemotron-science-reasoning", "nemotron-terminal")

# Rows must render under the cap on every family's tokenizer, so all four
# architectures train on the same row-id set.
_LENGTH_FILTER_FAMILIES = ("grug", "qwen3_moe", "qwen3_5_moe", "nemotron_h")


def _default_render_lengths(messages: list[Message]) -> dict[str, int]:
    """Rendered token lengths per family, through the pinned real tokenizers."""
    from transformers import AutoTokenizer  # noqa: PLC0415 - deferred: heavy at import time

    lengths = {}
    for family in _LENGTH_FILTER_FAMILIES:
        repo, revision = _FAMILY_TOKENIZER_REVISIONS[family]
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        lengths[family] = len(
            tokenizer.apply_chat_template(messages, chat_template=shared_template(FAMILY_TOKEN_MAPS[family]))
        )
    return lengths


_FAMILY_TOKENIZER_REVISIONS = {
    "grug": ("open-athena/snowball-67b-a2b-base-262k-qk175", "11c9430da3075ef4596f92ee0af142d9d08215c1"),
    "qwen3_moe": ("Qwen/Qwen3-30B-A3B-Base", "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"),
    "qwen3_5_moe": ("Qwen/Qwen3.5-35B-A3B-Base", "0f0813072d2358973511097385626f21fcb6d422"),
    "nemotron_h": ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16", "e0ac9ee3dfd02be21b5479edfc2f671ed269d0a2"),
}


def row_id(source: str, messages: list[Message]) -> str:
    """A stable row identity: the sha256 of the canonical protocol row."""
    canonical = json.dumps({"source": source, "messages": messages}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def convert_source(
    spec: SourceSpec,
    output_dir: str,
    *,
    max_rows: int | None = None,
    load_dataset: Callable | None = None,
    render_lengths: Callable[[list[Message]], dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Convert one source to gzipped protocol rows and return its manifest entry.

    ``load_dataset`` and ``render_lengths`` are injectable for tests; the default
    length filter renders every row with all four pinned family tokenizers and
    drops rows over the cap, recording the dropped ids.
    """
    if load_dataset is None:
        from datasets import load_dataset  # noqa: PLC0415 - deferred: heavy at import time

    if render_lengths is None:
        render_lengths = _default_render_lengths

    dataset = load_dataset(spec.repo, spec.config, revision=spec.revision, split="train", streaming=True)
    rows_written = 0
    rows_seen = 0
    dropped_ids: list[str] = []
    output_path = f"{output_dir}/{spec.name}.jsonl.gz"
    hasher = hashlib.sha256()
    use_tmp = not output_path.startswith("s3://")
    write_path = output_path + ".tmp" if use_tmp else output_path
    with gzip.open(write_path, "wt") as handle:
        for row in dataset:
            rows_seen += 1
            messages = spec.adapt(row)
            identity = row_id(spec.name, messages)
            lengths = render_lengths(messages)
            if max(lengths.values()) > MAX_RENDERED_TOKENS:
                dropped_ids.append(identity)
                continue
            line = json.dumps({"source": spec.name, "row_id": identity, "messages": messages}, sort_keys=True) + "\n"
            handle.write(line)
            hasher.update(line.encode())
            rows_written += 1
            if max_rows is not None and rows_written >= max_rows:
                break
    if rows_written == 0:
        raise ValueError(f"{spec.name}: every row was dropped (length filter or empty source); nothing to train on")
    if use_tmp:
        os.replace(write_path, output_path)
    return {
        "name": spec.name,
        "repo": spec.repo,
        "revision": spec.revision,
        "config": spec.config,
        "rows": rows_written,
        "seen": rows_seen,
        "dropped_over_cap": len(dropped_ids),
        "dropped_share": len(dropped_ids) / rows_seen if rows_seen else 0.0,
        "dropped_row_ids_sha256": hashlib.sha256(json.dumps(dropped_ids, sort_keys=True).encode()).hexdigest(),
        "sha256": hasher.hexdigest(),
        "path": output_path,
    }


def build_manifest(
    output_dir: str,
    *,
    max_rows: int | None = None,
    load_dataset: Callable | None = None,
    render_lengths: Callable[[list[Message]], dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Convert every source in stage order and write ``manifest.json``."""
    entries = []
    for spec in sorted(STAGE0_SOURCES, key=lambda source: STAGE_ORDER.index(source.name)):
        entries.append(
            convert_source(spec, output_dir, max_rows=max_rows, load_dataset=load_dataset, render_lengths=render_lengths)
        )
    manifest = {
        "stage_order": list(STAGE_ORDER),
        "sources": entries,
        "total_rows": sum(entry["rows"] for entry in entries),
        "total_dropped_over_cap": sum(entry["dropped_over_cap"] for entry in entries),
        "sequence_cap_tokens": MAX_RENDERED_TOKENS,
        "template_sha256": {family: template_sha256(tokens) for family, tokens in FAMILY_TOKEN_MAPS.items()},
        "unified_format": {
            "messages": "list of {role, content, reasoning_content?, tool_calls?}",
            "reasoning": "assistant reasoning_content, stripped from both source marker styles",
            "rendering": "shared chat template; see experiments/post_training/chat_template.py",
        },
    }
    with open(f"{output_dir}/manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest
