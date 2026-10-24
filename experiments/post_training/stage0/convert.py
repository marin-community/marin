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

import functools
import hashlib
import itertools
import json
import logging
import multiprocessing
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import fsspec
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from experiments.post_training.chat_template import (
    FAMILY_TOKEN_MAPS,
    FAMILY_TOKENIZER_REVISIONS,
    shared_template,
    template_sha256,
)

logger = logging.getLogger(__name__)

Message = dict[str, Any]
MAX_RENDERED_TOKENS = 32768

THINK_STYLES: tuple[tuple[str, str], ...] = (
    ("<|start_think|>", "<|end_think|>"),
    ("<think>", "</think>"),
)


class UnclosedThinkError(ValueError):
    """A message opens a think block that never closes."""


# Nemotron-Terminal's parquet row groups hold >2 GB of nested data, which the datasets
# streaming reader cannot convert; reading in small record batches can.
_PARQUET_BATCH_ROWS = 64
_PROGRESS_ROWS = 20_000


def hub_parquet_rows(repo: str, config: str | None, *, revision: str, split: str, streaming: bool) -> Iterator[dict]:
    """Rows of one dataset config at a pinned revision, file by file in sorted order.

    ``config`` names the repo directory holding the config's parquet files (``data``
    for the default config), as in all three Stage 0 sources' README configs. Only the
    ``train`` split exists for them. Files download one at a time and are deleted after
    reading. Matches ``datasets.load_dataset``'s call shape so tests can inject either.
    """
    if split != "train" or not streaming:
        raise ValueError("the Stage 0 sources have one streamed train split")
    prefix = f"{config or 'data'}/"
    files = sorted(
        path
        for path in HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
        if path.startswith(prefix) and path.endswith(".parquet")
    )
    if not files:
        raise ValueError(f"{repo}@{revision} has no parquet files under {prefix}")
    for path in files:
        with tempfile.TemporaryDirectory() as directory:
            local = hf_hub_download(repo, path, repo_type="dataset", revision=revision, local_dir=directory)
            for batch in pq.ParquetFile(local).iter_batches(batch_size=_PARQUET_BATCH_ROWS):
                yield from batch.to_pylist()
            os.remove(local)


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
        raise UnclosedThinkError(f"unclosed think block: {content[:80]!r}")
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


@functools.cache
def _family_renderers() -> dict[str, tuple[Any, str]]:
    """Each length-filter family's pinned tokenizer and shared template, loaded once per process."""
    renderers = {}
    for family in _LENGTH_FILTER_FAMILIES:
        repo, revision = FAMILY_TOKENIZER_REVISIONS[family]
        renderers[family] = (
            AutoTokenizer.from_pretrained(repo, revision=revision),
            shared_template(FAMILY_TOKEN_MAPS[family]),
        )
    return renderers


def rendered_token_ids(messages: list[Message]) -> dict[str, list[int]]:
    """The rendered token ids per family, through the pinned real tokenizers."""
    return {
        family: tokenizer.apply_chat_template(messages, chat_template=template, tokenize=True, return_dict=False)
        for family, (tokenizer, template) in _family_renderers().items()
    }


def rendered_lengths(messages: list[Message]) -> dict[str, int]:
    """Rendered token lengths per family, through the pinned real tokenizers."""
    return {family: len(ids) for family, ids in rendered_token_ids(messages).items()}


def row_id(source: str, messages: list[Message]) -> str:
    """A stable row identity: the sha256 of the canonical protocol row."""
    canonical = json.dumps({"source": source, "messages": messages}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def source_filename(source: str) -> str:
    """The converted file name for one source — how every trainer addresses the CL12 files."""
    return f"{source}.jsonl.gz"


def raw_row_id(row: dict) -> str:
    """A stable identity for a source row that cannot be converted (no protocol messages exist)."""
    return hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class _PreparedRow:
    """One source row after adaptation and the length check; ``line`` is set only when kept."""

    identity: str
    line: str | None
    unclosed_think: bool = False


def _prepare_row(
    source: str,
    adapt: Callable[[dict], list[Message]],
    render_lengths: Callable[[list[Message]], dict[str, int]],
    row: dict,
) -> _PreparedRow:
    try:
        messages = adapt(row)
    except UnclosedThinkError:
        return _PreparedRow(identity=raw_row_id(row), line=None, unclosed_think=True)
    identity = row_id(source, messages)
    if max(render_lengths(messages).values()) > MAX_RENDERED_TOKENS:
        return _PreparedRow(identity=identity, line=None)
    line = json.dumps({"source": source, "row_id": identity, "messages": messages}, sort_keys=True) + "\n"
    return _PreparedRow(identity=identity, line=line)


def _prepared_rows(
    prepare: Callable[[dict], _PreparedRow], rows: Iterable[dict], num_workers: int
) -> Iterator[_PreparedRow]:
    """``prepare`` over ``rows`` in source order, across ``num_workers`` processes."""
    if num_workers == 1:
        yield from map(prepare, rows)
        return
    # Rendering four tokenizers per row dominates the conversion; spawn keeps workers
    # free of the parent's tokenizer and fsspec state.
    with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
        yield from pool.imap(prepare, rows, chunksize=16)


def convert_source(
    spec: SourceSpec,
    output_dir: str,
    *,
    max_rows: int | None = None,
    load_dataset: Callable = hub_parquet_rows,
    render_lengths: Callable[[list[Message]], dict[str, int]] = rendered_lengths,
    num_workers: int = 1,
) -> dict[str, Any]:
    """Convert one source to gzipped protocol rows and return its manifest entry.

    Two row filters apply, and the manifest records both drop sets by id: rows whose
    rendered length on any family tokenizer exceeds the cap, and rows with an unclosed
    think block (training on a reasoning span that never ends would teach the model
    not to close its own thinking). ``load_dataset`` and ``render_lengths`` are
    injectable for tests; with ``num_workers > 1`` both ``spec.adapt`` and
    ``render_lengths`` must be picklable top-level functions.
    """
    dataset = load_dataset(spec.repo, spec.config, revision=spec.revision, split="train", streaming=True)
    rows_written = 0
    rows_seen = 0
    over_cap_ids: list[str] = []
    unclosed_ids: list[str] = []
    output_path = f"{output_dir}/{source_filename(spec.name)}"
    hasher = hashlib.sha256()
    prepare = functools.partial(_prepare_row, spec.name, spec.adapt, render_lengths)
    # Compression is explicit, so a remote path writes gzipped bytes like a local one.
    with fsspec.open(output_path, "wt", compression="gzip") as handle:
        for prepared in _prepared_rows(prepare, dataset, num_workers):
            rows_seen += 1
            if rows_seen % _PROGRESS_ROWS == 0:
                logger.info("%s: %d rows seen, %d kept", spec.name, rows_seen, rows_written)
            if prepared.unclosed_think:
                unclosed_ids.append(prepared.identity)
                continue
            if prepared.line is None:
                over_cap_ids.append(prepared.identity)
                continue
            handle.write(prepared.line)
            hasher.update(prepared.line.encode())
            rows_written += 1
            if max_rows is not None and rows_written >= max_rows:
                break
    if rows_written == 0:
        raise ValueError(f"{spec.name}: every row was dropped (length filter or empty source); nothing to train on")
    return {
        "name": spec.name,
        "repo": spec.repo,
        "revision": spec.revision,
        "config": spec.config,
        "rows": rows_written,
        "seen": rows_seen,
        "dropped_over_cap": len(over_cap_ids),
        "dropped_share": len(over_cap_ids) / rows_seen if rows_seen else 0.0,
        "dropped_row_ids_sha256": _ids_sha256(over_cap_ids),
        "dropped_unclosed_think": len(unclosed_ids),
        "dropped_unclosed_think_row_ids_sha256": _ids_sha256(unclosed_ids),
        "sha256": hasher.hexdigest(),
        "path": output_path,
    }


def _ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()


TOKEN_CHECK_ROWS = 100


def token_check(messages_rows: Iterable[list[Message]]) -> dict[str, dict[str, int]]:
    """The CL12 token-level check over real converted rows, per family.

    Each reasoning span must sit between exactly one pair of the family's think
    tokens, no think block may be empty (a spurious injected block), and every turn
    must close. Returns per-family counts of rows checked and rows failing each rule.
    """
    renderers = _family_renderers()
    think_ids = {}
    for family, (tokenizer, _) in renderers.items():
        tokens = FAMILY_TOKEN_MAPS[family]
        (think_open,) = tokenizer.encode(tokens.think_open, add_special_tokens=False)
        (think_close,) = tokenizer.encode(tokens.think_close, add_special_tokens=False)
        (turn_close,) = tokenizer.encode(tokens.turn_close, add_special_tokens=False)
        think_ids[family] = (think_open, think_close, turn_close)
    report = {
        family: {"rows": 0, "think_span_mismatch": 0, "empty_think": 0, "unclosed_turn": 0} for family in renderers
    }
    for messages in messages_rows:
        reasoning_turns = sum(1 for message in messages if message.get("reasoning_content"))
        for family, ids in rendered_token_ids(messages).items():
            think_open, think_close, turn_close = think_ids[family]
            counts = report[family]
            counts["rows"] += 1
            if ids.count(think_open) != reasoning_turns or ids.count(think_close) != reasoning_turns:
                counts["think_span_mismatch"] += 1
            if any(a == think_open and b == think_close for a, b in itertools.pairwise(ids)):
                counts["empty_think"] += 1
            if ids.count(turn_close) < len(messages):
                counts["unclosed_turn"] += 1
    return report


def _first_rows(path: str, count: int) -> Iterator[list[Message]]:
    with fsspec.open(path, "rt", compression="gzip") as handle:
        for index, line in enumerate(handle):
            if index >= count:
                return
            yield json.loads(line)["messages"]


def build_manifest(
    output_dir: str,
    *,
    max_rows: int | None = None,
    load_dataset: Callable = hub_parquet_rows,
    render_lengths: Callable[[list[Message]], dict[str, int]] = rendered_lengths,
    num_workers: int = 1,
    check_tokens: bool = True,
) -> dict[str, Any]:
    """Convert every source in stage order, token-check each, and write ``manifest.json``."""
    entries = []
    for spec in sorted(STAGE0_SOURCES, key=lambda source: STAGE_ORDER.index(source.name)):
        entry = convert_source(
            spec,
            output_dir,
            max_rows=max_rows,
            load_dataset=load_dataset,
            render_lengths=render_lengths,
            num_workers=num_workers,
        )
        if check_tokens:
            entry["token_check"] = token_check(_first_rows(entry["path"], TOKEN_CHECK_ROWS))
        entries.append(entry)
    manifest = {
        "stage_order": list(STAGE_ORDER),
        "sources": entries,
        "total_rows": sum(entry["rows"] for entry in entries),
        "total_dropped_over_cap": sum(entry["dropped_over_cap"] for entry in entries),
        "total_dropped_unclosed_think": sum(entry["dropped_unclosed_think"] for entry in entries),
        "sequence_cap_tokens": MAX_RENDERED_TOKENS,
        "max_rows_per_source": max_rows,
        "template_sha256": {family: template_sha256(tokens) for family, tokens in FAMILY_TOKEN_MAPS.items()},
        "unified_format": {
            "messages": "list of {role, content, reasoning_content?, tool_calls?}",
            "reasoning": "assistant reasoning_content, stripped from both source marker styles",
            "rendering": "shared chat template; see experiments/post_training/chat_template.py",
        },
    }
    with fsspec.open(f"{output_dir}/manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest
