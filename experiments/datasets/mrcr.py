# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI MRCR as four paired conditions in one tokenized validation cache.

Each source row in the 262K context cap yields four unpacked documents that score one gold response under
needles absent/present and distractors absent/present:

    full_context:    worked examples + target conversation + final query -> gold response
    query_only:      worked examples + final query                       -> gold response
    needle_only:     worked examples + repeated-request turns + query    -> gold response
    distractor_only: worked examples + remaining turns + query           -> gold response

The two-shot prompt supplies the benchmark's random answer prefix, and only the response body is scored. The
cache and ``manifest.jsonl`` share one row order, so a cache row's source and strata are its manifest line.
Sources run longest first, so a memory or layout failure shows on the first batches.
"""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypedDict

import numpy as np
import pyarrow.parquet as pq
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.store.cache import write_levanter_cache
from levanter.tokenizers import load_tokenizer
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.data import dataset_main, hf_download
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.marin_tokenizer import marin_tokenizer

MRCR_DATASET_REVISION = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"
MRCR_RAW_VERSION = "2026.07.14.1"
MRCR_VERSION = "2026.11.04"
MRCR_NEEDLE_COUNTS: tuple[int, ...] = (2, 4, 8)
MRCR_DISTANCE_BOUNDS: tuple[int, ...] = (32_768, 65_536, 131_072)
MRCR_CONTEXT_CAP = 262_144
MRCR_CONTEXT_CAP_LOWER = 131_072
"""The experiment's 262K cell holds the sources whose full render exceeds the next smaller cap."""
MRCR_PROMPT_VARIANT = "two_shot"
MRCR_EXPECTED_SOURCES = 299
MRCR_EXPECTED_SCORED_TOKENS = 115_834
MRCR_CACHE_SPLIT = "validation"
MRCR_MANIFEST_NAME = "manifest.jsonl"
MRCR_STATS_NAME = "stats.json"
MRCR_PREAMBLE_PREFIX = "Here are some examples of conversations succeeded by a follow-up question answered correctly:"

_EXAMPLE_START = "======EXAMPLE======"
_EXAMPLE_END = "======END EXAMPLE======"
_MRCR_CHAT_TEMPLATE = (
    "{{ messages[0]['content'] }}{% generation %}{{ messages[1]['content'] }}{% endgeneration %}{{ eos_token }}"
)
_LENGTH_FILTER_BATCH_SIZE = 16
# The prefilter counts the prompt and target apart, as the exact length does; the margin is only a safety band.
_LENGTH_FILTER_MARGIN = 64


class MrcrCondition(StrEnum):
    FULL_CONTEXT = "full_context"
    QUERY_ONLY = "query_only"
    NEEDLE_ONLY = "needle_only"
    DISTRACTOR_ONLY = "distractor_only"


class _MrcrMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class MrcrTransformConfig:
    input_path: str
    output_path: str
    tokenizer: str
    context_cap: int = MRCR_CONTEXT_CAP
    context_cap_lower: int = MRCR_CONTEXT_CAP_LOWER
    distance_bounds: tuple[int, ...] = MRCR_DISTANCE_BOUNDS
    # Reproducing the original experiment's counts confirms the source and the tokenizer.
    expected_sources: int | None = None
    expected_scored_tokens: int | None = None


@dataclass(frozen=True)
class _Source:
    source_id: str
    n_needles: int
    distance_band: str
    evidence_distance_tokens: int
    canonical_length: int
    scored_tokens: int
    conditions: dict[MrcrCondition, dict[str, np.ndarray]]


def mrcr_format() -> ChatLmDatasetFormat:
    return ChatLmDatasetFormat(chat_template=_MRCR_CHAT_TEMPLATE, pack=False)


def _distance_band(distance: int, bounds: tuple[int, ...]) -> str:
    lower = 0
    for upper in bounds:
        if distance <= upper:
            return f"distance_{lower}_{upper}"
        lower = upper + 1
    return f"distance_{lower}_plus"


def _validate_preamble(content: str) -> None:
    if not content.startswith(MRCR_PREAMBLE_PREFIX):
        raise ValueError("MRCR prompt does not start with the official worked-example preamble")
    if content.count(_EXAMPLE_START) != 2 or content.count(_EXAMPLE_END) != 2:
        raise ValueError("MRCR preamble must contain exactly two complete official worked examples")
    if content[content.rindex(_EXAMPLE_END) + len(_EXAMPLE_END) :].strip():
        raise ValueError("The first MRCR message must contain only the official worked examples")


def _render_prompt(messages: list[_MrcrMessage], assistant_prefix: str) -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    position = 0
    for message in messages:
        header = f"{message['role'].capitalize()}: "
        part = f"{header}{message['content']}\n"
        start = position + len(header)
        spans.append((start, start + len(message["content"])))
        parts.append(part)
        position += len(part)
    parts.append(f"Assistant: {assistant_prefix}")
    return "".join(parts), spans


def _preprocess(tokenizer: Any, prompt: str, target: str) -> dict[str, np.ndarray]:
    """Tokenize the prompt and response body apart, so no scored token overlaps the supplied answer prefix.

    The chat processor masks by offsets into one whole-render tokenization, where a BPE merge across the
    boundary would score a token that starts in the prefix. #8701 scores the body tokenized on its own.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    ids = [*prompt_ids, *target_ids, tokenizer.eos_token_id]
    return {
        "input_ids": np.asarray(ids, dtype=np.int32),
        "assistant_masks": np.asarray([0] * len(prompt_ids) + [1] * len(target_ids) + [0], dtype=np.int32),
    }


def _evidence_distance(
    offset_tokenizer: Any, *, prompt: str, response_span: tuple[int, int], target: str, eos_token: str
) -> int:
    """Measure both endpoints from offsets produced by one full rendered tokenization."""
    encoded = offset_tokenizer(prompt + target + eos_token, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
    response_start, response_end = response_span
    response_tokens = [i for i, (start, end) in enumerate(offsets) if end > response_start and start < response_end]
    target_start = len(prompt)
    target_end = target_start + len(target)
    target_tokens = [i for i, (start, end) in enumerate(offsets) if end > target_start and start < target_end]
    if not response_tokens or not target_tokens:
        raise ValueError("Could not locate the selected response and scored target in full-render token offsets")
    return max(0, target_tokens[0] - response_tokens[-1] - 1)


def _source_id(*, parquet_path: str, row_index: int, prompt: str, answer: str, needles: int) -> str:
    identity = {
        "answer": answer,
        "dataset_revision": MRCR_DATASET_REVISION,
        "n_needles": needles,
        "parquet_path": parquet_path,
        "prompt": prompt,
        "row_index": row_index,
    }
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _condition_prompts(
    messages: list[_MrcrMessage], *, selected_request: str, needles: int, nonce: str
) -> dict[MrcrCondition, str]:
    conversation = messages[1:-1]
    if len(conversation) % 2:
        raise ValueError("MRCR target conversation must contain complete user/assistant pairs")
    pairs = [conversation[index : index + 2] for index in range(0, len(conversation), 2)]
    if any(request["role"] != "user" or response["role"] != "assistant" for request, response in pairs):
        raise ValueError("MRCR target conversation must alternate user and assistant messages")
    needle_turns = [message for pair in pairs if pair[0]["content"] == selected_request for message in pair]
    distractor_turns = [message for pair in pairs if pair[0]["content"] != selected_request for message in pair]
    if len(needle_turns) != 2 * needles:
        raise ValueError("n_needles does not match the number of selected-request occurrences")
    preamble, query = messages[0], messages[-1]
    return {
        MrcrCondition.FULL_CONTEXT: _render_prompt(messages, nonce)[0],
        MrcrCondition.QUERY_ONLY: _render_prompt([preamble, query], nonce)[0],
        MrcrCondition.NEEDLE_ONLY: _render_prompt([preamble, *needle_turns, query], nonce)[0],
        MrcrCondition.DISTRACTOR_ONLY: _render_prompt([preamble, *distractor_turns, query], nonce)[0],
    }


def _build_source(
    row: dict[str, Any],
    *,
    parquet_path: str,
    row_index: int,
    canonical: dict[str, np.ndarray],
    canonical_prompt: str,
    response_span: tuple[int, int],
    target: str,
    tokenizer: Any,
    offset_tokenizer: Any,
    distance_bounds: tuple[int, ...],
) -> _Source:
    messages: list[_MrcrMessage] = json.loads(row["prompt"])
    needles: int = row["n_needles"]
    selected_request = messages[row["desired_msg_index"]]["content"]
    prompts = _condition_prompts(messages, selected_request=selected_request, needles=needles, nonce=row["nonce"])
    conditions = {
        condition: canonical if condition == MrcrCondition.FULL_CONTEXT else _preprocess(tokenizer, prompt, target)
        for condition, prompt in prompts.items()
    }
    target_ids = canonical["input_ids"][canonical["assistant_masks"].astype(bool)]
    if target_ids.size == 0:
        raise ValueError("MRCR source has no scored tokens")
    for condition, processed in conditions.items():
        if not np.array_equal(target_ids, processed["input_ids"][processed["assistant_masks"].astype(bool)]):
            raise ValueError(f"MRCR scored tokens differ for condition {condition.value}")
    distance = _evidence_distance(
        offset_tokenizer,
        prompt=canonical_prompt,
        response_span=response_span,
        target=target,
        eos_token=tokenizer.eos_token or "",
    )
    return _Source(
        source_id=_source_id(
            parquet_path=parquet_path, row_index=row_index, prompt=row["prompt"], answer=row["answer"], needles=needles
        ),
        n_needles=needles,
        distance_band=_distance_band(distance, distance_bounds),
        evidence_distance_tokens=distance,
        canonical_length=len(canonical["input_ids"]),
        scored_tokens=int(target_ids.size),
        conditions=conditions,
    )


def _validated_row(row: dict[str, Any]) -> tuple[list[_MrcrMessage], str]:
    """Check one source row's shape and return its messages and scored response body."""
    messages: list[_MrcrMessage] = json.loads(row["prompt"])
    if row["n_needles"] not in MRCR_NEEDLE_COUNTS:
        raise ValueError(f"Unsupported MRCR needle count: {row['n_needles']}")
    if len(messages) < 4 or messages[0]["role"] != "user" or messages[-1]["role"] != "user":
        raise ValueError("MRCR prompt must begin with the worked examples and end with the final user query")
    _validate_preamble(messages[0]["content"])
    desired = row["desired_msg_index"]
    if desired < 1 or desired + 1 >= len(messages) - 1:
        raise ValueError("desired_msg_index does not identify a target-conversation request")
    if messages[desired]["role"] != "user" or messages[desired + 1]["role"] != "assistant":
        raise ValueError("desired_msg_index must identify a user request followed by an assistant response")
    if not row["answer"].startswith(row["nonce"]):
        raise ValueError("MRCR answer does not begin with random_string_to_prepend")
    target = row["answer"].removeprefix(row["nonce"])
    if not target or messages[desired + 1]["content"] != target:
        raise ValueError("The assistant response following desired_msg_index does not match the answer body")
    return messages, target


def _sources_in_cell(config: MrcrTransformConfig) -> Iterator[_Source]:
    input_root = StoragePath(config.input_path)
    input_files = sorted(StoragePath(prefix_join(config.input_path, "**/*.parquet")).expand_glob(), key=str)
    if not input_files:
        raise FileNotFoundError(f"No MRCR parquet files found under {config.input_path}")
    tokenizer = load_tokenizer(config.tokenizer)
    offset_tokenizer = tokenizer.as_hf_tokenizer()
    columns = ["prompt", "answer", "random_string_to_prepend", "n_needles", "desired_msg_index"]
    lower = config.context_cap_lower - _LENGTH_FILTER_MARGIN
    upper = config.context_cap + _LENGTH_FILTER_MARGIN
    for input_file in input_files:
        parquet_path = input_file.relative_to(input_root)
        row_index = 0
        with input_file.open("rb") as source:
            for batch in pq.ParquetFile(source).iter_batches(batch_size=_LENGTH_FILTER_BATCH_SIZE, columns=columns):
                rows = [{**row, "nonce": row["random_string_to_prepend"]} for row in batch.to_pylist()]
                validated = [_validated_row(row) for row in rows]
                renders = [
                    _render_prompt(messages, row["nonce"]) for row, (messages, _) in zip(rows, validated, strict=True)
                ]
                # The target is short, so the prompt alone decides the row unless it sits at the margin.
                lengths = [
                    len(ids) + len(tokenizer.encode(target))
                    for ids, (_, target) in zip(
                        tokenizer.encode_batch([prompt for prompt, _ in renders]), validated, strict=True
                    )
                ]
                for row, (_, target), (prompt, spans), length in zip(rows, validated, renders, lengths, strict=True):
                    index = row_index
                    row_index += 1
                    if not lower < length <= upper:
                        continue
                    canonical = _preprocess(tokenizer, prompt, target)
                    if not config.context_cap_lower < len(canonical["input_ids"]) <= config.context_cap:
                        continue
                    yield _build_source(
                        row,
                        parquet_path=parquet_path,
                        row_index=index,
                        canonical=canonical,
                        canonical_prompt=prompt,
                        response_span=spans[row["desired_msg_index"] + 1],
                        target=target,
                        tokenizer=tokenizer,
                        offset_tokenizer=offset_tokenizer,
                        distance_bounds=config.distance_bounds,
                    )


def _check_expected(config: MrcrTransformConfig, sources: list[_Source]) -> None:
    if not sources:
        raise ValueError(f"No MRCR source falls in ({config.context_cap_lower}, {config.context_cap}]")
    scored = sum(source.scored_tokens for source in sources)
    if config.expected_sources is not None and len(sources) != config.expected_sources:
        raise ValueError(f"Expected {config.expected_sources} MRCR sources, found {len(sources)}")
    if config.expected_scored_tokens is not None and scored != config.expected_scored_tokens:
        raise ValueError(f"Expected {config.expected_scored_tokens} scored MRCR tokens, found {scored}")


def transform_mrcr(config: MrcrTransformConfig) -> None:
    """Write the four-condition cache, its row manifest and summary stats for one context cell."""
    sources = sorted(_sources_in_cell(config), key=lambda source: (-source.canonical_length, source.source_id))
    _check_expected(config, sources)
    # Each source's four rows stay adjacent, so any whole number of sources keeps every pair complete.
    rows = [(source, condition) for source in sources for condition in MrcrCondition]
    cache_path = prefix_join(config.output_path, MRCR_CACHE_SPLIT)
    write_levanter_cache(
        (source.conditions[condition] for source, condition in rows),
        cache_path,
        metadata={"tokenizer": config.tokenizer, "dataset_revision": MRCR_DATASET_REVISION},
    )
    manifest = (
        {
            "source_id": source.source_id,
            "condition": condition.value,
            "n_needles": source.n_needles,
            "distance_band": source.distance_band,
            "evidence_distance_tokens": source.evidence_distance_tokens,
            "canonical_full_length_tokens": source.canonical_length,
            "length_tokens": len(source.conditions[condition]["input_ids"]),
            "scored_tokens": source.scored_tokens,
        }
        for source, condition in rows
    )
    output = StoragePath(config.output_path)
    (output / MRCR_MANIFEST_NAME).write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in manifest))
    stats = {
        "dataset_revision": MRCR_DATASET_REVISION,
        "tokenizer": config.tokenizer,
        "prompt_variant": MRCR_PROMPT_VARIANT,
        "context_cap": config.context_cap,
        "context_cap_lower": config.context_cap_lower,
        "sources": len(sources),
        "scored_tokens": sum(source.scored_tokens for source in sources),
    }
    (output / MRCR_STATS_NAME).write_text(json.dumps(stats, indent=2, sort_keys=True))


def mrcr_dataset(*, tokenizer: str = marin_tokenizer) -> ArtifactStep[Artifact]:
    """The 262K two-shot MRCR condition cache, built from the pinned source revision."""
    raw = hf_download(
        "raw/openai/mrcr",
        hf_id="openai/mrcr",
        revision=MRCR_DATASET_REVISION,
        version=MRCR_RAW_VERSION,
        urls_glob=["*needle/*.parquet"],
        pin=f"raw/openai/mrcr/{MRCR_RAW_VERSION}",
    )

    def build_config(ctx: StepContext) -> MrcrTransformConfig:
        return MrcrTransformConfig(
            input_path=ctx.artifact_path(raw),
            output_path=ctx.output_path,
            tokenizer=tokenizer,
            expected_sources=MRCR_EXPECTED_SOURCES,
            expected_scored_tokens=MRCR_EXPECTED_SCORED_TOKENS,
        )

    return ArtifactStep(
        name=f"eval/mrcr-hero/data/{MRCR_PROMPT_VARIANT}-cap{MRCR_CONTEXT_CAP}",
        version=MRCR_VERSION,
        artifact_type=Artifact,
        run=transform_mrcr,
        build_config=build_config,
        deps=(raw,),
    )


if __name__ == "__main__":
    dataset_main({"two_shot_cap262144": mrcr_dataset()})
