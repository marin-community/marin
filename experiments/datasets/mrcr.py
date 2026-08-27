# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paired OpenAI MRCR likelihood datasets for base-model evaluation."""

import gzip
import hashlib
import io
import json
import os
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TextIO, TypedDict

import numpy as np
import pyarrow.parquet as pq
from fray import ResourceConfig
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.tokenizers import load_tokenizer
from levanter.utils import fsspec_utils
from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, InputName, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from rigging.filesystem import open_url

from experiments.marin_models import marin_tokenizer

MRCR_DATASET_REVISION = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"
MRCR_CONTEXT_CAPS: tuple[int, ...] = (8_192, 16_384, 32_768, 65_536, 131_072, 262_144)
MRCR_NEEDLE_COUNTS: tuple[int, ...] = (2, 4, 8)
MRCR_DISTANCE_BOUNDS: tuple[int, ...] = (32_768, 65_536, 131_072)
MRCR_BOOTSTRAP_SAMPLES = 10_000
MRCR_PREAMBLE_PREFIX = "Here are some examples of conversations succeeded by a follow-up question answered correctly:"

_EXAMPLE_START = "======EXAMPLE======"
_EXAMPLE_END = "======END EXAMPLE======"
_MRCR_CHAT_TEMPLATE = (
    "{{ messages[0]['content'] }}{% generation %}{{ messages[1]['content'] }}{% endgeneration %}{{ eos_token }}"
)


def _join(*parts: str) -> str:
    return "/".join(part.rstrip("/") if index == 0 else part.strip("/") for index, part in enumerate(parts))


class MrcrCondition(StrEnum):
    FULL_CONTEXT = "full_context"
    QUERY_ONLY = "query_only"
    NEEDLE_ONLY = "needle_only"
    DISTRACTOR_ONLY = "distractor_only"


class MrcrPromptVariant(StrEnum):
    TWO_SHOT = "two_shot"
    ONE_SHOT = "one_shot"
    TWO_SHOT_NO_PREFIX = "two_shot_no_prefix"


class _MrcrMessage(TypedDict):
    role: str
    content: str


def _copy_message(message: _MrcrMessage) -> _MrcrMessage:
    return {"role": message["role"], "content": message["content"]}


@dataclass(frozen=True)
class MrcrDatasetBundle:
    """Tokenized dataset steps and their explicit validation artifacts."""

    datasets: dict[str, ExecutorStep[TokenizeConfig]]
    manifests: dict[str, InputName]
    stats: InputName


@dataclass(frozen=True)
class MrcrTransformConfig:
    input_path: str
    output_path: str
    tokenizer: str
    context_caps: tuple[int, ...] = MRCR_CONTEXT_CAPS
    distance_bounds: tuple[int, ...] = MRCR_DISTANCE_BOUNDS
    prompt_variants: tuple[MrcrPromptVariant, ...] = (
        MrcrPromptVariant.TWO_SHOT,
        MrcrPromptVariant.ONE_SHOT,
        MrcrPromptVariant.TWO_SHOT_NO_PREFIX,
    )


def _mrcr_format() -> ChatLmDatasetFormat:
    return ChatLmDatasetFormat(chat_template=_MRCR_CHAT_TEMPLATE, pack=False)


def _distance_band(distance: int, bounds: tuple[int, ...]) -> str:
    lower = 0
    for upper in bounds:
        if distance <= upper:
            return f"distance_{lower}_{upper}"
        lower = upper + 1
    return f"distance_{lower}_plus"


def _distance_bands(bounds: tuple[int, ...]) -> tuple[str, ...]:
    return *(_distance_band(bound, bounds) for bound in bounds), f"distance_{bounds[-1] + 1}_plus"


def _distance_bands_for_cap(cap: int, bounds: tuple[int, ...]) -> tuple[str, ...]:
    bands = _distance_bands(bounds)
    lower_bounds = (0, *(bound + 1 for bound in bounds))
    return tuple(band for band, lower in zip(bands, lower_bounds, strict=True) if lower <= cap)


def _validate_preamble(content: str) -> tuple[str, str]:
    if not content.startswith(MRCR_PREAMBLE_PREFIX):
        raise ValueError("MRCR prompt does not start with the official worked-example preamble")

    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    while (index := content.find(_EXAMPLE_START, cursor)) >= 0:
        starts.append(index)
        cursor = index + len(_EXAMPLE_START)
    cursor = 0
    while (index := content.find(_EXAMPLE_END, cursor)) >= 0:
        ends.append(index)
        cursor = index + len(_EXAMPLE_END)

    if len(starts) != 2 or len(ends) != 2 or not (starts[0] < ends[0] < starts[1] < ends[1]):
        raise ValueError("MRCR preamble must contain exactly two complete official worked examples")
    if content[len(MRCR_PREAMBLE_PREFIX) : starts[0]].strip():
        raise ValueError("Unexpected content before the first official worked example")
    second_end = ends[1] + len(_EXAMPLE_END)
    if content[second_end:].strip():
        raise ValueError("The first MRCR message must contain only the official worked examples")

    first_end = ends[0] + len(_EXAMPLE_END)
    one_shot = content[:first_end] + content[second_end:]
    return content, one_shot


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


def _preprocess(preprocessor: Any, prompt: str, target: str) -> dict[str, np.ndarray]:
    processed = preprocessor(
        [
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ]
            }
        ]
    )[0]
    return {
        "input_ids": np.asarray(processed["input_ids"], dtype=np.int32),
        "assistant_masks": np.asarray(processed["assistant_masks"], dtype=np.int32),
    }


def _evidence_distance(
    offset_tokenizer: Any,
    *,
    prompt: str,
    response_span: tuple[int, int],
    target: str,
    eos_token: str,
) -> int:
    """Measure both endpoints from offsets produced by one full rendered tokenization."""

    rendered = prompt + target + eos_token
    encoded = offset_tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
    response_start, response_end = response_span
    response_tokens = [
        index for index, (start, end) in enumerate(offsets) if end > response_start and start < response_end
    ]
    target_start = len(prompt)
    target_end = target_start + len(target)
    target_tokens = [index for index, (start, end) in enumerate(offsets) if end > target_start and start < target_end]
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


def _cap_for_length(length: int, caps: tuple[int, ...]) -> int | None:
    return next((cap for cap in caps if length <= cap), None)


def _update_range(stats: dict[str, int], name: str, value: int) -> None:
    stats[f"{name}_min"] = min(stats.get(f"{name}_min", value), value)
    stats[f"{name}_max"] = max(stats.get(f"{name}_max", value), value)


def _writer(
    stack: ExitStack,
    writers: dict[str, TextIO],
    path: str,
    *,
    compressed: bool,
) -> TextIO:
    if path in writers:
        return writers[path]
    fsspec_utils.mkdirs(os.path.dirname(path))
    raw = stack.enter_context(open_url(path, "wb"))
    if compressed:
        gzip_file = stack.enter_context(gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0))
        writers[path] = stack.enter_context(io.TextIOWrapper(gzip_file, encoding="utf-8"))
    else:
        writers[path] = stack.enter_context(io.TextIOWrapper(raw, encoding="utf-8"))
    return writers[path]


def transform_mrcr(config: MrcrTransformConfig) -> None:
    """Build complete paired MRCR records in canonical tokenizer-specific bins."""

    if not config.context_caps or any(
        a >= b for a, b in zip(config.context_caps, config.context_caps[1:], strict=False)
    ):
        raise ValueError("MRCR context caps must be strictly increasing")
    if not config.distance_bounds or any(
        a >= b for a, b in zip(config.distance_bounds, config.distance_bounds[1:], strict=False)
    ):
        raise ValueError("MRCR distance bounds must be strictly increasing")

    input_files = sorted(fsspec_utils.expand_glob(_join(config.input_path, "**/*.parquet")))
    if not input_files:
        raise FileNotFoundError(f"No MRCR parquet files found under {config.input_path}")

    tokenizer = load_tokenizer(config.tokenizer)
    offset_tokenizer = tokenizer.as_hf_tokenizer()
    preprocessor = _mrcr_format().build_preprocessor(tokenizer)
    accepted: dict[str, dict[str, int]] = {}
    excluded = {"total": 0, **{f"{needles}needle": 0 for needles in MRCR_NEEDLE_COUNTS}}
    max_query_only_tokens = {variant.value: 0 for variant in config.prompt_variants}
    cap_counts = {cap: 0 for cap in config.context_caps}

    with ExitStack() as stack:
        writers: dict[str, TextIO] = {}
        manifest_writers: dict[str, TextIO] = {}
        for input_file in input_files:
            parquet_path = os.path.relpath(input_file.removeprefix("file://"), config.input_path.removeprefix("file://"))
            with open_url(input_file, "rb") as source:
                parquet = pq.ParquetFile(source)
                row_index = 0
                columns = [
                    "prompt",
                    "answer",
                    "random_string_to_prepend",
                    "n_needles",
                    "desired_msg_index",
                ]
                for batch in parquet.iter_batches(batch_size=16, columns=columns):
                    for row in batch.to_pylist():
                        raw_prompt = row["prompt"]
                        messages: list[_MrcrMessage] = json.loads(raw_prompt)
                        answer: str = row["answer"]
                        nonce: str = row["random_string_to_prepend"]
                        needles: int = row["n_needles"]
                        desired_index: int = row["desired_msg_index"]
                        if needles not in MRCR_NEEDLE_COUNTS:
                            raise ValueError(f"Unsupported MRCR needle count: {needles}")
                        if len(messages) < 4 or messages[0]["role"] != "user":
                            raise ValueError("MRCR prompt must begin with a user worked-example preamble")
                        two_shot_preamble, one_shot_preamble = _validate_preamble(messages[0]["content"])
                        if messages[-1]["role"] != "user":
                            raise ValueError("The last MRCR message must be the final user query")
                        if desired_index < 1 or desired_index + 1 >= len(messages) - 1:
                            raise ValueError("desired_msg_index does not identify a target-conversation request")
                        selected_request = messages[desired_index]
                        selected_response = messages[desired_index + 1]
                        if selected_request["role"] != "user" or selected_response["role"] != "assistant":
                            raise ValueError(
                                "desired_msg_index must identify a user request followed by an assistant response"
                            )
                        conversation = messages[1:-1]
                        if len(conversation) % 2:
                            raise ValueError("MRCR target conversation must contain complete user/assistant pairs")
                        conversation_pairs: list[tuple[_MrcrMessage, _MrcrMessage]] = []
                        for index in range(0, len(conversation), 2):
                            request, response = conversation[index : index + 2]
                            if request["role"] != "user" or response["role"] != "assistant":
                                raise ValueError("MRCR target conversation must alternate user and assistant messages")
                            conversation_pairs.append((request, response))
                        needle_pairs = [
                            pair for pair in conversation_pairs if pair[0]["content"] == selected_request["content"]
                        ]
                        distractor_pairs = [
                            pair for pair in conversation_pairs if pair[0]["content"] != selected_request["content"]
                        ]
                        if len(needle_pairs) != needles:
                            raise ValueError("n_needles does not match the number of selected-request occurrences")
                        if not answer.startswith(nonce):
                            raise ValueError("MRCR answer does not begin with random_string_to_prepend")
                        target = answer.removeprefix(nonce)
                        if not target:
                            raise ValueError("Removing the MRCR random prefix left an empty response body")
                        if selected_response["content"] != target:
                            raise ValueError(
                                "The assistant response following desired_msg_index does not match the answer body"
                            )

                        source_id = _source_id(
                            parquet_path=parquet_path,
                            row_index=row_index,
                            prompt=raw_prompt,
                            answer=answer,
                            needles=needles,
                        )
                        final_query = messages[-1]["content"]
                        exact_directive = f"Prepend {nonce} to "
                        if not final_query.startswith(exact_directive):
                            raise ValueError("The final MRCR query cannot be rewritten by the exact no-prefix rule")
                        no_prefix_query = "Return " + final_query.removeprefix(exact_directive)

                        canonical_messages = [_copy_message(message) for message in messages]
                        canonical_prompt, canonical_spans = _render_prompt(canonical_messages, nonce)
                        canonical = _preprocess(preprocessor, canonical_prompt, target)
                        canonical_length = len(canonical["input_ids"])
                        cap = _cap_for_length(canonical_length, config.context_caps)
                        if cap is None:
                            excluded["total"] += 1
                            excluded[f"{needles}needle"] += 1
                            row_index += 1
                            continue
                        cap_counts[cap] += 1
                        distance = _evidence_distance(
                            offset_tokenizer,
                            prompt=canonical_prompt,
                            response_span=canonical_spans[desired_index + 1],
                            target=target,
                            eos_token=tokenizer.eos_token or "",
                        )
                        distance_band = _distance_band(distance, config.distance_bounds)

                        for variant in config.prompt_variants:
                            preamble = one_shot_preamble if variant == MrcrPromptVariant.ONE_SHOT else two_shot_preamble
                            query = no_prefix_query if variant == MrcrPromptVariant.TWO_SHOT_NO_PREFIX else final_query
                            assistant_prefix = "" if variant == MrcrPromptVariant.TWO_SHOT_NO_PREFIX else nonce
                            variant_messages: list[_MrcrMessage] = [
                                {"role": "user", "content": preamble},
                                *[_copy_message(message) for message in messages[1:-1]],
                                {"role": "user", "content": query},
                            ]
                            full_prompt, _ = _render_prompt(variant_messages, assistant_prefix)
                            query_prompt, _ = _render_prompt(
                                [variant_messages[0], variant_messages[-1]], assistant_prefix
                            )
                            needle_prompt, _ = _render_prompt(
                                [
                                    variant_messages[0],
                                    *[_copy_message(message) for pair in needle_pairs for message in pair],
                                    variant_messages[-1],
                                ],
                                assistant_prefix,
                            )
                            distractor_prompt, _ = _render_prompt(
                                [
                                    variant_messages[0],
                                    *[_copy_message(message) for pair in distractor_pairs for message in pair],
                                    variant_messages[-1],
                                ],
                                assistant_prefix,
                            )
                            condition_prompts = {
                                MrcrCondition.FULL_CONTEXT: full_prompt,
                                MrcrCondition.QUERY_ONLY: query_prompt,
                                MrcrCondition.NEEDLE_ONLY: needle_prompt,
                                MrcrCondition.DISTRACTOR_ONLY: distractor_prompt,
                            }
                            processed_conditions = {
                                condition: _preprocess(preprocessor, prompt, target)
                                for condition, prompt in condition_prompts.items()
                            }
                            full = processed_conditions[MrcrCondition.FULL_CONTEXT]
                            if len(full["input_ids"]) > cap:
                                raise ValueError("A prompt variant exceeded its canonical two-shot context cap")
                            full_target_ids = full["input_ids"][full["assistant_masks"].astype(bool)]
                            scored_tokens = int(full["assistant_masks"].sum())
                            for condition, processed in processed_conditions.items():
                                target_ids = processed["input_ids"][processed["assistant_masks"].astype(bool)]
                                if not np.array_equal(full_target_ids, target_ids):
                                    raise ValueError(f"MRCR target tokenization differs for condition {condition.value}")
                                if scored_tokens <= 0 or scored_tokens != int(processed["assistant_masks"].sum()):
                                    raise ValueError(f"MRCR scored-token mask differs for condition {condition.value}")
                            query_only = processed_conditions[MrcrCondition.QUERY_ONLY]
                            max_query_only_tokens[variant.value] = max(
                                max_query_only_tokens[variant.value], len(query_only["input_ids"])
                            )

                            cell = f"{variant.value}/cap_{cap}/{needles}needle/{distance_band}"
                            cell_stats = accepted.setdefault(cell, {"examples": 0, "scored_tokens": 0})
                            cell_stats["examples"] += 1
                            cell_stats["scored_tokens"] += scored_tokens
                            _update_range(cell_stats, "canonical_full_length_tokens", canonical_length)
                            _update_range(cell_stats, "variant_full_length_tokens", len(full["input_ids"]))
                            _update_range(cell_stats, "evidence_distance_tokens", distance)
                            manifest_record = {
                                "source_id": source_id,
                                "canonical_full_length_tokens": canonical_length,
                                "variant_full_length_tokens": len(full["input_ids"]),
                                "evidence_distance_tokens": distance,
                                "scored_tokens": scored_tokens,
                            }
                            manifest_path = _join(config.output_path, cell, "manifest.jsonl")
                            manifest = _writer(stack, manifest_writers, manifest_path, compressed=False)
                            manifest.write(json.dumps(manifest_record, sort_keys=True) + "\n")

                            for condition, prompt in condition_prompts.items():
                                record = {
                                    "messages": [
                                        {"role": "user", "content": prompt},
                                        {"role": "assistant", "content": target},
                                    ],
                                    "source_id": source_id,
                                    "prompt_variant": variant.value,
                                    "context_cap": cap,
                                    "n_needles": needles,
                                    "canonical_full_length_tokens": canonical_length,
                                    "variant_full_length_tokens": len(full["input_ids"]),
                                    "evidence_distance_tokens": distance,
                                    "distance_band": distance_band,
                                    "condition": condition.value,
                                }
                                output = _writer(
                                    stack,
                                    writers,
                                    _join(config.output_path, cell, f"{condition.value}.jsonl.gz"),
                                    compressed=True,
                                )
                                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        row_index += 1

        empty_caps = [cap for cap, count in cap_counts.items() if count == 0]
        if empty_caps:
            raise ValueError(f"Requested MRCR context caps have no accepted examples: {empty_caps}")

        stats_path = _join(config.output_path, "stats.json")
        fsspec_utils.mkdirs(os.path.dirname(stats_path))
        with open_url(stats_path, "w") as output:
            json.dump(
                {
                    "dataset_revision": MRCR_DATASET_REVISION,
                    "tokenizer": config.tokenizer,
                    "accepted": accepted,
                    f"excluded_over_{config.context_caps[-1]}": excluded,
                    "max_query_only_tokens": max_query_only_tokens,
                },
                output,
                indent=2,
                sort_keys=True,
            )
            output.write("\n")


def _mrcr_tags(
    variant: MrcrPromptVariant,
    cap: int,
    needles: int,
    distance_band: str,
    condition: MrcrCondition,
) -> tuple[str, ...]:
    prefix = f"mrcr/{variant.value}"
    suffix = condition.value
    return (
        f"{prefix}/{suffix}",
        f"{prefix}/cap_{cap}/{suffix}",
        f"{prefix}/{needles}needle/{suffix}",
        f"{prefix}/{distance_band}/{suffix}",
        f"{prefix}/cap_{cap}/{needles}needle/{distance_band}/{suffix}",
    )


def mrcr_datasets(
    *,
    tokenizer: str = marin_tokenizer,
    context_caps: tuple[int, ...] = MRCR_CONTEXT_CAPS,
    needle_counts: tuple[int, ...] = MRCR_NEEDLE_COUNTS,
    distance_bounds: tuple[int, ...] = MRCR_DISTANCE_BOUNDS,
    prompt_variants: tuple[MrcrPromptVariant, ...] = (MrcrPromptVariant.TWO_SHOT,),
) -> MrcrDatasetBundle:
    """Return complete, unpacked paired validation caches plus manifests and stats."""

    raw = ExecutorStep(
        name="raw/openai/mrcr",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id="openai/mrcr",
            revision=MRCR_DATASET_REVISION,
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
            hf_urls_glob=["*needle/*.parquet"],
        ),
    )
    processed = ExecutorStep(
        name="processed/openai/mrcr-paired",
        fn=remote(transform_mrcr, resources=ResourceConfig.with_cpu(cpu=8, ram="32g", disk="10g")),
        config=MrcrTransformConfig(
            input_path=raw.as_input_name(),  # type: ignore[arg-type]
            output_path=this_output_path(),  # type: ignore[arg-type]
            tokenizer=tokenizer,
            context_caps=context_caps,
            distance_bounds=distance_bounds,
            prompt_variants=prompt_variants,
        ),
    )

    datasets: dict[str, ExecutorStep[TokenizeConfig]] = {}
    manifests: dict[str, InputName] = {}
    for variant in prompt_variants:
        for cap in context_caps:
            for needles in needle_counts:
                for distance_band in _distance_bands_for_cap(cap, distance_bounds):
                    cell = f"{variant.value}/cap_{cap}/{needles}needle/{distance_band}"
                    manifests[cell] = processed.cd(f"{cell}/manifest.jsonl")
                    for condition in MrcrCondition:
                        key = f"{cell}/{condition.value}"
                        datasets[key] = ExecutorStep(
                            name=f"tokenized/mrcr/{key}",
                            fn=tokenize,
                            config=TokenizeConfig(
                                train_paths=[],
                                validation_paths=[processed.cd(f"{cell}/{condition.value}.jsonl.gz")],
                                cache_path=this_output_path(),
                                tokenizer=versioned(tokenizer),
                                format=_mrcr_format(),
                                tags=list(_mrcr_tags(variant, cap, needles, distance_band, condition)),
                                num_shards=1,
                                levanter_batch_size=1,
                            ),
                        )
    return MrcrDatasetBundle(datasets=datasets, manifests=manifests, stats=processed.cd("stats.json"))
