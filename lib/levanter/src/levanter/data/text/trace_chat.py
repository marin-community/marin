# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Trace chat preprocessing for per-span-type language-model evaluation.

This module keeps the trace-specific labeled evaluation path together: it renders chat
templates, recovers token spans for each source message, assigns exclusive integer labels
to trace regions, and builds packed `LabeledLmExample` batches for `LabeledEvaluator`.

For example, an agent trace with a user request, an assistant tool call, a tool
observation, and a final assistant answer can produce separate loss labels for
assistant prose, tool-call JSON, observations, and final answers. The evaluator
then reports per-label losses such as `assistant_text/loss`, `tool_call/loss`,
and aggregate groups like `assistant/loss`.
"""

import ast
import functools
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis

from levanter.data._preprocessor import BatchProcessor
from levanter.data.dataset import MappedAsyncDataset
from levanter.data.packing import GreedyPrepackedDataset
from levanter.data.sharded_datasource import ShardedDataSource
from levanter.data.text.datasets import _resolve_pack_config
from levanter.data.text.examples import GrugAttentionMask, LabeledLmExample, LossLabelSpec
from levanter.store.cache import CacheOptions, TreeCache
from levanter.tokenizers import MarinTokenizer


TRACE_LABEL_DONT_SCORE = 0
TRACE_LABEL_ASSISTANT_TEXT = 1
TRACE_LABEL_ASSISTANT_TOOL_CALL = 2
TRACE_LABEL_OBSERVATION = 3
TRACE_LABEL_PATCH = 4
TRACE_LABEL_OUTCOME = 5
TRACE_LABEL_FINAL_ASSISTANT = 6
_FIRST_CUSTOM_TRACE_LABEL = 32


class ProcessedTraceChatDict(TypedDict):
    input_ids: np.ndarray
    loss_labels: np.ndarray


def _single_cpu_sharding() -> jax.sharding.SingleDeviceSharding:
    return jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])


_TEXT_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_AGGREGATE_TAGS = {"assistant", "tool", "tool_call", "observation", "assistant_text", "final_assistant"}


def _normalize_chat_message(message: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    normalized.setdefault("content", "")
    normalized.setdefault("reasoning_content", None)
    if "tool_calls" in normalized and normalized["tool_calls"] is None:
        normalized["tool_calls"] = []
    return normalized


def _parsed_text_tool_call(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(text.strip())
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Could not parse <tool_call> payload: {text!r}") from e

    if not isinstance(payload, Mapping):
        raise ValueError(f"<tool_call> payload must be a mapping, got {type(payload).__name__}.")

    function_payload = payload.get("function")
    if isinstance(function_payload, Mapping):
        name = function_payload.get("name")
        arguments = function_payload.get("arguments", function_payload.get("args"))
    else:
        name = payload.get("name")
        arguments = payload.get("arguments", payload.get("args"))

    if not isinstance(name, str):
        raise ValueError("<tool_call> payload must include a string function name.")
    if arguments is None:
        raise ValueError("<tool_call> payload must include function arguments.")

    tool_call: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }
    tool_call_id = payload.get("id")
    if isinstance(tool_call_id, str):
        tool_call["id"] = tool_call_id
    return tool_call


def _message_without_tool_calls(message: Mapping[str, Any], content: str) -> dict[str, Any]:
    chunk = dict(message)
    chunk["content"] = content
    chunk["tool_calls"] = []
    return chunk


def _split_text_tool_call_message(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if message.get("role") != "assistant" or not isinstance(content, str) or message.get("tool_calls"):
        return [dict(message)]

    matches = list(_TEXT_TOOL_CALL_PATTERN.finditer(content))
    if not matches:
        return [dict(message)]

    split_messages: list[dict[str, Any]] = []
    cursor = 0
    parsed_any = False
    for match in matches:
        prefix = content[cursor : match.start()]
        if prefix.strip():
            split_messages.append(_message_without_tool_calls(message, prefix))

        tool_call = _parsed_text_tool_call(match.group(1))
        # Prefix/suffix text stays in adjacent assistant messages; this chunk represents only the structured call.
        split_messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
        parsed_any = True
        cursor = match.end()

    suffix = content[cursor:]
    if suffix.strip():
        split_messages.append(_message_without_tool_calls(message, suffix))

    if not parsed_any:
        return [dict(message)]
    return split_messages


def _split_text_tool_call_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    split_messages: list[dict[str, Any]] = []
    for message in messages:
        split_messages.extend(_split_text_tool_call_message(message))
    return split_messages


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None

    haystack_ids = list(haystack)
    needle_ids = list(needle)
    needle_len = len(needle)
    for start in range(len(haystack) - needle_len + 1):
        if haystack_ids[start : start + needle_len] == needle_ids:
            return start
    return None


def _normalize_chat_kwargs(kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    kwargs_dict = dict(kwargs) if kwargs is not None else {}
    kwargs_dict.setdefault("tools", None)
    return kwargs_dict


def loss_label_spec_for_trace_tags(loss_tags: Sequence[str]) -> LossLabelSpec:
    id_to_name = {
        TRACE_LABEL_DONT_SCORE: "dont_score",
        TRACE_LABEL_ASSISTANT_TEXT: "assistant_text",
        TRACE_LABEL_ASSISTANT_TOOL_CALL: "assistant_tool_call",
        TRACE_LABEL_OBSERVATION: "observation",
        TRACE_LABEL_PATCH: "patch",
        TRACE_LABEL_OUTCOME: "outcome",
        TRACE_LABEL_FINAL_ASSISTANT: "final_assistant",
    }

    custom_label_ids: dict[str, int] = {}
    next_label = _FIRST_CUSTOM_TRACE_LABEL
    for tag in loss_tags:
        if tag in id_to_name.values() or tag in _AGGREGATE_TAGS or tag in {"patch", "outcome"}:
            continue
        custom_label_ids[tag] = next_label
        id_to_name[next_label] = tag
        next_label += 1

    aggregates: dict[str, tuple[int, ...]] = {}
    for tag in loss_tags:
        if tag == "assistant":
            aggregates[tag] = (
                TRACE_LABEL_ASSISTANT_TEXT,
                TRACE_LABEL_ASSISTANT_TOOL_CALL,
                TRACE_LABEL_FINAL_ASSISTANT,
            )
        elif tag == "assistant_text":
            aggregates[tag] = (TRACE_LABEL_ASSISTANT_TEXT, TRACE_LABEL_FINAL_ASSISTANT)
        elif tag == "tool_call":
            aggregates[tag] = (TRACE_LABEL_ASSISTANT_TOOL_CALL,)
        elif tag in {"tool", "observation"}:
            aggregates[tag] = (TRACE_LABEL_OBSERVATION,)
        elif tag == "final_assistant":
            aggregates[tag] = (TRACE_LABEL_FINAL_ASSISTANT,)
        elif tag == "patch":
            aggregates[tag] = (TRACE_LABEL_PATCH,)
        elif tag == "outcome":
            aggregates[tag] = (TRACE_LABEL_OUTCOME,)
        elif tag in custom_label_ids:
            aggregates[tag] = (custom_label_ids[tag],)

    return LossLabelSpec(id_to_name=id_to_name, aggregates=aggregates)


@dataclass(frozen=True)
class TraceChatEvaluationFormat:
    """Evaluation-only config for agent traces with exclusive token labels."""

    messages_field: str = "messages"
    chat_template: str | None = None
    system_prompt: str | None = None
    chat_template_kwargs: str | None = "chat_template_kwargs"
    pack: bool | int | None = None
    loss_tags: tuple[str, ...] = (
        "assistant",
        "assistant_text",
        "tool_call",
        "observation",
        "patch",
        "final_assistant",
        "outcome",
    )
    message_loss_tags_field: str | None = "loss_tags"
    include_role_tags: bool = True
    include_final_assistant_tag: bool = True
    parse_text_tool_calls: bool = True
    slice_strategy: Literal["left", "right", "raise"] = "left"

    def build_preprocessor(
        self, tokenizer: MarinTokenizer, *, enforce_eos: bool = True, enforce_bos: bool = True
    ) -> BatchProcessor[dict, dict]:
        del enforce_eos, enforce_bos
        return TraceChatProcessor(
            tokenizer,
            messages_field=self.messages_field,
            chat_template=self.chat_template,
            system_prompt_field=self.system_prompt,
            chat_template_kwargs_field=self.chat_template_kwargs,
            loss_tags=self.loss_tags,
            message_loss_tags_field=self.message_loss_tags_field,
            include_role_tags=self.include_role_tags,
            include_final_assistant_tag=self.include_final_assistant_tag,
            parse_text_tool_calls=self.parse_text_tool_calls,
        )

    def loss_label_spec(self) -> LossLabelSpec:
        return loss_label_spec_for_trace_tags(self.loss_tags)


class TraceChatDataset(MappedAsyncDataset[tuple[ProcessedTraceChatDict, ProcessedTraceChatDict], LabeledLmExample]):
    """A dataset that yields packed trace examples with exclusive loss labels."""

    def __init__(
        self,
        cache: TreeCache[ProcessedTraceChatDict],
        Pos: Axis,
        max_segments_per_example: int = 64,
        slice_strategy: Literal["left", "right", "raise"] = "left",
        block_cross_document_attention: bool = True,
    ):
        self.packed: GreedyPrepackedDataset[ProcessedTraceChatDict] = GreedyPrepackedDataset(
            cache.jagged_array_tree(),
            Pos.size,
            max_segments_per_example=max_segments_per_example,
            slice_strategy=slice_strategy,
        )
        self.Pos = Pos
        self.block_cross_document_attention = block_cross_document_attention

        sharding = _single_cpu_sharding()

        @functools.partial(eqx.filter_jit)
        def _create_trace_example(e: tuple[ProcessedTraceChatDict, ProcessedTraceChatDict]) -> LabeledLmExample:
            example, seg_ids = e
            tokens = example["input_ids"]
            segment_ids = seg_ids["input_ids"]

            labels = jnp.roll(example["loss_labels"], -1, axis=-1)
            next_segment_ids = jnp.roll(segment_ids, -1, axis=-1)
            same_segment_next = (segment_ids == next_segment_ids) & (segment_ids >= 0)
            same_segment_next = same_segment_next.at[-1].set(False)
            labels = jnp.where(same_segment_next, labels, 0)

            attn_mask = GrugAttentionMask.causal()
            if block_cross_document_attention:
                attn_mask = attn_mask.with_segment_ids(segment_ids)

            out = LabeledLmExample(tokens=tokens, loss_labels=labels, attn_mask=attn_mask)
            out = jax.lax.with_sharding_constraint(out, sharding)
            return out

        super().__init__(self.packed, _create_trace_example)


def build_trace_chat_dataset_cache(
    cache_dir: str,
    source: ShardedDataSource[dict],
    trace_format: TraceChatEvaluationFormat,
    tokenizer: MarinTokenizer,
    options: CacheOptions = CacheOptions.default(),
) -> TreeCache[ProcessedTraceChatDict]:
    processor = trace_format.build_preprocessor(tokenizer)
    return cast(
        TreeCache[ProcessedTraceChatDict], TreeCache.build_or_load(cache_dir, source, processor, options=options)
    )


def dataset_for_trace_chat_format(
    trace_format: TraceChatEvaluationFormat,
    Pos: Axis,
    cache: TreeCache[ProcessedTraceChatDict],
    *,
    block_cross_document_attention: bool = True,
) -> TraceChatDataset:
    pack = True if trace_format.pack is None else trace_format.pack
    max_segments, slice_strategy = _resolve_pack_config(pack, packed_slice_strategy=trace_format.slice_strategy)
    return TraceChatDataset(
        cache,
        Pos,
        max_segments_per_example=max_segments,
        slice_strategy=slice_strategy,
        block_cross_document_attention=block_cross_document_attention,
    )


class TraceChatProcessor(BatchProcessor[dict, dict]):
    """Processor that renders chat traces and emits exclusive loss labels."""

    def __init__(
        self,
        tokenizer: MarinTokenizer,
        chat_template: str | None = None,
        messages_field: str = "messages",
        system_prompt_field: str | None = "system",
        chat_template_kwargs_field: str | None = "chat_template_kwargs",
        loss_tags: Sequence[str] = (),
        message_loss_tags_field: str | None = "loss_tags",
        include_role_tags: bool = True,
        include_final_assistant_tag: bool = True,
        parse_text_tool_calls: bool = True,
    ):
        if chat_template is None and tokenizer.chat_template is None:
            raise ValueError("No chat template provided and tokenizer has no default chat template")
        if not loss_tags:
            raise ValueError("TraceChatProcessor requires at least one loss tag")

        self.tokenizer = tokenizer
        self.chat_template = chat_template or tokenizer.chat_template
        self.messages_field = messages_field
        self.system_prompt_field = system_prompt_field
        self.chat_template_kwargs_field = chat_template_kwargs_field
        self.loss_tags = tuple(loss_tags)
        self.message_loss_tags_field = message_loss_tags_field
        self.include_role_tags = include_role_tags
        self.include_final_assistant_tag = include_final_assistant_tag
        self.parse_text_tool_calls = parse_text_tool_calls
        self.label_spec = loss_label_spec_for_trace_tags(loss_tags)

        if self.chat_template is None:
            raise ValueError("No chat template provided and tokenizer has no default chat template")

    def __call__(self, batch: Sequence[dict]) -> Sequence[dict]:
        out: list[dict] = []
        for example in batch:
            conversation, kwargs_dict = self._normalize_conversation(example)
            tokenized = self.tokenizer.apply_chat_template_with_masks(
                [conversation],
                chat_template=self.chat_template,
                return_message_spans=True,
                **kwargs_dict,
            )
            input_ids = tokenized["input_ids"][0]
            assistant_mask = tokenized["assistant_masks"][0]
            spans = tokenized["message_spans"][0]
            labels = self._loss_labels(conversation, spans, assistant_mask, input_ids, len(input_ids), kwargs_dict)

            out.append(
                {
                    "input_ids": np.array(input_ids, dtype=np.int32),
                    "loss_labels": labels.astype(np.int32),
                }
            )
        return out

    def _normalize_conversation(self, example: dict) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        example_messages = example[self.messages_field]
        normalized_messages = [_normalize_chat_message(message) for message in example_messages]
        if self.parse_text_tool_calls:
            normalized_messages = _split_text_tool_call_messages(normalized_messages)

        if self.system_prompt_field is not None and self.system_prompt_field in example:
            system_content = example[self.system_prompt_field]
            if system_content is not None:
                if isinstance(system_content, Mapping):
                    system_message = dict(system_content)
                    system_message["role"] = "system"
                    if "content" not in system_message:
                        raise ValueError("System prompt mapping must include 'content'.")
                else:
                    system_message = {"role": "system", "content": system_content}
                system_message = _normalize_chat_message(system_message)
                normalized_messages = [system_message, *normalized_messages]

        if self.chat_template_kwargs_field is not None and self.chat_template_kwargs_field in example:
            raw_kwargs = example[self.chat_template_kwargs_field]
            if raw_kwargs is not None:
                if not isinstance(raw_kwargs, Mapping):
                    raise ValueError("chat_template_kwargs must be a mapping when present.")
                kwargs_dict = _normalize_chat_kwargs(raw_kwargs)
            else:
                kwargs_dict = _normalize_chat_kwargs(None)
        else:
            kwargs_dict = _normalize_chat_kwargs(None)

        for forbidden in ("tokenize", "return_assistant_tokens_mask", "return_dict", "return_message_spans"):
            if forbidden in kwargs_dict:
                raise ValueError(f"chat_template_kwargs may not override '{forbidden}'.")
        kwargs_dict.pop("add_generation_prompt", None)
        kwargs_dict.pop("chat_template", None)

        return normalized_messages, kwargs_dict

    def _loss_labels(
        self,
        conversation: Sequence[Mapping[str, Any]],
        spans: Sequence[tuple[int, int]],
        assistant_mask: Sequence[int],
        input_ids: Sequence[int],
        full_length: int,
        kwargs_dict: Mapping[str, Any],
    ) -> np.ndarray:
        labels = np.zeros((full_length,), dtype=np.int32)
        assistant_positions = np.asarray(assistant_mask, dtype=np.int32)

        final_assistant_idx = self._final_assistant_index(conversation)
        for idx, (message, (start, end)) in enumerate(zip(conversation, spans, strict=True)):
            if start >= end:
                continue

            label_id = self._explicit_message_label(message)
            if label_id is not None:
                self._assign_label(labels, label_id, start, end, assistant_positions, message)
                continue

            if not self.include_role_tags:
                continue

            role = message.get("role")
            if role == "assistant":
                if message.get("tool_calls"):
                    text_positions = self._assistant_text_positions(
                        conversation,
                        spans,
                        assistant_positions,
                        message_index=idx,
                        full_length=full_length,
                        input_ids=input_ids,
                        kwargs_dict=kwargs_dict,
                    )
                    for pos in range(start, min(end, full_length)):
                        if pos >= len(assistant_positions) or not assistant_positions[pos]:
                            continue
                        labels[pos] = (
                            TRACE_LABEL_ASSISTANT_TEXT if text_positions[pos] else TRACE_LABEL_ASSISTANT_TOOL_CALL
                        )
                elif self.include_final_assistant_tag and idx == final_assistant_idx:
                    self._assign_label(labels, TRACE_LABEL_FINAL_ASSISTANT, start, end, assistant_positions, message)
                else:
                    self._assign_label(labels, TRACE_LABEL_ASSISTANT_TEXT, start, end, assistant_positions, message)
            elif role in {"tool", "function", "ipython"}:
                labels[start:end] = TRACE_LABEL_OBSERVATION

        return labels

    def _explicit_message_label(self, message: Mapping[str, Any]) -> int | None:
        if self.message_loss_tags_field is None or self.message_loss_tags_field not in message:
            return None

        raw_tags = message[self.message_loss_tags_field]
        if raw_tags is None:
            return None
        if isinstance(raw_tags, str):
            message_tags = {raw_tags}
        else:
            message_tags = {str(tag) for tag in raw_tags}

        label_by_tag = {name: label_id for label_id, name in self.label_spec.id_to_name.items()}
        for tag in self.loss_tags:
            if tag in message_tags and tag in label_by_tag:
                return label_by_tag[tag]
        return None

    def _assign_label(
        self,
        labels: np.ndarray,
        label_id: int,
        start: int,
        end: int,
        assistant_positions: np.ndarray,
        message: Mapping[str, Any],
    ) -> None:
        upper = min(end, labels.shape[0])
        if message.get("role") == "assistant":
            for pos in range(start, upper):
                if pos < len(assistant_positions) and assistant_positions[pos]:
                    labels[pos] = label_id
        else:
            labels[start:upper] = label_id

    def _assistant_text_positions(
        self,
        conversation: Sequence[Mapping[str, Any]],
        spans: Sequence[tuple[int, int]],
        assistant_positions: np.ndarray,
        *,
        message_index: int,
        full_length: int,
        input_ids: Sequence[int],
        kwargs_dict: Mapping[str, Any],
    ) -> np.ndarray:
        message = conversation[message_index]
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return np.zeros((full_length,), dtype=np.int32)

        start, end = spans[message_index]
        content_positions = self._content_token_positions(content, input_ids, start, end, assistant_positions)
        if content_positions is not None:
            return content_positions

        content_only_prefix = [dict(previous_message) for previous_message in conversation[:message_index]]
        content_only_prefix.append(_message_without_tool_calls(message, content))
        tokenized = self.tokenizer.apply_chat_template_with_masks(
            [content_only_prefix],
            chat_template=self.chat_template,
            return_message_spans=True,
            **kwargs_dict,
        )
        content_only_positions = np.zeros((full_length,), dtype=np.int32)
        content_only_assistant_mask = np.asarray(tokenized["assistant_masks"][0], dtype=np.int32)
        upper = min(end, content_only_assistant_mask.shape[0], full_length)
        for pos in range(start, upper):
            if assistant_positions[pos] and content_only_assistant_mask[pos]:
                content_only_positions[pos] = 1
        return content_only_positions

    def _content_token_positions(
        self,
        content: str,
        input_ids: Sequence[int],
        start: int,
        end: int,
        assistant_positions: np.ndarray,
    ) -> np.ndarray | None:
        window = list(input_ids[start:end])
        candidates = (content, content.strip())
        for candidate in dict.fromkeys(candidates):
            if not candidate:
                continue
            content_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            offset = _find_subsequence(window, content_ids)
            if offset is None:
                continue

            positions = np.zeros((len(input_ids),), dtype=np.int32)
            content_start = start + offset
            content_end = content_start + len(content_ids)
            for pos in range(content_start, min(content_end, len(input_ids))):
                if pos < len(assistant_positions) and assistant_positions[pos]:
                    positions[pos] = 1
            return positions
        return None

    @staticmethod
    def _final_assistant_index(conversation: Sequence[Mapping[str, Any]]) -> int | None:
        for index in range(len(conversation) - 1, -1, -1):
            if conversation[index].get("role") == "assistant":
                return index
        return None

    @property
    def output_exemplar(self):
        return {
            "input_ids": np.zeros((0,), dtype=np.int32),
            "loss_labels": np.zeros((0,), dtype=np.int32),
        }

    @property
    def num_cpus(self) -> int:
        return 1

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": self.tokenizer.vocab_size,
            "chat_template": self.chat_template,
            "messages_field": self.messages_field,
            "system_prompt_field": self.system_prompt_field,
            "chat_template_kwargs_field": self.chat_template_kwargs_field,
            "loss_tags": list(self.loss_tags),
            "message_loss_tags_field": self.message_loss_tags_field,
            "include_role_tags": self.include_role_tags,
            "include_final_assistant_tag": self.include_final_assistant_tag,
            "parse_text_tool_calls": self.parse_text_tool_calls,
            "label_spec": {
                "id_to_name": {str(label_id): name for label_id, name in self.label_spec.id_to_name.items()},
                "aggregates": {name: list(label_ids) for name, label_ids in self.label_spec.aggregates.items()},
            },
        }
