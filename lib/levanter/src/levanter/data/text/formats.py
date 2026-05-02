# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import numpy as np
from draccus import ChoiceRegistry

from levanter.data._preprocessor import BatchProcessor
from levanter.tokenizers import MarinTokenizer

from ._batch_tokenizer import BatchTokenizer


class LmDatasetFormatBase(ChoiceRegistry):
    @classmethod
    def default_choice_name(cls) -> str | None:
        return "text"

    @property
    def token_data_key(self) -> str:
        return "input_ids"

    def build_preprocessor(
        self, tokenizer: MarinTokenizer, *, enforce_eos: bool = True, enforce_bos: bool = True
    ) -> BatchProcessor[dict, dict]:
        raise ValueError(f"Unknown format {self}")


@LmDatasetFormatBase.register_subclass("text")
@dataclass(frozen=True)
class TextLmDatasetFormat(LmDatasetFormatBase):
    """Dataset configuration for raw text examples."""

    text_key: str = "text"  # key for the text field in the jsonl file

    def build_preprocessor(
        self, tokenizer: MarinTokenizer, *, enforce_eos: bool = True, enforce_bos: bool = True
    ) -> BatchProcessor[dict, dict]:
        return BatchTokenizer(tokenizer, enforce_bos=enforce_bos, enforce_eos=enforce_eos, text_field=self.text_key)


@LmDatasetFormatBase.register_subclass("chat")
@dataclass(frozen=True)
class ChatLmDatasetFormat(LmDatasetFormatBase):
    """Dataset configuration for multi-turn chat transcripts."""

    messages_field: str = "messages"
    chat_template: str | None = None
    system_prompt: str | None = None
    chat_template_kwargs: str | None = "chat_template_kwargs"
    pack: bool | int | Literal["pad"] | None = None  # None => default pack behavior (currently pack)
    mask_user_turns: bool = True

    def build_preprocessor(
        self, tokenizer: MarinTokenizer, *, enforce_eos: bool = True, enforce_bos: bool = True
    ) -> BatchProcessor[dict, dict]:
        return ChatProcessor(
            tokenizer,
            messages_field=self.messages_field,
            chat_template=self.chat_template,
            system_prompt_field=self.system_prompt,
            chat_template_kwargs_field=self.chat_template_kwargs,
            mask_user_turns=self.mask_user_turns,
        )


@LmDatasetFormatBase.register_subclass("prebuilt")
@dataclass(frozen=True)
class PrebuiltLmDatasetFormat(LmDatasetFormatBase):
    """Dataset configuration for caches that already contain tokenized sequences.

    Attributes:
        input_ids_key: Field name containing token ids.
        loss_weights_key: Optional field name containing loss weights.
        loss_weight_transform: Optional callable to transform loss weights before training.
    """

    input_ids_key: str = "input_ids"
    loss_weights_key: str | None = None
    loss_weight_transform: Callable[[np.ndarray], np.ndarray] | None = None

    @property
    def token_data_key(self) -> str:
        return self.input_ids_key

    def build_preprocessor(
        self, tokenizer: MarinTokenizer, *, enforce_eos: bool = True, enforce_bos: bool = True
    ) -> BatchProcessor[dict, dict]:
        del tokenizer, enforce_eos, enforce_bos
        return PrebuiltCacheProcessor(self.input_ids_key, self.loss_weights_key)


class PrebuiltCacheProcessor(BatchProcessor[dict, dict]):
    """
    Processor that normalizes prebuilt cache records to consistent dtypes.
    """

    def __init__(self, input_ids_key: str, loss_weights_key: str | None):
        self.input_ids_key = input_ids_key
        self.loss_weights_key = loss_weights_key
        self._exemplar = {input_ids_key: np.zeros((0,), dtype=np.int32)}
        if loss_weights_key is not None:
            self._exemplar[loss_weights_key] = np.zeros((0,), dtype=np.float32)

    def __call__(self, batch: Sequence[dict]) -> Sequence[dict]:
        out = []
        for example in batch:
            if self.input_ids_key not in example:
                raise ValueError(f"Missing required field '{self.input_ids_key}' in prebuilt example.")
            item = {
                self.input_ids_key: np.asarray(example[self.input_ids_key], dtype=np.int32),
            }
            if self.loss_weights_key is not None:
                if self.loss_weights_key not in example:
                    raise ValueError(f"Missing required field '{self.loss_weights_key}' in prebuilt example.")
                item[self.loss_weights_key] = np.asarray(example[self.loss_weights_key], dtype=np.float32)
            out.append(item)
        return out

    @property
    def output_exemplar(self) -> dict:
        return self._exemplar

    @property
    def num_cpus(self) -> int:
        return 1

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "input_ids_key": self.input_ids_key,
            "loss_weights_key": self.loss_weights_key,
        }


class ProcessedChatDict(TypedDict):
    input_ids: np.ndarray
    assistant_masks: np.ndarray


class ProcessedTraceChatDict(TypedDict):
    input_ids: np.ndarray
    trace_masks: dict[str, np.ndarray]


_TEXT_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _normalize_chat_message(message: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    normalized.setdefault("content", "")
    normalized.setdefault("reasoning_content", None)
    normalized.setdefault("tool_calls", [])
    if normalized["tool_calls"] is None:
        normalized["tool_calls"] = []
    return normalized


def _parsed_text_tool_call(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(text.strip())
        except (SyntaxError, ValueError):
            return None

    if not isinstance(payload, Mapping):
        return None

    function_payload = payload.get("function")
    if isinstance(function_payload, Mapping):
        name = function_payload.get("name")
        arguments = function_payload.get("arguments", function_payload.get("args"))
    else:
        name = payload.get("name")
        arguments = payload.get("arguments", payload.get("args"))

    if not isinstance(name, str) or arguments is None:
        return None

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
        if tool_call is None:
            split_messages.append(_message_without_tool_calls(message, match.group(0)))
        else:
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


def _normalize_chat_kwargs(kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    kwargs_dict = dict(kwargs) if kwargs is not None else {}
    kwargs_dict.setdefault("tools", None)
    return kwargs_dict


class ChatProcessor(BatchProcessor[dict, dict]):
    """
    Processor that converts chat data into token ids and assistant masks via chat templates.
    """

    def __init__(
        self,
        tokenizer: MarinTokenizer,
        chat_template: str | None = None,
        messages_field: str = "messages",
        system_prompt_field: str | None = "system",
        chat_template_kwargs_field: str | None = "chat_template_kwargs",
        mask_user_turns: bool = True,
    ):
        if chat_template is None and tokenizer.chat_template is None:
            raise ValueError("No chat template provided and tokenizer has no default chat template")
        self.tokenizer = tokenizer
        self.chat_template = chat_template or tokenizer.chat_template
        self.messages_field = messages_field
        self.system_prompt_field = system_prompt_field
        self.chat_template_kwargs_field = chat_template_kwargs_field
        self.mask_user_turns = mask_user_turns

        if self.chat_template is None:
            raise ValueError("No chat template provided and tokenizer has no default chat template")

        if mask_user_turns and not re.search(r"\{%-?\s*generation\s*-?%}", self.chat_template):
            raise ValueError(
                "Chat template must contain {%generation%} to indicate the position of the assistant message "
                "if mask_user_turns is True."
            )

    def __call__(self, batch: Sequence[dict]) -> Sequence[dict]:
        messages: list[list[dict[str, Any]]] = []
        chat_kwargs_list: list[Mapping[str, Any] | None] = []
        for example in batch:
            example_messages = example[self.messages_field]
            normalized_messages = [_normalize_chat_message(message) for message in example_messages]

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

            messages.append(normalized_messages)

            example_kwargs: Mapping[str, Any] | None = None
            if self.chat_template_kwargs_field is not None and self.chat_template_kwargs_field in example:
                raw_kwargs = example[self.chat_template_kwargs_field]
                if raw_kwargs is not None:
                    if not isinstance(raw_kwargs, Mapping):
                        raise ValueError("chat_template_kwargs must be a mapping when present.")
                    example_kwargs = dict(raw_kwargs)
            chat_kwargs_list.append(example_kwargs)

        use_per_example_kwargs = any(kwargs for kwargs in chat_kwargs_list)

        if not use_per_example_kwargs:
            tokenized = self.tokenizer.apply_chat_template_with_masks(
                messages,
                chat_template=self.chat_template,
                **_normalize_chat_kwargs(None),
            )
        else:
            input_ids_batches: list[list[int]] = []
            assistant_mask_batches: list[list[int]] = []

            for conversation, example_kwargs in zip(messages, chat_kwargs_list):
                kwargs_dict = _normalize_chat_kwargs(example_kwargs)
                for forbidden in ("tokenize", "return_assistant_tokens_mask", "return_dict"):
                    if forbidden in kwargs_dict:
                        raise ValueError(f"chat_template_kwargs may not override '{forbidden}'.")

                chat_template_override = kwargs_dict.pop("chat_template", self.chat_template)
                if chat_template_override is None:
                    raise ValueError("Chat template must be provided either in the dataset format or per example.")

                # Remove add_generation_prompt if present; it's not used for mask computation.
                kwargs_dict.pop("add_generation_prompt", None)

                tokenized_single = self.tokenizer.apply_chat_template_with_masks(
                    [conversation],
                    chat_template=chat_template_override,
                    **kwargs_dict,
                )
                input_ids_batches.extend(tokenized_single["input_ids"])
                assistant_mask_batches.extend(tokenized_single["assistant_masks"])

            tokenized = {"input_ids": input_ids_batches, "assistant_masks": assistant_mask_batches}

        masks = tokenized["assistant_masks"]
        for seq, mask_for_seq in zip(batch, masks):
            if not np.any(mask_for_seq):
                raise ValueError(f"Chat did not contain an assistant message for sequence {seq}")

        out: list[dict] = []
        for ids, mask in zip(tokenized["input_ids"], masks):
            out.append(
                {
                    "input_ids": np.array(ids, dtype=np.int32),
                    "assistant_masks": np.array(mask, dtype=np.int32),
                }
            )
        return out

    @property
    def output_exemplar(self):
        return {
            "input_ids": np.zeros((0,), dtype=np.int32),
            "assistant_masks": np.zeros((0,), dtype=np.int32),
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
        }


@dataclass(frozen=True)
class TraceChatEvaluationFormat:
    """Evaluation-only configuration for multi-turn agent traces with named loss masks.

    This format expects OpenAI-style conversation records. In addition to
    rendering the conversation with the chat template, it emits one token mask
    per configured loss tag. It is intentionally not an `LmDatasetFormatBase`:
    trace masks are consumed by `MaskedEvaluator`, not by normal LM training or
    `TaggedEvaluator`.
    """

    messages_field: str = "messages"
    chat_template: str | None = None
    system_prompt: str | None = None
    chat_template_kwargs: str | None = "chat_template_kwargs"
    pack: bool | int | Literal["pad"] | None = None
    loss_tags: tuple[str, ...] = (
        "assistant",
        "assistant_text",
        "tool",
        "observation",
        "action",
        "tool_call",
        "bash",
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


class TraceChatProcessor(BatchProcessor[dict, dict]):
    """Processor that renders chat traces and emits named token masks."""

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

        if self.chat_template is None:
            raise ValueError("No chat template provided and tokenizer has no default chat template")

    def __call__(self, batch: Sequence[dict]) -> Sequence[dict]:
        out: list[dict] = []
        for example in batch:
            conversation, kwargs_dict = self._normalize_conversation(example)
            tokenized = self.tokenizer.apply_chat_template_with_masks(
                [conversation],
                chat_template=self.chat_template,
                **kwargs_dict,
            )
            input_ids = tokenized["input_ids"][0]
            assistant_mask = tokenized["assistant_masks"][0]
            spans = self._message_spans(conversation, len(input_ids), kwargs_dict)
            masks = self._loss_masks(conversation, spans, assistant_mask, len(input_ids), kwargs_dict)

            out.append(
                {
                    "input_ids": np.array(input_ids, dtype=np.int32),
                    "trace_masks": {tag: np.array(masks[tag], dtype=np.int32) for tag in self.loss_tags},
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

        for forbidden in ("tokenize", "return_assistant_tokens_mask", "return_dict"):
            if forbidden in kwargs_dict:
                raise ValueError(f"chat_template_kwargs may not override '{forbidden}'.")
        kwargs_dict.pop("add_generation_prompt", None)
        kwargs_dict.pop("chat_template", None)

        return normalized_messages, kwargs_dict

    def _message_spans(
        self, conversation: list[dict[str, Any]], full_length: int, kwargs_dict: Mapping[str, Any]
    ) -> list[tuple[int, int]]:
        prefix_lengths = [0]
        for end in range(1, len(conversation) + 1):
            prefix = conversation[:end]
            tokenized_prefix = self.tokenizer.apply_chat_template_with_masks(
                [prefix],
                chat_template=self.chat_template,
                **kwargs_dict,
            )
            prefix_lengths.append(min(len(tokenized_prefix["input_ids"][0]), full_length))

        spans: list[tuple[int, int]] = []
        for start, end in zip(prefix_lengths[:-1], prefix_lengths[1:]):
            spans.append((min(start, full_length), min(max(end, start), full_length)))
        return spans

    def _loss_masks(
        self,
        conversation: Sequence[Mapping[str, Any]],
        spans: Sequence[tuple[int, int]],
        assistant_mask: Sequence[int],
        full_length: int,
        kwargs_dict: Mapping[str, Any],
    ) -> dict[str, list[int]]:
        masks = {tag: [0] * full_length for tag in self.loss_tags}
        assistant_positions = np.asarray(assistant_mask, dtype=np.int32)

        final_assistant_idx = self._final_assistant_index(conversation)
        for idx, (message, (start, end)) in enumerate(zip(conversation, spans)):
            if start >= end:
                continue

            tags = self._tags_for_message(message, idx == final_assistant_idx)
            for tag in tags:
                if tag not in masks:
                    continue
                mask = masks[tag]
                for pos in range(start, end):
                    if message.get("role") == "assistant" and pos < len(assistant_positions):
                        mask[pos] = int(assistant_positions[pos])
                    else:
                        mask[pos] = 1

        if "assistant" in masks:
            masks["assistant"] = assistant_positions.astype(np.int32).tolist()

        if "assistant_text" in masks:
            masks["assistant_text"] = self._assistant_text_mask(
                conversation,
                spans,
                assistant_positions,
                full_length,
                kwargs_dict,
            )

        if "final_assistant" in masks and final_assistant_idx is not None:
            start, end = spans[final_assistant_idx]
            final_mask = [0] * full_length
            for pos in range(start, end):
                if pos < len(assistant_positions) and assistant_positions[pos]:
                    final_mask[pos] = 1
            masks["final_assistant"] = final_mask

        return masks

    def _assistant_text_mask(
        self,
        conversation: Sequence[Mapping[str, Any]],
        spans: Sequence[tuple[int, int]],
        assistant_positions: np.ndarray,
        full_length: int,
        kwargs_dict: Mapping[str, Any],
    ) -> list[int]:
        text_mask = [0] * full_length
        for idx, (message, (start, end)) in enumerate(zip(conversation, spans)):
            if message.get("role") != "assistant" or start >= end:
                continue

            if not message.get("tool_calls"):
                for pos in range(start, end):
                    if pos < len(assistant_positions) and assistant_positions[pos]:
                        text_mask[pos] = 1
                continue

            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue

            content_only_positions = self._assistant_content_only_positions(
                conversation,
                message_index=idx,
                full_length=full_length,
                kwargs_dict=kwargs_dict,
            )
            upper = min(end, len(content_only_positions), full_length)
            for pos in range(start, upper):
                if assistant_positions[pos] and content_only_positions[pos]:
                    text_mask[pos] = 1

        return text_mask

    def _assistant_content_only_positions(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        message_index: int,
        full_length: int,
        kwargs_dict: Mapping[str, Any],
    ) -> np.ndarray:
        message = conversation[message_index]
        content = message.get("content")
        if not isinstance(content, str):
            return np.zeros((full_length,), dtype=np.int32)

        content_only_prefix = [dict(previous_message) for previous_message in conversation[:message_index]]
        content_only_prefix.append(_message_without_tool_calls(message, content))
        # Supported chat templates render assistant text before structured tool calls, so
        # retokenizing the prefix without tool calls isolates the assistant-text span.
        tokenized = self.tokenizer.apply_chat_template_with_masks(
            [content_only_prefix],
            chat_template=self.chat_template,
            **kwargs_dict,
        )
        content_only_positions = np.zeros((full_length,), dtype=np.int32)
        assistant_mask = np.asarray(tokenized["assistant_masks"][0], dtype=np.int32)
        limit = min(full_length, assistant_mask.shape[0])
        content_only_positions[:limit] = assistant_mask[:limit]
        return content_only_positions

    def _tags_for_message(self, message: Mapping[str, Any], is_final_assistant: bool) -> set[str]:
        tags: set[str] = set()

        if self.message_loss_tags_field is not None and self.message_loss_tags_field in message:
            raw_tags = message[self.message_loss_tags_field]
            if raw_tags is None:
                pass
            elif isinstance(raw_tags, str):
                tags.add(raw_tags)
            else:
                tags.update(str(tag) for tag in raw_tags)

        if self.include_role_tags:
            role = message.get("role")
            if role == "assistant":
                tags.add("assistant")
                tags.add("assistant_text")
                if message.get("tool_calls"):
                    tags.add("tool_call")
            elif role in {"tool", "function", "ipython"}:
                tags.add("tool")
                tags.add("observation")

        if self.include_final_assistant_tag and is_final_assistant:
            tags.add("final_assistant")

        return tags

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
            "trace_masks": {tag: np.zeros((0,), dtype=np.int32) for tag in self.loss_tags},
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
        }


def preprocessor_for_format(
    format: LmDatasetFormatBase, tokenizer: MarinTokenizer, *, enforce_eos: bool = True, enforce_bos: bool = True
) -> BatchProcessor[dict, dict]:
    return format.build_preprocessor(tokenizer, enforce_eos=enforce_eos, enforce_bos=enforce_bos)
