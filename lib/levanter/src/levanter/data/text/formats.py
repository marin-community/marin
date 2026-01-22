# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, TypedDict

import numpy as np
from draccus import ChoiceRegistry

from levanter.data._preprocessor import BatchProcessor
from ._batch_tokenizer import BatchTokenizer
from levanter.utils.hf_utils import HfTokenizer, num_cpus_used_by_tokenizer


class LmDatasetFormatBase(ChoiceRegistry):
    @classmethod
    def default_choice_name(cls) -> Optional[str]:
        return "text"


@LmDatasetFormatBase.register_subclass("text")
@dataclass(frozen=True)
class TextLmDatasetFormat(LmDatasetFormatBase):
    """Dataset configuration for raw text examples."""

    text_key: str = "text"  # key for the text field in the jsonl file


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


class ChatProcessor(BatchProcessor[dict, dict]):
    """
    Processor that converts chat data into token ids and assistant masks via chat templates.
    """

    def __init__(
        self,
        tokenizer: HfTokenizer,
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
            normalized_messages = list(example_messages)

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
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                chat_template=self.chat_template,
                return_assistant_tokens_mask=True,
                return_dict=True,
            )
        else:
            input_ids_batches: list[Sequence[int]] = []
            assistant_mask_batches: list[Sequence[int]] = []

            for conversation, example_kwargs in zip(messages, chat_kwargs_list):
                kwargs_dict = dict(example_kwargs) if example_kwargs is not None else {}
                for forbidden in ("tokenize", "return_assistant_tokens_mask", "return_dict"):
                    if forbidden in kwargs_dict:
                        raise ValueError(f"chat_template_kwargs may not override '{forbidden}'.")

                chat_template_override = kwargs_dict.pop("chat_template", self.chat_template)
                if chat_template_override is None:
                    raise ValueError("Chat template must be provided either in the dataset format or per example.")

                apply_kwargs = {
                    **kwargs_dict,
                    "tokenize": True,
                    "return_assistant_tokens_mask": True,
                    "return_dict": True,
                    "chat_template": chat_template_override,
                }

                tokenized_single = self.tokenizer.apply_chat_template([conversation], **apply_kwargs)
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
        return num_cpus_used_by_tokenizer(self.tokenizer)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": len(self.tokenizer),
            "chat_template": self.chat_template,
            "messages_field": self.messages_field,
            "system_prompt_field": self.system_prompt_field,
            "chat_template_kwargs_field": self.chat_template_kwargs_field,
        }


def preprocessor_for_format(
    format: LmDatasetFormatBase, tokenizer: HfTokenizer, *, enforce_eos: bool = True, enforce_bos: bool = True
) -> BatchProcessor[dict, dict]:
    match format:
        case TextLmDatasetFormat(text_key=key):
            return BatchTokenizer(tokenizer, enforce_bos=enforce_bos, enforce_eos=enforce_eos, text_field=key)
        case PrebuiltLmDatasetFormat(input_ids_key=input_ids_key, loss_weights_key=loss_weights_key):
            return PrebuiltCacheProcessor(input_ids_key, loss_weights_key)
        case ChatLmDatasetFormat(
            messages_field=m,
            chat_template=ct,
            system_prompt=sp,
            chat_template_kwargs=ct_kwargs,
            mask_user_turns=mt,
        ):
            return ChatProcessor(
                tokenizer,
                messages_field=m,
                chat_template=ct,
                system_prompt_field=sp,
                chat_template_kwargs_field=ct_kwargs,
                mask_user_turns=mt,
            )  # type: ignore
        case _:
            raise ValueError(f"Unknown format {format}")
