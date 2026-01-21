# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import re
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from typing import Any, Literal, Mapping, Optional, Sequence, TypedDict

import numpy as np
import regex
from draccus import ChoiceRegistry
from tokenizers import normalizers
from transformers import BatchEncoding, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from levanter.data._preprocessor import BatchProcessor
from levanter.utils.hf_utils import HfTokenizer, num_cpus_used_by_tokenizer

LONG_STRING_WORKAROUND = 10_000
ws = regex.compile(r"\s")


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


def _maybe_force_tokenizer_parallelism(tokenizer: PreTrainedTokenizerBase):
    if tokenizer.is_fast and os.getenv("TOKENIZERS_PARALLELISM") is None:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"


class BatchTokenizer(BatchProcessor[dict, dict]):
    """
    Batch processor that tokenizes raw text, optionally adding BOS/EOS tokens.
    """

    def __init__(
        self,
        tokenizer: HfTokenizer,
        text_field: str = "text",
        enforce_bos: bool = True,
        enforce_eos: bool = True,
        *,
        override_resources=None,
        _workaround_len: int = LONG_STRING_WORKAROUND,
        return_attention_mask: bool = False,
        padding=False,
        max_length=None,
    ):
        _maybe_force_tokenizer_parallelism(tokenizer)
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.override_resources = override_resources
        self.return_attention_mask = return_attention_mask
        self.padding = padding
        self.max_length = max_length if max_length is not None else self.tokenizer.model_max_length

        if tokenizer.bos_token_id is None:
            enforce_bos = False
        if tokenizer.eos_token_id is None:
            enforce_eos = False

        if enforce_eos or enforce_bos:
            input_ids = tokenizer("hi there")["input_ids"]
            should_append_eos = input_ids[-1] != tokenizer.eos_token_id and enforce_eos
            should_append_bos = input_ids[0] != tokenizer.bos_token_id and enforce_bos
        else:
            should_append_eos = False
            should_append_bos = False

        self._need_to_add_eos = should_append_eos
        self._need_to_add_bos = should_append_bos
        self._workaround_len = _workaround_len

    def __call__(self, batch: Sequence[dict]) -> list[dict]:
        batch_text = [example[self.text_field] for example in batch]

        if self._need_to_add_bos:
            batch_text = [self.tokenizer.bos_token + " " + d for d in batch_text]
        if self._need_to_add_eos:
            batch_text = [d + " " + self.tokenizer.eos_token for d in batch_text]

        if self._needs_long_sequence_workaround:
            batch_text, needs_merge = self._break_for_long_sequences(batch_text)
        else:
            needs_merge = []

        if self.padding is not False:
            encoding = self.tokenizer(
                batch_text,
                return_attention_mask=self.return_attention_mask,
                verbose=False,
                padding=self.padding,
                max_length=self.max_length,
                truncation=True,
            )  # type: ignore
        else:
            encoding = self.tokenizer(batch_text, return_attention_mask=self.return_attention_mask, verbose=False)  # type: ignore

        if needs_merge:
            new_encoding = self._merge_split_encodings(batch_text, encoding, needs_merge)
            encoding = BatchEncoding(new_encoding)

        unbatched = [dict(zip(encoding, t)) for t in zip(*[encoding[k] for k in encoding])]
        return unbatched

    def _break_for_long_sequences(self, batch: Sequence[str]):
        orig_lengths = [len(d) for d in batch]
        orig_batch = batch
        batch_out: list[str] = []
        needs_merge: list[bool] = []
        for i, d in enumerate(orig_batch):
            needs_merge.append(False)
            orig_len = orig_lengths[i]
            while len(d) > self._workaround_len:
                match = ws.search(d, self._workaround_len)
                split = match.start() if match is not None else len(d)
                batch_out.append(d[:split])
                needs_merge.append(True)
                d = d[split:]
                orig_len -= split
            batch_out.append(d)
        return batch_out, needs_merge

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": len(self.tokenizer),
            "return_attention_mask": self.return_attention_mask,
            "padding": self.padding,
            "max_length": self.max_length,
            "append_bos": self._need_to_add_bos,
            "append_eos": self._need_to_add_eos,
        }

    @property
    def output_exemplar(self) -> dict:
        return dict(**self.tokenizer("hi there", return_attention_mask=self.return_attention_mask, verbose=False))

    @property
    def name_or_path(self):
        return self.tokenizer.name_or_path

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @cached_property
    def _needs_long_sequence_workaround(self):
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            normalizer = self.tokenizer.backend_tokenizer.normalizer
            if normalizer is None:
                return False
            return isinstance(normalizer, (normalizers.Replace, normalizers.Sequence))
        return False

    @property
    def num_cpus(self) -> int:
        if self.override_resources is not None:
            cpus = self.override_resources.get("num_cpus", None)
            if cpus is not None:
                return cpus
        return num_cpus_used_by_tokenizer(self.tokenizer)

    @property
    def num_gpus(self) -> int:
        if self.override_resources is not None:
            return self.override_resources.get("num_gpus", 0)
        return 0

    @staticmethod
    def _merge_split_encodings(batch, encoding, needs_merge):
        new_encoding = {}
        for k, v in encoding.items():
            if len(v) == 0:
                continue
            if isinstance(v[0], np.ndarray):
                v_out = []
                vs_to_merge = []
                for i in range(len(batch)):
                    if not needs_merge[i]:
                        if len(vs_to_merge) > 0:
                            v_out.append(np.concatenate(vs_to_merge))
                        vs_to_merge = []
                    vs_to_merge.append(v[i])
                if len(vs_to_merge) > 0:
                    v_out.append(np.concatenate(vs_to_merge))
                new_encoding[k] = v_out
            elif isinstance(v[0], list):
                v_out = []
                vs_to_merge = []
                for i in range(len(batch)):
                    if not needs_merge[i]:
                        if len(vs_to_merge) > 0:
                            v_out.append(list(chain(*vs_to_merge)))  # type: ignore[name-defined]
                        vs_to_merge = []
                    vs_to_merge.append(v[i])
                if len(vs_to_merge) > 0:
                    v_out.append(list(chain(*vs_to_merge)))  # type: ignore[name-defined]
                new_encoding[k] = v_out
            else:
                raise ValueError(f"Unknown type {type(v[0])}")
        return new_encoding


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
