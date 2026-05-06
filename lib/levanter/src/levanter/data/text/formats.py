# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from draccus import ChoiceRegistry

from levanter.data._preprocessor import BatchProcessor
from levanter.tokenizers import MarinTokenizer

from ._batch_tokenizer import BatchTokenizer
from .chat import ChatProcessor, TraceChatProcessor


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


def preprocessor_for_format(
    format: LmDatasetFormatBase, tokenizer: MarinTokenizer, *, enforce_eos: bool = True, enforce_bos: bool = True
) -> BatchProcessor[dict, dict]:
    return format.build_preprocessor(tokenizer, enforce_eos=enforce_eos, enforce_bos=enforce_bos)
