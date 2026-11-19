# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Discover how a model's chat template marks reasoning inside its reply text."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from taskcompendium.evaluation.response import ReasoningDelimiters
from verifyit.reasoning_markup import REASONING_DELIMITERS


class ChatTemplateTokenizer(Protocol):
    @property
    def chat_template(self) -> str | dict[str, str] | None: ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str: ...

    def apply_chat_template(
        self, conversation: list[dict[str, Any]], *, tokenize: bool, add_generation_prompt: bool, **kwargs: Any
    ) -> Any: ...


@dataclass(frozen=True)
class InlineReasoningFormat:
    """A chat template's reasoning delimiters, which a reply carries inline when no reasoning parser runs.

    ``special`` delimiters are special tokens, so the server must decode with ``skip_special_tokens``
    off for them to reach the reply text.
    """

    delimiters: ReasoningDelimiters
    special: bool
    tokenizer: ChatTemplateTokenizer

    def prompt_opens_span(self, messages: Sequence[Mapping[str, Any]], chat_template_kwargs: Mapping[str, Any]) -> bool:
        """Whether the rendered generation prompt ends inside a reasoning span."""
        prompt = self.tokenizer.apply_chat_template(
            [dict(message) for message in messages], tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        assert isinstance(prompt, str)
        return prompt.rfind(self.delimiters.start) > prompt.rfind(self.delimiters.end)


def inline_reasoning_format(tokenizer: ChatTemplateTokenizer) -> InlineReasoningFormat | None:
    """Return the reasoning delimiters the tokenizer's chat template renders, or ``None`` if it has none."""
    template = tokenizer.chat_template
    if not isinstance(template, str):
        return None
    for start, end in REASONING_DELIMITERS:
        if start in template and end in template:
            hidden = tokenizer.decode(tokenizer.encode(start, add_special_tokens=False), skip_special_tokens=True)
            return InlineReasoningFormat(ReasoningDelimiters(start, end), start not in hidden, tokenizer)
    return None
