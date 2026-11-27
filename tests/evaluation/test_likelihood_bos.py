# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A likelihood context starts with BOS exactly when the model's tokenizer or chat template uses one."""

from dataclasses import dataclass

import pytest
from marin.evaluation.task_worker import context_bos_token_id


@dataclass(frozen=True)
class FixtureTokenizer:
    bos_token_id: int | None
    adds_bos: bool
    chat_template: str | None

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        tokens = [ord(character) for character in text]
        return [self.bos_token_id, *tokens] if add_special_tokens and self.adds_bos else tokens


@pytest.mark.parametrize(
    ("tokenizer", "expected"),
    [
        # Llama 3: the tokenizer adds BOS.
        (FixtureTokenizer(7, True, "{{ bos_token }}{{ messages }}"), 7),
        # Gemma 4: only the chat template writes BOS.
        (FixtureTokenizer(2, False, "{{ bos_token }}<|turn>{{ messages }}"), 2),
        # Nemotron 3.5: a BOS token that neither the tokenizer nor the template uses.
        (FixtureTokenizer(1, False, "{{ messages }}"), None),
        # Qwen 3.6: no BOS token.
        (FixtureTokenizer(None, False, "{{ messages }}"), None),
    ],
)
def test_context_bos_follows_the_tokenizer_or_chat_template(tokenizer, expected):
    assert context_bos_token_id(tokenizer) == expected
