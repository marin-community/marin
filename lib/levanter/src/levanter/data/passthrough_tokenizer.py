# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses


@dataclasses.dataclass(frozen=True)
class PassthroughTokenizer:
    """Tokenizer for pre-tokenized datasets where integers are passed as text.

    Implements the MarinTokenizer protocol. Input text is expected to be
    space-separated integers which are parsed directly into token IDs.
    """

    _vocab_size: int

    def __len__(self) -> int:
        return self._vocab_size

    @property
    def name_or_path(self) -> str:
        return "passthrough"

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def bos_token_id(self) -> int | None:
        return None

    @property
    def eos_token_id(self) -> int | None:
        return None

    @property
    def pad_token_id(self) -> int | None:
        return None

    @property
    def bos_token(self) -> str | None:
        return None

    @property
    def eos_token(self) -> str | None:
        return None

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if not text.strip():
            return []
        return [int(t) for t in text.split()]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return " ".join(str(i) for i in ids)

    def encode_batch(self, texts: list[str], *, add_special_tokens: bool = False) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def get_vocab(self) -> dict[str, int]:
        return {str(i): i for i in range(self._vocab_size)}

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        if isinstance(ids, int):
            return str(ids)
        return [str(i) for i in ids]

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return int(tokens)
        return [int(t) for t in tokens]

    @property
    def all_special_ids(self) -> list[int]:
        return []

    @property
    def chat_template(self) -> str | None:
        return None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> str | list[int]:
        raise ValueError("PassthroughTokenizer does not support chat templates")

    def apply_chat_template_with_masks(
        self,
        conversations: list[list[dict[str, str]]],
        *,
        chat_template: str | None = None,
        **kwargs,
    ) -> dict[str, list[list[int]]]:
        raise ValueError("PassthroughTokenizer does not support chat templates")

    def as_hf_tokenizer(self):
        raise ValueError("PassthroughTokenizer cannot be converted to an HF tokenizer")
