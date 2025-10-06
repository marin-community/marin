# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import numpy as np
from transformers import PreTrainedTokenizer


class PassthroughTokenizer(PreTrainedTokenizer):
    """
    A tokenizer that takes plain-text integers, parses them as integers, and returns them as-is.
    """

    def __init__(self, vocab_size, **kwargs):
        self._vocab = {i: i for i in range(vocab_size)}
        self._vocab_size = vocab_size
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self):
        return self._vocab

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, ...]:
        return ()

    def _tokenize(self, text, **kwargs):
        tokens = np.fromstring(text, dtype=int, sep=" ")
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)
