# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenizers for Kelp tree diffusion models."""


class SimpleTokenizer:
    """A simple byte-level tokenizer.

    Maps characters to token IDs via their ordinal value, with an offset of 2
    to reserve ID 0 for padding and ID (vocab_size - 1) for the [MASK] token.

    For production use with larger models, use LlamaTokenizer from
    experiments.kelp.data.stack_edu instead.
    """

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.mask_token_id = vocab_size - 1
        self.unk_token_id = 1

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs (byte-level)."""
        ids = []
        for c in text:
            code = ord(c)
            if code < self.vocab_size - 2:  # Reserve 0 for pad, vocab_size-1 for mask
                ids.append(code + 2)  # Offset by 2
            else:
                ids.append(self.unk_token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        chars = []
        for i in ids:
            if i == self.pad_token_id or i == self.mask_token_id:
                continue
            if i == self.unk_token_id:
                chars.append("?")
            elif i >= 2:
                chars.append(chr(i - 2))
        return "".join(chars)
