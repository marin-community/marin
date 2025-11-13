# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import re
from typing import Any, Protocol

from levanter.utils.logging import silence_transformer_nag
from levanter.utils.py_utils import logical_cpu_core_count

silence_transformer_nag()

_HF_TOKENIZER_OFF_VALUES = {"off", "false", "f", "no", "n", "0"}


class HfTokenizer(Protocol):
    """
    Protocol defining the interface for Hugging Face tokenizers.

    This protocol captures the common interface used across Levanter and Marin for both
    PreTrainedTokenizer and PreTrainedTokenizerFast, without requiring a direct type union.

    Note: Most attributes are effectively read-only in practice, but the Protocol doesn't
    enforce this to allow compatibility with various tokenizer implementations.
    """

    # Token ID attributes
    pad_token_id: int | None
    eos_token_id: int | None
    bos_token_id: int | None
    all_special_ids: list[int]

    # Token string attributes
    pad_token: str | None
    eos_token: str | None
    bos_token: str | None
    unk_token: str | None

    # Configuration attributes
    vocab_size: int
    model_max_length: int
    chat_template: str | None
    name_or_path: str
    is_fast: bool
    backend_tokenizer: Any

    # Core tokenization methods
    def encode(self, text: Any, text_pair: Any = None, add_special_tokens: Any = True, **kwargs: Any) -> Any: ...
    def decode(self, token_ids: Any, skip_special_tokens: Any = False, **kwargs: Any) -> str: ...
    def __call__(self, text: Any, text_pair: Any = None, add_special_tokens: Any = True, **kwargs: Any) -> Any: ...
    def apply_chat_template(
        self, conversation: Any, tokenize: bool = True, add_generation_prompt: bool = False, **kwargs: Any
    ) -> Any: ...

    # Token conversion methods
    def convert_tokens_to_ids(self, tokens: Any) -> Any: ...
    def convert_ids_to_tokens(self, ids: Any) -> Any: ...

    # Batch processing methods
    def pad(self, encoded_inputs: Any, padding: Any = True, max_length: Any = None, **kwargs: Any) -> Any: ...
    def batch_decode(self, sequences: Any, skip_special_tokens: Any = False, **kwargs: Any) -> Any: ...

    # Special token management
    def add_special_tokens(self, special_tokens_dict: Any, **kwargs: Any) -> int: ...

    # Checkpoint methods
    def save_pretrained(self, save_directory: str, **kwargs: Any) -> Any: ...

    # Special methods
    def __len__(self) -> int: ...


def num_cpus_used_by_tokenizer(tokenizer: HfTokenizer) -> int:
    if getattr(tokenizer, "is_fast", False):
        if os.getenv("TOKENIZERS_PARALLELISM", "true").lower() in _HF_TOKENIZER_OFF_VALUES:
            return 1
        else:
            # This is a bit hacky, but HF's fast tokenizers are parallelized under the hood.
            # we reserve a couple of cores just so Ray has somewhere to run the coordinator.
            # Empirically it doesn't usually exceed 16-20, and it's useful to have some slack
            return min(max(1, logical_cpu_core_count() - 4), 12)
    else:
        return 1


def byte_length_of_token(tokenizer, idx: int) -> int:
    # this is a pain because we want the prefix spaces, but we don't want extra noise for bytes
    # e.g. in llama
    # >>> t.convert_ids_to_tokens(q[2])
    # '▁this'
    # >>> t.convert_ids_to_tokens(25)
    # '<0x16>'
    # We want the _ (as a single byte, not the 3 it's encoded as) but not the <0x16>, which should instead be a single byte \x16
    # decode strips the prefix spaces, but does correctly handle the <0x16> case
    # we can avoid prefix space issues by prepending another token before decoding, then stripping
    repr = tokenizer.convert_ids_to_tokens(idx)
    if idx in tokenizer.all_special_ids:
        # NB: special tokens don't have bytes, but they contribute to perplexity/bits
        return 0
    # handle bytes specially. This is a bit of a hack, but there's no other way
    elif m := re.match(r"<0x([0-9A-Fa-f]+)>", repr):
        return len(bytes.fromhex(m.group(1)))
    else:
        extra_token = tokenizer(".", add_special_tokens=False)["input_ids"][0]
        excess_bytes = len(".".encode("utf-8"))
        decoded = tokenizer.decode([extra_token, idx]).encode("utf-8")
        return len(decoded) - excess_bytes
