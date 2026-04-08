# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import re
from typing import TYPE_CHECKING, Any, TypeAlias

from levanter.tokenizers import MarinTokenizer
from levanter.utils.logging import silence_transformer_nag


silence_transformer_nag()

# HfTokenizer is retained only for callers that need the actual HF transformers
# tokenizer object (hf_checkpoints, lora save_pretrained, etc.).
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    HfTokenizer: TypeAlias = PreTrainedTokenizerFast | PreTrainedTokenizer
else:
    HfTokenizer: TypeAlias = Any


def byte_length_of_token(tokenizer: MarinTokenizer, idx: int) -> int:
    """Compute the UTF-8 byte length of a single token.

    Uses convert_ids_to_tokens to get the raw BPE representation, then handles
    special tokens (0 bytes), hex-encoded byte tokens like <0x16>, and normal
    tokens (decoded via a prefix trick to preserve leading spaces).
    """
    token_repr = tokenizer.convert_ids_to_tokens(idx)
    if idx in tokenizer.all_special_ids:
        return 0
    if m := re.match(r"<0x([0-9A-Fa-f]+)>", token_repr):
        return len(bytes.fromhex(m.group(1)))

    extra_token = tokenizer.encode(".", add_special_tokens=False)[0]
    excess_bytes = len(".".encode("utf-8"))
    decoded = tokenizer.decode([extra_token, idx]).encode("utf-8")
    return len(decoded) - excess_bytes
