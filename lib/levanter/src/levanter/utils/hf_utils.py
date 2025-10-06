# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import re
from typing import TypeAlias

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from levanter.utils.logging import silence_transformer_nag
from levanter.utils.py_utils import logical_cpu_core_count


silence_transformer_nag()

_HF_TOKENIZER_OFF_VALUES = {"off", "false", "f", "no", "n", "0"}

HfTokenizer: TypeAlias = PreTrainedTokenizerFast | PreTrainedTokenizer
"""
Type alias for a Hugging Face tokenizer. This is a union of the two tokenizer types.
While there is PreTrainedTokenizerBase, it doesn't have all methods that are implemented in both
PreTrainedTokenizer and PreTrainedTokenizerFast. grumble grumble.
"""


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
