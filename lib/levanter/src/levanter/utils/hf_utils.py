# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import re
from typing import TYPE_CHECKING, Any, TypeAlias

from levanter.utils.logging import silence_transformer_nag


silence_transformer_nag()

# Deprecated: prefer MarinTokenizer from levanter.tokenizers for new code.
# HfTokenizer is retained for callers that still pass HF transformers tokenizers
# (eval_harness, hf_checkpoints, main scripts, etc.) and will be removed once they migrate.
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    HfTokenizer: TypeAlias = PreTrainedTokenizerFast | PreTrainedTokenizer
else:
    # Avoid importing transformers at runtime: it unconditionally imports torch, which
    # fails on CPU-only workers that lack CUDA libs. HfTokenizer is only used in type
    # annotations (no isinstance checks), so Any is sufficient at runtime.
    HfTokenizer: TypeAlias = Any


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
