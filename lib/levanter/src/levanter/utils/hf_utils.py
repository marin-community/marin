# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import re
from typing import TYPE_CHECKING, Any, TypeAlias

from levanter.tokenizers import MarinTokenizer
from levanter.utils.logging import silence_transformer_nag


silence_transformer_nag()

_TOKENMONSTER_DICTIONARY_CACHE: dict[int, dict] = {}

# HfTokenizer is retained only for callers that need the actual HF transformers
# tokenizer object (hf_checkpoints, lora save_pretrained, etc.).
if TYPE_CHECKING:
    # transformers is an optional dep; keep guard to avoid import at type-check time only
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
    tokenmonster_length = _tokenmonster_byte_length_of_token(tokenizer, idx)
    if tokenmonster_length is not None:
        return tokenmonster_length

    token_repr = tokenizer.convert_ids_to_tokens(idx)
    if idx in getattr(tokenizer, "all_special_ids", ()):
        return 0
    if m := re.match(r"<0x([0-9A-Fa-f]+)>", token_repr):
        return len(bytes.fromhex(m.group(1)))

    extra_token = tokenizer.encode(".", add_special_tokens=False)[0]
    excess_bytes = len(".".encode("utf-8"))
    decoded = tokenizer.decode([extra_token, idx]).encode("utf-8")
    return len(decoded) - excess_bytes


def _tokenmonster_byte_length_of_token(tokenizer: MarinTokenizer, idx: int) -> int | None:
    """Return TokenMonster source-byte contribution when metadata is available.

    TokenMonster tokens can carry capcode/control markers in their raw token
    strings. Those markers are not source text bytes, and the delete-space
    marker can reduce the byte length of the following decoded token. Using
    ``decode([prefix, token])`` or the raw token string therefore inflates BPB
    denominators for capitalized words. TokenMonster exposes the already
    decoded token text; use that and account for standalone delete markers.
    """
    id_to_token_decoded = getattr(tokenizer, "id_to_token_decoded", None)
    if id_to_token_decoded is None:
        return None

    convert_ids_to_tokens = getattr(tokenizer, "convert_ids_to_tokens", None)
    id_to_token = getattr(tokenizer, "id_to_token", None)
    if convert_ids_to_tokens is not None:
        token_repr = convert_ids_to_tokens(idx)
    elif id_to_token is not None:
        token_repr = id_to_token(idx)
    else:
        return None
    token_decoded = id_to_token_decoded(idx)

    token_type = None
    get_dictionary = getattr(tokenizer, "get_dictionary", None)
    if get_dictionary is not None:
        dictionary = _TOKENMONSTER_DICTIONARY_CACHE.setdefault(id(tokenizer), get_dictionary())
        token_info = dictionary.get(idx, {})
        token_type = token_info.get("type")

    if token_type in {2, "special"} or idx in getattr(tokenizer, "all_special_ids", ()):
        return 0

    if isinstance(token_repr, str) and token_repr == "\ufffd" and idx < 256:
        return 1

    if token_decoded == "" and isinstance(token_repr, str) and token_repr.startswith("D"):
        return -1

    return len(token_decoded.encode("utf-8"))
