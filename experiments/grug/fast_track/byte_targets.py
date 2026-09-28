# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-token byte targets for the byte-level auxiliary loss (``GrugModelConfig.byte_aux_bytes``)."""

import logging
import re

import numpy as np
from levanter.tokenizers import MarinTokenizer

logger = logging.getLogger(__name__)

_HEX_BYTE_TOKEN = re.compile(r"<0x([0-9A-Fa-f]+)>")


def token_bytes(tokenizer: MarinTokenizer, idx: int, dot: int, special: frozenset[int]) -> bytes:
    """UTF-8 bytes of one token: empty for special tokens, the literal byte for ``<0xNN>`` tokens, else the
    decode of ``[".", idx]`` minus the leading dot (the prefix keeps a token's leading space)."""
    if idx in special:
        return b""
    if m := _HEX_BYTE_TOKEN.fullmatch(tokenizer.convert_ids_to_tokens(idx)):
        return bytes.fromhex(m.group(1))
    return tokenizer.decode([dot, idx]).encode("utf-8")[1:]


def token_byte_table(tokenizer: MarinTokenizer, vocab_size: int, num_bytes: int) -> np.ndarray:
    """``[vocab_size, num_bytes]`` int32: each token's first ``num_bytes`` bytes, ``-1`` past its end."""
    dot = tokenizer.encode(".", add_special_tokens=False)[0]
    special = frozenset(tokenizer.all_special_ids)
    table = np.full((vocab_size, num_bytes), -1, np.int32)
    lengths = np.zeros(vocab_size, np.int32)
    for idx in range(min(vocab_size, len(tokenizer))):
        b = token_bytes(tokenizer, idx, dot, special)
        lengths[idx] = len(b)
        table[idx, : min(len(b), num_bytes)] = list(b[:num_bytes])
    logger.info(
        "byte targets: mean token length %.2f bytes, %.1f%% of tokens longer than %d bytes, %d empty",
        float(lengths.mean()),
        100.0 * float((lengths > num_bytes).mean()),
        num_bytes,
        int((lengths == 0).sum()),
    )
    return table
