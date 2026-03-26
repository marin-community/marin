# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for exporting HuggingFace-compatible checkpoints."""

import logging

from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

GenerationConfigDict = dict[str, int | list[int]]


def build_generation_config(
    tokenizer: PreTrainedTokenizerBase,
    eos_token_ids: list[int] | None,
) -> GenerationConfigDict | None:
    """Build a validated generation_config dict from explicit EOS token IDs.

    The returned dict is suitable for writing as ``generation_config.json``
    alongside an HF checkpoint.  It tells inference tools like vLLM which
    tokens should stop generation (e.g. both ``<|end_of_text|>`` and
    ``<|eot_id|>`` for chat models).

    Normalization guarantees:
    - Output ``eos_token_id`` is always sorted and deduplicated.
    - The tokenizer's own ``eos_token_id`` is auto-added if not already present.

    Args:
        tokenizer: The tokenizer that will be saved with the checkpoint.
        eos_token_ids: Explicit list of EOS token IDs, or ``None`` to skip.

    Returns:
        A config dict ready for JSON serialization, or ``None`` if
        *eos_token_ids* is ``None``.

    Raises:
        ValueError: If the list is empty, contains non-ints, or contains
            IDs outside the tokenizer's vocabulary range.
    """
    if eos_token_ids is None:
        return None

    if not eos_token_ids:
        raise ValueError("hf_generation_eos_token_ids must be non-empty when set")

    vocab_size = len(tokenizer)
    for tid in eos_token_ids:
        if not isinstance(tid, int):
            raise ValueError(f"hf_generation_eos_token_ids contains non-int: {tid!r}")
        if not (0 <= tid < vocab_size):
            raise ValueError(f"Token ID {tid} out of range [0, {vocab_size})")

    ids = set(eos_token_ids)

    tok_eos = tokenizer.eos_token_id
    if tok_eos is None:
        logger.warning("Tokenizer has no eos_token_id; generation config will use only the provided IDs")
    elif tok_eos not in ids:
        logger.info("Auto-adding tokenizer eos_token_id=%d to generation config", tok_eos)
        ids.add(tok_eos)

    gen_config: GenerationConfigDict = {"eos_token_id": sorted(ids)}
    if tokenizer.bos_token_id is not None:
        gen_config["bos_token_id"] = tokenizer.bos_token_id
    return gen_config
