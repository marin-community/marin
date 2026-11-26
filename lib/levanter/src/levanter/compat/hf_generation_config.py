# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Generation defaults and stop tokens for Hugging Face checkpoint exports."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

GENERATION_CONFIG_FILENAME = "generation_config.json"

END_OF_TURN_TOKENS = ("<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<end_of_turn>")
"""Turn-boundary tokens added as stop tokens when the exported chat template emits them."""


GenerationConfigDict = dict[str, Any]


class ExportTokenizer(Protocol):
    """Tokenizer fields needed to derive generation stop tokens."""

    @property
    def bos_token_id(self) -> int | None: ...

    @property
    def eos_token_id(self) -> int | None: ...

    def get_vocab(self) -> dict[str, int]: ...

    def __len__(self) -> int: ...


def stop_token_ids(tokenizer: ExportTokenizer, chat_template: str | None) -> list[int]:
    """Return the tokenizer EOS id and turn-boundary tokens emitted by the chat template."""
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    if chat_template:
        vocab = tokenizer.get_vocab()
        ids.update(vocab[token] for token in END_OF_TURN_TOKENS if token in chat_template and token in vocab)
    return sorted(ids)


def _token_id_list(value: int | list[int] | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return list(value)


def export_generation_config(
    tokenizer: ExportTokenizer,
    *,
    chat_template: str | None,
    explicit: Mapping[str, Any] | None = None,
    sampling_defaults: Mapping[str, Any] | None = None,
) -> GenerationConfigDict:
    """Complete preserved generation settings with export defaults and stop tokens.

    Explicit fields win over caller-supplied sampling defaults. ``eos_token_id``
    is the union of the explicit ids, the tokenizer EOS id, and turn-boundary
    tokens emitted by the exported chat template. Greedy configs do not receive
    sampling defaults.

    Raises:
        ValueError: If a stop token id is outside the tokenizer vocabulary.
    """
    config: GenerationConfigDict = dict(explicit or {})
    eos_ids = sorted(set(_token_id_list(config.get("eos_token_id"))) | set(stop_token_ids(tokenizer, chat_template)))
    vocab_size = len(tokenizer)
    for token_id in eos_ids:
        if not isinstance(token_id, int) or not 0 <= token_id < vocab_size:
            raise ValueError(f"eos_token_id {token_id!r} is outside the tokenizer vocabulary [0, {vocab_size})")
    if eos_ids:
        config["eos_token_id"] = eos_ids
    if "bos_token_id" not in config and tokenizer.bos_token_id is not None:
        config["bos_token_id"] = tokenizer.bos_token_id
    if config.get("do_sample", True):
        for key, value in (sampling_defaults or {}).items():
            config.setdefault(key, value)
    return config


def write_generation_config(directory: str, config: Mapping[str, Any]) -> Path:
    """Write a deterministic ``generation_config.json`` in ``directory``."""
    path = Path(directory) / GENERATION_CONFIG_FILENAME
    path.write_text(json.dumps(dict(config), indent=2, sort_keys=True) + "\n")
    return path
