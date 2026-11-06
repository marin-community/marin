# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical ``generation_config.json`` for Hugging Face checkpoint exports.

Serving stacks such as vLLM read sampling defaults and stop tokens from
``generation_config.json``. Without sampling fields they sample at temperature
1.0 with no nucleus cutoff, and without the chat template's end-of-turn token
they stop only at the document boundary and continue into invented turns.
Marin HF exports write this file through :func:`export_generation_config`, so
the defaults live here and nowhere else.

This module is pure Python so that both the JAX (Levanter) and Torch (SkyRL)
stacks can share it.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

GENERATION_CONFIG_FILENAME = "generation_config.json"

END_OF_TURN_TOKENS = ("<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<end_of_turn>")
"""Turn-boundary tokens added as stop tokens when the exported chat template emits them."""


@dataclass(frozen=True)
class SamplingDefaults:
    temperature: float
    top_p: float


EXPORT_SAMPLING_DEFAULTS = SamplingDefaults(temperature=0.6, top_p=0.95)
"""Sampling written into every export. Matches Marin's eval-policy reasoning-model defaults."""

# Values are JSON scalars or token-id lists; preserved source configs may carry any HF generation field.
GenerationConfigDict = dict[str, Any]


class ExportTokenizer(Protocol):
    """The tokenizer surface needed to derive stop tokens (HF tokenizers and Levanter's MarinTokenizer)."""

    @property
    def bos_token_id(self) -> int | None: ...

    @property
    def eos_token_id(self) -> int | None: ...

    def get_vocab(self) -> dict[str, int]: ...

    def __len__(self) -> int: ...


def stop_token_ids(tokenizer: ExportTokenizer, chat_template: str | None) -> list[int]:
    """Return the tokenizer's EOS id plus every end-of-turn token that ``chat_template`` emits."""
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    if chat_template:
        vocab = tokenizer.get_vocab()
        ids.update(vocab[token] for token in END_OF_TURN_TOKENS if token in chat_template and token in vocab)
    return sorted(ids)


def _token_id_list(value: Any) -> list[int]:
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
) -> GenerationConfigDict:
    """Build the ``generation_config.json`` contents for an export.

    Fields in ``explicit`` (a caller's configuration or the source checkpoint's
    generation config) win, except ``eos_token_id``: it is the union of the
    explicit ids, the tokenizer's EOS id, and the end-of-turn tokens of
    ``chat_template``, so an export always stops at the end of a turn. Missing
    ``bos_token_id`` comes from the tokenizer. Missing ``do_sample``,
    ``temperature``, and ``top_p`` come from :data:`EXPORT_SAMPLING_DEFAULTS`,
    unless ``explicit`` sets ``do_sample: false``.

    Args:
        tokenizer: The tokenizer shipped with the export.
        chat_template: The chat template shipped with the export, if any.
        explicit: Generation fields to preserve.

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
        config.setdefault("do_sample", True)
        config.setdefault("temperature", EXPORT_SAMPLING_DEFAULTS.temperature)
        config.setdefault("top_p", EXPORT_SAMPLING_DEFAULTS.top_p)
    return config


def write_generation_config(directory: str, config: Mapping[str, Any]) -> Path:
    """Write ``config`` as ``generation_config.json`` in a local ``directory``."""
    path = Path(directory) / GENERATION_CONFIG_FILENAME
    path.write_text(json.dumps(dict(config), indent=2, sort_keys=True) + "\n")
    return path
