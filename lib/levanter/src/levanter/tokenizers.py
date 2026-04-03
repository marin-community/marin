# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
MarinTokenizer abstraction layer.

Provides a Protocol-based tokenizer interface that decouples callers from
HuggingFace's transformers library. The HF backend uses `tokenizers.Tokenizer`
(the Rust library) directly, avoiding the torch import that transformers pulls in.

Usage:
    from levanter.tokenizers import load_tokenizer
    tok = load_tokenizer("meta-llama/Llama-3.1-8B")
    ids = tok.encode("hello world")
"""

import dataclasses
import functools
import json
import os
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import jinja2
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from tokenizers import Tokenizer as HfBaseTokenizer


@runtime_checkable
class MarinTokenizer(Protocol):
    @property
    def name_or_path(self) -> str: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def bos_token_id(self) -> int | None: ...

    @property
    def eos_token_id(self) -> int | None: ...

    @property
    def pad_token_id(self) -> int | None: ...

    @property
    def bos_token(self) -> str | None: ...

    @property
    def eos_token(self) -> str | None: ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str: ...

    def encode_batch(self, texts: list[str], *, add_special_tokens: bool = False) -> list[list[int]]: ...

    def get_vocab(self) -> dict[str, int]: ...

    @property
    def chat_template(self) -> str | None: ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> str | list[int]: ...


@dataclasses.dataclass(frozen=True)
class HfMarinTokenizer:
    """MarinTokenizer backed by the HF tokenizers (Rust) library."""

    _tokenizer: HfBaseTokenizer
    _name_or_path: str
    _bos_id: int | None
    _eos_id: int | None
    _pad_id: int | None
    _bos_token: str | None
    _eos_token: str | None
    _chat_template: str | None
    _vocab_size: int

    @property
    def name_or_path(self) -> str:
        return self._name_or_path

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def bos_token_id(self) -> int | None:
        return self._bos_id

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_id

    @property
    def pad_token_id(self) -> int | None:
        return self._pad_id

    @property
    def bos_token(self) -> str | None:
        return self._bos_token

    @property
    def eos_token(self) -> str | None:
        return self._eos_token

    @property
    def chat_template(self) -> str | None:
        return self._chat_template

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: list[str], *, add_special_tokens: bool = False) -> list[list[int]]:
        # Copy strings to release references to potentially large source buffers,
        # mitigating memory retention from sliced strings.
        texts = ["".join(s) for s in texts]
        encodings = self._tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [enc.ids for enc in encodings]

    def get_vocab(self) -> dict[str, int]:
        return self._tokenizer.get_vocab()

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> str | list[int]:
        if self._chat_template is None:
            raise ValueError(f"Tokenizer {self._name_or_path} has no chat template")
        env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        template = env.from_string(self._chat_template)
        rendered = template.render(
            messages=conversation,
            add_generation_prompt=add_generation_prompt,
            bos_token=self._bos_token or "",
            eos_token=self._eos_token or "",
            **kwargs,
        )
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


@dataclasses.dataclass(frozen=True)
class TokieMarinTokenizer:
    """Experimental MarinTokenizer backed by tokie.

    Tokie is alpha-stage software. This backend exists for correctness verification
    and performance benchmarking against the HF backend. Do not use in production
    until tokie's accuracy warning is removed.

    Limitations vs HfMarinTokenizer:
    - decode() ignores skip_special_tokens (tokie doesn't support it)
    - Decoding token IDs for special tokens (e.g. bos/eos) may panic in tokie
    """

    _tokenizer: Any  # tokie.Tokenizer (optional dep, no top-level import)
    _name_or_path: str
    _bos_token: str | None
    _eos_token: str | None
    _pad_token: str | None
    _bos_id: int | None
    _eos_id: int | None
    _pad_id: int | None
    _vocab_size: int
    _chat_template: str | None

    @property
    def name_or_path(self) -> str:
        return self._name_or_path

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def bos_token_id(self) -> int | None:
        return self._bos_id

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_id

    @property
    def pad_token_id(self) -> int | None:
        return self._pad_id

    @property
    def bos_token(self) -> str | None:
        return self._bos_token

    @property
    def eos_token(self) -> str | None:
        return self._eos_token

    @property
    def chat_template(self) -> str | None:
        return self._chat_template

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        # tokie's decode() does not support skip_special_tokens.
        # Filter out known special token IDs manually to avoid panics in tokie's
        # decoder when it encounters IDs outside its vocabulary range.
        if skip_special_tokens:
            special_ids = {self._bos_id, self._eos_id, self._pad_id} - {None}
            ids = [i for i in ids if i not in special_ids]
        return self._tokenizer.decode(ids)

    def encode_batch(self, texts: list[str], *, add_special_tokens: bool = False) -> list[list[int]]:
        encodings = self._tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [enc.ids for enc in encodings]

    def get_vocab(self) -> dict[str, int]:
        return self._tokenizer.get_vocab()

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> str | list[int]:
        if self._chat_template is None:
            raise ValueError(f"Tokenizer {self._name_or_path} has no chat template")
        env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        template = env.from_string(self._chat_template)
        rendered = template.render(
            messages=conversation,
            add_generation_prompt=add_generation_prompt,
            bos_token=self._bos_token or "",
            eos_token=self._eos_token or "",
            **kwargs,
        )
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


class TokenizerBackend(StrEnum):
    HF = "hf"
    TOKIE = "tokie"


@functools.lru_cache(maxsize=32)
def load_tokenizer(
    name_or_path: str,
    *,
    backend: TokenizerBackend = TokenizerBackend.HF,
) -> MarinTokenizer:
    """Load a tokenizer by HF model name or local path.

    Cached per (name_or_path, backend). The HF backend uses tokenizers.Tokenizer
    directly (not the transformers wrapper), avoiding the torch import.
    """
    if backend == TokenizerBackend.HF:
        return _load_hf_tokenizer(name_or_path)
    if backend == TokenizerBackend.TOKIE:
        return _load_tokie_tokenizer(name_or_path)
    raise ValueError(f"Unknown backend: {backend}")


def _load_hf_base_tokenizer(name_or_path: str) -> HfBaseTokenizer:
    """Load an HfBaseTokenizer, handling both local paths and HF hub names."""
    local_path = os.path.join(name_or_path, "tokenizer.json")
    if os.path.isfile(local_path):
        return HfBaseTokenizer.from_file(local_path)
    return HfBaseTokenizer.from_pretrained(name_or_path)


def _load_hf_tokenizer(name_or_path: str) -> HfMarinTokenizer:
    tok = _load_hf_base_tokenizer(name_or_path)
    config = _load_tokenizer_config(name_or_path)

    bos_token = _resolve_special_token(config, "bos_token")
    eos_token = _resolve_special_token(config, "eos_token")
    pad_token = _resolve_special_token(config, "pad_token")

    vocab = tok.get_vocab()
    bos_id = vocab.get(bos_token) if bos_token is not None else None
    eos_id = vocab.get(eos_token) if eos_token is not None else None
    pad_id = vocab.get(pad_token) if pad_token is not None else None

    return HfMarinTokenizer(
        _tokenizer=tok,
        _name_or_path=name_or_path,
        _bos_id=bos_id,
        _eos_id=eos_id,
        _pad_id=pad_id,
        _bos_token=bos_token,
        _eos_token=eos_token,
        _chat_template=config.get("chat_template"),
        _vocab_size=tok.get_vocab_size(),
    )


def _resolve_special_token(config: dict, key: str) -> str | None:
    """Extract a special token string from tokenizer_config.json.

    The value can be a plain string or a dict like {"content": "<s>", ...}.
    """
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("content")
    return value


def _load_tokenizer_config(name_or_path: str) -> dict:
    """Load tokenizer_config.json from HF hub or local path."""
    local_path = os.path.join(name_or_path, "tokenizer_config.json")
    if os.path.isfile(local_path):
        with open(local_path) as f:
            return json.load(f)

    try:
        path = hf_hub_download(name_or_path, "tokenizer_config.json")
    except (EntryNotFoundError, RepositoryNotFoundError):
        return {}

    with open(path) as f:
        return json.load(f)


def _load_tokie_tokenizer(name_or_path: str) -> TokieMarinTokenizer:
    import tokie

    tok = tokie.Tokenizer.from_pretrained(name_or_path)
    config = _load_tokenizer_config(name_or_path)

    bos_token = _resolve_special_token(config, "bos_token")
    eos_token = _resolve_special_token(config, "eos_token")
    pad_token = _resolve_special_token(config, "pad_token")

    # tokie's vocab_size and token_to_id exclude added/special tokens, so we
    # resolve IDs from the added_tokens_decoder in tokenizer_config.json.
    added_tokens = _build_added_token_map(config)
    bos_id = _resolve_special_token_id_from_config(config, "bos_token_id", bos_token, added_tokens)
    eos_id = _resolve_special_token_id_from_config(config, "eos_token_id", eos_token, added_tokens)
    pad_id = _resolve_special_token_id_from_config(config, "pad_token_id", pad_token, added_tokens)

    # tokie.vocab_size excludes added tokens. The true vocab size includes them.
    vocab_size = tok.vocab_size + len(added_tokens)

    return TokieMarinTokenizer(
        _tokenizer=tok,
        _name_or_path=name_or_path,
        _bos_token=bos_token,
        _eos_token=eos_token,
        _pad_token=pad_token,
        _bos_id=bos_id,
        _eos_id=eos_id,
        _pad_id=pad_id,
        _vocab_size=vocab_size,
        _chat_template=config.get("chat_template"),
    )


def _build_added_token_map(config: dict) -> dict[str, int]:
    """Build a {token_string: token_id} map from the added_tokens_decoder in tokenizer_config.json."""
    added_tokens_decoder = config.get("added_tokens_decoder", {})
    result: dict[str, int] = {}
    for id_str, token_info in added_tokens_decoder.items():
        content = token_info.get("content") if isinstance(token_info, dict) else None
        if content is not None:
            result[content] = int(id_str)
    return result


def _resolve_special_token_id_from_config(
    config: dict, key: str, token_str: str | None, added_tokens: dict[str, int]
) -> int | None:
    """Resolve a special token ID from config fields or the added_tokens_decoder map."""
    config_id = config.get(key)
    if config_id is not None:
        return int(config_id)
    if token_str is not None:
        return added_tokens.get(token_str)
    return None
