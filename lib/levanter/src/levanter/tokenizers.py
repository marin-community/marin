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
import re
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import jinja2
import jinja2.ext
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

    def __len__(self) -> int: ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str: ...

    def encode_batch(self, texts: list[str], *, add_special_tokens: bool = False) -> list[list[int]]: ...

    def get_vocab(self) -> dict[str, int]: ...

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]: ...

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]: ...

    @property
    def all_special_ids(self) -> list[int]: ...

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

    def apply_chat_template_with_masks(
        self,
        conversations: list[list[dict[str, str]]],
        *,
        chat_template: str | None = None,
        **kwargs,
    ) -> dict[str, list[list[int]]]: ...


# Sentinel used to mark generation (assistant) boundaries in rendered templates.
_GENERATION_SENTINEL_START = "__MARIN_GEN_START_7f3a9c__"
_GENERATION_SENTINEL_END = "__MARIN_GEN_END_7f3a9c__"


class _GenerationSentinelExtension(jinja2.ext.Extension):
    """Jinja2 extension that wraps {% generation %}...{% endgeneration %} block content
    with sentinel strings, preserving the same whitespace behavior as HF's AssistantTracker."""

    tags = {"generation"}

    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.CallBlock:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_wrap_generation"), [], [], body).set_lineno(lineno)

    @staticmethod
    def _wrap_generation(caller: jinja2.runtime.Macro) -> str:
        return _GENERATION_SENTINEL_START + caller() + _GENERATION_SENTINEL_END


class _GenerationStripExtension(jinja2.ext.Extension):
    """Jinja2 extension that renders {% generation %}...{% endgeneration %} blocks
    as plain content (no sentinels), for use in apply_chat_template without masks."""

    tags = {"generation"}

    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.CallBlock:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_passthrough"), [], [], body).set_lineno(lineno)

    @staticmethod
    def _passthrough(caller: jinja2.runtime.Macro) -> str:
        return caller()


def _make_jinja_env(extensions: list[type]) -> jinja2.Environment:
    """Create a jinja2 environment matching HF's template rendering settings."""
    return jinja2.Environment(
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=extensions,
    )


def _apply_chat_template_with_masks(
    tokenizer: "MarinTokenizer",
    conversations: list[list[dict[str, str]]],
    *,
    chat_template: str | None = None,
    **kwargs,
) -> dict[str, list[list[int]]]:
    """Render chat template for batched conversations, returning input_ids and assistant_masks.

    Uses a jinja2 extension to wrap {% generation %}...{% endgeneration %} block content
    with sentinel strings, then uses the sentinel positions to determine which tokens
    correspond to assistant content.
    """
    template_str = chat_template or tokenizer.chat_template
    if template_str is None:
        raise ValueError(f"Tokenizer {tokenizer.name_or_path} has no chat template")

    env = _make_jinja_env([_GenerationSentinelExtension])
    compiled = env.from_string(template_str)

    all_ids: list[list[int]] = []
    all_masks: list[list[int]] = []

    for conversation in conversations:
        rendered = compiled.render(
            messages=conversation,
            add_generation_prompt=False,
            bos_token=tokenizer.bos_token or "",
            eos_token=tokenizer.eos_token or "",
            **kwargs,
        )

        ids: list[int] = []
        mask: list[int] = []
        is_assistant = False

        parts = re.split(
            f"({re.escape(_GENERATION_SENTINEL_START)}|{re.escape(_GENERATION_SENTINEL_END)})",
            rendered,
        )

        # Each segment is encoded independently. BPE merges that would span a
        # sentinel boundary are lost, which can produce slightly different token
        # IDs at the boundary vs encoding the full string. This matches HF's
        # AssistantTracker behavior which has the same limitation.
        for part in parts:
            if part == _GENERATION_SENTINEL_START:
                is_assistant = True
                continue
            if part == _GENERATION_SENTINEL_END:
                is_assistant = False
                continue
            if not part:
                continue
            segment_ids = tokenizer.encode(part, add_special_tokens=False)
            ids.extend(segment_ids)
            mask.extend([1 if is_assistant else 0] * len(segment_ids))

        all_ids.append(ids)
        all_masks.append(mask)

    return {"input_ids": all_ids, "assistant_masks": all_masks}


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
    _all_special_ids: list[int]
    _id_to_token: dict[int, str] = dataclasses.field(default_factory=dict, repr=False)

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

    def __len__(self) -> int:
        return self._vocab_size

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

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        if isinstance(ids, int):
            return self._id_to_token.get(ids, f"<unk:{ids}>")
        return [self._id_to_token.get(i, f"<unk:{i}>") for i in ids]

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        vocab = self._tokenizer.get_vocab()
        if isinstance(tokens, str):
            return vocab.get(tokens, -1)
        return [vocab.get(t, -1) for t in tokens]

    @property
    def all_special_ids(self) -> list[int]:
        return self._all_special_ids

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
        env = _make_jinja_env([_GenerationStripExtension])
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

    def apply_chat_template_with_masks(
        self,
        conversations: list[list[dict[str, str]]],
        *,
        chat_template: str | None = None,
        **kwargs,
    ) -> dict[str, list[list[int]]]:
        return _apply_chat_template_with_masks(self, conversations, chat_template=chat_template, **kwargs)


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
    _all_special_ids: list[int]
    _id_to_token: dict[int, str] = dataclasses.field(default_factory=dict, repr=False)

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

    def __len__(self) -> int:
        return self._vocab_size

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

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        if isinstance(ids, int):
            return self._id_to_token.get(ids, f"<unk:{ids}>")
        return [self._id_to_token.get(i, f"<unk:{i}>") for i in ids]

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        vocab = self._tokenizer.get_vocab()
        if isinstance(tokens, str):
            return vocab.get(tokens, -1)
        return [vocab.get(t, -1) for t in tokens]

    @property
    def all_special_ids(self) -> list[int]:
        return self._all_special_ids

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
        env = _make_jinja_env([_GenerationStripExtension])
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

    def apply_chat_template_with_masks(
        self,
        conversations: list[list[dict[str, str]]],
        *,
        chat_template: str | None = None,
        **kwargs,
    ) -> dict[str, list[list[int]]]:
        return _apply_chat_template_with_masks(self, conversations, chat_template=chat_template, **kwargs)


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


def _collect_special_ids(
    config: dict,
    vocab: dict[str, int],
    bos_id: int | None,
    eos_id: int | None,
    pad_id: int | None,
) -> list[int]:
    """Collect all special token IDs from known special tokens and added_tokens_decoder."""
    ids: set[int] = set()
    for token_id in (bos_id, eos_id, pad_id):
        if token_id is not None:
            ids.add(token_id)

    # Include tokens marked as special in added_tokens_decoder
    for id_str, token_info in config.get("added_tokens_decoder", {}).items():
        if isinstance(token_info, dict) and token_info.get("special", False):
            ids.add(int(id_str))

    return sorted(ids)


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

    all_special_ids = _collect_special_ids(config, vocab, bos_id, eos_id, pad_id)
    id_to_token = {v: k for k, v in vocab.items()}

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
        _all_special_ids=all_special_ids,
        _id_to_token=id_to_token,
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

    # Use hf_hub_download for auth support (gated models like Llama 3).
    # tokie's from_pretrained doesn't pass HF_TOKEN.
    local_json = os.path.join(name_or_path, "tokenizer.json")
    if not os.path.isfile(local_json):
        local_json = hf_hub_download(name_or_path, "tokenizer.json")
    tok = tokie.Tokenizer.from_json(local_json)
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

    # For tokie, we don't have a complete vocab dict, so pass empty vocab for _collect_special_ids.
    # The added_tokens_decoder in config + bos/eos/pad are sufficient.
    all_special_ids = _collect_special_ids(config, {}, bos_id, eos_id, pad_id)

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
        _all_special_ids=all_special_ids,
        _id_to_token={v: k for k, v in tok.get_vocab().items()},
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
