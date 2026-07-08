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

import contextlib
import dataclasses
import functools
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import jinja2
import jinja2.ext
import jinja2.sandbox
from huggingface_hub import __version__ as _hf_hub_version
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from rigging.filesystem import StoragePath, filesystem, open_url
from tokenizers import Tokenizer as HfBaseTokenizer

logger = logging.getLogger(__name__)


# Borrowed from meta-llama/llama3 tokenizer.py: bound the size of any single
# string passed into the underlying tokenizer to avoid pathological inputs
# (e.g. multi-MB runs of whitespace from broken HTML→text extraction) blowing
# up Rust tokenizer working memory. We split on whitespace/non-whitespace
# transitions and cap each homogeneous run.
_MAX_ENCODE_CHARS = 400_000
_MAX_HOMOGENEOUS_RUN_CHARS = 25_000


# Match runs of N+ whitespace OR N+ non-whitespace chars. These are the only
# points where the input MUST be split to keep each substring's longest
# homogeneous run bounded; everything else passes through untouched so that
# BPE merges (e.g. " world" leading-space tokens) on normal text are
# preserved exactly as the underlying tokenizer would produce them.
_OVERLONG_RUN_RE = re.compile(rf"\s{{{_MAX_HOMOGENEOUS_RUN_CHARS},}}|\S{{{_MAX_HOMOGENEOUS_RUN_CHARS},}}")


def _safe_split_for_tokenizer(text: str) -> list[str]:
    """Split ``text`` into substrings safe to feed to a Rust BPE tokenizer.

    Each substring contains no more than ``_MAX_HOMOGENEOUS_RUN_CHARS``
    consecutive whitespace or non-whitespace characters. Inputs whose runs
    are all within the cap are returned unchanged as a single-element list,
    so normal text round-trips byte-identically through the tokenizer.
    """
    if len(text) <= _MAX_HOMOGENEOUS_RUN_CHARS:
        return [text]

    parts: list[str] = []
    last = 0
    for m in _OVERLONG_RUN_RE.finditer(text):
        if m.start() > last:
            parts.append(text[last : m.start()])
        run = m.group()
        for i in range(0, len(run), _MAX_HOMOGENEOUS_RUN_CHARS):
            parts.append(run[i : i + _MAX_HOMOGENEOUS_RUN_CHARS])
        last = m.end()
    if last < len(text):
        parts.append(text[last:])
    return parts or [text]


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
    ) -> dict[str, Any]: ...

    def as_hf_tokenizer(self) -> Any:
        """Return a HuggingFace PreTrainedTokenizerFast for this tokenizer.

        Useful for operations that require the HF API (save_pretrained,
        add_tokens, generation config, etc.).
        """
        ...


# Sentinel used to mark generation (assistant) boundaries in rendered templates.
_GENERATION_SENTINEL_START = "__MARIN_GEN_START_7f3a9c__"
_GENERATION_SENTINEL_END = "__MARIN_GEN_END_7f3a9c__"
_MESSAGE_SENTINEL_START = "__MARIN_MSG_START_7f3a9c_"
_MESSAGE_SENTINEL_END = "__MARIN_MSG_END_7f3a9c_"
_MESSAGE_INDEX_ATTR = "marin_message_index"
_MESSAGE_LOOP_COLLECTIONS = {"messages", "loop_messages"}
_MESSAGE_SENTINEL_RE = re.compile(
    rf"{re.escape(_MESSAGE_SENTINEL_START)}\d+__|{re.escape(_MESSAGE_SENTINEL_END)}\d+__"
)


class _GenerationSentinelExtension(jinja2.ext.Extension):
    """Jinja2 extension that wraps {% generation %}...{% endgeneration %} block content
    with sentinel strings, preserving the same whitespace behavior as HF's AssistantTracker."""

    tags = {"generation"}

    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.CallBlock:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        block = jinja2.nodes.CallBlock(self.call_method("_wrap_generation"), [], [], body)
        block.set_lineno(lineno)
        return block

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
        block = jinja2.nodes.CallBlock(self.call_method("_passthrough"), [], [], body)
        block.set_lineno(lineno)
        return block

    @staticmethod
    def _passthrough(caller: jinja2.runtime.Macro) -> str:
        return caller()


def _make_jinja_env(extensions: list[type]) -> jinja2.Environment:
    """Create a jinja2 environment matching HF's template rendering settings."""
    env = jinja2.sandbox.SandboxedEnvironment(
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=extensions,
    )
    # jinja2 types globals narrowly from its DEFAULT_NAMESPACE; pyrefly rejects assigning
    # other callables. Update via the mapping interface, which accepts t.Any values.
    env.globals.update(
        {
            "raise_exception": _raise_chat_template_exception,
            "strftime_now": lambda fmt: time.strftime(fmt),
        }
    )
    return env


def _raise_chat_template_exception(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)


def _block_name(block_tokens: list[tuple[str, str]]) -> str | None:
    for token_type, value in block_tokens:
        if token_type == "whitespace":
            continue
        if token_type == "name":
            return value
        return None
    return None


def _message_loop_variable(block_tokens: list[tuple[str, str]]) -> str | None:
    tokens = [(token_type, value) for token_type, value in block_tokens if token_type != "whitespace"]
    if len(tokens) != 4:
        return None
    if tokens[0] != ("name", "for") or tokens[2] != ("name", "in"):
        return None
    if tokens[3][0] != "name" or tokens[3][1] not in _MESSAGE_LOOP_COLLECTIONS:
        return None
    token_type, variable = tokens[1]
    if token_type != "name":
        return None
    return variable


def _chat_template_parts(chat_template: str) -> list[tuple[str, str, list[tuple[str, str]]]]:
    env = _make_jinja_env([])
    parts: list[tuple[str, str, list[tuple[str, str]]]] = []
    block_text: list[str] | None = None
    block_tokens: list[tuple[str, str]] = []

    for _, token_type, value in env.lex(chat_template):
        if token_type == "block_begin":
            block_text = [value]
            block_tokens = []
            continue

        if block_text is not None:
            block_text.append(value)
            if token_type == "block_end":
                parts.append(("block", "".join(block_text), block_tokens))
                block_text = None
                block_tokens = []
            else:
                block_tokens.append((token_type, value))
            continue

        parts.append(("text", value, []))

    return parts


def _message_sentinel_template(prefix: str, loop_variable: str) -> str:
    return '{{ "' + prefix + '" ~ ' + loop_variable + "." + _MESSAGE_INDEX_ATTR + ' ~ "__" }}'


def _append_message_end_sentinel(parts: list[str], loop_variable: str) -> None:
    sentinel = _message_sentinel_template(_MESSAGE_SENTINEL_END, loop_variable)
    if not parts:
        parts.append(sentinel)
        return

    previous = parts.pop()
    stripped = previous.rstrip()
    parts.append(stripped)
    parts.append(sentinel)
    parts.append(previous[len(stripped) :])


def _instrument_message_loop(chat_template: str) -> str:
    """Add per-message sentinels around the top-level `{% for message in messages %}` body.

    Hugging Face exposes assistant masks via `{% generation %}` blocks, but it does not expose
    token spans for each rendered chat message. Trace-labeled eval needs those spans so it can
    map source-message labels onto tokens after the tokenizer's chat template has added role
    headers, separators, and tool-call formatting. We use Jinja's lexer to identify block
    boundaries instead of regex-parsing the template language, then insert inert string
    sentinels that are removed after tokenization.
    """

    parts = _chat_template_parts(chat_template)
    instrumented: list[str] = []
    message_loop_depth: int | None = None
    message_loop_variable = ""

    for part_type, part, block_tokens in parts:
        if part_type != "block":
            if part:
                instrumented.append(part)
            continue

        if message_loop_depth is None:
            loop_variable = _message_loop_variable(block_tokens)
            if loop_variable is not None:
                message_loop_variable = loop_variable
                instrumented.append(part)
                instrumented.append(_message_sentinel_template(_MESSAGE_SENTINEL_START, message_loop_variable))
                message_loop_depth = 1
                continue
        else:
            block_name = _block_name(block_tokens)
            if block_name == "for":
                message_loop_depth += 1
            elif block_name == "endfor":
                message_loop_depth -= 1
                if message_loop_depth == 0:
                    _append_message_end_sentinel(instrumented, message_loop_variable)
                    instrumented.append(part)
                    message_loop_depth = None
                    message_loop_variable = ""
                    continue

        instrumented.append(part)

    return "".join(instrumented)


def _message_sentinel_index(sentinel: str, prefix: str) -> int:
    return int(sentinel[len(prefix) : -2])


class _ChatTemplateMessage(dict):
    def __init__(self, message: dict[str, str], message_index: int | None):
        super().__init__(message)
        self._message_index = message_index

    def __getattr__(self, key: str) -> Any:
        if key == _MESSAGE_INDEX_ATTR and self._message_index is not None:
            return self._message_index
        if key in self:
            return self[key]
        if key in {"reasoning_content", "tool_calls"}:
            return None
        raise AttributeError(key)


def _chat_template_messages(
    conversation: list[dict[str, str]], *, include_indices: bool
) -> list[_ChatTemplateMessage]:
    return [
        _ChatTemplateMessage(message, index if include_indices else None) for index, message in enumerate(conversation)
    ]


def _apply_chat_template_with_masks(
    tokenizer: "MarinTokenizer",
    conversations: list[list[dict[str, str]]],
    *,
    chat_template: str | None = None,
    return_message_spans: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Render chat templates for batched conversations and return token-level masks.

    The returned `assistant_masks` mark tokens rendered inside `{% generation %}` blocks.
    When `return_message_spans` is set, the returned `message_spans` list contains
    half-open token spans `(start, end)` for each source message after chat-template rendering.
    These spans are used by trace-labeled evals to project per-message labels onto the exact
    rendered prompt tokens, including role headers and tool-call formatting.
    """
    template_str = chat_template or tokenizer.chat_template
    if template_str is None:
        raise ValueError(f"Tokenizer {tokenizer.name_or_path} has no chat template")

    render_template = _instrument_message_loop(template_str) if return_message_spans else template_str
    env = _make_jinja_env([_GenerationSentinelExtension])
    compiled = env.from_string(render_template)

    all_ids: list[list[int]] = []
    all_masks: list[list[int]] = []
    all_message_spans: list[list[tuple[int, int]]] = []

    for conversation in conversations:
        render_conversation = _chat_template_messages(conversation, include_indices=return_message_spans)
        rendered = compiled.render(
            messages=render_conversation,
            add_generation_prompt=False,
            bos_token=tokenizer.bos_token or "",
            eos_token=tokenizer.eos_token or "",
            **kwargs,
        )

        ids: list[int] = []
        mask: list[int] = []
        is_assistant = False
        message_starts: dict[int, int] = {}
        message_spans = [(0, 0) for _ in conversation]

        parts = re.split(
            (
                f"({re.escape(_GENERATION_SENTINEL_START)}|"
                f"{re.escape(_GENERATION_SENTINEL_END)}|"
                f"{_MESSAGE_SENTINEL_RE.pattern})"
            ),
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
            if _MESSAGE_SENTINEL_RE.fullmatch(part):
                if part.startswith(_MESSAGE_SENTINEL_START):
                    message_index = _message_sentinel_index(part, _MESSAGE_SENTINEL_START)
                    if message_index < len(message_spans):
                        message_starts[message_index] = len(ids)
                else:
                    message_index = _message_sentinel_index(part, _MESSAGE_SENTINEL_END)
                    start = message_starts.pop(message_index, len(ids))
                    if message_index < len(message_spans):
                        message_spans[message_index] = (start, len(ids))
                continue
            if not part:
                continue
            segment_ids = tokenizer.encode(part, add_special_tokens=False)
            ids.extend(segment_ids)
            mask.extend([1 if is_assistant else 0] * len(segment_ids))

        if return_message_spans:
            observed_spans = [(start, end) for start, end in message_spans if start != end]
            if observed_spans:
                first_observed_start = observed_spans[0][0]
                for index, span in enumerate(message_spans):
                    if span == (0, 0) and index < len(message_spans) - 1:
                        message_spans[index] = (0, first_observed_start)
                    else:
                        break
            all_message_spans.append(message_spans)
        all_ids.append(ids)
        all_masks.append(mask)

    result: dict[str, Any] = {"input_ids": all_ids, "assistant_masks": all_masks}
    if return_message_spans:
        result["message_spans"] = all_message_spans
    return result


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
    _vocab: dict[str, int] = dataclasses.field(default_factory=dict, repr=False)

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
        parts = _safe_split_for_tokenizer(text)
        if len(parts) <= 1:
            return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
        # Multi-chunk path: encode each chunk without specials and prepend BOS
        # at the end. We don't append EOS — Llama-style BPE tokenizers used
        # here don't add EOS via the post-processor, matching the llama3
        # reference. If a future tokenizer's post-processor appends EOS, the
        # multi-chunk path would silently drop it.
        ids: list[int] = []
        encodings = self._tokenizer.encode_batch(parts, add_special_tokens=False)
        for enc in encodings:
            ids.extend(enc.ids)
        if add_special_tokens and self._bos_id is not None:
            ids = [self._bos_id, *ids]
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: list[str], *, add_special_tokens: bool = False) -> list[list[int]]:
        # Copy strings to release references to potentially large source buffers,
        # mitigating memory retention from sliced strings.
        texts = ["".join(s) for s in texts]

        # Flatten all parts across all texts into one batch so the underlying
        # Rust encoder can parallelize across them via rayon. ``origin[i]``
        # tracks which original text part ``i`` belongs to so we can scatter
        # the encoded ids back into per-text lists.
        flat_parts: list[str] = []
        origin: list[int] = []
        for orig_idx, text in enumerate(texts):
            for part in _safe_split_for_tokenizer(text):
                flat_parts.append(part)
                origin.append(orig_idx)

        encodings = self._tokenizer.encode_batch(flat_parts, add_special_tokens=False)

        results: list[list[int]] = [[] for _ in texts]
        for orig_idx, enc in zip(origin, encodings, strict=True):
            results[orig_idx].extend(enc.ids)

        if add_special_tokens and self._bos_id is not None:
            results = [[self._bos_id, *r] for r in results]
        return results

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        if isinstance(ids, int):
            return self._id_to_token.get(ids, f"<unk:{ids}>")
        return [self._id_to_token.get(i, f"<unk:{i}>") for i in ids]

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self._vocab.get(tokens, -1)
        return [self._vocab.get(t, -1) for t in tokens]

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

    def as_hf_tokenizer(self) -> Any:
        from transformers import AutoTokenizer  # noqa: PLC0415  # guarded: avoid eager torch

        tokenizer = AutoTokenizer.from_pretrained(self._name_or_path, trust_remote_code=True)
        if self._chat_template is not None and getattr(tokenizer, "chat_template", None) != self._chat_template:
            tokenizer.chat_template = self._chat_template
        return tokenizer


class TokenizerBackend(StrEnum):
    HF = "hf"


@functools.lru_cache(maxsize=32)
def load_tokenizer(
    name_or_path: str,
    *,
    backend: TokenizerBackend = TokenizerBackend.HF,
) -> MarinTokenizer:
    """Load a tokenizer by HF model name or local path.

    Files are staged once via mirror://tokenizers/ (GCS/S3) before falling back
    to HF Hub. Cached per (name_or_path, backend).
    """
    local_dir = _stage_tokenizer(name_or_path) if not os.path.isdir(name_or_path) else name_or_path
    if backend == TokenizerBackend.HF:
        tok = _load_hf_tokenizer(local_dir)
        return dataclasses.replace(tok, _name_or_path=name_or_path)
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


_MIRROR_TOKENIZER_PREFIX = "tokenizers"

# Glob patterns for the full set of files that may belong to a tokenizer.
# Broad enough to cover sentencepiece, BPE, wordpiece, tiktoken and chat
# templates; excludes model weights, model config, READMEs, images, etc.
# Used as ``allow_patterns`` for HF Hub ``snapshot_download``.
_TOKENIZER_ALLOW_PATTERNS = [
    "tokenizer*",  # tokenizer.json, tokenizer_config.json, tokenizer.model
    "chat_template*",  # chat_template.jinja, chat_template.json
    "special_tokens*",  # special_tokens_map.json
    "added_tokens*",  # added_tokens.json
    "vocab*",  # vocab.json, vocab.txt
    "merges*",  # merges.txt
    "spiece*",  # spiece.model (T5-style sentencepiece)
    "*.tiktoken",  # tiktoken format
]


def _fetch_file_atomic(src_url: str, dest_path: str) -> bool:
    """Atomically fetch src_url to dest_path via a .tmp sibling.

    Returns False if the source does not exist; re-raises all other errors.
    Prevents partial writes from poisoning the local cache on any failure.
    """
    tmp = dest_path + ".tmp"
    try:
        data = StoragePath(src_url).read_bytes()
        with open(tmp, "wb") as dst:
            dst.write(data)
        os.replace(tmp, dest_path)
        return True
    except FileNotFoundError:
        return False
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _copy_file_atomic(src_path: str, dest_path: str) -> None:
    """Atomically copy a local file via a .tmp sibling."""
    tmp = dest_path + ".tmp"
    try:
        shutil.copy2(src_path, tmp)
        os.replace(tmp, dest_path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _populate_mirror_file(local_path: str, mirror_url: str) -> None:
    """Best-effort push of a local file to the mirror. Swallows any failure."""
    try:
        with open(local_path, "rb") as src, open_url(mirror_url, "wb") as dst:
            dst.write(src.read())
    except Exception:
        logger.debug("Could not populate mirror at %s", mirror_url, exc_info=True)


def _try_load_tokenizer_from_dir(local_dir: str) -> bool:
    """Try to load a tokenizer from a local directory.

    Uses ``HfBaseTokenizer.from_file`` as the gate: if it can parse the
    ``tokenizer.json`` file, the tokenizer is usable. This catches missing
    files, 0-byte cache-poisoned files, and corrupt data — all of which
    should fall through to the next source.
    """
    tokenizer_json = os.path.join(local_dir, "tokenizer.json")
    if not os.path.isfile(tokenizer_json):
        return False
    try:
        HfBaseTokenizer.from_file(tokenizer_json)
        return True
    except Exception:
        return False


def _stage_from_mirror(name_or_path: str, local_dir: str) -> bool:
    """Copy tokenizer files from mirror:// to *local_dir*.

    Discovers whatever files the mirror holds via ``ls()`` (no hardcoded
    file list) and fetches them all atomically.  Returns True if any files
    were copied.
    """
    mirror_dir = f"{_MIRROR_TOKENIZER_PREFIX}/{name_or_path}/hf-hub-{_hf_hub_version}"
    mirror_base = f"mirror://{mirror_dir}"
    copied = False
    try:
        mirror_fs = filesystem("mirror")
        if mirror_fs.exists(mirror_dir):
            for entry in mirror_fs.ls(mirror_dir, detail=False):
                filename = os.path.basename(entry.rstrip("/"))
                if not filename:
                    continue
                if _fetch_file_atomic(f"{mirror_base}/{filename}", os.path.join(local_dir, filename)):
                    copied = True
            if copied:
                logger.info(
                    "Copied %s tokenizer files from mirror %s",
                    name_or_path,
                    mirror_base,
                )
    except Exception as e:
        logger.warning("Could not stage tokenizer from mirror %s: %s", mirror_base, e)
    return copied


def _stage_from_hf(name_or_path: str, local_dir: str) -> None:
    """Download tokenizer files from HF Hub and populate the mirror.

    Uses ``snapshot_download`` with tokenizer-file allow-patterns to fetch
    every tokenizer-relevant file the repo ships, then copies them into
    *local_dir* atomically and pushes to the mirror as a best-effort
    side-effect for future workers.

    Raises ``RepositoryNotFoundError`` / ``OSError`` if the repo or
    network is unreachable (matches pre-mirror behaviour).
    """
    snapshot_dir = snapshot_download(name_or_path, allow_patterns=_TOKENIZER_ALLOW_PATTERNS)

    mirror_base = f"mirror://{_MIRROR_TOKENIZER_PREFIX}/{name_or_path}/hf-hub-{_hf_hub_version}"

    for filename in sorted(os.listdir(snapshot_dir)):
        src_path = os.path.join(snapshot_dir, filename)
        if not os.path.isfile(src_path):
            continue
        dest = os.path.join(local_dir, filename)
        _copy_file_atomic(src_path, dest)
        _populate_mirror_file(dest, f"{mirror_base}/{filename}")


# Serializes tokenizer staging across threads. ``lru_cache`` deduplicates
# *repeat* calls but does not serialize concurrent *first* calls for the same
# key, so without this lock N threads (e.g. one per dataset component in
# ``build_caches``) race to write the same staging directory. That race
# corrupts the tokenizer.json a sibling is mid-read of and forces a fatal HF
# fall-through for mirror-only refs. A single process-wide lock is sufficient:
# staging is I/O-bound and an app rarely fetches more than one tokenizer, so
# serializing all staging costs nothing in practice.
_STAGE_LOCK = threading.Lock()


@functools.lru_cache(maxsize=32)
def _stage_tokenizer(name_or_path: str) -> str:
    """Download the full set of tokenizer files to a stable local directory.

    Uses actual tokenizer loading (``HfBaseTokenizer.from_file``) as the
    success gate — no hardcoded file-list checks.  Resolution order:

      1. Local cache — a prior call already staged this tokenizer on disk.
      2. mirror://tokenizers/{org}/{model}/hf-hub-{ver}/ — discovered via ``ls()``, fetches
         whatever files a previous worker populated (any shape).
      3. HF Hub via ``snapshot_download`` — fetches every tokenizer-relevant
         file the repo ships, then populates the mirror for future workers.

    The local cache directory is keyed by the ``huggingface_hub`` library
    version so that a library upgrade busts the cache and re-downloads.
    Once staged, downstream loaders operate purely on local files — no
    HF Hub network calls (HEAD revalidation, etc.) are made.

    Safe to call concurrently for the same ref: staging is serialized so only
    one thread downloads while the others reuse the staged files.

    Returns the local directory path. ``lru_cache`` makes subsequent calls free.
    """
    local_dir = os.path.join(
        tempfile.gettempdir(),
        "levanter_tokenizers",
        name_or_path,
        f"hf-hub-{_hf_hub_version}",
    )
    os.makedirs(local_dir, exist_ok=True)

    with _STAGE_LOCK:
        # 1. Local cache hit (also the double-checked fast path for threads that
        #    waited on the lock while another thread staged this same ref).
        if _try_load_tokenizer_from_dir(local_dir):
            return local_dir

        # 2. Mirror: copy whatever files are present, then try loading.
        if _stage_from_mirror(name_or_path, local_dir) and _try_load_tokenizer_from_dir(local_dir):
            return local_dir

        # 3. HF Hub: full download, populate mirror as side-effect.
        _stage_from_hf(name_or_path, local_dir)
        return local_dir


def _load_hf_base_tokenizer(local_dir: str) -> HfBaseTokenizer:
    """Load HfBaseTokenizer from a pre-staged local directory using from_file.

    ``tokenizers.Tokenizer.from_pretrained`` only accepts Hub identifiers and
    has no ``local_files_only`` mode, so we locate tokenizer.json directly.
    """
    tokenizer_json = os.path.join(local_dir, "tokenizer.json")
    if not os.path.isfile(tokenizer_json):
        raise FileNotFoundError(f"tokenizer.json not found in staged directory: {local_dir}")
    return HfBaseTokenizer.from_file(tokenizer_json)


def _load_chat_template_jinja(name_or_path: str) -> str | None:
    """Load chat template from a standalone .jinja file.

    HF transformers>=4.43 saves large chat templates to a separate
    ``chat_template.jinja`` file instead of inlining them in
    ``tokenizer_config.json``.
    """
    local_path = os.path.join(name_or_path, "chat_template.jinja")
    if os.path.isfile(local_path):
        with open(local_path) as f:
            return f.read()

    if os.path.isdir(name_or_path):
        return None

    try:
        path = hf_hub_download(name_or_path, "chat_template.jinja")
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None

    with open(path) as f:
        return f.read()


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

    chat_template = config.get("chat_template") or _load_chat_template_jinja(name_or_path)

    return HfMarinTokenizer(
        _tokenizer=tok,
        _name_or_path=name_or_path,
        _bos_id=bos_id,
        _eos_id=eos_id,
        _pad_id=pad_id,
        _bos_token=bos_token,
        _eos_token=eos_token,
        _chat_template=chat_template,
        _vocab_size=tok.get_vocab_size(),
        _all_special_ids=all_special_ids,
        _id_to_token=id_to_token,
        _vocab=vocab,
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

    if os.path.isdir(name_or_path):
        return {}

    try:
        path = hf_hub_download(name_or_path, "tokenizer_config.json")
    except (EntryNotFoundError, RepositoryNotFoundError):
        return {}

    with open(path) as f:
        return json.load(f)
