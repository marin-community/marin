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
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import fsspec
import jinja2
import jinja2.ext
from huggingface_hub import __version__ as _hf_hub_version
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from tokenizers import Tokenizer as HfBaseTokenizer

logger = logging.getLogger(__name__)


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

    def as_hf_tokenizer(self) -> Any:
        """Return a HuggingFace PreTrainedTokenizerFast for this tokenizer.

        Useful for operations that require the HF API (save_pretrained,
        add_tokens, generation config, etc.).
        """
        ...


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
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._name_or_path, trust_remote_code=True)
        if self._chat_template is not None and getattr(tokenizer, "chat_template", None) != self._chat_template:
            tokenizer.chat_template = self._chat_template
        return tokenizer


@dataclasses.dataclass(frozen=True)
class KitokenMarinTokenizer:
    """MarinTokenizer backed by kitoken.

    Kitoken is a fast tokenizer supporting BPE, Unigram, and WordPiece models,
    compatible with SentencePiece, HuggingFace Tokenizers, Tiktoken, and Tekken formats.

    Limitations vs HfMarinTokenizer:
    - add_special_tokens manually prepends BOS (kitoken's encode_specials is different)
    - decode() returns bytes from kitoken; we convert to str with utf-8
    """

    _tokenizer: Any  # kitoken.Kitoken (optional dep, no top-level import)
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
    _prepend_bos: bool
    _vocab: dict[str, int] = dataclasses.field(default_factory=dict, repr=False)
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
        # encode_specials=True tells kitoken to recognize special token strings
        # (e.g. "<|end_of_text|>") in the input, matching HF's default behavior.
        # This is orthogonal to add_special_tokens which controls BOS/EOS wrapping.
        ids = self._tokenizer.encode(text, True)
        if add_special_tokens and self._prepend_bos and self._bos_id is not None:
            ids = [self._bos_id] + ids
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        # kitoken's decode_specials=True emits control tokens; False strips them.
        # MarinTokenizer's skip_special_tokens=True means strip, so invert.
        raw = self._tokenizer.decode(ids, not skip_special_tokens)
        return raw.decode("utf-8", errors="replace")

    def encode_batch(self, texts: list[str], *, add_special_tokens: bool = False) -> list[list[int]]:
        texts = ["".join(s) for s in texts]
        results = self._tokenizer.encode_all(texts, True)
        if add_special_tokens and self._prepend_bos and self._bos_id is not None:
            return [[self._bos_id] + ids for ids in results]
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
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._name_or_path, trust_remote_code=True)
        if self._chat_template is not None and getattr(tokenizer, "chat_template", None) != self._chat_template:
            tokenizer.chat_template = self._chat_template
        return tokenizer


class TokenizerBackend(StrEnum):
    HF = "hf"
    KITOKEN = "kitoken"


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
    if backend == TokenizerBackend.KITOKEN:
        tok = _load_kitoken_tokenizer(local_dir)
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
        with fsspec.open(src_url, "rb") as src:
            data = src.read()
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
        with open(local_path, "rb") as src, fsspec.open(mirror_url, "wb") as dst:
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
        mirror_fs = fsspec.filesystem("mirror")
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

    Returns the local directory path. ``lru_cache`` makes subsequent calls free.
    """
    local_dir = os.path.join(
        tempfile.gettempdir(),
        "levanter_tokenizers",
        name_or_path,
        f"hf-hub-{_hf_hub_version}",
    )
    os.makedirs(local_dir, exist_ok=True)

    # 1. Local cache hit.
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


def _post_processor_prepends_bos(name_or_path: str, bos_token: str | None) -> bool:
    """Check whether the tokenizer.json post-processor prepends BOS on add_special_tokens.

    HF tokenizers use a post-processor (TemplateProcessing) to decide what
    special tokens to add. If the post-processor's 'single' template starts
    with the BOS token, add_special_tokens=True should prepend BOS. Models
    like OLMo and Phi have no post-processor, so they don't prepend BOS.
    """
    if bos_token is None:
        return False

    local_json = os.path.join(name_or_path, "tokenizer.json")
    if not os.path.isfile(local_json):
        if os.path.isdir(name_or_path):
            return False
        try:
            local_json = hf_hub_download(name_or_path, "tokenizer.json")
        except (EntryNotFoundError, RepositoryNotFoundError):
            return False

    with open(local_json) as f:
        data = json.load(f)

    pp = data.get("post_processor")
    if pp is None:
        return False

    # Walk through Sequence or direct TemplateProcessing
    processors = [pp]
    if pp.get("type") == "Sequence":
        processors = pp.get("processors", [])

    for proc in processors:
        if proc.get("type") == "TemplateProcessing":
            single = proc.get("single", [])
            if single and isinstance(single[0], dict):
                special = single[0].get("SpecialToken", {})
                if special.get("id") == bos_token:
                    return True
    return False


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


def _load_kitoken_tokenizer(name_or_path: str) -> KitokenMarinTokenizer:
    import kitoken

    local_json = os.path.join(name_or_path, "tokenizer.json")
    if not os.path.isfile(local_json):
        if os.path.isdir(name_or_path):
            raise FileNotFoundError(f"Local tokenizer directory is missing tokenizer.json: {name_or_path}")
        local_json = hf_hub_download(name_or_path, "tokenizer.json")
    tok = kitoken.Kitoken.from_tokenizers_file(local_json)
    config = _load_tokenizer_config(name_or_path)

    bos_token = _resolve_special_token(config, "bos_token")
    eos_token = _resolve_special_token(config, "eos_token")
    pad_token = _resolve_special_token(config, "pad_token")

    # kitoken's get_vocab includes both regular and special tokens.
    # Resolve special token IDs from the config, falling back to the vocab.
    added_tokens = _build_added_token_map(config)
    bos_id = _resolve_special_token_id_from_config(config, "bos_token_id", bos_token, added_tokens)
    eos_id = _resolve_special_token_id_from_config(config, "eos_token_id", eos_token, added_tokens)
    pad_id = _resolve_special_token_id_from_config(config, "pad_token_id", pad_token, added_tokens)

    vocab = tok.get_vocab()
    vocab_size = tok.vocab_size

    all_special_ids = _collect_special_ids(config, vocab, bos_id, eos_id, pad_id)
    prepend_bos = _post_processor_prepends_bos(name_or_path, bos_token)

    return KitokenMarinTokenizer(
        _tokenizer=tok,
        _name_or_path=name_or_path,
        _bos_token=bos_token,
        _eos_token=eos_token,
        _pad_token=pad_token,
        _bos_id=bos_id,
        _eos_id=eos_id,
        _pad_id=pad_id,
        _vocab_size=vocab_size,
        _chat_template=config.get("chat_template") or _load_chat_template_jinja(name_or_path),
        _all_special_ids=all_special_ids,
        _prepend_bos=prepend_bos,
        _vocab=vocab,
        _id_to_token={v: k for k, v in vocab.items()},
    )
