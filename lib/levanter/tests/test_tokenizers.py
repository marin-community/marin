# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive test suite for MarinTokenizer backends (HF, kitoken).

Tests are parameterized across all available backends. Backends that are not
installed are skipped gracefully. The test model is meta-llama/Llama-3.1-8B,
which requires HF authentication (tests skip if auth is missing).
"""

import json
import os
import pathlib
import re
import shutil
from unittest.mock import patch

import pytest
from huggingface_hub import __version__ as _hf_hub_version

from levanter.tokenizers import (
    MarinTokenizer,
    TokenizerBackend,
    _load_tokenizer_config,
    _stage_from_hf,
    _stage_from_mirror,
    _stage_tokenizer,
    _try_load_tokenizer_from_dir,
    load_tokenizer,
)

try:
    import kitoken as _kitoken  # noqa: F401

    HAS_KITOKEN = True
except ImportError:
    HAS_KITOKEN = False


MODEL_NAME = "meta-llama/Llama-3.1-8B"


def _can_load_model() -> bool:
    """Check if we can authenticate and load the gated model."""
    try:
        load_tokenizer.cache_clear()
        load_tokenizer(MODEL_NAME)
        return True
    except Exception:
        return False


# Set once at import time to avoid repeated network calls.
_MODEL_AVAILABLE = _can_load_model()

requires_model = pytest.mark.skipif(not _MODEL_AVAILABLE, reason="HF auth or network unavailable for gated model")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Pre-load available backends once at module import.  The loaded tokenizers are
# cached in _BACKEND_TOKENIZERS so every fixture reuses the same instances
# instead of hitting the network/disk on each test.
_BACKEND_TOKENIZERS: dict[str, MarinTokenizer] = {}
_AVAILABLE_BACKENDS: list[str] = []
if _MODEL_AVAILABLE:
    load_tokenizer.cache_clear()
    _BACKEND_TOKENIZERS["hf"] = load_tokenizer(MODEL_NAME, backend=TokenizerBackend.HF)
    _AVAILABLE_BACKENDS.append("hf")
    if HAS_KITOKEN:
        _BACKEND_TOKENIZERS["kitoken"] = load_tokenizer(MODEL_NAME, backend=TokenizerBackend.KITOKEN)
        _AVAILABLE_BACKENDS.append("kitoken")


@pytest.fixture(scope="module", params=_AVAILABLE_BACKENDS if _AVAILABLE_BACKENDS else ["_skip_all"])
def backend_tokenizer(request):
    """Parameterized fixture yielding each available backend tokenizer.

    Module-scoped so each backend is loaded once per test module, not per test.
    """
    name = request.param
    if name == "_skip_all":
        pytest.skip("No tokenizer backends available")
    return _BACKEND_TOKENIZERS[name]


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

# 100+ diverse strings for cross-backend verification and encoding tests.
DIVERSE_TEXTS = [
    # Basic ASCII
    "hello world",
    "",
    " ",
    "   ",
    "a",
    "Z",
    "Hello, World!",
    "The quick brown fox jumps over the lazy dog.",
    "0123456789",
    "3.14159265358979",
    "-1.5e10",
    "!@#$%^&*()_+-=[]{}|;':\",./<>?",
    # Whitespace variants
    "\n",
    "\t\t\n",
    "\r\n",
    "\r",
    "line1\nline2\nline3",
    "tab1\ttab2\ttab3",
    "col1\tcol2\tcol3\nval1\tval2\tval3",
    "word1  word2   word3    word4",
    "  leading and trailing spaces  ",
    "\n\n\n",
    # Non-breaking and unicode whitespace
    "the\u00a0non-breaking\u00a0space",
    "thin\u2009space",
    "em\u2003space",
    # European languages
    "Bonjour le monde",
    "Hallo Welt",
    "Hola mundo",
    "café résumé naïve",
    "Привет мир",
    # CJK
    "こんにちは世界",
    "你好世界",
    "안녕하세요 세계",
    "Tokyo 東京 Seoul 서울 Beijing 北京",
    # Arabic and RTL
    "مرحبا بالعالم",
    "שלום עולם",
    # Indic
    "नमस्ते दुनिया",
    "สวัสดีชาวโลก",
    # Mixed scripts
    "English и Русский and 日本語",
    # Emoji
    "🌍🌎🌏",
    "Hello 🌍 World 🚀",
    "\U0001f600\U0001f601\U0001f602",
    "😀😁😂🤣😃😄😅😆😉😊😋😎😍😘🥰",
    # Compound emoji (ZWJ sequences)
    "👨‍👩‍👧‍👦",
    "👩‍💻",
    # Flag emoji
    "🇺🇸",
    "🇯🇵🇫🇷🇩🇪",
    # Combining characters and diacritics
    "e\u0301",  # e + combining acute accent (vs. é)
    "n\u0303",  # n + combining tilde (vs. ñ)
    "\u0065\u0301 vs \u00e9",  # comparing composed vs decomposed
    # Zero-width characters
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # byte-order mark
    "a\u200bb\u200bc",  # ZWSP between chars
    # Null bytes
    "\x00",
    "null\x00byte",
    # Full-width vs half-width
    "\uff21\uff22\uff23",  # full-width ABC
    "\uff71\uff72\uff73",  # half-width katakana
    # Math and music symbols
    "∫ f(x) dx = F(x) + C",
    "α β γ δ ε ζ η θ",
    "∑_{i=0}^{n} x_i",
    "f(x) = x^2 + 2x + 1",
    "♪♫♬",
    "𝕳𝖊𝖑𝖑𝖔",  # mathematical fraktur
    # Private use area
    "\ue000\ue001\ue002",
    # Special token strings embedded in regular text
    "<|begin_of_text|>",
    "Some text with <|end_of_text|> in the middle",
    # Programming constructs
    "def foo(x: int) -> int:\n    return x + 1",
    "import torch\nmodel = torch.nn.Linear(10, 20)",
    "SELECT * FROM users WHERE id = 1;",
    '{"key": "value", "list": [1, 2, 3]}',
    "<html><body><p>Hello</p></body></html>",
    "function hello() { return 'world'; }",
    "console.log(`template ${literal}`);",
    # Repeated patterns
    "aaaaaaaaaaaaaaaaaaaaa",
    "abcabcabcabcabcabcabc",
    "aaabbbccc",
    "the the the the the",
    "A" * 500,
    "foo bar " * 100,
    # Long text
    "The quick brown fox jumps over the lazy dog. " * 20,
    "a b c d e f g h i j k l m n o p q r s t u v w x y z " * 10,
    # Punctuation and formatting
    "Hello... World!!! How are you???",
    "Mr. Smith went to Washington, D.C., on Jan. 1st.",
    "left\u2019s right\u201d quotes",
    "(((nested))) [[[brackets]]] {{{braces}}}",
    "---",
    "===",
    "***",
    "~~~",
    # URLs and paths
    "https://www.example.com/path?q=hello&lang=en#section",
    "/usr/local/bin/python3",
    "C:\\Users\\foo\\Desktop\\file.txt",
    "user@example.com",
    "path/to/file.txt",
    # Numbers and formatting
    "123456789",
    "1,000,000",
    "1.000.000",
    "$1,234.56",
    "50%",
    "0x1A2B3C4D",
    "2026-04-03T12:00:00Z",
    # Case variants
    "camelCaseVariable",
    "snake_case_variable",
    "SCREAMING_SNAKE_CASE",
    "kebab-case-identifier",
    "MiXeD CaSe TeXt",
    "under_score and-dash",
    # Escapes and special chars
    "\\n\\t\\r\\0",
    "\xff",
    "&amp; &lt; &gt; &quot;",
    # Base64 and encoded content
    "SGVsbG8gV29ybGQ=",
    # Single short words
    "I",
    "an",
    "the",
    "http",
    "https",
    "www",
    ".com",
    # Long words
    "internationalization",
    "supercalifragilisticexpialidocious",
    # Markdown / structured text
    "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
    "* bullet 1\n* bullet 2\n* bullet 3",
    "# Heading\n\n**bold** and *italic* and `code`",
    # LaTeX
    "\\begin{equation} E = mc^2 \\end{equation}",
    # SQL
    "SELECT COUNT(*) AS cnt FROM table GROUP BY col HAVING cnt > 5 ORDER BY cnt DESC LIMIT 10;",
    # Lorem ipsum
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt.",
    # Boolean-like strings
    "True False None null undefined NaN Infinity",
    # Chat-like content
    "<|user|>\nWhat is the capital of France?\n<|assistant|>\nParis is the capital of France.",
]

assert len(DIVERSE_TEXTS) >= 100, f"Need >= 100 test strings, have {len(DIVERSE_TEXTS)}"

# Subset for tests that don't need the full list.
BASIC_TEXTS = [
    "Hello, world! This is a test.",
    "The quick brown fox jumps over the lazy dog.",
    "def foo(): pass",
    "123 + 456 = 579",
    "café résumé",
    "你好世界",
    "😀🚀",
]

# Unicode edge case texts for focused tests.
UNICODE_TEXTS = [
    ("chinese", "你好世界"),
    ("japanese", "こんにちは"),
    ("korean", "안녕하세요"),
    ("arabic", "مرحبا"),
    ("hindi", "नमस्ते"),
    ("thai", "สวัสดี"),
    ("hebrew", "שלום"),
    ("russian", "Привет"),
    ("emoji_simple", "😀"),
    ("emoji_zwj", "👨‍👩‍👧‍👦"),
    ("emoji_flag", "🇺🇸"),
    ("combining_accent", "e\u0301"),
    ("bom", "\ufeff"),
    ("zwsp", "\u200b"),
    ("null_byte", "\x00"),
    ("fullwidth", "\uff21\uff22\uff23"),
    ("math_fraktur", "𝕳𝖊𝖑𝖑𝖔"),
    ("private_use", "\ue000"),
    ("mixed_script", "English 日本語 العربية"),
]

WHITESPACE_TEXTS = [
    ("tab", "\t"),
    ("newline", "\n"),
    ("crlf", "\r\n"),
    ("cr", "\r"),
    ("multi_space", "     "),
    ("mixed_line_endings", "a\nb\r\nc\rd"),
    ("leading_trailing", "  hello  "),
    ("nbsp", "\u00a0"),
    ("thin_space", "\u2009"),
    ("em_space", "\u2003"),
    ("tabs_and_spaces", "\t  \t  \t"),
]

CODE_SAMPLES = [
    ("python", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n"),
    ("javascript", "const x = [1, 2, 3].map(n => n * 2).filter(n => n > 2);"),
    (
        "html",
        '<!DOCTYPE html>\n<html lang="en">\n<head><title>Test</title></head>\n<body><p>Hello</p></body>\n</html>',
    ),
    ("json", '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}'),
    ("xml", '<?xml version="1.0"?>\n<root><item id="1">value</item></root>'),
    ("sql", "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name;"),
    ("url", "https://api.example.com/v2/users?page=1&limit=50&sort=created_at&order=desc"),
    (
        "stack_trace",
        'Traceback (most recent call last):\n  File "main.py", line 42, in <module>\n    result = process(data)\nValueError: invalid input',
    ),
    ("markdown", "# Title\n\n## Section 1\n\n- item 1\n- item 2\n\n```python\nprint('hello')\n```\n"),
    ("log_line", "[2026-04-03 12:00:00.123] ERROR main.py:42 - Connection refused: host=db.example.com port=5432"),
]

REAL_WORLD_TEXTS = [
    # Wikipedia-style excerpt
    (
        "wikipedia",
        "The Turing machine is a mathematical model of computation describing an abstract machine "
        "that manipulates symbols on a strip of tape according to a table of rules. Despite the "
        "model's simplicity, it is capable of implementing any computer algorithm.",
    ),
    # Mixed-language paragraph
    (
        "multilingual",
        "In Tokyo (東京), the cherry blossoms (桜) bloom in spring. Les Parisiens enjoy café "
        "culture, while in München they drink Weißbier. مرحباً من القاهرة!",
    ),
    # Email-style text
    (
        "email",
        "From: alice@example.com\nTo: bob@example.com\nSubject: Re: Meeting tomorrow\n\n"
        "Hi Bob,\n\nCan we reschedule to 3pm? I have a conflict at 2pm.\n\nThanks,\nAlice",
    ),
]


# ---------------------------------------------------------------------------
# 1. Protocol conformance (parameterized across backends)
# ---------------------------------------------------------------------------


@requires_model
def test_protocol_conformance(backend_tokenizer):
    assert isinstance(backend_tokenizer, MarinTokenizer)


# ---------------------------------------------------------------------------
# 2. Basic properties (parameterized across backends)
# ---------------------------------------------------------------------------


@requires_model
def test_name_or_path(backend_tokenizer):
    assert backend_tokenizer.name_or_path == MODEL_NAME


@requires_model
def test_len_equals_vocab_size(backend_tokenizer):
    assert len(backend_tokenizer) == backend_tokenizer.vocab_size


@requires_model
def test_vocab_size_is_128256(backend_tokenizer):
    assert backend_tokenizer.vocab_size == 128256


@requires_model
def test_bos_token(backend_tokenizer):
    assert backend_tokenizer.bos_token == "<|begin_of_text|>"


@requires_model
def test_eos_token(backend_tokenizer):
    assert backend_tokenizer.eos_token == "<|end_of_text|>"


@requires_model
def test_bos_token_id(backend_tokenizer):
    assert backend_tokenizer.bos_token_id == 128000


@requires_model
def test_eos_token_id(backend_tokenizer):
    assert backend_tokenizer.eos_token_id == 128001


@requires_model
def test_pad_token_id_is_none(backend_tokenizer):
    assert backend_tokenizer.pad_token_id is None


@requires_model
def test_all_special_ids_is_list(backend_tokenizer):
    ids = backend_tokenizer.all_special_ids
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


@requires_model
def test_bos_in_special_ids(backend_tokenizer):
    assert backend_tokenizer.bos_token_id in backend_tokenizer.all_special_ids


@requires_model
def test_eos_in_special_ids(backend_tokenizer):
    assert backend_tokenizer.eos_token_id in backend_tokenizer.all_special_ids


# ---------------------------------------------------------------------------
# 3. Basic encoding/decoding (parameterized across backends)
# ---------------------------------------------------------------------------


@requires_model
@pytest.mark.parametrize("text", BASIC_TEXTS, ids=[t[:30] for t in BASIC_TEXTS])
def test_encode_returns_list_of_ints(backend_tokenizer, text):
    ids = backend_tokenizer.encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


@requires_model
@pytest.mark.parametrize("text", BASIC_TEXTS, ids=[t[:30] for t in BASIC_TEXTS])
def test_encode_decode_roundtrip(backend_tokenizer, text):
    ids = backend_tokenizer.encode(text)
    decoded = backend_tokenizer.decode(ids)
    assert decoded == text


@requires_model
def test_encode_empty_string(backend_tokenizer):
    ids = backend_tokenizer.encode("")
    assert ids == []


@requires_model
def test_encode_single_char(backend_tokenizer):
    ids = backend_tokenizer.encode("a")
    assert len(ids) >= 1


@requires_model
def test_encode_single_space(backend_tokenizer):
    ids = backend_tokenizer.encode(" ")
    assert len(ids) >= 1


@requires_model
def test_encode_very_long_string(backend_tokenizer):
    text = "Hello world. " * 1000  # ~13k chars
    ids = backend_tokenizer.encode(text)
    assert len(ids) > 100
    decoded = backend_tokenizer.decode(ids)
    assert decoded == text


@requires_model
def test_encode_repeated_chars(backend_tokenizer):
    text = "a" * 1000
    ids = backend_tokenizer.encode(text)
    assert len(ids) >= 1
    decoded = backend_tokenizer.decode(ids)
    assert decoded == text


@requires_model
def test_encode_numbers(backend_tokenizer):
    text = "1234567890"
    ids = backend_tokenizer.encode(text)
    assert len(ids) >= 1
    assert backend_tokenizer.decode(ids) == text


@requires_model
def test_encode_punctuation(backend_tokenizer):
    text = "!@#$%^&*()"
    ids = backend_tokenizer.encode(text)
    assert len(ids) >= 1
    assert backend_tokenizer.decode(ids) == text


# ---------------------------------------------------------------------------
# 4. Unicode edge cases (parameterized across backends)
# ---------------------------------------------------------------------------


@requires_model
@pytest.mark.parametrize("name,text", UNICODE_TEXTS, ids=[u[0] for u in UNICODE_TEXTS])
def test_unicode_encode_nonempty(backend_tokenizer, name, text):
    ids = backend_tokenizer.encode(text)
    assert len(ids) >= 1, f"Unicode text '{name}' should produce at least one token"


@requires_model
@pytest.mark.parametrize(
    "name,text",
    [u for u in UNICODE_TEXTS if u[0] not in ("null_byte", "bom", "zwsp", "private_use")],
    ids=[u[0] for u in UNICODE_TEXTS if u[0] not in ("null_byte", "bom", "zwsp", "private_use")],
)
def test_unicode_roundtrip(backend_tokenizer, name, text):
    """Encode-decode roundtrip should preserve unicode text.

    Some characters (null bytes, BOM, ZWSP, private use) may not roundtrip
    perfectly depending on the tokenizer's normalizer, so we exclude them here.
    """
    ids = backend_tokenizer.encode(text)
    decoded = backend_tokenizer.decode(ids)
    assert decoded == text, f"Roundtrip failed for '{name}': {repr(text)} -> {repr(decoded)}"


@requires_model
def test_mixed_scripts_encode(backend_tokenizer):
    text = "English 日本語 العربية Кириллица"
    ids = backend_tokenizer.encode(text)
    assert len(ids) >= 4  # At least one token per script segment


@requires_model
def test_combining_characters_encode(backend_tokenizer):
    """Both composed (é) and decomposed (e + combining accent) should encode."""
    composed = "é"
    decomposed = "e\u0301"
    ids_c = backend_tokenizer.encode(composed)
    ids_d = backend_tokenizer.encode(decomposed)
    assert len(ids_c) >= 1
    assert len(ids_d) >= 1


@requires_model
def test_zwj_emoji_encode(backend_tokenizer):
    """ZWJ emoji sequences should encode to some tokens."""
    text = "👨‍👩‍👧‍👦"
    ids = backend_tokenizer.encode(text)
    assert len(ids) >= 1


@requires_model
def test_flag_emoji_encode(backend_tokenizer):
    text = "🇺🇸🇯🇵"
    ids = backend_tokenizer.encode(text)
    assert len(ids) >= 1


@requires_model
def test_surrogate_adjacent_codepoints(backend_tokenizer):
    """Codepoints near the surrogate range should encode without error."""
    text = "\ud7ff\ue000"  # Just outside surrogate range
    ids = backend_tokenizer.encode(text)
    assert isinstance(ids, list)


@requires_model
def test_fullwidth_vs_halfwidth(backend_tokenizer):
    """Full-width and half-width forms should both encode."""
    fw = "\uff21\uff22\uff23"  # ABC full-width
    hw = "ABC"
    ids_fw = backend_tokenizer.encode(fw)
    ids_hw = backend_tokenizer.encode(hw)
    assert len(ids_fw) >= 1
    assert len(ids_hw) >= 1


@requires_model
def test_musical_symbols(backend_tokenizer):
    text = "♪♫♬𝄞"
    ids = backend_tokenizer.encode(text)
    assert len(ids) >= 1


@requires_model
def test_mathematical_bold(backend_tokenizer):
    text = "𝐀𝐁𝐂"  # Mathematical bold
    ids = backend_tokenizer.encode(text)
    assert len(ids) >= 1


# ---------------------------------------------------------------------------
# 5. Whitespace handling (parameterized across backends)
# ---------------------------------------------------------------------------


@requires_model
@pytest.mark.parametrize("name,text", WHITESPACE_TEXTS, ids=[w[0] for w in WHITESPACE_TEXTS])
def test_whitespace_encodes(backend_tokenizer, name, text):
    ids = backend_tokenizer.encode(text)
    assert len(ids) >= 1, f"Whitespace text '{name}' should produce at least one token"


@requires_model
def test_multiple_consecutive_spaces(backend_tokenizer):
    text = "a     b"
    ids = backend_tokenizer.encode(text)
    decoded = backend_tokenizer.decode(ids)
    assert decoded == text


@requires_model
def test_mixed_line_endings(backend_tokenizer):
    text = "line1\nline2\r\nline3\rline4"
    ids = backend_tokenizer.encode(text)
    decoded = backend_tokenizer.decode(ids)
    assert decoded == text


# ---------------------------------------------------------------------------
# 6. Special tokens
# ---------------------------------------------------------------------------


@requires_model
def test_add_special_tokens_prepends_bos(backend_tokenizer):
    text = "hello"
    ids_plain = backend_tokenizer.encode(text, add_special_tokens=False)
    ids_special = backend_tokenizer.encode(text, add_special_tokens=True)
    assert len(ids_special) > len(ids_plain)
    assert ids_special[0] == backend_tokenizer.bos_token_id


@requires_model
def test_add_special_tokens_false_no_bos(backend_tokenizer):
    text = "hello"
    ids = backend_tokenizer.encode(text, add_special_tokens=False)
    if backend_tokenizer.bos_token_id is not None:
        assert ids[0] != backend_tokenizer.bos_token_id


def _longest_homogeneous_run(s: str) -> int:
    """Return the length of the longest run of consecutive whitespace OR
    consecutive non-whitespace characters in ``s``."""
    if not s:
        return 0
    longest = 1
    current = 1
    is_space = s[0].isspace()
    for ch in s[1:]:
        ch_is_space = ch.isspace()
        if ch_is_space == is_space:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 1
            is_space = ch_is_space
    return longest


@requires_model
def test_multi_chunk_path_preserves_bos(backend_tokenizer, monkeypatch):
    """When the safe-split path activates, BOS handling must still match the
    single-chunk path (BOS prepended exactly once when add_special_tokens=True,
    absent otherwise) and the decoded text must round-trip.

    Forces the multi-chunk path by feeding text with a long run of whitespace
    and capping the homogeneous-run limit so the splitter cuts it up.
    """
    import levanter.tokenizers as tk

    if backend_tokenizer.bos_token_id is None:
        pytest.skip("Backend has no BOS token to verify against")

    # Pathological-ish: real text bracketing a 1k-space run. With the default
    # 25k cap this stays one part; with the patched 100-char cap below it
    # gets split into ~10 parts.
    text = "The quick brown fox jumps. " + (" " * 1_000) + "And then the lazy dog naps."

    # Sanity: this text should NOT trigger the split path with default limits.
    assert len(tk._safe_split_for_tokenizer(text)) == 1

    # Force the multi-chunk path by capping homogeneous runs at 100 chars.
    monkeypatch.setattr(tk, "_MAX_HOMOGENEOUS_RUN_CHARS", 100)
    monkeypatch.setattr(tk, "_SAFE_CHUNK_RE", re.compile(r"\s{1,100}|\S{1,100}"))
    parts = tk._safe_split_for_tokenizer(text)
    assert len(parts) > 1, "Expected multi-chunk path to activate"
    assert "".join(parts) == text
    # Each part respects the run cap (the cap is on max consecutive run length
    # within a part, not on part length itself).
    for p in parts:
        assert _longest_homogeneous_run(p) <= 100

    multi_plain = backend_tokenizer.encode(text, add_special_tokens=False)
    multi_special = backend_tokenizer.encode(text, add_special_tokens=True)

    # BOS handling: present iff add_special_tokens=True, exactly once at the front.
    assert multi_plain[0] != backend_tokenizer.bos_token_id
    assert multi_special[0] == backend_tokenizer.bos_token_id
    assert multi_special.count(backend_tokenizer.bos_token_id) == 1
    assert backend_tokenizer.bos_token_id not in multi_plain

    # The multi-chunk add_special_tokens=True path must equal the
    # add_special_tokens=False path with a single BOS prepended.
    assert multi_special == [backend_tokenizer.bos_token_id] + multi_plain

    # Decoded text must round-trip losslessly even with broken BPE merges.
    assert backend_tokenizer.decode(multi_plain) == text
    assert backend_tokenizer.decode(multi_special, skip_special_tokens=True) == text


@requires_model
def test_normal_text_unchanged_by_splitter(backend_tokenizer):
    """For text where no homogeneous run exceeds the cap, the splitter must
    not change tokenization at all. ``encode(text)`` should return exactly
    what a single direct call to the underlying Rust tokenizer would.

    A regex like ``\\s{1,N}|\\S{1,N}`` splits at every whitespace transition,
    which severs leading-space-prefix BPE merges (e.g. " world" → 1 token vs
    " " + "world" → 2 different tokens) and roughly doubles the token count
    on normal English text. This test guards against that regression.
    """
    text = "The quick brown fox jumps over the lazy dog."

    via_split = backend_tokenizer.encode(text, add_special_tokens=False)

    # Bypass our splitter and call the underlying Rust tokenizer directly.
    inner = backend_tokenizer._tokenizer
    if hasattr(inner, "encode_batch"):
        # HF tokenizers backend
        direct = inner.encode(text, add_special_tokens=False).ids
    else:
        # kitoken backend: encode(text, encode_specials=True)
        direct = inner.encode(text, True)

    assert via_split == direct, (
        f"splitter changed tokenization on normal text: "
        f"{len(via_split)} tokens via splitter vs {len(direct)} direct"
    )


@requires_model
def test_encode_batch_scatters_parts_back_to_originals(backend_tokenizer, monkeypatch):
    """encode_batch must reassemble per-text token sequences correctly when
    some texts are split into multiple parts and others aren't."""
    import levanter.tokenizers as tk

    monkeypatch.setattr(tk, "_MAX_HOMOGENEOUS_RUN_CHARS", 100)
    monkeypatch.setattr(tk, "_SAFE_CHUNK_RE", re.compile(r"\s{1,100}|\S{1,100}"))

    short = "The quick brown fox."
    pathological = "start" + (" " * 500) + "end"
    texts = [short, pathological, short, pathological, short]

    # Sanity: with the patched cap, the pathological text gets split.
    assert len(tk._safe_split_for_tokenizer(pathological)) > 1
    assert len(tk._safe_split_for_tokenizer(short)) == 1

    batch_ids = backend_tokenizer.encode_batch(texts, add_special_tokens=False)
    assert len(batch_ids) == len(texts)

    # Each row must equal the per-text encode result (ground truth).
    for got, original in zip(batch_ids, texts, strict=True):
        expected = backend_tokenizer.encode(original, add_special_tokens=False)
        assert got == expected
        # And every row must round-trip losslessly.
        assert backend_tokenizer.decode(got) == original

    # add_special_tokens=True must prepend BOS to every row exactly once.
    if backend_tokenizer.bos_token_id is not None:
        batch_special = backend_tokenizer.encode_batch(texts, add_special_tokens=True)
        for row in batch_special:
            assert row[0] == backend_tokenizer.bos_token_id
            assert row.count(backend_tokenizer.bos_token_id) == 1


def test_safe_split_caps_runs_and_roundtrips(monkeypatch):
    """The splitter must cap homogeneous runs within each part and round-trip
    losslessly on a pathological all-whitespace input."""
    import levanter.tokenizers as tk

    # 1 MB of spaces with two real words at the ends — the realistic shape of
    # the FDLP/lps47065 OOM document.
    text = "hello" + (" " * 1_000_000) + "world"

    parts = tk._safe_split_for_tokenizer(text)
    assert "".join(parts) == text
    assert len(parts) > 1
    for p in parts:
        run = _longest_homogeneous_run(p)
        assert run <= tk._MAX_HOMOGENEOUS_RUN_CHARS, f"run {run} exceeds cap {tk._MAX_HOMOGENEOUS_RUN_CHARS}"

    # Also confirm the outer 400k chunking is respected when the input has no
    # long homogeneous runs (so only the outer cap fires).
    no_runs = "abcde" * 200_000  # 1M chars, longest run = 1
    parts2 = tk._safe_split_for_tokenizer(no_runs)
    assert "".join(parts2) == no_runs
    assert len(parts2) > 1
    for p in parts2:
        assert len(p) <= tk._MAX_ENCODE_CHARS


@requires_model
def test_decode_skip_special_tokens(backend_tokenizer):
    text = "hello"
    ids_with_bos = [backend_tokenizer.bos_token_id] + backend_tokenizer.encode(text)
    decoded_skip = backend_tokenizer.decode(ids_with_bos, skip_special_tokens=True)
    assert "hello" in decoded_skip


@requires_model
def test_encode_text_containing_special_token_string(backend_tokenizer):
    """Encoding text that literally contains special token strings should work."""
    text = "The token <|begin_of_text|> appears in this sentence."
    ids = backend_tokenizer.encode(text, add_special_tokens=False)
    assert isinstance(ids, list)
    assert len(ids) > 0


# ---------------------------------------------------------------------------
# 7. Batch encoding (parameterized across backends)
# ---------------------------------------------------------------------------


@requires_model
def test_encode_batch_matches_individual(backend_tokenizer):
    texts = ["Hello world", "foo bar baz", "The quick brown fox"]
    batch_result = backend_tokenizer.encode_batch(texts)
    individual = [backend_tokenizer.encode(t) for t in texts]
    assert batch_result == individual


@requires_model
def test_encode_batch_empty_list(backend_tokenizer):
    assert backend_tokenizer.encode_batch([]) == []


@requires_model
def test_encode_batch_single_element(backend_tokenizer):
    texts = ["hello"]
    batch = backend_tokenizer.encode_batch(texts)
    assert len(batch) == 1
    assert batch[0] == backend_tokenizer.encode("hello")


@requires_model
def test_encode_batch_with_empty_strings(backend_tokenizer):
    texts = ["hello", "", "world"]
    batch = backend_tokenizer.encode_batch(texts)
    assert len(batch) == 3
    assert batch[1] == []


@requires_model
def test_encode_batch_mixed_lengths(backend_tokenizer):
    texts = ["a", "hello world this is a longer sentence", "b"]
    batch = backend_tokenizer.encode_batch(texts)
    assert len(batch) == 3
    assert len(batch[0]) < len(batch[1])


@requires_model
def test_encode_batch_large(backend_tokenizer):
    """Batch encoding 1000 items should work and match individual encoding."""
    texts = [f"sentence number {i} with some extra words" for i in range(1000)]
    batch = backend_tokenizer.encode_batch(texts)
    assert len(batch) == 1000
    # Spot-check a few
    for i in [0, 499, 999]:
        assert batch[i] == backend_tokenizer.encode(texts[i])


@requires_model
def test_encode_batch_add_special_tokens(backend_tokenizer):
    texts = ["hello", "world"]
    batch_plain = backend_tokenizer.encode_batch(texts, add_special_tokens=False)
    batch_special = backend_tokenizer.encode_batch(texts, add_special_tokens=True)
    for plain, special in zip(batch_plain, batch_special):
        assert len(special) > len(plain)
        assert special[0] == backend_tokenizer.bos_token_id


# ---------------------------------------------------------------------------
# 8. Vocab operations (parameterized across backends)
# ---------------------------------------------------------------------------


@requires_model
def test_get_vocab_returns_dict(backend_tokenizer):
    vocab = backend_tokenizer.get_vocab()
    assert isinstance(vocab, dict)
    assert len(vocab) > 0


@requires_model
def test_get_vocab_values_are_ints(backend_tokenizer):
    vocab = backend_tokenizer.get_vocab()
    sample = list(vocab.items())[:100]
    for token, idx in sample:
        assert isinstance(token, str)
        assert isinstance(idx, int)


@requires_model
def test_convert_ids_to_tokens_single(backend_tokenizer):
    """Converting a single token ID returns a string."""
    token = backend_tokenizer.convert_ids_to_tokens(0)
    assert isinstance(token, str)


@requires_model
def test_convert_ids_to_tokens_list(backend_tokenizer):
    tokens = backend_tokenizer.convert_ids_to_tokens([0, 1, 2])
    assert isinstance(tokens, list)
    assert len(tokens) == 3
    assert all(isinstance(t, str) for t in tokens)


@requires_model
def test_convert_tokens_to_ids_single(backend_tokenizer):
    vocab = backend_tokenizer.get_vocab()
    some_token = next(iter(vocab.keys()))
    idx = backend_tokenizer.convert_tokens_to_ids(some_token)
    assert isinstance(idx, int)
    assert idx == vocab[some_token]


@requires_model
def test_convert_tokens_to_ids_list(backend_tokenizer):
    vocab = backend_tokenizer.get_vocab()
    tokens = list(vocab.keys())[:5]
    ids = backend_tokenizer.convert_tokens_to_ids(tokens)
    assert isinstance(ids, list)
    assert len(ids) == 5
    for token, idx in zip(tokens, ids):
        assert idx == vocab[token]


@requires_model
def test_convert_tokens_to_ids_unknown_returns_minus_one(backend_tokenizer):
    """Unknown tokens should return -1."""
    idx = backend_tokenizer.convert_tokens_to_ids("__DEFINITELY_NOT_A_TOKEN__")
    assert idx == -1


@requires_model
def test_convert_ids_to_tokens_unknown_returns_unk_format(backend_tokenizer):
    """Unknown token IDs should return a formatted unk string."""
    token = backend_tokenizer.convert_ids_to_tokens(999999999)
    assert "unk" in token.lower() or "999999999" in token


@requires_model
def test_token_id_roundtrip(backend_tokenizer):
    """convert_tokens_to_ids(convert_ids_to_tokens(id)) should return original id for valid ids.

    We pick IDs that are in the regular vocab (not byte-fallback tokens).
    IDs 256+ are safe non-byte-fallback token IDs for Llama 3.
    """
    for token_id in [256, 500, 1000, 5000]:
        token = backend_tokenizer.convert_ids_to_tokens(token_id)
        assert "unk" not in token.lower(), f"Token {token_id} should be in vocab, got {token}"
        recovered = backend_tokenizer.convert_tokens_to_ids(token)
        assert recovered == token_id, f"Roundtrip failed for id {token_id}: {token} -> {recovered}"


@requires_model
def test_vocab_dict_subset_of_vocab_size(backend_tokenizer):
    """get_vocab() length should be <= vocab_size."""
    vocab = backend_tokenizer.get_vocab()
    assert len(vocab) <= backend_tokenizer.vocab_size


# ---------------------------------------------------------------------------
# 9. Chat template
# ---------------------------------------------------------------------------

CHAT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


@pytest.fixture(scope="module")
def chat_tokenizer() -> MarinTokenizer:
    try:
        load_tokenizer.cache_clear()
        return load_tokenizer(CHAT_MODEL)
    except Exception:
        pytest.skip("Cannot load chat model")


def test_chat_template_renders_string(chat_tokenizer):
    conversation = [{"role": "user", "content": "What is 2+2?"}]
    result = chat_tokenizer.apply_chat_template(conversation, tokenize=False)
    assert isinstance(result, str)
    assert "What is 2+2?" in result


def test_chat_template_tokenizes(chat_tokenizer):
    conversation = [{"role": "user", "content": "What is 2+2?"}]
    result = chat_tokenizer.apply_chat_template(conversation, tokenize=True)
    assert isinstance(result, list)
    assert all(isinstance(i, int) for i in result)
    assert len(result) > 0


def test_chat_template_with_system_message(chat_tokenizer):
    conversation = [
        {"role": "user", "content": "Hi there"},
    ]
    result = chat_tokenizer.apply_chat_template(conversation, tokenize=False)
    assert isinstance(result, str)


def test_chat_template_multi_turn(chat_tokenizer):
    conversation = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ]
    result = chat_tokenizer.apply_chat_template(conversation, tokenize=False)
    assert "What is 2+2?" in result
    assert "4" in result
    assert "And 3+3?" in result


def test_chat_template_add_generation_prompt(chat_tokenizer):
    conversation = [{"role": "user", "content": "Hello"}]
    without = chat_tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    with_prompt = chat_tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    # With generation prompt should be at least as long
    assert len(with_prompt) >= len(without)


@requires_model
def test_chat_template_no_template_raises(backend_tokenizer):
    """Llama 3.1 base model has no chat template; should raise ValueError."""
    if backend_tokenizer.chat_template is None:
        with pytest.raises(ValueError, match="no chat template"):
            backend_tokenizer.apply_chat_template([{"role": "user", "content": "hi"}])


_SIMPLE_CHAT_TEMPLATE = """\
{%- for message in messages -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{%- if message['role'] == 'assistant' -%}
{% generation %}{{ message['content'] }}{% endgeneration %}
{%- else -%}
{{ message['content'] }}
{%- endif -%}
<|eot_id|>
{%- endfor -%}
"""


@requires_model
def test_apply_chat_template_with_masks_structure(backend_tokenizer):
    """apply_chat_template_with_masks should return dict with input_ids and assistant_masks."""
    conversations = [
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    ]
    result = backend_tokenizer.apply_chat_template_with_masks(conversations, chat_template=_SIMPLE_CHAT_TEMPLATE)
    assert "input_ids" in result
    assert "assistant_masks" in result
    assert len(result["input_ids"]) == 1
    assert len(result["assistant_masks"]) == 1
    assert len(result["input_ids"][0]) == len(result["assistant_masks"][0])


@requires_model
def test_apply_chat_template_with_masks_has_assistant_content(backend_tokenizer):
    conversations = [
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    ]
    result = backend_tokenizer.apply_chat_template_with_masks(conversations, chat_template=_SIMPLE_CHAT_TEMPLATE)
    mask = result["assistant_masks"][0]
    assert any(m == 1 for m in mask), "Mask should have 1s for assistant content"
    assert any(m == 0 for m in mask), "Mask should have 0s for non-assistant content"


@requires_model
def test_apply_chat_template_with_masks_batch(backend_tokenizer):
    conversations = [
        [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ],
        [
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ],
    ]
    result = backend_tokenizer.apply_chat_template_with_masks(conversations, chat_template=_SIMPLE_CHAT_TEMPLATE)
    assert len(result["input_ids"]) == 2
    assert len(result["assistant_masks"]) == 2


# ---------------------------------------------------------------------------
# 10. Code samples and real-world text
# ---------------------------------------------------------------------------


@requires_model
@pytest.mark.parametrize("name,code", CODE_SAMPLES, ids=[c[0] for c in CODE_SAMPLES])
def test_code_sample_roundtrip(backend_tokenizer, name, code):
    ids = backend_tokenizer.encode(code)
    decoded = backend_tokenizer.decode(ids)
    assert decoded == code, f"Code sample '{name}' roundtrip failed"


@requires_model
@pytest.mark.parametrize("name,text", REAL_WORLD_TEXTS, ids=[r[0] for r in REAL_WORLD_TEXTS])
def test_real_world_roundtrip(backend_tokenizer, name, text):
    ids = backend_tokenizer.encode(text)
    decoded = backend_tokenizer.decode(ids)
    assert decoded == text, f"Real-world text '{name}' roundtrip failed"


# ---------------------------------------------------------------------------
# 11. Roundtrip properties
# ---------------------------------------------------------------------------


@requires_model
@pytest.mark.parametrize("text", BASIC_TEXTS + ["αβγδ", "こんにちは", "Hello 🌍"], ids=lambda t: t[:30])
def test_idempotent_encoding(backend_tokenizer, text):
    """encode(decode(encode(x))) == encode(x): encoding is idempotent after one roundtrip."""
    ids1 = backend_tokenizer.encode(text)
    decoded = backend_tokenizer.decode(ids1)
    ids2 = backend_tokenizer.encode(decoded)
    assert ids1 == ids2, f"Idempotent encoding failed for {repr(text)}"


# ---------------------------------------------------------------------------
# 12. Factory function and caching
# ---------------------------------------------------------------------------


@requires_model
def test_load_tokenizer_caching():
    load_tokenizer.cache_clear()
    tok1 = load_tokenizer(MODEL_NAME)
    tok2 = load_tokenizer(MODEL_NAME)
    assert tok1 is tok2


@requires_model
def test_local_tokenizer_encode_batch(tmp_path):
    """Ensure encode_batch works with local tokenizer paths (no hub round-trip)."""
    staged_dir = _stage_tokenizer(MODEL_NAME)
    local_dir = tmp_path / "tokenizer"
    local_dir.mkdir()
    for fname in os.listdir(staged_dir):
        src = os.path.join(staged_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, local_dir / fname)

    load_tokenizer.cache_clear()
    with patch(
        "levanter.tokenizers.hf_hub_download",
        side_effect=AssertionError("Local tokenizer paths should not hit HF Hub"),
    ):
        tok = load_tokenizer(os.fspath(local_dir))

        texts = [f"sentence number {i}" for i in range(20)]
        batch_result = tok.encode_batch(texts)
        individual = [tok.encode(t) for t in texts]
        assert batch_result == individual


# ---------------------------------------------------------------------------
# 13. Error handling
# ---------------------------------------------------------------------------


def test_exception_propagation_on_network_error():
    """_load_tokenizer_config should propagate network errors, not swallow them."""

    class FakeNetworkError(OSError):
        pass

    with patch("levanter.tokenizers.hf_hub_download", side_effect=FakeNetworkError("connection refused")):
        with pytest.raises(FakeNetworkError, match="connection refused"):
            _load_tokenizer_config("some-nonexistent-model/that-will-fail")


# ---------------------------------------------------------------------------
# 14. Regression tests
# ---------------------------------------------------------------------------

CHAT_MODEL_WITH_PAD = "mistralai/Mistral-7B-Instruct-v0.2"


def test_padding_uses_pad_token_id_not_zero():
    """Verify BatchTokenizer pads input_ids with pad_token_id, not hardcoded 0."""
    from levanter.data.text._batch_tokenizer import BatchTokenizer

    try:
        tok = load_tokenizer(CHAT_MODEL_WITH_PAD)
    except Exception:
        pytest.skip("Cannot load chat model")

    bt = BatchTokenizer(
        tok,
        enforce_bos=False,
        enforce_eos=False,
        return_attention_mask=True,
        padding="max_length",
        max_length=32,
    )

    result = bt([{"text": "hi"}])
    assert len(result) == 1
    ids = result[0]["input_ids"]
    mask = result[0]["attention_mask"]

    assert len(ids) == 32
    assert len(mask) == 32

    pad_start = mask.index(0) if 0 in mask else len(mask)
    assert pad_start < 32, "Sequence should be shorter than max_length to have padding"

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    for i in range(pad_start, 32):
        assert ids[i] == pad_id, f"Position {i}: expected pad_token_id={pad_id}, got {ids[i]}"
        assert mask[i] == 0, f"Position {i}: attention_mask should be 0 for padding"


def test_chat_processor_with_marin_tokenizer():
    """ChatProcessor must work with MarinTokenizer (not just HfTokenizer)."""
    if not _MODEL_AVAILABLE:
        pytest.skip("HF auth or network unavailable")

    from levanter.data.text.formats import ChatProcessor

    tok = load_tokenizer(MODEL_NAME)
    assert isinstance(tok, MarinTokenizer)

    processor = ChatProcessor(tok, chat_template=_SIMPLE_CHAT_TEMPLATE, mask_user_turns=True)

    conversation = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    results = processor([{"messages": conversation}])
    assert len(results) == 1
    assert "input_ids" in results[0]
    assert "assistant_masks" in results[0]
    assert len(results[0]["input_ids"]) > 0

    mask = results[0]["assistant_masks"]
    assert any(m == 1 for m in mask), "assistant_masks should mark assistant content"


@requires_model
@pytest.mark.parametrize("text", BASIC_TEXTS, ids=[t[:30] for t in BASIC_TEXTS])
def test_encode_batch_respects_prepend_bos(backend_tokenizer, text):
    """encode_batch with add_special_tokens must agree with encode.

    When _prepend_bos is False (e.g. models without a TemplateProcessing
    post-processor), encode_batch should not prepend BOS either. This
    regression test patches _prepend_bos=False to exercise the code path
    that was previously unchecked in encode_batch.
    """
    import dataclasses as _dc

    from levanter.tokenizers import KitokenMarinTokenizer

    if not isinstance(backend_tokenizer, KitokenMarinTokenizer):
        pytest.skip("Bug only affects KitokenMarinTokenizer")

    # Simulate a model whose post-processor does NOT prepend BOS.
    patched = _dc.replace(backend_tokenizer, _prepend_bos=False)

    single = patched.encode(text, add_special_tokens=True)
    batch = patched.encode_batch([text], add_special_tokens=True)
    assert batch[0] == single, (
        f"encode_batch diverged from encode with _prepend_bos=False: "
        f"encode returned {single[:5]}..., encode_batch returned {batch[0][:5]}..."
    )


_GEMMA_TOKENIZERS: dict[str, MarinTokenizer] = {}
_GEMMA_BACKENDS: list[str] = []
try:
    load_tokenizer.cache_clear()
    for _b in [TokenizerBackend.HF, TokenizerBackend.KITOKEN]:
        _GEMMA_TOKENIZERS[_b.value] = load_tokenizer("google/gemma-3-4b-it", backend=_b)
        _GEMMA_BACKENDS.append(_b.value)
except Exception:
    pass


@pytest.fixture(scope="module", params=_GEMMA_BACKENDS if _GEMMA_BACKENDS else ["_skip_all"])
def gemma_tokenizer(request):
    name = request.param
    if name == "_skip_all":
        pytest.skip("Cannot load gemma-3-4b-it tokenizer")
    return _GEMMA_TOKENIZERS[name]


# Regression test for Systemcluster/kitoken#3: SentencePiece BPE merge rank mishandling.
_GEMMA_EXPECTED_IDS = {
    "In [[political philosophy]], the concept of [[limited government]]": [
        902,
        15385,
        61754,
        19548,
        36878,
        506,
        3495,
        529,
        15385,
        21028,
        3251,
        10660,
    ],
    "The quick brown fox jumps over the lazy dog.": [
        818,
        3823,
        8864,
        37423,
        38167,
        1024,
        506,
        31770,
        4799,
        236761,
    ],
}


@pytest.mark.parametrize(
    "text,expected_ids",
    _GEMMA_EXPECTED_IDS.items(),
    ids=[t[:40] for t in _GEMMA_EXPECTED_IDS],
)
def test_sentencepiece_bpe_gemma3(gemma_tokenizer, text, expected_ids):
    actual = gemma_tokenizer.encode(text, add_special_tokens=False)
    assert actual == expected_ids


@requires_model
def test_encode_batch_correctness_many_strings(backend_tokenizer):
    texts = [
        "hello world",
        "a" * 500,
        "foo bar " * 50,
        "",
        "  leading spaces",
        "trailing spaces  ",
        "multi\nline\ntext",
        "special chars: !@#$%^&*()",
        "unicode: 日本語テスト",
        "emoji: 🎉🎊",
    ]
    non_empty = [t for t in texts if t]
    batch = backend_tokenizer.encode_batch(non_empty)
    individual = [backend_tokenizer.encode(t) for t in non_empty]
    assert batch == individual


# ---------------------------------------------------------------------------
# 15. Staging / mirror fallback tests
# ---------------------------------------------------------------------------

# Minimal tokenizer.json accepted by HfBaseTokenizer.from_file.
_MINIMAL_TOKENIZER_JSON = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": None,
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "BPE",
        "dropout": None,
        "unk_token": None,
        "continuing_subword_prefix": None,
        "end_of_word_suffix": None,
        "fuse_unk": False,
        "byte_fallback": False,
        "vocab": {},
        "merges": [],
    },
}


@pytest.fixture
def fake_tokenizer_dir(tmp_path):
    """Local directory with a minimal valid tokenizer.json + tokenizer_config.json."""
    d = tmp_path / "tokenizer_src"
    d.mkdir()
    (d / "tokenizer.json").write_text(json.dumps(_MINIMAL_TOKENIZER_JSON))
    (d / "tokenizer_config.json").write_text("{}")
    return d


@pytest.fixture(autouse=False)
def clear_stage_cache():
    _stage_tokenizer.cache_clear()
    yield
    _stage_tokenizer.cache_clear()


def test_try_load_tokenizer_from_dir_valid(fake_tokenizer_dir):
    assert _try_load_tokenizer_from_dir(str(fake_tokenizer_dir))


def test_try_load_tokenizer_from_dir_empty(tmp_path):
    assert not _try_load_tokenizer_from_dir(str(tmp_path))


def test_try_load_tokenizer_from_dir_corrupt(tmp_path):
    (tmp_path / "tokenizer.json").write_text("not json")
    assert not _try_load_tokenizer_from_dir(str(tmp_path))


def test_stage_from_hf_copies_files_and_populates_mirror(tmp_path, fake_tokenizer_dir):
    """_stage_from_hf copies snapshot files to local_dir and pushes each to the mirror."""
    local_dir = tmp_path / "staged"
    local_dir.mkdir()
    mirror_calls: list[str] = []

    with (
        patch("levanter.tokenizers.snapshot_download", return_value=str(fake_tokenizer_dir)),
        patch("levanter.tokenizers._populate_mirror_file", side_effect=lambda _p, url: mirror_calls.append(url)),
    ):
        _stage_from_hf("org/model", str(local_dir))

    assert (local_dir / "tokenizer.json").exists()
    assert (local_dir / "tokenizer_config.json").exists()
    assert len(mirror_calls) == 2
    assert all("org/model" in u for u in mirror_calls)
    assert all(f"hf-hub-{_hf_hub_version}" in u for u in mirror_calls)


def test_stage_from_mirror_copies_files(tmp_path, fake_tokenizer_dir):
    """_stage_from_mirror fetches files from the mirror filesystem into local_dir."""
    local_dir = tmp_path / "staged"
    local_dir.mkdir()
    mirror_dir = f"tokenizers/org/model/hf-hub-{_hf_hub_version}"
    fake_entries = [f"{mirror_dir}/tokenizer.json", f"{mirror_dir}/tokenizer_config.json"]

    class FakeMirrorFS:
        def exists(self, path):
            return path == mirror_dir

        def ls(self, path, detail=False):
            return fake_entries

    def fake_fetch(src_url, dest_path):
        fname = os.path.basename(dest_path)
        src = fake_tokenizer_dir / fname
        if src.exists():
            shutil.copy2(src, dest_path)
            return True
        return False

    with (
        patch("levanter.tokenizers.fsspec.filesystem", return_value=FakeMirrorFS()),
        patch("levanter.tokenizers._fetch_file_atomic", side_effect=fake_fetch),
    ):
        result = _stage_from_mirror("org/model", str(local_dir))

    assert result is True
    assert (local_dir / "tokenizer.json").exists()


def test_stage_from_mirror_absent(tmp_path):
    """_stage_from_mirror returns False when the mirror dir does not exist."""
    local_dir = tmp_path / "staged"
    local_dir.mkdir()

    class FakeMirrorFS:
        def exists(self, path):
            return False

        def ls(self, path, detail=False):
            return []

    with patch("levanter.tokenizers.fsspec.filesystem", return_value=FakeMirrorFS()):
        result = _stage_from_mirror("org/model", str(local_dir))

    assert result is False
    assert not list(local_dir.iterdir())


def test_stage_tokenizer_local_cache_hit(tmp_path, fake_tokenizer_dir, clear_stage_cache):
    """_stage_tokenizer returns immediately when the local cache is already valid."""
    local_dir = tmp_path / "levanter_tokenizers" / "org" / "model" / f"hf-hub-{_hf_hub_version}"
    local_dir.mkdir(parents=True)
    shutil.copy2(fake_tokenizer_dir / "tokenizer.json", local_dir / "tokenizer.json")

    with (
        patch("levanter.tokenizers.tempfile.gettempdir", return_value=str(tmp_path)),
        patch("levanter.tokenizers._stage_from_mirror") as mock_mirror,
        patch("levanter.tokenizers._stage_from_hf") as mock_hf,
    ):
        result = _stage_tokenizer("org/model")

    assert result == str(local_dir)
    mock_mirror.assert_not_called()
    mock_hf.assert_not_called()


def test_stage_tokenizer_mirror_hit(tmp_path, fake_tokenizer_dir, clear_stage_cache):
    """_stage_tokenizer uses mirror files and skips HF Hub when mirror is populated."""

    def fake_stage_from_mirror(name_or_path, local_dir):
        shutil.copy2(fake_tokenizer_dir / "tokenizer.json", os.path.join(local_dir, "tokenizer.json"))
        return True

    with (
        patch("levanter.tokenizers.tempfile.gettempdir", return_value=str(tmp_path)),
        patch("levanter.tokenizers._stage_from_mirror", side_effect=fake_stage_from_mirror),
        patch("levanter.tokenizers._stage_from_hf") as mock_hf,
    ):
        _stage_tokenizer("org/model")

    mock_hf.assert_not_called()


def test_stage_tokenizer_falls_through_to_hf(tmp_path, fake_tokenizer_dir, clear_stage_cache):
    """_stage_tokenizer calls HF Hub when both local cache and mirror are empty."""

    def fake_stage_from_hf(name_or_path, local_dir):
        shutil.copy2(fake_tokenizer_dir / "tokenizer.json", os.path.join(local_dir, "tokenizer.json"))

    with (
        patch("levanter.tokenizers.tempfile.gettempdir", return_value=str(tmp_path)),
        patch("levanter.tokenizers._stage_from_mirror", return_value=False),
        patch("levanter.tokenizers._stage_from_hf", side_effect=fake_stage_from_hf) as mock_hf,
    ):
        _stage_tokenizer("org/model")

    mock_hf.assert_called_once()


def test_stage_tokenizer_corrupt_mirror_falls_through_to_hf(tmp_path, fake_tokenizer_dir, clear_stage_cache):
    """_stage_tokenizer falls through to HF Hub when mirror returns a corrupt tokenizer."""

    def corrupt_mirror(name_or_path, local_dir):
        pathlib.Path(local_dir, "tokenizer.json").write_text("not json")
        return True

    def fake_stage_from_hf(name_or_path, local_dir):
        shutil.copy2(fake_tokenizer_dir / "tokenizer.json", os.path.join(local_dir, "tokenizer.json"))

    with (
        patch("levanter.tokenizers.tempfile.gettempdir", return_value=str(tmp_path)),
        patch("levanter.tokenizers._stage_from_mirror", side_effect=corrupt_mirror),
        patch("levanter.tokenizers._stage_from_hf", side_effect=fake_stage_from_hf) as mock_hf,
    ):
        _stage_tokenizer("org/model")

    mock_hf.assert_called_once()


def test_stage_from_mirror_tolerates_broken_fs(tmp_path):
    """_stage_from_mirror returns False (does not raise) when the mirror FS throws."""
    local_dir = tmp_path / "staged"
    local_dir.mkdir()

    class BrokenMirrorFS:
        def exists(self, path):
            raise OSError("mirror unreachable")

    with patch("levanter.tokenizers.fsspec.filesystem", return_value=BrokenMirrorFS()):
        result = _stage_from_mirror("org/model", str(local_dir))

    assert result is False
    assert not list(local_dir.iterdir())
