# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from levanter.tokenizers import (
    HfMarinTokenizer,
    MarinTokenizer,
    TokenizerBackend,
    _load_tokenizer_config,
    load_tokenizer,
)

try:
    import tokie as _tokie  # noqa: F401

    HAS_TOKIE = True
except ImportError:
    HAS_TOKIE = False

MODEL_NAME = "meta-llama/Llama-3.1-8B"


@pytest.fixture(scope="module")
def tokenizer() -> HfMarinTokenizer:
    return load_tokenizer(MODEL_NAME)


def test_protocol_conformance(tokenizer):
    assert isinstance(tokenizer, MarinTokenizer)


def test_name_or_path(tokenizer):
    assert tokenizer.name_or_path == MODEL_NAME


def test_vocab_size(tokenizer):
    assert tokenizer.vocab_size == 128256


def test_special_tokens(tokenizer):
    assert tokenizer.bos_token == "<|begin_of_text|>"
    assert tokenizer.eos_token == "<|end_of_text|>"
    assert tokenizer.bos_token_id == 128000
    assert tokenizer.eos_token_id == 128001
    # Llama 3.1 has no pad token
    assert tokenizer.pad_token_id is None


def test_encode_decode_roundtrip(tokenizer):
    text = "Hello, world! This is a test."
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text


def test_encode_add_special_tokens(tokenizer):
    text = "hello"
    ids_no_special = tokenizer.encode(text, add_special_tokens=False)
    ids_with_special = tokenizer.encode(text, add_special_tokens=True)
    # With special tokens should have BOS prepended
    assert len(ids_with_special) > len(ids_no_special)
    assert ids_with_special[0] == tokenizer.bos_token_id


def test_encode_batch_matches_individual(tokenizer):
    texts = ["Hello world", "foo bar baz", "The quick brown fox"]
    batch_result = tokenizer.encode_batch(texts)
    individual_results = [tokenizer.encode(t) for t in texts]
    assert batch_result == individual_results


def test_encode_batch_empty(tokenizer):
    assert tokenizer.encode_batch([]) == []


def test_get_vocab(tokenizer):
    vocab = tokenizer.get_vocab()
    assert isinstance(vocab, dict)
    assert len(vocab) == tokenizer.vocab_size
    assert vocab["<|begin_of_text|>"] == tokenizer.bos_token_id


def test_load_tokenizer_caching():
    tok1 = load_tokenizer(MODEL_NAME)
    tok2 = load_tokenizer(MODEL_NAME)
    assert tok1 is tok2


@pytest.mark.skipif(not HAS_TOKIE, reason="tokie not installed")
def test_load_tokenizer_tokie():
    load_tokenizer.cache_clear()
    tok = load_tokenizer(MODEL_NAME, backend=TokenizerBackend.TOKIE)
    assert tok.name_or_path == MODEL_NAME


def test_chat_template_no_template():
    # Llama 3.1 base model has no chat template
    tok = load_tokenizer(MODEL_NAME)
    if tok.chat_template is None:
        with pytest.raises(ValueError, match="no chat template"):
            tok.apply_chat_template([{"role": "user", "content": "hi"}])


CHAT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


@pytest.fixture(scope="module")
def chat_tokenizer() -> HfMarinTokenizer:
    return load_tokenizer(CHAT_MODEL)


def test_chat_template_renders(chat_tokenizer):
    conversation = [{"role": "user", "content": "What is 2+2?"}]
    result_str = chat_tokenizer.apply_chat_template(conversation, tokenize=False)
    assert isinstance(result_str, str)
    assert "What is 2+2?" in result_str


def test_chat_template_tokenizes(chat_tokenizer):
    conversation = [{"role": "user", "content": "What is 2+2?"}]
    result_ids = chat_tokenizer.apply_chat_template(conversation, tokenize=True)
    assert isinstance(result_ids, list)
    assert all(isinstance(i, int) for i in result_ids)
    assert len(result_ids) > 0


# ---------------------------------------------------------------------------
# Tokie backend correctness verification
# ---------------------------------------------------------------------------

# 100+ diverse strings for cross-backend verification.
DIVERSE_TEXTS = [
    "hello world",
    "",
    " ",
    "   ",
    "\n",
    "\t\t\n",
    "a",
    "Hello, World!",
    "The quick brown fox jumps over the lazy dog.",
    "Bonjour le monde",
    "Hallo Welt",
    "Hola mundo",
    "Привет мир",
    "こんにちは世界",
    "你好世界",
    "مرحبا بالعالم",
    "🌍🌎🌏",
    "Hello 🌍 World 🚀",
    "English и Русский and 日本語",
    "café résumé naïve",
    "123456789",
    "3.14159265358979",
    "-1.5e10",
    "!@#$%^&*()_+-=[]{}|;':\",./<>?",
    "\\n\\t\\r\\0",
    "def foo(x: int) -> int:\n    return x + 1",
    "import torch\nmodel = torch.nn.Linear(10, 20)",
    "SELECT * FROM users WHERE id = 1;",
    '{"key": "value", "list": [1, 2, 3]}',
    "<html><body><p>Hello</p></body></html>",
    "aaaaaaaaaaaaaaaaaaaaa",
    "abcabcabcabcabcabcabc",
    "word1  word2   word3    word4",
    "line1\nline2\nline3",
    "tab1\ttab2\ttab3",
    "The quick brown fox jumps over the lazy dog. " * 20,
    "a b c d e f g h i j k l m n o p q r s t u v w x y z " * 10,
    "aaabbbccc",
    "the the the the the",
    "Hello... World!!! How are you???",
    "Mr. Smith went to Washington, D.C., on Jan. 1st.",
    "https://www.example.com/path?q=hello&lang=en#section",
    "/usr/local/bin/python3",
    "f(x) = x^2 + 2x + 1",
    "∫ f(x) dx = F(x) + C",
    "α β γ δ ε ζ η θ",
    "camelCaseVariable",
    "snake_case_variable",
    "SCREAMING_SNAKE_CASE",
    "kebab-case-identifier",
    "\x00",
    "\xff",
    "null\x00byte",
    "I",
    "an",
    "the",
    "internationalization",
    "supercalifragilisticexpialidocious",
    "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
    "* bullet 1\n* bullet 2\n* bullet 3",
    "# Heading\n\n**bold** and *italic* and `code`",
    "<|user|>\nWhat is the capital of France?\n<|assistant|>\nParis is the capital of France.",
    "\u200b",
    "\ufeff",
    "\U0001f600\U0001f601\U0001f602",
    "שלום עולם",
    "สวัสดีชาวโลก",
    "안녕하세요 세계",
    "नमस्ते दुनिया",
    "http",
    "https",
    "www",
    ".com",
    "1,000,000",
    "1.000.000",
    "$1,234.56",
    "50%",
    "\\begin{equation} E = mc^2 \\end{equation}",
    "&amp; &lt; &gt; &quot;",
    "SGVsbG8gV29ybGQ=",
    "---",
    "===",
    "***",
    "~~~",
    "col1\tcol2\tcol3\nval1\tval2\tval3",
    # Additional strings to reach 100+
    "A" * 500,
    "foo bar " * 100,
    "  leading and trailing spaces  ",
    "\n\n\n",
    "MiXeD CaSe TeXt",
    "under_score and-dash",
    "path/to/file.txt",
    "user@example.com",
    "2026-04-03T12:00:00Z",
    "0x1A2B3C4D",
    "True False None null undefined NaN Infinity",
    "the\u00a0non-breaking\u00a0space",
    "left\u2019s right\u201d quotes",
    "(((nested))) [[[brackets]]] {{{braces}}}",
    "C:\\Users\\foo\\Desktop\\file.txt",
    "SELECT COUNT(*) AS cnt FROM table GROUP BY col HAVING cnt > 5 ORDER BY cnt DESC LIMIT 10;",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt.",
    "😀😁😂🤣😃😄😅😆😉😊😋😎😍😘🥰",
    "∑_{i=0}^{n} x_i",
    "Tokyo 東京 Seoul 서울 Beijing 北京",
]


@pytest.fixture(scope="module")
def tokie_tokenizer():
    if not HAS_TOKIE:
        pytest.skip("tokie not installed")
    load_tokenizer.cache_clear()
    return load_tokenizer(MODEL_NAME, backend=TokenizerBackend.TOKIE)


@pytest.mark.skipif(not HAS_TOKIE, reason="tokie not installed")
def test_tokie_protocol_conformance(tokie_tokenizer):
    assert isinstance(tokie_tokenizer, MarinTokenizer)


@pytest.mark.skipif(not HAS_TOKIE, reason="tokie not installed")
def test_tokie_vocab_size_matches_hf(tokenizer, tokie_tokenizer):
    assert tokie_tokenizer.vocab_size == tokenizer.vocab_size


@pytest.mark.skipif(not HAS_TOKIE, reason="tokie not installed")
def test_tokie_special_tokens_match_hf(tokenizer, tokie_tokenizer):
    assert tokie_tokenizer.bos_token == tokenizer.bos_token
    assert tokie_tokenizer.eos_token == tokenizer.eos_token
    assert tokie_tokenizer.bos_token_id == tokenizer.bos_token_id
    assert tokie_tokenizer.eos_token_id == tokenizer.eos_token_id


@pytest.mark.skipif(not HAS_TOKIE, reason="tokie not installed")
def test_tokie_encode_matches_hf(tokenizer, tokie_tokenizer):
    """Encode 100+ diverse strings with both backends and assert identical token IDs."""
    assert len(DIVERSE_TEXTS) >= 100, f"Need >= 100 test strings, have {len(DIVERSE_TEXTS)}"

    mismatches = []
    for text in DIVERSE_TEXTS:
        hf_ids = tokenizer.encode(text, add_special_tokens=False)
        tokie_ids = tokie_tokenizer.encode(text, add_special_tokens=False)
        if hf_ids != tokie_ids:
            mismatches.append(
                {
                    "text": repr(text[:80]),
                    "hf_ids": hf_ids[:20],
                    "tokie_ids": tokie_ids[:20],
                    "hf_len": len(hf_ids),
                    "tokie_len": len(tokie_ids),
                }
            )

    if mismatches:
        summary = "\n".join(
            f"  {m['text']}: hf={m['hf_ids']}{'...' if m['hf_len'] > 20 else ''} "
            f"tokie={m['tokie_ids']}{'...' if m['tokie_len'] > 20 else ''}"
            for m in mismatches[:20]
        )
        pytest.fail(f"{len(mismatches)}/{len(DIVERSE_TEXTS)} encoding mismatches:\n{summary}")


@pytest.mark.skipif(not HAS_TOKIE, reason="tokie not installed")
def test_tokie_encode_batch_matches_hf(tokenizer, tokie_tokenizer):
    """Batch encoding should produce the same results as individual HF encoding."""
    texts = [t for t in DIVERSE_TEXTS if t]
    hf_batch = tokenizer.encode_batch(texts, add_special_tokens=False)
    tokie_batch = tokie_tokenizer.encode_batch(texts, add_special_tokens=False)

    mismatches = []
    for i, (hf_ids, tokie_ids) in enumerate(zip(hf_batch, tokie_batch)):
        if hf_ids != tokie_ids:
            mismatches.append(i)

    if mismatches:
        examples = mismatches[:5]
        details = "\n".join(f"  index {i}: {repr(texts[i][:60])}" for i in examples)
        pytest.fail(f"{len(mismatches)}/{len(texts)} batch encoding mismatches:\n{details}")


@pytest.mark.skipif(not HAS_TOKIE, reason="tokie not installed")
def test_tokie_decode_roundtrip(tokenizer, tokie_tokenizer):
    """Decode should produce the same output as HF for non-special token IDs."""
    texts = ["hello world", "The quick brown fox", "def foo(): pass", "123 + 456 = 579"]
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        hf_decoded = tokenizer.decode(ids)
        tokie_decoded = tokie_tokenizer.decode(ids)
        assert (
            tokie_decoded == hf_decoded
        ), f"Decode mismatch for {repr(text)}: hf={repr(hf_decoded)}, tokie={repr(tokie_decoded)}"


@pytest.mark.skipif(not HAS_TOKIE, reason="tokie not installed")
def test_tokie_encode_with_special_tokens(tokenizer, tokie_tokenizer):
    """Encoding with add_special_tokens=True should match HF behavior."""
    text = "hello world"
    hf_ids = tokenizer.encode(text, add_special_tokens=True)
    tokie_ids = tokie_tokenizer.encode(text, add_special_tokens=True)
    assert hf_ids == tokie_ids, f"Special token encoding mismatch: hf={hf_ids}, tokie={tokie_ids}"


@pytest.mark.skipif(not HAS_TOKIE, reason="tokie not installed")
def test_tokie_get_vocab(tokenizer, tokie_tokenizer):
    """Vocab dictionaries should have substantial overlap.

    Known limitation: tokie's get_vocab() excludes added/special tokens and has
    ~118 ID mismatches for single-byte tokens (tokie's internal byte-fallback
    representation differs from HF's). Encoding correctness is unaffected.
    """
    hf_vocab = tokenizer.get_vocab()
    tokie_vocab = tokie_tokenizer.get_vocab()
    common_keys = set(hf_vocab.keys()) & set(tokie_vocab.keys())
    mismatches = [k for k in common_keys if hf_vocab[k] != tokie_vocab[k]]
    # tokie has known byte-fallback token ID discrepancies in get_vocab (~118 for Llama 3).
    # This does not affect encoding/decoding correctness. Fail only if the
    # mismatch count is unexpectedly large, indicating a deeper problem.
    assert (
        len(mismatches) < 200
    ), f"{len(mismatches)} vocab ID mismatches (expected <200 due to byte-fallback tokens): {mismatches[:10]}"


# ---------------------------------------------------------------------------
# Regression tests for codex findings
# ---------------------------------------------------------------------------

CHAT_MODEL_WITH_PAD = "mistralai/Mistral-7B-Instruct-v0.2"


def test_padding_uses_pad_token_id_not_zero():
    """Verify BatchTokenizer pads input_ids with pad_token_id, not hardcoded 0."""
    from levanter.data.text._batch_tokenizer import BatchTokenizer

    tok = load_tokenizer(CHAT_MODEL_WITH_PAD)
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

    # Find where padding starts (attention_mask transitions to 0)
    pad_start = mask.index(0) if 0 in mask else len(mask)
    assert pad_start < 32, "Sequence should be shorter than max_length to have padding"

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    for i in range(pad_start, 32):
        assert ids[i] == pad_id, f"Position {i}: expected pad_token_id={pad_id}, got {ids[i]}"
        assert mask[i] == 0, f"Position {i}: attention_mask should be 0 for padding"


def test_local_tokenizer_encode_batch():
    """Ensure encode_batch works with local tokenizer paths (no hub round-trip)."""
    from huggingface_hub import hf_hub_download

    # Download tokenizer.json to a local path, then load from that directory.
    path = hf_hub_download(MODEL_NAME, "tokenizer.json")
    import os

    local_dir = os.path.dirname(path)

    load_tokenizer.cache_clear()
    tok = load_tokenizer(local_dir)

    # Use >16 texts to exercise any batch-size code paths
    texts = [f"sentence number {i}" for i in range(20)]
    batch_result = tok.encode_batch(texts)
    individual = [tok.encode(t) for t in texts]
    assert batch_result == individual


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


def test_chat_processor_with_marin_tokenizer():
    """ChatProcessor must work with MarinTokenizer (not just HfTokenizer)."""
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

    # The assistant mask should have at least one 1 (for the assistant content)
    mask = results[0]["assistant_masks"]
    assert any(m == 1 for m in mask), "assistant_masks should mark assistant content"


def test_vocab_size_property():
    """vocab_size should return the correct count without needing len(tokenizer)."""
    tok = load_tokenizer(MODEL_NAME)
    assert tok.vocab_size == 128256
    vocab = tok.get_vocab()
    assert tok.vocab_size == len(vocab)


def test_exception_propagation_on_network_error():
    """_load_tokenizer_config should propagate network errors, not swallow them."""

    class FakeNetworkError(OSError):
        pass

    with patch("levanter.tokenizers.hf_hub_download", side_effect=FakeNetworkError("connection refused")):
        with pytest.raises(FakeNetworkError, match="connection refused"):
            _load_tokenizer_config("some-nonexistent-model/that-will-fail")


def test_encode_batch_correctness_many_strings(tokenizer):
    """Verify encode_batch produces correct results for many strings (string-copy workaround coverage)."""
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
    # Filter empty strings (encode_batch may differ from encode on empty)
    non_empty = [t for t in texts if t]
    batch = tokenizer.encode_batch(non_empty)
    individual = [tokenizer.encode(t) for t in non_empty]
    assert batch == individual, "encode_batch must match individual encode for all strings"
