#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare MarinTokenizer backends against the canonical HF backend.

Reads text from stdin, encodes with the kitoken backend,
and exits with an error if it produces different token IDs than HF.
Deltas are written to logs/tokenizer/{backend}.json.

Usage:
    echo "hello world" | uv run scripts/tokenizer_compare.py --model meta-llama/Llama-3.1-8B
    cat corpus.txt | uv run scripts/tokenizer_compare.py --model google/gemma-3-4b-it
    uv run scripts/tokenizer_compare.py --fuzz              # rotate across all models
    uv run scripts/tokenizer_compare.py --fuzz -n 5000      # 5000 iterations
    uv run scripts/tokenizer_compare.py --fuzz --model X    # fuzz single model
    uv run scripts/tokenizer_compare.py --fuzz --tests chat_template,decode_roundtrip
"""

import argparse
import json
import os
import random
import sys
import time

from levanter.tokenizers import TokenizerBackend, load_tokenizer

BACKENDS = [
    ("kitoken", TokenizerBackend.KITOKEN),
]

LOG_DIR = "logs/tokenizer"

# Diverse tokenizer families: BPE variants, SentencePiece, tiktoken-derived.
# Gated models (Llama) require HF auth; the fuzzer skips models that fail to load.
FUZZ_MODELS = [
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-4b-it",
    "google/gemma-4-E4B-it",
    "openai-community/gpt2",
    "mistralai/Mistral-7B-v0.3",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "deepseek-ai/DeepSeek-R1",
    "microsoft/Phi-4-mini-instruct",
    # "tiiuae/falcon-7b",
    "allenai/OLMo-2-0325-32B",
]


def _load_backend(name: str, model: str, backend: TokenizerBackend):
    try:
        return load_tokenizer(model, backend=backend)
    except Exception as e:
        print(f"warning: skipping {name}/{model}: {e}", file=sys.stderr)
        return None


def _write_error(
    backend_name: str,
    model: str,
    text: str,
    hf_ids: list[int],
    other_ids: list[int],
    hf_tok,
    other_tok,
    *,
    tag: str = "",
    extra: dict | None = None,
):
    os.makedirs(LOG_DIR, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    path = os.path.join(LOG_DIR, f"{backend_name}{suffix}.json")

    first_diff = None
    for i in range(max(len(hf_ids), len(other_ids))):
        hf_id = hf_ids[i] if i < len(hf_ids) else None
        other_id = other_ids[i] if i < len(other_ids) else None
        if hf_id != other_id:
            first_diff = {
                "position": i,
                "hf_token_id": hf_id,
                "other_token_id": other_id,
                "hf_token": hf_tok.convert_ids_to_tokens(hf_id) if hf_id is not None else None,
                "other_token": other_tok.convert_ids_to_tokens(other_id) if other_id is not None else None,
            }
            break

    record = {
        "backend": backend_name,
        "model": model,
        "input_text": text[:10000],
        "input_length": len(text),
        "hf_ids": hf_ids,
        "other_ids": other_ids,
        "hf_length": len(hf_ids),
        "other_length": len(other_ids),
        "first_divergence": first_diff,
    }
    if extra:
        record.update(extra)

    with open(path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return path


def _compare_one(
    text: str, model: str, hf_tok, backends: list[tuple[str, object]], *, tag: str = ""
) -> list[tuple[str, str]]:
    """Encode text with HF and all backends, return list of (name, error_path) for mismatches."""
    hf_ids = hf_tok.encode(text)
    failures = []
    for name, tok in backends:
        ids = tok.encode(text)
        if ids != hf_ids:
            path = _write_error(name, model, text, hf_ids, ids, hf_tok, tok, tag=tag)
            failures.append((name, path))
    return failures


# ---------------------------------------------------------------------------
# Additional comparison functions
# ---------------------------------------------------------------------------


def _compare_encode_special_tokens(
    text: str, model: str, hf_tok, backends: list[tuple[str, object]], *, tag: str = ""
) -> list[tuple[str, str]]:
    """Compare encode(text, add_special_tokens=True) across backends."""
    hf_ids = hf_tok.encode(text, add_special_tokens=True)
    failures = []
    for name, tok in backends:
        ids = tok.encode(text, add_special_tokens=True)
        if ids != hf_ids:
            path = _write_error(name, model, text, hf_ids, ids, hf_tok, tok, tag=f"special_{tag}")
            failures.append((name, path))
    return failures


def _compare_decode_roundtrip(
    text: str, model: str, hf_tok, backends: list[tuple[str, object]], *, tag: str = ""
) -> list[tuple[str, str]]:
    """Check decode(encode(text)) roundtrip fidelity vs HF."""
    hf_ids = hf_tok.encode(text)
    hf_decoded = hf_tok.decode(hf_ids)
    failures = []
    for name, tok in backends:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        if decoded != hf_decoded:
            path = _write_error(
                name,
                model,
                text,
                hf_ids,
                ids,
                hf_tok,
                tok,
                tag=f"roundtrip_{tag}",
                extra={"hf_decoded": hf_decoded, "other_decoded": decoded},
            )
            failures.append((name, path))
    return failures


def _compare_decode_skip_special(
    ids_with_specials: list[int], model: str, hf_tok, backends: list[tuple[str, object]], *, tag: str = ""
) -> list[tuple[str, str]]:
    """Compare decode(ids, skip_special_tokens=True) with mixed special/normal token sequences."""
    hf_decoded = hf_tok.decode(ids_with_specials, skip_special_tokens=True)
    failures = []
    for name, tok in backends:
        try:
            decoded = tok.decode(ids_with_specials, skip_special_tokens=True)
        except Exception as e:
            path = _write_error(
                name,
                model,
                f"decode_skip_special_CRASH:{e}",
                ids_with_specials,
                ids_with_specials,
                hf_tok,
                tok,
                tag=f"decode_skip_{tag}",
                extra={"hf_decoded": hf_decoded, "error": str(e)},
            )
            failures.append((name, path))
            continue
        if decoded != hf_decoded:
            path = _write_error(
                name,
                model,
                f"decode_skip_special:{ids_with_specials[:50]}...",
                ids_with_specials,
                ids_with_specials,
                hf_tok,
                tok,
                tag=f"decode_skip_{tag}",
                extra={"hf_decoded": hf_decoded, "other_decoded": decoded},
            )
            failures.append((name, path))
    return failures


def _compare_chat_template(
    conversation: list[dict],
    model: str,
    hf_tok,
    backends: list[tuple[str, object]],
    *,
    add_generation_prompt: bool = True,
    tag: str = "",
) -> list[tuple[str, str]]:
    """Compare apply_chat_template(conv, tokenize=True) across backends."""
    if hf_tok.chat_template is None:
        return []
    try:
        hf_ids = hf_tok.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        return []
    failures = []
    for name, tok in backends:
        try:
            ids = tok.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            continue
        if ids != hf_ids:
            text_repr = json.dumps(conversation, ensure_ascii=False)[:5000]
            path = _write_error(
                name,
                model,
                text_repr,
                hf_ids,
                ids,
                hf_tok,
                tok,
                tag=f"chat_{tag}",
                extra={"conversation": conversation, "add_generation_prompt": add_generation_prompt},
            )
            failures.append((name, path))
    return failures


def _compare_vocab_token_roundtrip(
    token_ids: list[int], model: str, hf_tok, backends: list[tuple[str, object]], *, tag: str = ""
) -> list[tuple[str, str]]:
    """Compare convert_ids_to_tokens consistency for sampled token IDs."""
    failures = []
    for name, tok in backends:
        mismatches = []
        for tid in token_ids:
            hf_token = hf_tok.convert_ids_to_tokens(tid)
            other_token = tok.convert_ids_to_tokens(tid)
            if hf_token != other_token:
                mismatches.append({"id": tid, "hf": hf_token, "other": other_token})
        if mismatches:
            path = _write_error(
                name,
                model,
                f"vocab_roundtrip:{len(mismatches)}_mismatches",
                [m["id"] for m in mismatches],
                [m["id"] for m in mismatches],
                hf_tok,
                tok,
                tag=f"vocab_{tag}",
                extra={"mismatches": mismatches[:20]},
            )
            failures.append((name, path))
    return failures


# ---------------------------------------------------------------------------
# Fuzzer input generators
# ---------------------------------------------------------------------------

# Unicode ranges worth exercising: ASCII, Latin-1 supplement, CJK, Arabic,
# Devanagari, emoji, mathematical symbols, private use area, surrogates-adjacent.
_UNICODE_RANGES = [
    (0x0020, 0x007E),  # printable ASCII
    (0x00A0, 0x00FF),  # Latin-1 supplement
    (0x0100, 0x024F),  # Latin extended
    (0x0370, 0x03FF),  # Greek
    (0x0400, 0x04FF),  # Cyrillic
    (0x0600, 0x06FF),  # Arabic
    (0x0900, 0x097F),  # Devanagari
    (0x0E00, 0x0E7F),  # Thai
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0x1F600, 0x1F64F),  # emoticons
    (0x1F300, 0x1F5FF),  # misc symbols & pictographs
    (0xE000, 0xE0FF),  # private use area
    (0x2000, 0x206F),  # general punctuation (includes ZWSP, ZWJ, etc.)
    (0x2100, 0x214F),  # letterlike symbols
    (0x1D400, 0x1D7FF),  # mathematical alphanumeric symbols
]

_SPECIAL_CHARS = [
    "\x00",
    "\t",
    "\n",
    "\r",
    "\r\n",
    "\u200b",
    "\u200c",
    "\u200d",
    "\ufeff",  # zero-width chars, BOM
    "\u00a0",  # non-breaking space
]

_TEMPLATE_FRAGMENTS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|user|>",
    "<|assistant|>",
    "<s>",
    "</s>",
    "[INST]",
    "[/INST]",
    "<|im_start|>",
    "<|im_end|>",
]

# Model-family special token strings that might appear in user text.
_MODEL_SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|finetune_right_pad_id|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<bos>",
    "<eos>",
    "<start_of_turn>",
    "<end_of_turn>",
    "<s>",
    "</s>",
    "[INST]",
    "[/INST]",
    "<|pad|>",
    "<|unk|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
]

_SYSTEM_PROMPTS = [
    "You are a helpful assistant.",
    "You are a coding assistant. Respond with code when possible.",
    "You are Qwen, created by Alibaba Cloud.",
    "",
    "Answer concisely.",
]

_USER_MESSAGES = [
    "What is 2+2?",
    "Write a Python function to sort a list.",
    "Explain quantum computing in simple terms.",
    "```python\ndef foo():\n    pass\n```\nWhat does this do?",
    "What is the capital of France?",
    "Translate 'hello world' to Japanese.",
    'Fix this JSON: {"key": "value",}',
]

_ASSISTANT_MESSAGES = [
    "4",
    "The capital of France is Paris.",
    "```python\ndef sort_list(lst):\n    return sorted(lst)\n```",
    "I'd be happy to help!",
    "Here's a brief explanation:",
]

_CODE_TEMPLATES = [
    "```python\n{code}\n```",
    "```json\n{code}\n```",
    "```\n{code}\n```",
    "```sql\n{code}\n```",
]

_CODE_SNIPPETS = [
    "def foo(x):\n    return x + 1",
    '{"key": "value", "nested": {"a": 1}}',
    "SELECT * FROM users WHERE id = ?;",
    "import numpy as np\narr = np.zeros((3, 4))",
    "for i in range(10):\n    print(i)",
    "class Foo:\n    def __init__(self):\n        self.x = 1",
    "const f = async () => await fetch('/api');",
]


def _random_unicode_char(rng: random.Random) -> str:
    lo, hi = rng.choice(_UNICODE_RANGES)
    cp = rng.randint(lo, hi)
    return chr(cp)


def _gen_random_unicode(rng: random.Random) -> str:
    length = rng.randint(1, 500)
    return "".join(_random_unicode_char(rng) for _ in range(length))


def _gen_random_bytes(rng: random.Random) -> str:
    """Random bytes decoded as utf-8 with replacement -- tests the garbage-in path."""
    length = rng.randint(1, 200)
    raw = bytes(rng.randint(0, 255) for _ in range(length))
    return raw.decode("utf-8", errors="replace")


def _gen_mixed(rng: random.Random) -> str:
    """Mix ASCII, unicode, special chars, and template-like strings."""
    parts = []
    for _ in range(rng.randint(2, 20)):
        kind = rng.randint(0, 5)
        if kind == 0:
            parts.append(rng.choice(_SPECIAL_CHARS))
        elif kind == 1:
            parts.append(rng.choice(_TEMPLATE_FRAGMENTS))
        elif kind == 2:
            word_len = rng.randint(1, 30)
            parts.append("".join(chr(rng.randint(0x61, 0x7A)) for _ in range(word_len)))
        elif kind == 3:
            parts.append("".join(_random_unicode_char(rng) for _ in range(rng.randint(1, 50))))
        elif kind == 4:
            parts.append(" " * rng.randint(1, 10))
        else:
            parts.append("\n" * rng.randint(1, 5))
    return "".join(parts)


def _gen_repeated(rng: random.Random) -> str:
    """Repeated patterns -- stress-tests BPE merge behavior."""
    base = "".join(_random_unicode_char(rng) for _ in range(rng.randint(1, 10)))
    return base * rng.randint(2, 200)


def _gen_boundary(rng: random.Random) -> str:
    """Single characters or very short strings from interesting unicode ranges."""
    return "".join(_random_unicode_char(rng) for _ in range(rng.randint(1, 3)))


def _gen_special_token_text(rng: random.Random) -> str:
    """Text with embedded special token strings from various model families."""
    parts = []
    for _ in range(rng.randint(2, 8)):
        if rng.random() < 0.4:
            parts.append(rng.choice(_MODEL_SPECIAL_TOKENS))
        else:
            word_len = rng.randint(1, 20)
            parts.append("".join(chr(rng.randint(0x61, 0x7A)) for _ in range(word_len)))
    return " ".join(parts)


def _gen_code_block(rng: random.Random) -> str:
    """Code/JSON/SQL wrapped in markdown fences."""
    template = rng.choice(_CODE_TEMPLATES)
    code = rng.choice(_CODE_SNIPPETS)
    return template.format(code=code)


_GENERATORS = [
    _gen_random_unicode,
    _gen_random_bytes,
    _gen_mixed,
    _gen_repeated,
    _gen_boundary,
    _gen_special_token_text,
    _gen_code_block,
]


def generate_fuzz_input(rng: random.Random) -> str:
    gen = rng.choice(_GENERATORS)
    return gen(rng)


# ---------------------------------------------------------------------------
# Conversation generators (for chat template tests)
# ---------------------------------------------------------------------------


def _gen_conversation(rng: random.Random) -> list[dict[str, str]]:
    """Generate a realistic multi-turn chat conversation."""
    messages: list[dict[str, str]] = []
    if rng.random() < 0.4:
        messages.append({"role": "system", "content": rng.choice(_SYSTEM_PROMPTS)})
    num_turns = rng.randint(1, 5)
    for i in range(num_turns):
        messages.append({"role": "user", "content": rng.choice(_USER_MESSAGES)})
        if i < num_turns - 1 or rng.random() < 0.5:
            messages.append({"role": "assistant", "content": rng.choice(_ASSISTANT_MESSAGES)})
    return messages


def _gen_conversation_with_fuzz_content(rng: random.Random) -> list[dict[str, str]]:
    """Chat conversation with fuzzed content (special tokens, unicode, etc.)."""
    messages: list[dict[str, str]] = []
    num_turns = rng.randint(1, 3)
    for i in range(num_turns):
        messages.append({"role": "user", "content": generate_fuzz_input(rng)})
        if i < num_turns - 1:
            messages.append({"role": "assistant", "content": generate_fuzz_input(rng)})
    return messages


def _gen_tool_call_conversation(rng: random.Random) -> list[dict]:
    """Chat conversation with tool_call/tool role messages."""
    tools = [
        {"name": "get_weather", "args": '{"city": "Paris"}', "result": "Sunny, 22C"},
        {"name": "calculate", "args": '{"expr": "2+2"}', "result": "4"},
        {"name": "search", "args": '{"query": "python sort"}', "result": "Use sorted()"},
    ]
    tool = rng.choice(tools)
    messages: list[dict] = []
    if rng.random() < 0.3:
        messages.append({"role": "system", "content": "You have access to tools."})
    messages.append({"role": "user", "content": rng.choice(_USER_MESSAGES)})
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{rng.randint(1, 9999)}",
                    "type": "function",
                    "function": {"name": tool["name"], "arguments": tool["args"]},
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "name": tool["name"],
            "tool_call_id": messages[-1]["tool_calls"][0]["id"],
            "content": tool["result"],
        }
    )
    messages.append({"role": "assistant", "content": rng.choice(_ASSISTANT_MESSAGES)})
    return messages


def _gen_ids_with_specials(rng: random.Random, hf_tok, backends: list[tuple[str, object]]) -> list[int]:
    """Token ID sequence mixing normal tokens with special token IDs.

    Uses minimum vocab_size across all backends to avoid out-of-range panics.
    """
    min_vocab = hf_tok.vocab_size
    for _, tok in backends:
        min_vocab = min(min_vocab, tok.vocab_size)
    special_ids = list(hf_tok.all_special_ids) if hf_tok.all_special_ids else []
    # Filter special IDs to those within all backends' vocab range
    special_ids = [sid for sid in special_ids if sid < min_vocab]
    ids = []
    for _ in range(rng.randint(5, 50)):
        if rng.random() < 0.3 and special_ids:
            ids.append(rng.choice(special_ids))
        else:
            ids.append(rng.randint(256, min(min_vocab - 1, 50000)))
    return ids


# ---------------------------------------------------------------------------
# Multi-model tokenizer cache
# ---------------------------------------------------------------------------

_tokenizer_cache: dict[tuple[str, TokenizerBackend], object | None] = {}


def _get_tokenizer(model: str, backend: TokenizerBackend):
    """Load a tokenizer, caching across calls. Returns None for backends that fail to load."""
    key = (model, backend)
    if key not in _tokenizer_cache:
        try:
            _tokenizer_cache[key] = load_tokenizer(model, backend=backend)
        except Exception as e:
            print(f"warning: cannot load {backend.value}/{model}: {e}", file=sys.stderr)
            _tokenizer_cache[key] = None
    return _tokenizer_cache[key]


def _load_model_set(
    models: list[str],
) -> list[tuple[str, object, list[tuple[str, object]]]]:
    """Load HF + alternative backends for each model. Returns (model, hf_tok, backends) triples."""
    result = []
    for model in models:
        hf_tok = _get_tokenizer(model, TokenizerBackend.HF)
        if hf_tok is None:
            continue
        loaded_backends = []
        for name, backend in BACKENDS:
            tok = _get_tokenizer(model, backend)
            if tok is not None:
                loaded_backends.append((name, tok))
        if loaded_backends:
            result.append((model, hf_tok, loaded_backends))
    return result


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}m{seconds % 60:02.0f}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


def _print_status(
    i: int,
    total_failures: int,
    elapsed: float,
    model: str,
    *,
    num_iterations: int | None,
    test_type: str = "",
):
    rate = i / elapsed if elapsed > 0 else 0
    progress = f"{i}/{num_iterations}" if num_iterations else str(i)
    pct = f" ({100 * i / num_iterations:.0f}%)" if num_iterations else ""
    short_model = model.rsplit("/", 1)[-1]
    line = (
        f"\r  {progress}{pct} fuzzes | {rate:.0f}/s | "
        f"{total_failures} fail | {_format_duration(elapsed)} | {short_model} | {test_type}"
    )
    print(f"{line:<90}", end="", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Weighted test type selection
# ---------------------------------------------------------------------------

_FUZZ_TESTS = [
    ("encode", 40),
    ("encode_special", 15),
    ("decode_roundtrip", 15),
    ("decode_skip_special", 10),
    ("chat_template", 15),
    ("vocab_roundtrip", 5),
]


def _pick_test_type(rng: random.Random, tests: list[tuple[str, int]]) -> str:
    total = sum(w for _, w in tests)
    r = rng.randint(1, total)
    cumulative = 0
    for name, weight in tests:
        cumulative += weight
        if r <= cumulative:
            return name
    return tests[-1][0]


def _minimize_text(text: str, is_failing) -> str:
    """Delta-debugging: remove chunks of decreasing size while the failure persists."""
    current = text
    chunk_size = max(1, len(current) // 2)
    while chunk_size >= 1:
        i = 0
        while i + chunk_size <= len(current):
            candidate = current[:i] + current[i + chunk_size :]
            if candidate and is_failing(candidate):
                current = candidate
            else:
                i += 1
        chunk_size //= 2
    return current


def _minimize_ids(ids: list[int], is_failing) -> list[int]:
    """Delta-debugging: remove chunks of decreasing size from a token ID sequence."""
    current = list(ids)
    chunk_size = max(1, len(current) // 2)
    while chunk_size >= 1:
        i = 0
        while i + chunk_size <= len(current):
            candidate = current[:i] + current[i + chunk_size :]
            if candidate and is_failing(candidate):
                current = candidate
            else:
                i += 1
        chunk_size //= 2
    return current


def _run_fuzz_iteration(
    rng: random.Random,
    test_type: str,
    model: str,
    hf_tok,
    backends: list[tuple[str, object]],
    tag_prefix: str,
    *,
    minimize: bool = True,
) -> list[tuple[str, str]]:
    """Run a single fuzz iteration of the given test type. Returns failure list."""
    if test_type == "encode":
        text = generate_fuzz_input(rng)
        failures = _compare_one(text, model, hf_tok, backends, tag=tag_prefix)
        if failures and minimize:
            orig_len = len(text)
            text = _minimize_text(text, lambda t: any(tok.encode(t) != hf_tok.encode(t) for _, tok in backends))
            if len(text) < orig_len:
                print(f"  minimized: {orig_len} -> {len(text)} chars", file=sys.stderr)
                failures = _compare_one(text, model, hf_tok, backends, tag=tag_prefix)
        return failures

    elif test_type == "encode_special":
        text = generate_fuzz_input(rng)
        failures = _compare_encode_special_tokens(text, model, hf_tok, backends, tag=tag_prefix)
        if failures and minimize:
            orig_len = len(text)
            text = _minimize_text(
                text,
                lambda t: any(
                    tok.encode(t, add_special_tokens=True) != hf_tok.encode(t, add_special_tokens=True)
                    for _, tok in backends
                ),
            )
            if len(text) < orig_len:
                print(f"  minimized: {orig_len} -> {len(text)} chars", file=sys.stderr)
                failures = _compare_encode_special_tokens(text, model, hf_tok, backends, tag=tag_prefix)
        return failures

    elif test_type == "decode_roundtrip":
        text = generate_fuzz_input(rng)
        failures = _compare_decode_roundtrip(text, model, hf_tok, backends, tag=tag_prefix)
        if failures and minimize:
            orig_len = len(text)

            def _roundtrip_fails(t):
                hf_ids = hf_tok.encode(t)
                hf_dec = hf_tok.decode(hf_ids)
                return any(tok.decode(tok.encode(t)) != hf_dec for _, tok in backends)

            text = _minimize_text(text, _roundtrip_fails)
            if len(text) < orig_len:
                print(f"  minimized: {orig_len} -> {len(text)} chars", file=sys.stderr)
                failures = _compare_decode_roundtrip(text, model, hf_tok, backends, tag=tag_prefix)
        return failures

    elif test_type == "decode_skip_special":
        ids = _gen_ids_with_specials(rng, hf_tok, backends)
        failures = _compare_decode_skip_special(ids, model, hf_tok, backends, tag=tag_prefix)
        if failures and minimize:
            orig_len = len(ids)

            def _skip_special_fails(candidate_ids):
                hf_dec = hf_tok.decode(candidate_ids, skip_special_tokens=True)
                for _, tok in backends:
                    try:
                        if tok.decode(candidate_ids, skip_special_tokens=True) != hf_dec:
                            return True
                    except Exception:
                        return True
                return False

            ids = _minimize_ids(ids, _skip_special_fails)
            if len(ids) < orig_len:
                print(f"  minimized: {orig_len} -> {len(ids)} token IDs", file=sys.stderr)
                failures = _compare_decode_skip_special(ids, model, hf_tok, backends, tag=tag_prefix)
        return failures

    elif test_type == "chat_template":
        choice = rng.random()
        if choice < 0.35:
            conv = _gen_conversation(rng)
        elif choice < 0.7:
            conv = _gen_conversation_with_fuzz_content(rng)
        else:
            conv = _gen_tool_call_conversation(rng)
        gen_prompt = rng.choice([True, False])
        return _compare_chat_template(
            conv,
            model,
            hf_tok,
            backends,
            add_generation_prompt=gen_prompt,
            tag=tag_prefix,
        )

    elif test_type == "vocab_roundtrip":
        sample_ids = [rng.randint(256, min(hf_tok.vocab_size - 1, 50000)) for _ in range(20)]
        return _compare_vocab_token_roundtrip(sample_ids, model, hf_tok, backends, tag=tag_prefix)

    return []


def run_fuzz(
    model_set: list[tuple[str, object, list[tuple[str, object]]]],
    *,
    num_iterations: int | None,
    seed: int,
    active_tests: list[tuple[str, int]],
    minimize: bool = True,
):
    rng = random.Random(seed)
    i = 0
    total_failures = 0
    t0 = time.monotonic()

    try:
        while num_iterations is None or i < num_iterations:
            model, hf_tok, backends = rng.choice(model_set)
            test_type = _pick_test_type(rng, active_tests)
            model_slug = model.replace("/", "_")
            tag = f"fuzz_{model_slug}_{i:06d}"

            failures = _run_fuzz_iteration(rng, test_type, model, hf_tok, backends, tag, minimize=minimize)

            if failures:
                total_failures += 1
                print(f"\r{' ' * 90}\r", end="", file=sys.stderr)
                for name, path in failures:
                    print(f"[iter {i}] {model} MISMATCH {name} ({test_type}) — {path}", file=sys.stderr)
                if num_iterations is None:
                    break
            i += 1
            if i % 100 == 0:
                now = time.monotonic()
                _print_status(i, total_failures, now - t0, model, num_iterations=num_iterations, test_type=test_type)
    except KeyboardInterrupt:
        pass

    elapsed = time.monotonic() - t0
    rate = i / elapsed if elapsed > 0 else 0
    print(f"\r{' ' * 90}\r", end="", file=sys.stderr)
    print(f"{i} iterations in {_format_duration(elapsed)} ({rate:.0f}/s), {total_failures} failure(s)", file=sys.stderr)
    return total_failures


def main():
    parser = argparse.ArgumentParser(description="Compare tokenizer backends against HF")
    parser.add_argument(
        "--model", default=None, help="HF model name or local path (default: rotate across all models in fuzz mode)"
    )
    parser.add_argument("--fuzz", action="store_true", help="Fuzz mode: generate random inputs")
    parser.add_argument(
        "-n", type=int, default=None, help="Number of fuzz iterations (default: run until mismatch or ctrl-c)"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for fuzz mode")
    parser.add_argument("--backend", choices=["kitoken", "all"], default="all", help="Which backend(s) to test")
    parser.add_argument("--no-minimize", action="store_true", help="Disable input minimization on failure")
    parser.add_argument(
        "--tests",
        default="all",
        help="Comma-separated test types: encode,encode_special,decode_roundtrip,"
        "decode_skip_special,chat_template,vocab_roundtrip (default: all)",
    )
    args = parser.parse_args()

    if args.backend != "all":
        global BACKENDS
        BACKENDS = [(n, b) for n, b in BACKENDS if n == args.backend]

    active_tests = _FUZZ_TESTS
    if args.tests != "all":
        requested = set(args.tests.split(","))
        valid_names = {name for name, _ in _FUZZ_TESTS}
        unknown = requested - valid_names
        if unknown:
            print(f"error: unknown test type(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"valid types: {', '.join(valid_names)}", file=sys.stderr)
            sys.exit(1)
        active_tests = [(n, w) for n, w in _FUZZ_TESTS if n in requested]

    if args.fuzz:
        models = [args.model] if args.model else FUZZ_MODELS
        print(f"loading tokenizers for {len(models)} model(s)...", file=sys.stderr)
        model_set = _load_model_set(models)
        if not model_set:
            print("error: no models with working backends", file=sys.stderr)
            sys.exit(1)
        for model, _, backends in model_set:
            print(f"  {model}: {', '.join(n for n, _ in backends)}", file=sys.stderr)
        test_names = ", ".join(n for n, _ in active_tests)
        print(f"seed: {args.seed}, iterations: {args.n or '∞'}, tests: {test_names}", file=sys.stderr)
        failures = run_fuzz(
            model_set, num_iterations=args.n, seed=args.seed, active_tests=active_tests, minimize=not args.no_minimize
        )
        sys.exit(1 if failures else 0)

    # Stdin mode — requires --model
    model = args.model or "meta-llama/Llama-3.1-8B"
    hf_tok = _get_tokenizer(model, TokenizerBackend.HF)
    if hf_tok is None:
        print(f"error: cannot load HF tokenizer for {model}", file=sys.stderr)
        sys.exit(1)

    loaded_backends = []
    for name, backend in BACKENDS:
        tok = _get_tokenizer(model, backend)
        if tok is not None:
            loaded_backends.append((name, tok))

    if not loaded_backends:
        print("error: no alternative backends available", file=sys.stderr)
        sys.exit(1)

    text = sys.stdin.read()
    if not text:
        print("error: empty stdin", file=sys.stderr)
        sys.exit(1)

    print(f"model: {model}", file=sys.stderr)
    print(f"backends: hf (canonical), {', '.join(n for n, _ in loaded_backends)}", file=sys.stderr)
    print(f"input: {len(text)} chars", file=sys.stderr)

    hf_ids = hf_tok.encode(text)
    print(f"hf: {len(hf_ids)} tokens", file=sys.stderr)

    failures = []
    for name, tok in loaded_backends:
        ids = tok.encode(text)
        print(f"{name}: {len(ids)} tokens", file=sys.stderr)
        if ids != hf_ids:
            path = _write_error(name, model, text, hf_ids, ids, hf_tok, tok)
            failures.append((name, path))
            print(f"MISMATCH {name} — wrote {path}", file=sys.stderr)
        else:
            print(f"{name}: ok", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} backend(s) diverged from HF:", file=sys.stderr)
        for name, path in failures:
            print(f"  {name}: {path}", file=sys.stderr)
        sys.exit(1)

    print("\nall backends match", file=sys.stderr)


if __name__ == "__main__":
    main()
