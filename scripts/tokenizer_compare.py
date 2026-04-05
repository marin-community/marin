#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare MarinTokenizer backends against the canonical HF backend.

Reads text from stdin, encodes with all three backends (HF, tokie, kitoken),
and exits with an error if any backend produces different token IDs than HF.
Deltas are written to logs/tokenizer/{backend}.json.

Usage:
    echo "hello world" | uv run scripts/tokenizer_compare.py --model meta-llama/Llama-3.1-8B
    cat corpus.txt | uv run scripts/tokenizer_compare.py --model google/gemma-3-4b-it
    uv run scripts/tokenizer_compare.py --fuzz              # rotate across all models
    uv run scripts/tokenizer_compare.py --fuzz -n 5000      # 5000 iterations
    uv run scripts/tokenizer_compare.py --fuzz --model X    # fuzz single model
"""

import argparse
import json
import os
import random
import sys
import time

from levanter.tokenizers import TokenizerBackend, load_tokenizer

BACKENDS = [
    ("tokie", TokenizerBackend.TOKIE),
    ("kitoken", TokenizerBackend.KITOKEN),
]

LOG_DIR = "logs/tokenizer"

# Diverse tokenizer families: BPE variants, SentencePiece, tiktoken-derived.
# Gated models (Llama) require HF auth; the fuzzer skips models that fail to load.
FUZZ_MODELS = [
    "meta-llama/Llama-3.1-8B",
    "google/gemma-3-4b-it",
    "google/gemma-4-E4B-it",
    "openai-community/gpt2",
    "mistralai/Mistral-7B-v0.3",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "deepseek-ai/DeepSeek-R1",
    "microsoft/Phi-4-mini-instruct",
    "tiiuae/falcon-7b",
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


def _random_unicode_char(rng: random.Random) -> str:
    lo, hi = rng.choice(_UNICODE_RANGES)
    cp = rng.randint(lo, hi)
    return chr(cp)


def _gen_random_unicode(rng: random.Random) -> str:
    length = rng.randint(1, 500)
    return "".join(_random_unicode_char(rng) for _ in range(length))


def _gen_random_bytes(rng: random.Random) -> str:
    """Random bytes decoded as utf-8 with replacement — tests the garbage-in path."""
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
    """Repeated patterns — stress-tests BPE merge behavior."""
    base = "".join(_random_unicode_char(rng) for _ in range(rng.randint(1, 10)))
    return base * rng.randint(2, 200)


def _gen_boundary(rng: random.Random) -> str:
    """Single characters or very short strings from interesting unicode ranges."""
    return "".join(_random_unicode_char(rng) for _ in range(rng.randint(1, 3)))


_GENERATORS = [_gen_random_unicode, _gen_random_bytes, _gen_mixed, _gen_repeated, _gen_boundary]


def generate_fuzz_input(rng: random.Random) -> str:
    gen = rng.choice(_GENERATORS)
    return gen(rng)


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
):
    rate = i / elapsed if elapsed > 0 else 0
    progress = f"{i}/{num_iterations}" if num_iterations else str(i)
    pct = f" ({100 * i / num_iterations:.0f}%)" if num_iterations else ""
    short_model = model.rsplit("/", 1)[-1]
    line = (
        f"\r  {progress}{pct} fuzzes | {rate:.0f}/s | "
        f"{total_failures} fail | {_format_duration(elapsed)} | {short_model}"
    )
    print(f"{line:<80}", end="", file=sys.stderr, flush=True)


def run_fuzz(
    model_set: list[tuple[str, object, list[tuple[str, object]]]],
    *,
    num_iterations: int | None,
    seed: int,
):
    rng = random.Random(seed)
    i = 0
    total_failures = 0
    t0 = time.monotonic()
    last_status = 0.0

    try:
        while num_iterations is None or i < num_iterations:
            model, hf_tok, backends = rng.choice(model_set)
            text = generate_fuzz_input(rng)
            model_slug = model.replace("/", "_")
            failures = _compare_one(text, model, hf_tok, backends, tag=f"fuzz_{model_slug}_{i:06d}")
            if failures:
                total_failures += 1
                # Clear status line before printing mismatch
                print(f"\r{' ' * 80}\r", end="", file=sys.stderr)
                for name, path in failures:
                    print(f"[iter {i}] {model} MISMATCH {name} — {path}", file=sys.stderr)
                if num_iterations is None:
                    break
            i += 1
            now = time.monotonic()
            if now - last_status >= 0.25:
                _print_status(i, total_failures, now - t0, model, num_iterations=num_iterations)
                last_status = now
    except KeyboardInterrupt:
        pass

    elapsed = time.monotonic() - t0
    rate = i / elapsed if elapsed > 0 else 0
    print(f"\r{' ' * 80}\r", end="", file=sys.stderr)
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
    parser.add_argument("--backend", choices=["tokie", "kitoken", "all"], default="all", help="Which backend(s) to test")
    args = parser.parse_args()

    if args.backend != "all":
        global BACKENDS
        BACKENDS = [(n, b) for n, b in BACKENDS if n == args.backend]

    if args.fuzz:
        models = [args.model] if args.model else FUZZ_MODELS
        print(f"loading tokenizers for {len(models)} model(s)...", file=sys.stderr)
        model_set = _load_model_set(models)
        if not model_set:
            print("error: no models with working backends", file=sys.stderr)
            sys.exit(1)
        for model, _, backends in model_set:
            print(f"  {model}: {', '.join(n for n, _ in backends)}", file=sys.stderr)
        print(f"seed: {args.seed}, iterations: {args.n or '∞'}", file=sys.stderr)
        failures = run_fuzz(model_set, num_iterations=args.n, seed=args.seed)
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
