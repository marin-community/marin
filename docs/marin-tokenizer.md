# MarinTokenizer: Multi-Backend Tokenizer Status

**Branch:** `rpower/marin-tokenizer`
**Date:** 2026-04-05

## Overview

MarinTokenizer provides a unified `Protocol` interface over three tokenizer backends:

| Backend | Library | Language | Source |
|---------|---------|----------|--------|
| **HF** (canonical) | `transformers` | Python/Rust | HuggingFace |
| **tokie** | `tokie` | Rust | `vendor/tokie/` |
| **kitoken** | `kitoken` | Rust | `vendor/kitoken/` |

HF is the reference implementation. tokie and kitoken are Rust-native alternatives
targeting lower overhead for batch tokenization in data pipelines.

## Correctness Validation

### Fuzzer

`scripts/tokenizer_compare.py` compares backends against HF across 10 tokenizer
families:

- `meta-llama/Llama-3.1-8B` (tiktoken-derived BPE)
- `google/gemma-3-4b-it` (SentencePiece BPE)
- `google/gemma-4-E4B-it` (SentencePiece BPE)
- `openai-community/gpt2` (byte-level BPE)
- `mistralai/Mistral-7B-v0.3` (byte-fallback BPE)
- `Qwen/Qwen2.5-0.5B-Instruct` (tiktoken-based)
- `deepseek-ai/DeepSeek-R1` (custom BPE)
- `microsoft/Phi-4-mini-instruct` (custom BPE)
- `tiiuae/falcon-7b` (Punctuation+Digits+ByteLevel BPE)
- `allenai/OLMo-2-0325-32B` (BPE)

```bash
# Fuzz all models, infinite until mismatch
uv run scripts/tokenizer_compare.py --fuzz

# Fuzz N iterations across all models
uv run scripts/tokenizer_compare.py --fuzz -n 100000

# Fuzz single model, single backend
uv run scripts/tokenizer_compare.py --fuzz --model google/gemma-3-4b-it --backend kitoken -n 10000

# Stdin mode
echo "hello world" | uv run scripts/tokenizer_compare.py --model meta-llama/Llama-3.1-8B
```

Mismatch details are written to `logs/tokenizer/{backend}_fuzz_{model}_{iter}.json`
with both token ID sequences and the first divergence point.

### Cluster Benchmark

`experiments/tokenize/fineweb_benchmark.py` runs full-scale comparison on
fineweb-edu 10BT (~10.5M documents) via Iris/Zephyr. Output lands at
`gs://marin-us-central1/tmp/tokenizer-benchmark/`.

## kitoken Status: Production-Ready for Llama 3

### Current results

| Test | Result |
|------|--------|
| Fuzzer (100k iter, 10 models) | 0 failures on 9/10 models, 27 on falcon-7b |
| fineweb-edu 10BT (Llama 3.1) | **0 mismatches across 10.5M documents** |
| Unit tests (427 tests) | All passing |

### Fixes applied

1. **Special token recognition** (`tokenizers.py`): Changed `encode(text, False)` to
   `encode(text, True)` so kitoken recognizes `<|end_of_text|>` etc. as single
   tokens, matching HF's default behavior.

2. **Priority specials heuristic** (`vendor/kitoken/src/convert/tokenizers.rs`):
   Fixed `drop_priority_from_specials` to only drop Priority specials that are
   actual BPE merge products (single-byte chars or tokens in `bpe.merges`).
   Previously it dropped all or none, breaking tokenizers with many added_tokens
   (Gemma, Qwen, DeepSeek, Phi, OLMo).

3. **lstrip/rstrip propagation** (`vendor/kitoken/src/`): Added `lstrip`/`rstrip`
   fields to `SpecialToken`, propagated from HF's `AddedToken` config. Fixes
   whitespace handling around special tokens.

4. **GPT-2 byte-level BPE regex** (`vendor/kitoken/src/convert/tokenizers.rs`):
   Changed `\s?` to ` ?` (ASCII space only) in the ByteLevel pretokenizer regex,
   matching HF's actual behavior. Added missing `|\s+(?!\S)|\s+` alternatives.

### Known limitation

Falcon-7b has ~27 failures per 100k iterations due to an architectural mismatch:
HF's ByteLevel pretokenizer applies regex on byte-level character representations,
while kitoken applies regex on the original text then converts to bytes. Fixing
this requires a fundamental change to kitoken's encoding pipeline.

## tokie Status: Functional, Edge Cases Remain

### Current results

| Test | Result |
|------|--------|
| Fuzzer (10k iter, 10 models) | ~329 failures (3.3%) |
| Dominant failures | falcon-7b (259), DeepSeek (59) |

### Fixes applied

1. **Infinite loop on form feed/vertical tab** (`vendor/tokie/crates/pretokie/`):
   Whitespace scanning only recognized 4 of 6 ASCII whitespace bytes, missing
   `\x0b` (vertical tab) and `\x0c` (form feed). This caused an infinite loop
   when the pretokenizer encountered these bytes followed by newlines. Fixed by
   using Rust's `u8::is_ascii_whitespace()`.

2. **DeepSeek pretokenizer detection** (`vendor/tokie/crates/tokie/src/hf.rs`):
   Fixed multi-stage Split pretokenizer detection for DeepSeek's `\p{N}{1,3}`
   digit chunking.

3. **Continuation normalization** (`vendor/tokie/crates/tokie/src/normalizer.rs`):
   Added `normalize_continuation` to skip the `▁` prefix after special token
   boundaries, fixing Mistral failures.

4. **GPT-2 non-ASCII symbol handling** (`vendor/tokie/crates/pretokie/`): Fixed
   combining marks being incorrectly merged with following letters in GPT-2 mode.

5. **lstrip/rstrip support** (`vendor/tokie/crates/tokie/src/tokenizer.rs`):
   Added whitespace stripping around special tokens with `rstrip`/`lstrip` flags.

### Remaining work

- Falcon-7b needs a dedicated pretokenizer for its unusual
  `Punctuation(Contiguous) + ByteLevel + Digits + Split` chain (~2.6% failure rate)
- DeepSeek has edge cases with spaces before non-standard unicode (~0.6%)
- Scattered single-digit failures on Phi-4, Mistral, GPT-2, Llama, OLMo

## File Overview

| Path | Description |
|------|-------------|
| `lib/levanter/src/levanter/tokenizers.py` | `MarinTokenizer` Protocol + HF/tokie/kitoken wrappers |
| `lib/levanter/tests/test_tokenizers.py` | 427 unit tests across all backends |
| `scripts/tokenizer_compare.py` | Multi-model fuzzer and stdin comparison tool |
| `experiments/tokenize/fineweb_benchmark.py` | Cluster-scale benchmark on fineweb-edu |
| `vendor/kitoken/` | Kitoken Rust source (submodule) |
| `vendor/tokie/` | Tokie Rust source (submodule) |
