# MarinTokenizer Status

Branch: `rpower/marin-tokenizer` (18 commits ahead of main)

Last updated: 2026-04-04

## Architecture

The tokenizer abstraction lives in `lib/levanter/src/levanter/tokenizers.py`. The
central type is `MarinTokenizer`, a `@runtime_checkable` Protocol that defines the
contract every backend must satisfy:

```
MarinTokenizer (Protocol)
  name_or_path, vocab_size, bos/eos/pad token IDs and strings
  encode(text, add_special_tokens) -> list[int]
  decode(ids, skip_special_tokens) -> str
  encode_batch(texts, add_special_tokens) -> list[list[int]]
  get_vocab() -> dict[str, int]
  convert_ids_to_tokens / convert_tokens_to_ids
  all_special_ids
  chat_template, apply_chat_template, apply_chat_template_with_masks
  __len__
```

Callers use `load_tokenizer(name_or_path, backend=TokenizerBackend.HF)` which
returns a cached `MarinTokenizer`. The `TokenizerBackend` enum (`hf`, `kitoken`)
selects the implementation.

### Shared infrastructure

- `_load_tokenizer_config` reads `tokenizer_config.json` from hub or local path.
  All backends use it for special tokens, chat templates, and `added_tokens_decoder`.
- `_apply_chat_template_with_masks` is a free function shared by all backends. It
  uses a Jinja2 extension (`_GenerationSentinelExtension`) to wrap
  `{% generation %}` blocks with sentinel strings, then maps sentinels to binary
  assistant masks after encoding.
- `_collect_special_ids` builds the `all_special_ids` list from bos/eos/pad plus
  `added_tokens_decoder` entries marked `special: true`.

### Backend: HF (`HfMarinTokenizer`)

Wraps `tokenizers.Tokenizer` (the Rust `tokenizers` library). Does NOT use
`transformers.AutoTokenizer`, so it avoids the torch import.

Status: **stable, fully migrated**. All 49 changed files on this branch use
`load_tokenizer` instead of `AutoTokenizer`.

## Migration

All tokenizer usage across the codebase has been migrated to `load_tokenizer`:

- Levanter: `BatchTokenizer`, `ChatProcessor`, text data loading, SFT, DPO
- Marin: tokenize pipeline (`TokenizeConfig.tokenizer_backend`), RL code
  (rollout workers, train workers, environments), evaluation harness, utilities
- Tests: all test files updated

The `TokenizeConfig` dataclass in `marin.processing.tokenize` has a
`tokenizer_backend: TokenizerBackend` field that propagates through the Ray
pipeline via shared context.

## Benchmark

`experiments/tokenize/fineweb_benchmark.py` runs the same tokenization job with
both HF and kitoken backends over fineweb-edu 10BT.

- Tiny mode (`BENCHMARK_TINY=1`): 200 synthetic JSONL records, for local smoke
  testing.
- Full mode: reads from `gs://marin-us-central1/raw/fineweb-edu-87f0914/sample/10BT`.
- Uses `StepRunner` for execution; worker RAM bumped to 16GB (for the OOM fix).

## Tests

Test files:

- `lib/levanter/tests/test_tokenizers.py` -- HF backend unit tests, kitoken
  cross-backend correctness (100+ diverse strings), batch encoding, vocab
  comparison, chat template rendering, padding, local path loading, error
  propagation.
- `tests/test_marin_tokenizer.py` -- Marin-specific tokenizer creation tests
  (Llama 3 custom tokenizer with special token injection).
- `tests/test_marin_chat_template.py` -- Chat template tests with tool calls and
  ipython output.

The kitoken tests are gated on `HAS_KITOKEN` (import check) and skipped if kitoken is
not installed.

## Next Steps

1. **Comprehensive cross-backend testing**: Expand the diverse text corpus and
   add decode round-trip tests across all backends. Consider property-based
   testing with Hypothesis.

2. **Benchmark expansion**: Add throughput metrics (tokens/sec), memory profiling,
   and multi-model comparison (Llama 3, Mistral, Gemma) to the benchmark.
