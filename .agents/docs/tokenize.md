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
returns a cached `MarinTokenizer`. The `TokenizerBackend` enum (`hf`, `tokie`)
selects the implementation.

### Shared infrastructure

- `_load_tokenizer_config` reads `tokenizer_config.json` from hub or local path.
  Both backends use it for special tokens, chat templates, and `added_tokens_decoder`.
- `_apply_chat_template_with_masks` is a free function shared by both backends. It
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

### Backend: Tokie (`TokieMarinTokenizer`)

Wraps `tokie.Tokenizer`. Tokie is vendored as a git submodule at `vendor/tokie/`
(upstream: `github.com/chonkie-inc/tokie`). Installed as an editable path
dependency in `pyproject.toml`:

```
tokie = { path = "vendor/tokie/crates/tokie-python", editable = true }
```

Status: **working with caveats** (see Known Issues).

Loading flow: `_load_tokie_tokenizer` uses `hf_hub_download` to fetch
`tokenizer.json` (for auth on gated models), then calls `tokie.Tokenizer.from_json`.
Special token IDs are resolved from `tokenizer_config.json` because tokie's
`vocab_size` and `token_to_id` exclude added/special tokens.

Tokie's `decode()` does not support `skip_special_tokens`. The wrapper filters
known special token IDs manually before calling tokie's decoder.

## Known Issues

### 1. Vocab mismatch for byte-fallback tokens (get_vocab only)

HF's `get_vocab()` uses GPT-2's byte-to-unicode mapping for the 256 base byte
tokens (e.g., byte `0x20` maps to string `"Ġ"`). Tokie maps those same byte
values to their merge-result tokens instead. This causes ~118 key-to-ID
mismatches in the vocab dict for Llama 3.

**This does NOT affect `encode()` or `decode()`** -- only `get_vocab()` and by
extension `convert_tokens_to_ids` for those specific byte tokens. The test
`test_tokie_get_vocab` allows up to 200 mismatches as a known tolerance.

### 2. Threading removed from vendored tokie

The vendored tokie fork has threading support removed. All encoding is
single-threaded. This was done to avoid issues with Python's GIL and to simplify
the Rust-Python boundary.

### 3. OOM fix: string-copy in encode_batch

Both `HfMarinTokenizer.encode_batch` and `TokieMarinTokenizer.encode_batch` copy
input strings via `["".join(s) for s in texts]` before passing to the Rust
library. This releases references to potentially large source buffers (e.g.,
slices of a multi-GB JSONL line) that would otherwise be retained by Python's
string interning. Commit: `cfffbebb6`.

### 4. Auth fix: hf_hub_download for gated models

Tokie's `from_pretrained` does not pass `HF_TOKEN` for authentication.
`_load_tokie_tokenizer` works around this by using `hf_hub_download` to fetch
`tokenizer.json` with proper auth, then loading via `from_json`. Commit:
`2112105d9`.

### 5. Tokie decode panics on special token IDs

Tokie's decoder can panic when given IDs outside its core vocabulary (e.g., bos,
eos). The wrapper filters these out when `skip_special_tokens=True`.

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
both HF and tokie backends over fineweb-edu 10BT.

- Tiny mode (`BENCHMARK_TINY=1`): 200 synthetic JSONL records, for local smoke
  testing.
- Full mode: reads from `gs://marin-us-central1/raw/fineweb-edu-87f0914/sample/10BT`.
- Uses `StepRunner` for execution; worker RAM bumped to 16GB (for the OOM fix).

## Tests

Test files:

- `lib/levanter/tests/test_tokenizers.py` -- HF backend unit tests, tokie
  cross-backend correctness (100+ diverse strings), batch encoding, vocab
  comparison, chat template rendering, padding, local path loading, error
  propagation.
- `tests/test_marin_tokenizer.py` -- Marin-specific tokenizer creation tests
  (Llama 3 custom tokenizer with special token injection).
- `tests/test_marin_chat_template.py` -- Chat template tests with tool calls and
  ipython output.

The tokie tests are gated on `HAS_TOKIE` (import check) and skipped if tokie is
not installed.

## Next Steps

1. **Kitoken backend**: Another agent is adding a `KitokenMarinTokenizer`
   backend. It will follow the same pattern: implement the Protocol, add a
   `TokenizerBackend.KITOKEN` enum value, add a `_load_kitoken_tokenizer`
   factory function.

2. **Comprehensive cross-backend testing**: Expand the diverse text corpus and
   add decode round-trip tests across all backends. Consider property-based
   testing with Hypothesis.

3. **Benchmark expansion**: Add throughput metrics (tokens/sec), memory profiling,
   and multi-model comparison (Llama 3, Mistral, Gemma) to the benchmark.

4. **Vocab mismatch resolution**: Investigate whether tokie can be patched to use
   the same byte-to-unicode mapping as HF for `get_vocab()`, or document the
   divergence as permanent and adjust callers of `get_vocab()` accordingly.

5. **Tokie skip_special_tokens**: Upstream support for `skip_special_tokens` in
   tokie's `decode()` to remove the manual filtering workaround.
