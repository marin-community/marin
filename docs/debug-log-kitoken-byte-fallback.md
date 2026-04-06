# Debugging log for kitoken byte fallback decode

Vendor `kitoken` should match Hugging Face `ByteFallback` decode behavior for Gemma byte-fallback tokens.

## Initial status

`uv run scripts/tokenizer_compare.py --fuzz --backend=kitoken --model google/gemma-3-4b-it`
was still failing `decode_skip_special` on minimized cases like `[331, 383]` and `[348, 463]`.
HF decoded those to `"��"` while kitoken decoded them to `"]�"` and `"n�"`.

## Hypothesis 1

`kitoken` eagerly converts `<0xXX>` tokens into raw bytes at decoder initialization, then concatenates them directly.
HF's `ByteFallback` decoder instead treats contiguous byte-fallback tokens as a run and only emits decoded UTF-8 when the whole run is valid.

## Changes to make

- Patch `vendor/kitoken/src/decoder.rs` to preserve byte-fallback token identity and decode contiguous runs with HF-compatible semantics.
- Add a regression test in `vendor/kitoken/tests/test_gemma3_bpe.rs` for the minimized Gemma decode failures.

## Future Work

- [ ] Upstream the vendored fix back to the canonical `kitoken` repository if it is not already tracked there.

## Results

- Patched `vendor/kitoken/src/decoder.rs` so contiguous `<0xXX>` byte-fallback tokens are decoded as a run.
- If the whole run is valid UTF-8, the decoder emits the decoded bytes unchanged.
- If the run is not valid UTF-8, the decoder now emits one UTF-8 replacement character per token, matching Hugging Face `ByteFallback`.
- Added `test_gemma3_byte_fallback_decode_regression` in `vendor/kitoken/tests/test_gemma3_bpe.rs`.
- `cargo test --test test_gemma3_bpe` passed.
- `uv sync --reinstall-package kitoken` rebuilt the editable Python package successfully.
- `uv run scripts/tokenizer_compare.py --fuzz --backend=kitoken --model google/gemma-3-4b-it -n 10000` completed with `0 failure(s)`.
