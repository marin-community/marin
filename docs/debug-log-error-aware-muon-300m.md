# Debugging log for error-aware-muon-300m

Recover the 300M spectral-normalized, Sylvester error-feedback Muon sweep.

## Initial status

The first 300M launch, `/kaiyuew/muon-error-feedback-300m-spectral-sylvester-20260723-145300`, failed all 40 cells before TPU allocation. Each cell raised `ValueError: mixture components must share one tokenizer`, reporting `marin-community/marin-tokenizer` and `meta-llama/Meta-Llama-3.1-8B`.

## Hypothesis 1

The archived fineweb training cache is pinned to the Llama 3 tokenizer, but the shared speedrun helper constructs Paloma and Uncheatable validation caches with the Marin tokenizer.

## Changes to make

- Construct the validation caches with `llama3_tokenizer`, the tokenizer resolved by the archived training cache.
- Add a graph-construction regression test that materializes every validation cache and checks its tokenizer.

## Results

The focused optimizer and speedrun-submission tests pass (23 tests). The
regression test materializes all 23 validation cache configurations and confirms
they use `meta-llama/Meta-Llama-3.1-8B`.

Recovery job `/kaiyuew/muon-error-feedback-300m-spectral-sylvester-20260723-140500`
was submitted with a fresh `2026.07.23.2` output version. The CPU parent only
materializes the graph; the 40 child training jobs request their `v5p-8`
resources through the speedrun configuration. It again failed all 40 cells
before TPU allocation with the same tokenizer mismatch.

## Hypothesis 2

The existing `paloma/*-llama3@2026.06.28` and
`uncheatable_eval/*-llama3@2026.06.28` artifact records were created with the
Marin tokenizer. The tokenizing recipe is not part of an artifact identity, so
switching the recipe to the Llama 3 tokenizer reused those incompatible records.

## Changes to make

- Make the validation artifact name depend on the selected tokenizer.
- Advance the validation-cache version so the Llama 3 handles materialize new
  records rather than reuse the invalid `2026.06.28` artifacts.

## Results

The focused optimizer and speedrun-submission tests pass (24 tests). Llama 3
and Marin validation handles now produce distinct artifact identities, and the
new Llama 3 Paloma and Uncheatable cache prefixes do not yet exist in GCS.

The 2026.07.23.3 recovery launch constructed the fresh Llama 3 validation
caches successfully, then failed all 40 training steps before TPU allocation.
The persisted FineWeb record at
tokenized/subcache/fineweb-edu-10B-ac65f6/.executor_info identifies its
tokenizer as marin-community/marin-tokenizer; the fresh validation record
identifies meta-llama/Meta-Llama-3.1-8B.

## Hypothesis 3

The archived training cache, rather than the validation cache, is the Marin
tokenizer component in the mismatch. The speedrun validation must adopt that
actual persisted tokenizer record.

## Changes to make

- Build the default speedrun validation with the Marin tokenizer and retain the
  tokenizer-specific -marin artifact identity.
- Recognize the resulting c4_en-marin W&B metric when collecting results.

## Results

The focused optimizer and speedrun-submission tests pass (24 tests). Local
materialization now resolves both the archived FineWeb train cache and
Paloma's new c4_en-marin cache to marin-community/marin-tokenizer. The new
Marin cache prefixes are empty, so the next launch cannot reuse an
incompatible record.

## Future work

- [ ] Verify the fresh Marin validation caches materialize, then launch a new
  300M sweep version.
