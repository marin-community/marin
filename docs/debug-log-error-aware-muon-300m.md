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

## Future work

- [ ] Record child-job startup and W&B run links after the recovery launch.
