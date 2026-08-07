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

## Hypothesis 4

The spectral-normalized cubic and SVD-free Sylvester implementation would make
the Hessian correction stable enough to test at 300M.

## Results

Recovery job
`/kaiyuew/muon-error-feedback-300m-spectral-sylvester-20260723-201500`
materialized the fresh Marin validation caches and reached TPU training. The
parent finished with `RuntimeError: 16 step(s) failed`.

All five Muon controls and 18 of 20 blend cells reached the final global step
11,443. Blend gain `0.05` at learning rate `0.012` reached 1.056606 Paloma
C4-en BPB, compared with 1.058626 for paired Muon. Blend gain `0.5` at learning
rate `0.008` failed with `Loss is NaN` at step 5,393; blend gain `0.15` at
learning rate `0.008` did not produce a usable final W&B summary.

Every Hesscorr cell ended failed at global step 0. The task logs report
`RuntimeError: Loss is NaN`, so the 300M run does not test whether the 130M
Hesscorr result transfers to this scale.

## Hypothesis 5

At the first optimizer update, the normalized EMA is `(1 - momentum)` times
the gradient, so `gradient - EMA` is purely radial. The nuclear-norm Hessian
annihilates this direction, but the product-only Sylvester approximation
produces a clipped nonzero correction when the gradient matrix is rank
deficient.

## Changes to make

- Remove the radial tangent component before the Sylvester solve and return a
  zero correction when only float32 roundoff remains.
- Add a scale-relative eigenvalue floor to the polar Hessian used by the
  fixed-point and inverse iterations.
- Fall back to Frobenius normalization only when the power-normalized cubic
  iterate does not produce a finite SPD polar factor.
- Add a Hesscorr-only 300M retry group for the 15 unfinished cells.

## Results

A local rank-deficient first-update reproducer returned a correction at the
Frobenius clipping cap under the prior implementation. The corrected path
returns the exact Muon direction for the same radial update. The focused
optimizer and speedrun suite passes after the change.

## Future work

- [ ] Launch the 15-cell Hesscorr-only 300M retry and verify that each run
  advances beyond the first optimizer update.
