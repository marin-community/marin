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

## Hypothesis 6

The radial-component guard only handles the exact first-update geometry. During
the first momentum time constant, about `1 / (1 - momentum)` optimizer updates,
the EMA matrix remains close to zero and can still have many small singular
values. Applying the Sylvester correction during that stage inverts its
ill-conditioned polar factor. Delay the correction until the momentum buffer
has accumulated for one time constant, while continuing to apply the ordinary
Muon direction from the first update.

## Changes to make

- Track the optimizer update count in the error-aware Muon transform.
- Skip the Hessian correction for the first `round(1 / (1 - momentum))`
  updates, using a JIT-safe conditional so the unstable Sylvester branch is not
  evaluated.
- Add a regression test that observes ordinary Muon updates throughout the
  delay and a nonzero Hessian correction on the following update.

## Results

Before the scheduling change, the regression diverged from the plain-Muon
reference on the second update, inside the two-step momentum time constant for
`momentum=0.5`. With the update counter and conditional in place, both warmup
updates match Muon and the third update matches the independent SVD Hessian
reference. The full error-aware Muon test module passes (21 tests). The 300M
sweep now records a 50-update correction warmup for `momentum=0.98`.

The CPU launch gate repeated the boundary test with the exact 300M momentum:
updates 1 through 50 matched the independent plain-Muon reference, and update
51 matched the independent SVD Hessian reference. A separate 100-update,
bfloat16 stress probe used near-zero, nearly rank-one gradients; the minimum
momentum singular value at correction activation was `1.612e-8`, all 100
updates remained finite, the warmup was exactly equal to Muon, and the active
correction was nonzero. The full 21-test module passes with `JAX_PLATFORMS=cpu`,
and the changed-files lint, formatting, and Pyrefly checks pass.

## Hypothesis 7

The 50-step delay addresses the near-zero EMA stage but not TPU dot-product
rounding inside the first Sylvester solve. The solver promotes its inputs to
float32, but TPU matrix products can still use reduced internal precision unless
the dot precision is explicit. An almost singular polar Hessian is particularly
sensitive to that rounding, and the existing Frobenius clip cannot recover once
the solver has produced a non-finite matrix.

## Changes to make

- Run the polar-Hessian, Sylvester, and inverse matrix products at explicit
  highest precision.
- Form the polar Hessian from one product and its transpose so it is exactly
  symmetric after rounding.
- Treat a non-finite Sylvester result as an unusable correction and use the
  ordinary Muon direction for that layer; use a scaled Frobenius norm when
  clipping finite corrections so the norm calculation cannot overflow first.

## Results

The first warmup retry reached TPU training. The learning-rate 0.004 and 0.006
cells logged finite loss, gradient norms, and parameter norms through global
step 50, then both raised `RuntimeError: Loss is NaN` on the following update.
This pins the failure to the first enabled Sylvester correction rather than the
ordinary Muon warmup. The parent job was cancelled after the two failures so
the remaining cells would not consume scarce v5p capacity with the same code
snapshot.

With explicit highest-precision products and the finite fallback in place, all
21 focused tests pass on the CPU backend. A 100-update bfloat16 stress probe
with a nearly rank-deficient normalized EMA also remained finite throughout;
the first active correction differed from ordinary Muon by a Frobenius norm of
`0.4004`, confirming that the CPU gate did not pass merely by disabling the
correction.

## Gate and full-sweep result

The 100-step TPU gate
`/kaiyuew/muon-error-feedback-300m-hesscorr-stability-gate-flex-20260829-110500`
completed with `global_step=99`, finite loss, and no NaN or traceback. It crossed
the first corrected update at step 51, so the gate exercised the path that had
failed in the preceding warmup retry.

The full stabilized parent
`/kaiyuew/muon-error-feedback-300m-hesscorr-stable-20260828-222720`
then completed all 15 Hesscorr cells on preemptible v5p-8 workers in
`us-central1`. Every run reached `global_step=11443`, wrote final checkpoint
metadata, and produced finite held-out Paloma C4-en BPB. Iris reported zero
failed children and 124 aggregate preemptions.

| Variant | LR 0.004 | LR 0.006 | LR 0.008 | LR 0.010 | LR 0.012 |
|---|---:|---:|---:|---:|---:|
| Muon | 1.067245 | 1.062251 | 1.060028 | 1.059351 | 1.058626 |
| Blend 0.05 | 1.066771 | 1.061070 | 1.059204 | 1.057656 | 1.056606 |
| Hesscorr 0.1 | 1.066853 | 1.062055 | 1.059515 | 1.058882 | 1.057950 |
| Hesscorr 0.3 | 1.067066 | 1.061869 | 1.059530 | 1.058388 | 1.057701 |
| Hesscorr 1.0 | 1.067755 | 1.062778 | 1.060028 | 1.058927 | 1.058036 |

Hesscorr gains `0.1` and `0.3` beat paired Muon in all five learning-rate
comparisons, with mean BPB deltas `-0.000449` and `-0.000589`. Neither beat
blend `0.05` at any learning rate. The best Hesscorr cell, gain `0.3` at
learning rate `0.012`, reached `1.057701`: `0.000925` below Muon and `0.001095`
above blend. Hesscorr native training time was about 90% higher than Muon for
all three gains.

The stabilization fixes the numerical failure, but the quality-per-compute
result does not support scaling this Hesscorr implementation beyond the current
single-seed study. The complete per-run records, checkpoint metadata paths, and
paired summaries are in
`experiments/speedrun/prism_berkeley_qwen3_scaling/muon_error_feedback_results.json`.

A separate v4-8 launch in `us-central2` failed before W&B initialization when
the workers rejected the `us-central1` FineWeb cache. It was not retried and is
not evidence about optimizer stability or quality.
