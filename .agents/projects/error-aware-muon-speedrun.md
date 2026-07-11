# Error-aware Muon in the Qwen3 130M speedrun

## TL;DR

Add `blend` and `hesscorr` Muon policies to the standalone Qwen3 speedrun from PR #4933. Both policies use a normalized momentum EMA and the constant-coefficient five-step Muon iteration. `hesscorr` differentiates a separate convergent cubic iteration with `jax.jvp`. The 130M experiment crosses five archived learning rates with the handoff's nonzero feedback gains, plus one Muon baseline, for 40 runs. The completed sweep used 30 cubic steps for `hesscorr`; gain `0.1` improved C4-en BPB at four of five paired learning rates.

## Problem

Library Muon stores `beta * momentum + gradient`. Matrix sign is scale-invariant, so this is equivalent to a normalized EMA for plain Muon. It is not equivalent for error feedback because both policies use `gradient - momentum`.

The fast Muon iteration also cannot provide the Hessian correction. Its constant coefficients `(3.4445, -4.7750, 2.0315)` are deliberately non-convergent, and its JVP does not approach the nuclear-norm Hessian.

PR #4933 contains the Qwen3 model, dataset, resource, schedule, and result-collection setup needed for a 130M speedrun. Its checked-in script replays selected PRISM-Berkeley runs; it does not launch a new optimizer sweep.

## Approach

`muon_error_feedback_optimizer.py` adds an experiment-local `ErrorAwareMuonConfig`. Hidden linear weights with both flattened dimensions at least 8 use error-aware Muon. Embeddings, `lm_head`, small matrices, biases, norms, and other leaves use AdamW.

For a raw gradient `G`, the optimizer stores `M = beta * M_prev + (1 - beta) * G` in float32. The exact handoff path uses `nesterov=False`. The step policies are:

```python
if policy == "blend":
    step = quintic_ns(M + blend_gain * (G - M), steps=5)
elif policy == "hesscorr":
    base = quintic_ns(M, steps=5)
    cubic = lambda value: cubic_ns(value, steps=cubic_steps)
    _, correction = jax.jvp(cubic, (M,), (G - M,))
    cap = sqrt(min(M.shape))
    correction *= min(1, cap / max(fro_norm(correction), eps))
    step = base + correction_gain * correction
```

The quintic and cubic paths run in float32 for training. Tall matrices are transposed during both iterations so the Gram matrix uses the smaller dimension. Speedrun aspect-ratio scaling is applied after the complete policy step. Weight decay and learning-rate scaling remain outside the policy transform.

`muon_error_feedback_sweep.py` reuses the archived 130M Qwen3 geometry, FineWeb-Edu cache, v5p-8 resources, batch size 128, sequence length 4096, and 4,959 steps. It uses learning rates `{0.008, 0.012, 0.016, 0.020, 0.024}` and keeps Adam LR at `0.2 * muon_lr`, matching the archived W&B runs. The handoff gain grid becomes eight unique variants after zero-gain deduplication: one Muon baseline, four nonzero `blend` gains, and three nonzero `hesscorr` gains.

The algorithm intentionally differs from the archived PRISM optimizer in two places: Nesterov is disabled because the handoff defines a plain EMA, and the Muon epsilon is `1e-12` instead of `1e-5` to match the reference.

## Tests

`tests/test_muon_error_feedback_speedrun.py` checks:

- cubic matrix-sign values and JVPs against an independent SVD oracle for tall and wide matrices;
- constant-coefficient quintic parity with the supplied reference;
- the negative control showing that the quintic JVP is not the nuclear-norm Hessian;
- exact zero-gain reduction to Muon and direct `blend` policy behavior;
- finite, JIT-safe Hessian clipping at the required Frobenius cap;
- normalized float32 momentum state with bfloat16 updates;
- parameter routing and all 40 unique 130M sweep configurations.

The focused suite covers the new optimizer, completed result grid, and the existing PR #4933 submission path.

## Launch configuration

Running the experiment entry point submits 40 v5p-8 training runs and 40 result steps. The handoff does not define one nonzero gain, so the graph retains its full deduplicated gain grid.

The handoff fixes the cubic iteration count at 15. A float32 probe on a seeded random `512 x 512` matrix produced matrix-sign relative error `0.111` and Hessian-JVP relative error `0.620` at 15 steps. At 30 steps those errors fell to `5.2e-7` and `1.4e-6`. Clipping the exact and approximate corrections to `sqrt(512)` left relative error `0.650` and cosine similarity `0.789`, so the guard does not repair the direction. The sweep therefore sets `cubic_steps=30`; the optimizer API retains the handoff-faithful default of 15.

Each archived 130M run processes 2.60B tokens and used 4 TPU chips for 4,621 seconds. Forty runs are at least 104B tokens and about 205 chip-hours before accounting for the extra cubic JVP work.

## Completed results

All 40 training runs and all 40 result steps succeeded. Hesscorr gain `0.1` won four of five paired learning-rate comparisons against fresh Muon, with mean C4-en BPB delta `-0.000587`. The best run used learning rate `0.020` and reached `1.164666`, compared with `1.165484` for Muon at the same learning rate. Native speedrun training time increased by 4.33% on average across the five paired learning rates.

The full grid, baseline caveats, and next experiment are recorded in [`docs/reports/error-aware-muon-speedrun.md`](../../docs/reports/error-aware-muon-speedrun.md). The completed executor manifest is `gs://marin-us-central1/experiments/muon_error_feedback_sweep-d76bb7.json`.
