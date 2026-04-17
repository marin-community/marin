# Debug: TPU Type vs Microbatch Path Discrepancy in Per-Statement DPO

## Goal

Determine why paired per-statement LoRA-DPO runs that appear nominally matched
across `v5p-8` and `v6e-8` produce materially different training curves and
validation metrics.

This note is intended to be stricter and more methodological than
`.agents/logbooks/debug_accum_tpu_type.md`. Claims are separated into:

- **Established in this session**: directly revalidated from code, W&B, or local run logs.
- **Confound / caveat**: a real factor that weakens an argument but does not, by itself, explain the whole effect.
- **Hypothesis**: plausible explanation not yet conclusively demonstrated.

## Scope and Data Sources

This note revalidated the following sources:

- Experiment config:
  - `experiments/posttrain/per_stmt_dpo/common.py`
- Trainer and accumulation code:
  - `lib/levanter/src/levanter/trainer.py`
  - `lib/levanter/src/levanter/grad_accum.py`
  - `lib/levanter/src/levanter/trainer_state.py`
  - `lib/levanter/src/levanter/dpo.py`
  - `lib/levanter/src/levanter/callbacks/__init__.py`
  - `lib/levanter/src/levanter/metrics.py`
- Existing grad-accum test coverage:
  - `lib/levanter/tests/test_grad_accum.py`
- Local copies of W&B run logs under `scratch/wandb_compare/...`
- W&B API queries for config, summary metrics, and early-step histories

This note does **not** claim to have independently revalidated every historical
claim from prior notes across all experiment families. Where a statement is
based only on the revalidated subset below, it is labeled accordingly.

## Revalidated Run Set

### Primary comparison pair originally flagged

- `smh_lr1em06_s140_v6e8-a6a62e`
- `smh_lr1em06_s140_v5p8-cc7957`

### Fresh matched pairs revalidated for early-step behavior

- `smh_lr1em06_s70_v6e8-fbac2a`
- `smh_lr1em06_s70_v5p8-964129`
- `smh_lr1em07_s70_v6e8-4ac830`
- `smh_lr1em07_s70_v5p8-f3d60b`

### Additional matched `exp1a` pairs revalidated at summary level

- `smh_lr5em07_s70_v6e8-7ae68d`
- `smh_lr5em07_s70_v5p8-86109f`
- `smh_lr5em07_s140_v6e8-ddb5ce`
- `smh_lr5em07_s140_v5p8-5f928a`

## Experiment Configuration Facts

### Established in this session

The runs are **not** literally identical at execution level.

From `experiments/posttrain/per_stmt_dpo/common.py:74-93`:

- `v5p-8`: `per_device_parallelism = -1`, `per_device_eval_parallelism = 16`
- `v6e-8`: `per_device_parallelism = 4`, `per_device_eval_parallelism = 4`
- both: `train_batch_size = 64`

From `lib/levanter/src/levanter/trainer.py:897-900`:

- `microbatch_size = per_device_parallelism * data_axis_size` when `per_device_parallelism >= 0`

From `lib/levanter/src/levanter/trainer.py:955-957`:

- `eval_batch_size = per_device_eval_parallelism * data_axis_size`

Given the observed hardware topology:

- `v5p-8` run metadata corresponds to 4 TPU devices
- `v6e-8` run metadata corresponds to 8 TPU devices

Therefore the effective execution differs:

| TPU | `data_axis_size` | `per_device_parallelism` | train microbatch size | global batch | microsteps |
|---|---:|---:|---:|---:|---:|
| `v5p-8` | 4 | 16 (derived from `-1`) | 64 | 64 | 1 |
| `v6e-8` | 8 | 4 | 32 | 64 | 2 |

And eval also differs:

| TPU | `per_device_eval_parallelism` | eval batch size |
|---|---:|---:|
| `v5p-8` | 16 | 64 |
| `v6e-8` | 4 | 32 |

**Conclusion:** even when the global batch size and optimizer hyperparameters
match, the `v6e-8` run uses the microbatched/grad-accum path and the `v5p-8`
run does not.

## Revalidated Summary Metrics

### Fresh `1e-6, s70` pair

| Run | final `train/loss` | final `stmt_val/loss` | final `full_val/loss` |
|---|---:|---:|---:|
| `smh_lr1em06_s70_v6e8-fbac2a` | 0.320073 | 0.401942 | 0.612697 |
| `smh_lr1em06_s70_v5p8-964129` | 0.573534 | 0.600591 | 0.674550 |
| `v5p - v6e` | +0.253461 | +0.198649 | +0.061853 |

### Fresh `1e-7, s70` pair

| Run | final `train/loss` | final `stmt_val/loss` | final `full_val/loss` |
|---|---:|---:|---:|
| `smh_lr1em07_s70_v6e8-4ac830` | 0.362229 | 0.436456 | 0.624924 |
| `smh_lr1em07_s70_v5p8-f3d60b` | 0.683042 | 0.684264 | 0.691427 |
| `v5p - v6e` | +0.320813 | +0.247808 | +0.066502 |

### Additional `exp1a` matched pairs checked at summary level

| Pair key | `v5p train - v6e train` | `v5p stmt - v6e stmt` | `v5p full - v6e full` |
|---|---:|---:|---:|
| `smh_lr5em07_s70` | +0.292749 | +0.223825 | +0.065194 |
| `smh_lr5em07_s140` | +0.239616 | +0.195944 | +0.060763 |

### Established in this session

Across the four matched `exp1a` pairs revalidated in this session, `v5p-8` is
consistently worse than `v6e-8` on:

- final `train/loss`
- final `eval/stmt_val/loss`
- final `eval/full_val/loss`

This is a **systematic pattern in the revalidated subset**. It is not just one
bad run.

## Early-Step Behavior

### Fresh `1e-6, s70` pair

| Step | `v5p-8` train loss | `v6e-8` train loss |
|---|---:|---:|
| 0 | 0.693147 | 0.693147 |
| 1 | 0.693147 | 0.693147 |
| 2 | 0.695091 | 0.467247 |
| 3 | 0.695227 | 0.410079 |
| 4 | 0.690496 | 0.406596 |
| 5 | 0.688349 | 0.386113 |
| 10 | 0.671015 | 0.368791 |

### Fresh `1e-7, s70` pair

| Step | `v5p-8` train loss | `v6e-8` train loss |
|---|---:|---:|
| 0 | 0.693147 | 0.693147 |
| 1 | 0.693147 | 0.693147 |
| 2 | 0.694259 | 0.680598 |
| 3 | 0.695982 | 0.599591 |
| 4 | 0.692704 | 0.526153 |
| 5 | 0.693239 | 0.460644 |
| 10 | 0.691700 | 0.386168 |

### Established in this session

The divergence appears immediately after the first nontrivial update on **fresh**
runs. This matters because it removes checkpoint resume as the primary
explanation.

## Initialization and Step-0 Checks

### Established in this session

For the fresh `1e-6, s70` pair:

- `train/loss` at step 0 is identical: `0.693147`
- `eval/stmt_val/loss` at step 0 is identical: `0.693147`
- `eval/full_val/loss` at step 0 is identical: `0.693147`

Selected step-0 gradient diagnostics are also nearly identical:

- total gradient norm:
  - `v5p-8`: `28.856773`
  - `v6e-8`: `28.854567`
- `transformer.layers.0.self_attn.q_proj.lora.lora_B.weight` grad norm:
  - `v5p-8`: `0.0233466`
  - `v6e-8`: `0.0232554`
- `transformer.layers.0.self_attn.q_proj.lora.lora_A.weight` grad norm:
  - both effectively `0` at step 0, as expected with `zero_init_b=True`

**Interpretation:** the two runs begin from essentially the same initial state
and same first backward pass. The discrepancy emerges after update application,
batch sequencing, or later numerics, not from obviously different
initialization.

## Resume / Preemption Confound

### Confound / caveat

The original motivating `s140` pair is not a perfectly matched fresh-vs-fresh
comparison.

From local downloaded logs:

- `smh_lr1em06_s140_v6e8-a6a62e` starts fresh:
  - `No checkpoints found ...`
  - `No training checkpoint found. Initializing model from HF checkpoint ...`
- `smh_lr1em06_s140_v5p8-cc7957` resumes from checkpoint:
  - `Resuming from step 94, using checkpoint policy weights.`

The resumed `v5p` run also emits many `Unknown leaf type <class 'NoneType'>`
warnings during checkpoint restore before resuming.

### Established in this session

This is a real confound for the `s140` pair, but it is **not sufficient to
explain the overall pattern**, because the fresh `s70` pairs show the same
direction of divergence starting at step 2.

### Methodological consequence

The fresh `s70` pairs should be treated as the cleanest evidence. The resumed
`s140` pair is still useful, but it should not be the flagship example without
an explicit caveat.

## Evaluation Metric Caveat

### Confound / caveat

`eval/*/loss` is not a perfect apples-to-apples cross-hardware metric in these
runs.

From `lib/levanter/src/levanter/callbacks/__init__.py:32-88`:

- `eval_loss_loop` accumulates one scalar `loss.item()` per batch
- then divides by the number of batches `n`

This means it computes an **unweighted mean over batches**, not an example-
weighted mean over examples.

That matters here because eval batch size differs by TPU:

- `v5p-8`: eval batch size `64`
- `v6e-8`: eval batch size `32`

From `lib/levanter/src/levanter/metrics.py:75-82`, `Metric.from_value(..., MEAN)`
stores `_count = 1.0` for each observation. Therefore `eval/*/dpo_loss` has the
same caveat: it is also a mean over batch-level scalar observations, not a
true example-weighted average.

### Established in this session

This caveat is real, but it does **not** explain the main phenomenon by itself:

- step-0 eval values are identical across TPU types
- the most important discrepancy appears in **training loss** immediately after
  the first real update

### Methodological consequence

Eval-loss comparisons should still be reported, but always with a note that
they are batch-size-sensitive under the current callback implementation.

## Reference-Eval Cache Caveat

### Confound / caveat

The reference-eval cache path is not behaving as cleanly as intended.

From `lib/levanter/src/levanter/dpo.py:227-244`, the cache path is determined by
a hash of the reference identity payload.

However, in local logs for both TPU types, the runs try to load the **same**
cache directory and then rebuild it because metadata comparison fails on type
shape:

- `reference_identity['reference']`: stored as `{}` vs compared as `AdapterBaseReferenceConfig()`
- `reference_identity['adapter']`: stored as plain dict vs compared as `LoraAdaptationConfig(...)`

This causes repeated cache misses and rebuilds on both TPUs for the same path.

### Established in this session

This is a real bug / cleanliness issue in the eval preparation path.

### Not established

There is no evidence from this session that this cache issue is the main cause
of the TPU discrepancy:

- initial eval at step 0 still matches exactly
- the training-loss divergence appears before later eval comparisons matter

### Methodological consequence

The cache issue should be fixed, but it should be tracked as a secondary issue,
not as the leading explanation for the training discrepancy.

## What the Existing Grad-Accum Test Does and Does Not Prove

### Established in this session

`lib/levanter/tests/test_grad_accum.py:42-80` verifies that microbatching
matches a full-batch gradient for a simple sharded MLP.

That is useful, but it does **not** fully rule out a bug in this workload.

### Reason

The actual accumulation implementation in `lib/levanter/src/levanter/grad_accum.py:121-166`
includes:

- metric folding
- gradient accumulation through the quantization overwrite helpers
- the real DPO+LoRA parameter tree

The current test does not cover:

- LoRA module structure
- DPO loss path
- the same optimizer state / trainable-mask path
- the same large-model tree shape

### Methodological consequence

It is too strong to say “gradient accumulation is ruled out.” A more accurate
statement is:

> The generic microbatching logic has some test coverage and looks algebraically
> correct, but the specific DPO+LoRA microbatched execution path remains a live
> hypothesis.

## Data-Order Hypothesis

### Hypothesis

Different device topology may lead to different batch composition or batch
ordering even with the same seed.

### What was checked

- `train_dpo.py` derives `data_key` from `trainer.seed` unless `data_seed` is
  explicitly overridden.
- `data_loader.py` was inspected for topology-dependent ordering behavior.

### Established in this session

Code inspection did **not** reveal an obvious topology-dependent reordering bug
that would trivially explain a step-2 divergence. The loader reconstructs
examples by local index before stacking, which reduces the chance that raw
device iteration order alone changes batch contents.

### Not established

We have not yet logged exact example IDs per training step on both TPU types.
So data-order mismatch is still a live hypothesis, just not the strongest one.

## Parameter-Norm Observation

### Established in this session

Selected LoRA parameter norms remain close across TPU types over training in the
fresh `1e-6, s70` pair.

Example for `transformer.layers.0.self_attn.q_proj.lora.lora_B.weight`:

- step 10:
  - `v5p-8`: `0.0011658`
  - `v6e-8`: `0.0011389`
- step 60:
  - `v5p-8`: `0.0065561`
  - `v6e-8`: `0.0064075`

### Interpretation

This makes the discrepancy look less like a trivial “one run is applying
dramatically larger parameter updates everywhere” failure. The runs may differ
in a subtler way than simple update magnitude explosion.

This does **not** prove equivalence of the update path. It only constrains the
shape of the possible bug.

## Ranked Explanations

### 1. Most likely: the microbatched execution path is not behaviorally equivalent enough to the non-microbatched path for this workload

Why this is the current leading explanation:

- it is the clearest, stable execution difference between TPU types
- divergence starts immediately after the first real update on fresh runs
- initialization and step-0 gradients are nearly identical
- the pattern repeats across all four matched `exp1a` pairs revalidated here

This category includes several possible sub-mechanisms:

- a bug in accumulation/reduction for the real DPO+LoRA tree
- a subtle optimizer interaction with microbatched grads
- a topology-dependent numerical effect large enough to matter in this tiny DPO setup

### 2. Plausible but weaker: data-order or batch-composition differences

Why it remains plausible:

- very small preference dataset
- DPO on a single statement can be sensitive to early-example order

Why it is not the current leading explanation:

- no concrete topology-order bug found in code inspection
- effect appears unusually large and immediate

### 3. Secondary: evaluation-path reporting artifacts

Why it matters:

- eval batch sizes differ
- eval loss is averaged per batch, not per example
- reference-eval cache rebuild behavior is messy

Why it is not the main explanation:

- initial eval matches exactly
- training-loss divergence is already present without relying on eval

### 4. Weakest among current candidates: checkpoint resume artifact

Why it matters:

- the `s140` `v5p` run resumed
- restore logs are noisy

Why it is not the root cause:

- fresh `s70` pairs show the same phenomenon

## What Is Established vs Not Established

### Established

- The runs are not literally identical: `v6e-8` uses microbatching, `v5p-8` does not.
- The discrepancy is real, not chart noise.
- The discrepancy appears immediately on fresh runs.
- The discrepancy is not explained by the resumed `s140` run alone.
- Initial train loss, initial eval loss, and initial gradient norms are nearly identical.
- Eval-loss comparisons have a real batch-size-weighting caveat.
- Reference-eval cache metadata handling is unstable and causes rebuilds.
- Existing grad-accum tests do not fully cover the DPO+LoRA path.

### Not yet established

- Whether the root cause is a true bug in `grad_accum.py`
- Whether both TPU types consume exactly the same examples at each training step
- Whether a local or CPU-forced reproduction can isolate the issue without TPU hardware
- Whether the same discrepancy persists if both TPU types are forced onto the same microbatch regime

## Recommended Next Experiments

### Experiment 1: match the microbatch regime on `v5p-8`

Run `v5p-8` with `per_device_parallelism = 8`.

Expected consequence:

- 4 devices × 8 examples/device = microbatch size 32
- global batch 64 => 2 microsteps

This is the cleanest same-hardware test of the “microbatch path vs no
microbatch path” hypothesis because it makes `v5p-8` follow the same
microbatch-size / accumulation-count pattern as `v6e-8`.

### Experiment 2: log example identities for early steps

Instrument the train loader to log stable example identifiers for the first
several global steps on both TPU types.

Goal:

- determine whether step-0 / step-1 / step-2 batches are actually identical

### Experiment 3: add a targeted DPO+LoRA equivalence test

Construct a smaller synthetic DPO+LoRA case and compare:

- full-batch path
- 2-microstep path

The test should exercise:

- LoRA trainable masking
- DPO loss
- optimizer update path
- the same accumulation helpers used in real training

### Experiment 4: fix eval aggregation and reference-cache metadata separately

These should be cleaned up regardless of the root cause because they complicate
interpretation:

- make eval loss example-weighted rather than batch-weighted
- serialize reference identity in a stable way so cache loading is reliable

## Bottom Line

The strongest current conclusion is:

> This does not look like benign TPU-type variance. The most important
> execution-level difference is that `v6e-8` takes the microbatched / gradient-
> accumulation path while `v5p-8` does not, and that difference is the leading
> explanation for the observed divergence.

That is still a **hypothesis**, not a proof of a specific code bug. But it is
substantially stronger than saying “same config, weird TPU behavior.” The runs
are not the same execution, and the divergence begins exactly where that
execution difference first matters.
