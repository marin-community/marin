# Marin 32B Retrospective

> Total tokens trained: ≈6.437T (Phase 1: 2.679T + Phase 3/QK‑Norm: 2.684T + Phase 4/Mantis cooldown: 1.074T; excludes diagnostic restarts and the abandoned Bison cooldown attempt).

<!--

- [Introduction](#introduction)
- [Baseline Configuration](#baseline-configuration)
  - [Architecture](#architecture)
  - [Optimizer & Schedule](#optimizer--schedule)
  - [Data Mix & Batch Schedule](#data-mix--batch-schedule)
- [Training Phases](#training-phases)
  - [Phase 1: Loss Spikes in `exp1295_32b`](#phase-1-loss-spikes-in-exp1295_32b)
  - [Phase 2: Recovery Without Architecture Changes](#phase-2-recovery-without-architecture-changes)
    - ["Necromancy" Restart (`exp1390_32b_necro`)](#necromancy-restart-exp1390_32b_necro)
    - [Alternative Optimizers (`exp1388`, `exp1380`)](#alternative-optimizers-exp1388-exp1380)
  - [Phase 3: Switch to QK-Norm (`exp1395_qwen3_32b`)](#phase-3-switch-to-qk-norm-exp1395_qwen3_32b)
  - [Phase 4: Midtraining Runs](#phase-4-midtraining-runs)
    - [Attempt 1 — Bison Cooldown (`exp1529_32b_bison_cooldown`)](#attempt-1--bison-cooldown-exp1529_32b_bison_cooldown)
    - [Attempt 2 — Mantis Cooldown (`exp1529_32b_mantis_cooldown`)](#attempt-2--mantis-cooldown-exp1529_32b_mantis_cooldown)
- [Lessons Learned](#lessons-learned)
- [Glossary](#glossary) -->


## Introduction

This is a retrospective on Marin 32B, which is largely a scale-up of the 8B recipe. As with 8B, we followed the “Tootsie Roll” playbook: start training, instrument heavily, and make evidence-driven changes mid-flight. The intent here is to document what worked, what failed, and the mechanics of why and how we made changes so that others can learn from our process beyond the final result.

We deliberately reused the Nemotron-centric pretraining mixture and the AdamW-based schedule that behaved well at 8B. That mostly transferred. The notable exception was a loss-instability episode around 70k–80k steps in `exp1295_32b` that we ultimately resolved by introducing QK-Norm via a switch to the Qwen3 32B backbone. Later, during cooldown, we uncovered GSM8k contamination (from a cached Dolmino bundle) and shuffling pathologies from a linear permutation; both issues were addressed in the "Mantis" cooldown with a Feistel shuffle and a cleaner math mix.

If you’re not already reading this on ReadTheDocs, we recommend viewing it there for the right-hand ToC and better navigation.


## Baseline Configuration

### Architecture

- Initial backbone: Llama 32B (Llama‑3–style settings)
- Gradient checkpointing: offload carries to fit large global batches on a v4‑2048 slice

Llama 32B (Phase 1–2) parameters:

| **Parameter**          | **Value** |
|------------------------|-------|
| `seq_len`              | 4096  |
| `hidden_dim`           | 5120  |
| `intermediate_dim`     | 27648 |
| `num_heads`            | 40    |
| `num_kv_heads`         | 8     |
| `num_layers`           | 64    |
| `activation_function`  | `silu`|

Qwen3 32B (Phase 3+) parameters:

| **Parameter**          | **Value** |
|------------------------|-------|
| `seq_len`              | 4096  |
| `hidden_dim`           | 5120  |
| `intermediate_dim`     | 27648 |
| `num_heads`            | 40    |
| `num_kv_heads`         | 8     |
| `num_layers`           | 64    |
| `attention`            | QK‑Norm |
| `activation_function`  | `silu`|

We retained as many choices as reasonable from the 8B model!

### Optimizer & Schedule

Baseline (exp1295_32b) hyperparameters:

| **Hyperparameter**     | **Value** |
|------------------------|-------|
| Optimizer              | AdamW |
| Peak LR                | 7e‑4  |
| LR schedule            | linear warmup → hold → decay (WSD‑style) |
| Warmup                 | 1% of steps |
| Decay                  | 40% of steps |
| Weight decay           | 0.05  |
| Max grad norm          | 0.2   |
| Clip‑update‑norm       | on (σ=2.0, rolling=128); briefly off ~74k–80k |
| EMA beta               | 0.995 |
| z‑loss                 | 1e‑4  |
| Skip bad steps         | true  |

### Data Mix & Batch Schedule

- Pretraining mix: Nemotron‑CC + StarCoderData + ProofPile2 (Mirrors our 8B Phoenix data mix!)

Pretraining mixture (normalized share):

| Dataset                     | Percentage |
|----------------------------|------------|
| nemotron_cc/medium         | 30.69%     |
| nemotron_cc/hq_synth       | 24.70%     |
| nemotron_cc/medium_low     | 13.98%     |
| nemotron_cc/hq_actual      | 8.30%      |
| nemotron_cc/medium_high    | 7.49%      |
| nemotron_cc/low_actual     | 6.37%      |
| nemotron_cc/low_synth      | 5.70%      |
| starcoderdata              | 2.27%      |
| proofpile_2                | 0.50%      |

Batch schedule (seq_len = 4096):

| Start step | Global batch size |
|------------|-------------------|
| 0          | 8192              |
| 18,500     | 7680              |
| 21,010     | 8192              |


We intentionally kept data and schedule close to the 8B setup to isolate scale effects!


## Training Phases

![Training phases. See text for details.](../images/PLACEHOLDER_TIMELINE.png)

### Phase 1: Scaling up our existing recipe

For ~70k steps training behaved as expected though we had some loss spikes that people on Twitter/X told us to be worried about (Yay for open development!). While some other folks told us they looked fine, we added a bunch of clipping on update norms which seemed to mostly handle these spikes!

![Loss Spikes](../images/PLACEHOLDER_SPIKES.png)

Unfortunately, at 80k steps we began to see spikes that were unavoidable even with all the update clipping that we had added. This launched a bit of a fire-drill to see how and whether we could salvage this run!

- Tokens trained: ≈2.679T tokens (80,000 steps; 4096 seq len; batch schedule 0–18,499: 8192, 18,500–21,009: 7680, 21,010–79,999: 8192).

### Phase 2: Recovery Without Architecture Changes

We treated the 80k checkpoint as salvageable and attempted to coax the run back while keeping the Llama backbone.

#### "Necromancy" Restart (`exp1390_32b_necro`)

We  rebuilt optimizer state offline, seeding the warm‑start with update‑history tensors from the last good checkpoint so the clip‑update‑norm filter would have valid rolling statistics. This stabilized gradients for a few thousand steps, but the run relapsed into spikes—a sign the causes of spikes was more core to the model state.

#### Alternative Optimizers (`exp1388`, `exp1380`)

We next swapped optimizers without touching weights! There's been lots of recent work on better conditioned optimizers, including our own empirical validations in our ["Fantastic Optimizers"](https://arxiv.org/abs/2509.02046) paper

- `exp1388_nadamw32b` → NAdamW with β₁ = 0.95 and Nesterov momentum, intended to smooth oscillations while preserving stability at the original LR schedule.
- `exp1380_muon32b` → Muon with a higher effective LR = 2e‑3, but we retained the Adam‑style LR schedule to avoid a full retune.

Both restarts warm‑started from step 77,096 (pre‑spike). Both showed short‑term progress, then inevitable recurrence of loss spikes—consistent with a structural stability gap at 32B rather than a purely optimizer‑choice artifact. Identifying whether these intermediate checkpoints could have been salvaged or identifying what made them unsalvageable would be a great place for community researchers to utilize our intermediate releases!

- Tokens trained: Diagnostic restarts only (short runs of a few thousand steps); excluded from cumulative phase totals due to restart to 80k Llama checkpoint for Phase 3.

### Phase 3: Switch to QK-Norm (`exp1395_qwen3_32b`)

At this point we concluded that stabilizing a 32B Llama without architectural help wasn’t feasible under our constraints. We switched to Qwen3 32B, which adds QK‑Norm in attention, and warm‑started from the 80k Llama weights. This preserved useful signal in embeddings and MLPs while letting the normalized attention heads relearn.

- Prior reports—from [the OLMo team](https://arxiv.org/abs/2501.00656) and [Google DeepMind](https://arxiv.org/abs/2309.14322)—suggest QK‑Norm provides substantial headroom against loss spikes in large models. You might reasonably ask "Why not use QK-norm to begin with?". While we thought about it, we had some hubris from our 8B experience (stable without QK‑Norm) as well as an earlier trial 70B run (also stable without QK-norm). It is worth noting that the Llama 3 team seems to have trained much larger models without QK-Norm, so it seems possible it is not neccesary given the right tuning of underlying hyperparameters, which we continue to explore.

The switch imposed a one‑time loss penalty, but largely mitigated the spikes moving forward.

Warm‑start + rewarm parameters (exp1395_32b):

| **Setting**            | **Value** |
|------------------------|-------|
| Warm start checkpoint  | step‑80,000 (Llama 32B) |
| Cycles                 | [80k, 1,000,000,000] |
| Re‑warmup              | 1,000 steps |

- Tokens trained: ≈2.684T tokens (80,000 steps from 80k → 160k at 4096 seq len, global batch 8192).

### Phase 4: Midtraining Runs

With stability restored, we trained until we completed 1 Epoch over the Nemotron data and then resumed the 8B playbook. The first attempt (*Bison*) mirrored our Starling‑style cooldown and exposed two issues; the second (*Mantis*) fixed them.

#### Attempt 1 — Bison Cooldown (`exp1529_32b_bison_cooldown`)

**Checkpoint:** 160k from the Qwen run.

**Mixture:** 70% Nemotron PT + 30% Starling HQ.

Cooldown mixture (normalized share):

| Dataset                     | Percentage |
|----------------------------|------------|
| nemotron_cc/medium         | 21.49%     |
| nemotron_cc/hq_synth       | 17.29%     |
| nemotron_cc/medium_low     | 9.79%      |
| nemotron_cc/hq_actual      | 5.81%      |
| nemotron_cc/medium_high    | 5.24%      |
| nemotron_cc/low_actual     | 4.46%      |
| nemotron_cc/low_synth      | 3.99%      |
| arxiv_markdownified        | 7.41%      |
| dolmino/pes2o              | 7.41%      |
| finemath-3-plus            | 4.33%      |
| dolmino/flan               | 4.33%      |
| stackexchange_custom       | 2.18%      |
| dolmino/stackexchange      | 2.18%      |
| starcoderdata              | 1.59%      |
| all_math                   | 1.08%      |
| proofpile_2                | 0.35%      |
| wikipedia_markdown         | 0.47%      |
| dolmino/wiki               | 0.47%      |
| medu_science_qa            | 0.15%      |

Cooldown LR schedule:

| Setting           | Value |
|-------------------|-------|
| Warmup            | 0     |
| Decay window      | 160k → 192k (≈16.7% of total) |
| LR schedule       | linear |
| AdamC             | enabled during decay |

**Cooldown specifics (changes vs 8B):**

1. Z‑loss throughout. We previously observed late‑decay divergences at low LR in 8B; adding a small z‑loss stopped `lm_head` norm growth during cooldown.
2. [AdamC](https://arxiv.org/abs/2506.02285) during decay. We used the adjusted weight‑decay formulation only in decay phases to reduce gradient growth without retuning the entire schedule.

**Outcome:** The run was broadly strong versus OLMo 2 32B, with one extreme exception: GSM8k. Under the standard LM Eval Harness prompt, the model was ~22 points worse than the weakest baseline. Under OLMes‑style prompts, performance looked much more reasonable, but the extreme prompt fragility made us investigate further.

**Root cause — Contamination.** Our [Dolmino](https://huggingface.co/datasets/allenai/dolmino-mix-1124/tree/main/data/math) math bundle included GSM8k test items in a `test.json`. Although we later updated preprocessing to drop `test.json`, the dataset had already been cached on the cluster, introducing contamination for GSM8K on the Bison cooldown.

Dolmino’s GSM8k uses OLMes formatting, not LM Eval’s default. Therefore, rather than the expected improvement in results from training on the test, contamination made our model have dramatically increased surprisal on the original prompts structured tags (e.g., `16-8=<<16-7=9>>9`) that don’t appear in the contaminated data. This high surprisal seemed to cause our model to perform much worse than it did on the same questions with those tags removed.


![Shuffling](../images/PLACEHOLDER_SHUFFLE1.png)
**Additional anomaly — Shuffling.** Near 190k steps, training loss phase‑shifted while validation remained stable. This is normally observed when we change the underlying training data mix, but in this case we hadn't! We had separately begun to wonder whether our pseudo-random shuffle, which finds a co-prime step size across data indices, was leading to a somewhat unlucky shuffle where batches came from correlated data. This phase shift in cooldown increased our confidence this was happening!

- Tokens trained: ≈1.074T tokens (32,000 steps from 160k → 192k at 4096 seq len, global batch 8192).

#### Attempt 2 — Mantis Cooldown (`exp1529_32b_mantis_cooldown`)

We restarted from 160k with two major changes:

- Feistel shuffle. Switched from linear permutation to a Feistel‑based epochal shuffle, which re‑randomizes every dataset each epoch and breaks long‑range correlations.
- Cleaner math mix. Replaced Dolmino math with MegaMath splits and later added StackV2 Python (around 174k), redistributing the HQ budget accordingly.

Cooldown mixture (normalized share):

| Dataset                     | Percentage |
|----------------------------|------------|
| nemotron_cc/medium         | 21.49%     |
| nemotron_cc/hq_synth       | 17.29%     |
| nemotron_cc/medium_low     | 9.79%      |
| nemotron_cc/hq_actual      | 5.81%      |
| nemotron_cc/medium_high    | 5.24%      |
| nemotron_cc/low_actual     | 4.46%      |
| nemotron_cc/low_synth      | 3.99%      |
| megamath/web               | 5.57%      |
| arxiv_markdownified        | 4.54%      |
| megamath/text_code_block   | 4.24%      |
| dolmino/pes2o              | 4.54%      |
| megamath/web_pro           | 1.27%      |
| megamath/translated_code   | 0.61%      |
| megamath/qa                | 0.59%      |
| finemath-3-plus            | 2.66%      |
| dolmino/flan               | 2.66%      |
| stackexchange_custom       | 1.34%      |
| dolmino/stackexchange      | 1.34%      |
| starcoderdata              | 1.59%      |
| proofpile_2                | 0.35%      |
| wikipedia_markdown         | 0.29%      |
| dolmino/wiki               | 0.29%      |
| medu_science_qa            | 0.09%      |

Notes:
- At ~174k steps, we introduced `common_pile_stackv2_edu_filtered_python` and re‑normalized the HQ portion accordingly.
- Sampling permutation switched to Feistel.

We kept the optimizer schedule identical to Bison. With better shuffling and clean math, both failure modes disappeared, yielding a clean cooldown target for post‑training.

- Tokens trained: ≈1.074T tokens (32,000 steps from 160k → 192k at 4096 seq len, global batch 8192).

#### Shuffling: Linear vs. Feistel

Within each batch, we want sample examples that are as i.i.d. as possible from the full training distribution. This reduces within‑batch correlation and avoids long, correlated stretches that can bias updates or create non‑stationary “phases” in the loss curve. This also reduces gradient variance from batch to batch, which recent [NanoGPT speedruns](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months) have found beneficial

To achieve this in a reproducible way at runtime, we compute pseudo-random permutations over training data blocks inside of the data loader. We previously used an affine/LCG permutation, choosing integers `a` and `b` with `gcd(a, N) = 1` for dataset length `N`, and mapping indices by `p(x) = (a * x + b) % N`. This is a valid permutation (every index appears exactly once), cheap, and stateless.

The issue is that if our step size is very small and the data is not pre-shuffled, there can be clear phases in our training data!


In Mantis, we switched to a Feistel‑network permutation, another pseudo‑random permutation (PRP) over the index domain. Conceptually, Feistel splits the bit representation into halves and applies several mixing rounds with per‑round keys, yielding a bijection with much better mixing properties than an affine map. Empirically, this resolved the phase shift effect we had seen earlier.

![Shuffling](../images/PLACEHOLDER_SHUFFLE2.png)

## Base Model Results

We evaluate with LM Eval Harness defaults across a standard suite. Numbers may differ from model cards or OLMES due to prompt/format differences. “Average” is a simple mean over shown tasks.

| Model                                | Average | AGI Eval LSAT-AR | ARC Easy | ARC Challenge | BoolQ | CommonSense QA | COPA | HellaSwag | lambada_openai | OpenBookQA |  PIQA | WinoGrande |   WSC |  MMLU |  GPQA |   BBH | MMLU Pro | HumanEval | GSM8K |  MATH |
| :----------------------------------- | ------: | ---------------: | -------: | ------------: | ----: | -------------: | ---: | --------: | -------------: | ---------: | ----: | ---------: | ----: | ----: | ----: | ----: | -------: | --------: | ----: | ----: |
| **Marin 32B (Bison)**                |    63.0 |             23.4 |     87.8 |          65.8 |  88.9 |           82.3 | 94.0 |      86.6 |           77.4 |       46.6 |  86.1 |      78.61 | 82.42 |  72.9 | 32.13 |  55.2 |     41.9 |     29.27 | 54.71 | 10.35 |
| **Marin 32B (Mantis)**               |    65.2 |             24.8 |     88.0 |          65.7 |  89.4 |           82.8 | 93.0 |      86.9 |           77.2 |       46.4 |  85.9 |       79.3 |  79.5 |  74.7 |  34.0 |  59.6 |     45.1 |      42.7 |  69.1 |  15.3 |
| **OLMo 2 32B Base**                  |    63.2 |             22.6 |     85.9 |         61.86 |  83.0 |           78.6 | 93.0 |      85.9 |           78.3 |       47.2 | 83.08 |      78.85 | 86.81 | 71.85 | 32.21 | 56.07 |     42.0 |     23.78 | 76.35 | 12.69 |
| **Qwen 2.5 32B Base**                |    68.1 |            30.43 |    80.81 |         55.89 | 87.65 |          88.45 | 87.0 |     84.11 |          77.62 |       44.4 |  82.4 |       75.7 | 80.95 | 80.83 | 39.01 | 67.35 |     57.9 |     48.78 | 89.31 | 36.25 |
| **Gemma 3 27B PT**                   |    65.1 |            22.17 |    88.17 |         65.44 | 87.09 |          73.38 | 93.0 |     83.02 |          78.07 |       45.0 | 84.06 |      79.01 | 91.94 | 75.33 | 35.74 | 61.36 |    49.44 |      17.6 | 82.03 | 25.83 |
| **NVIDIA Nemotron Nano 12B v2 Base** |    68.6 |             28.7 |    83.59 |         60.58 | 84.83 |          76.09 | 85.0 |     81.42 |          72.93 |       45.8 | 82.81 |      74.35 | 85.35 |  77.9 | 36.58 | 62.02 |    53.13 |     59.15 | 84.08 | 68.28 |


## Glossary

- **AdamC** — An adjusted weight‑decay computation used during LR decay to reduce gradient growth.
- **Cooldown** — A training phase with gradually decreased LR, often paired with a higher‑quality, more structured data mix.
- **EMA (Exponential Moving Average)** — Weight averaging used for evaluation to reduce variance from hot parameters.
- **Feistel shuffle** — A deterministic, epoch‑wise re‑randomization scheme that avoids the autocorrelation of simple linear permutations.
- **QK‑Norm** — Normalization of query/key vectors in attention; improves headroom against spikes at large scales.
- **Tootsie Roll process** — Marin’s pragmatic strategy: start quickly, keep training, fold in changes mid‑flight as evidence accumulates.
- **Z‑loss** — A small penalty on the logit norm to prevent `lm_head` explosions during deep cooldowns.
