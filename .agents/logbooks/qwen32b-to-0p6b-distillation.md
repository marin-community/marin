---
topic: qwen32b-to-0p6b-distillation
issue: https://github.com/marin-community/marin/issues/7656
description: Compare five methods for transferring Qwen 32B into a 0.6B student.
author: rjpower
---

# Qwen 32B to 0.6B distillation: task logbook

## Scope

- Goal: measure the tokens, wall time, and GB200 GPU-hours required to transfer a 32B Qwen teacher into a 0.6B student.
- Primary metrics: held-out hard-label next-token NLL; time and tokens to fixed NLL thresholds.
- Diagnostics: teacher-student forward KL, tokens/s, teacher and student FLOPs, peak device memory, and fixed zero-shot evaluations.
- Constraints: Levanter training; Datakit data resident in CoreWeave storage; `cw-us-east-08a`; batch priority.
- Coordinating issue: https://github.com/marin-community/marin/issues/7656
- Series ID: `QD`.

## Current TL;DR

The local scaffold and all nine experiment arms are implemented and pass 12 focused behavior tests plus Pyrefly. No cluster run has launched yet; the next gate is a versioned commit followed by the regional data and one-node smoke stages. Qwen does not publish `Qwen3.5-32B` or `Qwen3.5-0.6B`. The exact requested sizes are `Qwen3-32B` and `Qwen3-0.6B`, so those are the working teacher and student.

## Current baseline

- Control `QD-C0`: scratch-initialized student trained with hard-label next-token loss.
- KD control `QD-C1`: the same initialization and training setup with full-vocabulary teacher-to-student forward KL.
- Practical controls `QD-C2` and `QD-C3`: the official `Qwen3-0.6B-Base` checkpoint trained with hard labels and forward KL, respectively.
- Student: the `Qwen3-0.6B-Base` architecture, scratch-initialized for `QD-C0`, `QD-C1`, `QD-001`, `QD-003`, and `QD-005`.
- Teacher: the official post-trained `Qwen3-32B` checkpoint; Qwen publishes no 32B Base checkpoint.
- Data, optimizer, batch, sequence length, seeds, and evaluation shards stay fixed across arms.

## Hypothesis queue

### Active

- `QD-001`: Projected hidden-state loss on every student block, mapped uniformly into the 64-layer teacher, improves early held-out NLL over logits-only KD. A four-anchor variant is the throughput ablation because direct decoder-LLM evidence for dense layer matching is weak.
- `QD-002`: Low-rank factorization followed by coordinate-preserving teacher submatrix selection gives a step-0 and early-training advantage over scratch initialization.
- `QD-003`: TAID's adaptive interpolation reduces the optimization penalty from the roughly 50-fold teacher/student capacity gap.
- `QD-004`: Weight-saliency-ranked, coordinate-preserving depth/width inheritance retains more useful function than independent per-weight factorization.
- `QD-005`: A 4B teacher beats the 32B teacher at equal student tokens because reducing the teacher/student capacity gap outweighs the weaker target distribution.

### Falsified / Dead End

- Echo corpus search: the configured database identity can connect but lacks permission to read the `chunks` table, so no team-history evidence was recovered from Echo.
- Naive per-matrix SVD as the sole initialization: prior work shows weight reconstruction error is poorly aligned with task loss and fails at compression ratios far smaller than this one. Keep only an activation-weighted version.
- Top-k cached teacher logits: Minitron found no benefit over full logits at large `k` and degradation at `k <= 100`; dense caching also creates a storage bottleneck.

### Promoted

None.

## Decision log

- 2026-07-26: Use held-out hard-label NLL as the primary quality metric. Teacher KL is diagnostic because teacher agreement can improve while generalization worsens.
- 2026-07-26: Use online, vocab-sharded teacher loss. Do not persist dense teacher logits or all teacher hidden states.
- 2026-07-26: Screen every arm for a fixed 100M tokens with two seeds and controller evaluation every approximately 12.5M tokens. Do not quality-stop arms before the common endpoint.
- 2026-07-26: Promote a treatment only if its two-seed mean final NLL is at least 0.5% below `QD-C1`, both seeds improve directionally, and the predefined zero-shot suite stays within task tolerances. Add a third seed near the threshold.
- 2026-07-26: Extend promoted treatments and all four controls to approximately 1.8B tokens. The shared token cap, not a validation plateau, is the primary stopping condition.
- 2026-07-26: Rank structured inherited coordinates by teacher weight energy. Activation-derived rotations do not commute through Qwen's RMSNorm and SwiGLU blocks without a more invasive model rewrite.
- 2026-07-26: Make all 28 mapped student layers the primary `QD-001` treatment to test the user's per-layer hypothesis directly. Retain four anchors only as a throughput ablation.
- 2026-07-26: Decouple the 12-step systems smoke from the full 100M-token cache with a fixed 1,000-document cache. Continue the full cache in parallel and pin its successful `2026.07.26.2` artifact for every screen arm.

## Negative results index

- Echo ACL failure: [initial forage](#2026-07-26-1525---initial-forage).
- Naive SVD and top-k logits rejected from the primary matrix: [background research brief](#background-research-brief).

## Background research brief

- Effort: high.
- Stop rule: stop when additional primary sources no longer change the ranked experiment slate.
- Date: 2026-07-26.

### Current Marin context

Levanter has native Qwen3 configuration and Hugging Face checkpoint conversion in `lib/levanter/src/levanter/models/qwen.py`. It does not have a Qwen3.5 model. Haliax `Stacked.scan_via` can emit mapped hidden states without changing the standard model forward. Levanter's trainable filter can carry a frozen teacher in the jitted model while excluding it from optimizer state and checkpoints.

`cw-us-east-08a` is configured as a GB200 NVL72 fleet. The requested B200 allocation therefore resolves to `GB200x4` worker nodes on this cluster.

### External prior art

- Minitron transfers structured subsets of a larger decoder model, then recovers quality with forward-KL distillation. Its ablations favor forward KL at temperature 1, find no gain from attention-relation loss, and find no gain or degradation from many mapped hidden layers. Source: https://arxiv.org/abs/2407.14679
- TAID interpolates student and teacher logits and reports gains over fixed forward/reverse KL in large capacity-gap settings without requiring student-generated data. Source: https://arxiv.org/abs/2501.16937
- Teacher-Assistant KD reports that an oversized teacher can produce a worse small student than an intermediate teacher. Source: https://arxiv.org/abs/1902.03393
- TinyBERT provides the projected hidden-state objective `MSE(H_student W_h, H_teacher)` and uniform layer mapping used by `QD-001`. Source: https://arxiv.org/abs/1909.10351
- SliceGPT uses activation-derived rotations before structured width deletion, supporting the activation-weighted basis in `QD-004`. Its tested compression ratio is much smaller than 32B→0.6B. Source: https://arxiv.org/abs/2401.15024
- Fisher-weighted SVD shows that naive weight reconstruction is misaligned with task loss; vanilla SVD collapses without fine-tuning at much smaller compression ratios. Source: https://arxiv.org/abs/2207.00112
- Generalized KD shows mixed on-policy data can improve sequence-level distillation, but it adds generation cost and is reserved for follow-up work. Source: https://arxiv.org/abs/2306.13649

### Evidence map

#### Claim: full-logit forward KL is the correct common KD control

- Support: Minitron's decoder-LLM ablation reports forward KL ahead of reverse KL, temperature 1 ahead of larger temperatures, and no gain from large top-k approximations.
- Contradiction: TAID and on-policy GKD report better results in other model/data regimes.
- Directness to Marin: high for architecture and objective; moderate for the larger 32B→0.6B capacity gap.
- Confidence: exploratory until `QD-C1` runs.
- Action: use forward KL for all initialization and hidden-state arms; compare TAID separately.

#### Claim: all-layer hidden matching should be an ablation

- Support: TinyBERT and Patient KD find intermediate-layer supervision useful for encoder models.
- Contradiction: Minitron finds multiple mapped decoder layers neutral or harmful and only a small gain from the last two layers.
- Directness to Marin: Minitron is the closer regime.
- Confidence: exploratory.
- Action: compare selected upper layers with all 28 uniformly mapped student layers; promote only on held-out NLL and throughput.

#### Claim: teacher-derived initialization can reduce recovery tokens

- Support: Minitron, Sheared LLaMA, SliceGPT, and DistilBERT retain useful function through inherited structure or activation-derived bases.
- Contradiction: naive SVD fails even at moderate compression without recovery training.
- Directness to Marin: moderate; the requested compression ratio is much larger.
- Confidence: exploratory.
- Action: test step-0 NLL and first-128-step learning curves before paying for long runs.

### Recommended experiment stages

1. Validate teacher/student HF parity, frozen-teacher checkpoint filtering, online forward KL, and mapped hidden-state memory on one `GB200x4` node.
2. Run `QD-C0` through `QD-C3` and `QD-001` through `QD-005` for 100M tokens on the same data order with two seeds.
3. Add a third seed to treatments near the promotion threshold.
4. Extend promoted treatments and all controls to approximately 1.8B tokens.
5. Evaluate final checkpoints with a fixed small zero-shot suite and report tokens, wall time, teacher FLOPs, student FLOPs, GPU-hours, NLL, and teacher KL.

### Source ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Minitron | paper | https://arxiv.org/abs/2407.14679 | structured inheritance and decoder-KD ablations | high | Closest objective/architecture evidence |
| TAID | paper/code | https://arxiv.org/abs/2501.16937 | adaptive capacity-gap curriculum | medium | Different model/data regime |
| Teacher Assistant KD | paper | https://arxiv.org/abs/1902.03393 | capacity-gap control | medium | Classifier evidence, supported by MiniLM |
| TinyBERT | paper | https://arxiv.org/abs/1909.10351 | hidden projection and layer mapping | medium | Encoder evidence |
| SliceGPT | paper/code | https://arxiv.org/abs/2401.15024 | activation-derived basis | medium | Smaller compression |
| Fisher-weighted SVD | paper | https://arxiv.org/abs/2207.00112 | reject naive SVD | medium | Smaller encoder model |
| Levanter Qwen3 | Marin code | `lib/levanter/src/levanter/models/qwen.py` | native teacher/student architecture | high | Qwen3, not Qwen3.5 |
| Haliax scan | Marin code | `lib/haliax/src/haliax/nn/scan.py` | hidden-state scan implementation | high | Existing reusable API |

## Entry log

### 2026-07-26 15:25 - Initial forage

- Hypothesis: existing Levanter primitives and regional data handles can support online distillation without storing teacher outputs.
- Commit hash: not yet committed.
- Commands: `rg` over `experiments/`, `lib/levanter/`, `lib/marin/`, and `docs/`; GitHub issue/PR search; official Qwen model-card search; Echo semantic and exact search.
- Config: read-only repository and external-source pass.
- Result: Qwen3 matches the requested sizes and is supported; Qwen3.5 does not match the names/sizes and is unsupported. No existing Marin LM distillation trainer or duplicate issue was found. Echo failed with `permission denied for table chunks`.
- Interpretation: proceed with a Qwen3 design while requesting model confirmation. Build online teacher loss around existing Haliax scan and Levanter trainable filters.
- Next action: verify regional Qwen-tokenized data, write and peer-review the design, then implement a one-node scaffold smoke.

### 2026-07-26 17:05 - Design review

- Hypothesis: a common online-teacher scaffold can compare the five treatments without method-specific data, stopping, or evaluation confounds.
- Commit hash: not yet committed.
- Commands: inspected official TAID code and Qwen model metadata; ran two context-isolated design reviews; revised the experiment matrix and stopping rule.
- Config: four controls, five treatments, 100M-token two-seed screen, fixed 1.8B-token promotion endpoint.
- Result: the reviewers found that optimizer filtering alone would still trace the teacher backward graph, dense activation rotations would not commute through Qwen nonlinearities, an independent 4B-to-0.6B run is not a teacher-assistant cascade, and the original single-seed step screen could select noise. The design now detaches the complete teacher result, uses coordinate-preserving weight-saliency initializers, labels the 4B run as a capacity-gap control, and uses paired fixed-token screens.
- Interpretation: the revised design tests the intended mechanisms while keeping treatment claims narrower than the evidence.
- Next action: publish the reviewed design and plan, then implement loss/filtering behavior tests before the one-node smoke.

### 2026-07-26 19:20 - Scaffold implementation

- Hypothesis: one online-teacher Levanter entry point can express the five treatments while keeping data order, optimizer, stopping, and evaluation identical.
- Commit hash: pending snapshot.
- Commands: focused Pytest suite; Pyrefly over the new trainer, objectives, initializers, Marin integration, and experiment driver; dry Marin graph resolution for the smoke stage.
- Config: full-logit forward KL, four selected residual anchors, rank-512 factorized initialization, weight-saliency structured initialization, official TAID controller, and 32B/4B teacher selection.
- Result: 12 focused behavior tests pass and Pyrefly reports zero errors. The dry graph resolves the regional Datakit source, Qwen tokenizer cache, pinned 32B and 0.6B checkpoints, and a 12-step `GB200x4` smoke.
- Interpretation: the CPU-level behavior and Marin graph are internally consistent. Device memory, exact sharding, checkpoint resume, and regional object-store access remain empirical smoke gates.
- Next action: create and push the immutable code snapshot, materialize the Qwen-tokenized cache on `cw-us-east-08a`, and run the one-node smoke.

### 2026-07-26 16:35 - Regional tokenizer staging failure

- Hypothesis: the pinned regional Qwen model artifact can also supply tokenizer files to Datakit workers without another model download.
- Commit hash: `d3c5b9de06`.
- Job: `/power/qwen-distill-data-d3c5b9` on `cw-us-east-08a`, batch priority.
- Result: the 0.6B checkpoint staged successfully, but every tokenize shard failed before reading data. `load_tokenizer` interpreted the `s3://.../models/Qwen...` directory as a Hugging Face repository ID.
- Interpretation: tokenizer staging supported local paths, the cross-region mirror, and Hugging Face IDs, but not explicit remote model directories. This is a boundary bug rather than a dataset failure.
- Next action: teach tokenizer staging to copy only tokenizer files from explicit S3/GCS directories, cover the path with behavior tests, then resubmit with a new immutable artifact version.

### 2026-07-26 17:05 - Full-batch systems smoke OOM

- Hypothesis: four-way vocabulary and model sharding is sufficient for an exact online-KL step at sequence length 2,048 and batch size 8.
- Commit hash: `651629ce09`.
- Job: `/power/qwen-distill-smoke-651629` on one `GB200x4` node.
- Result: data and all 17 teacher shards loaded, W&B initialized, and the train step traced and lowered. First execution failed on every device while allocating another 3.91 GiB; no step or checkpoint completed.
- Interpretation: the failure is a compiled-step peak rather than staging, scheduler, or HLO-lowering failure. Preserve the effective batch and exact objective, but split it into two microbatches of four.
- Next action: resubmit the 12-step smoke with `microbatch_size=4`; reduce to 2 only if the measured peak still exceeds capacity.

### 2026-07-26 17:20 - Microbatch smoke isolates retained initialization

- Hypothesis: reducing the compiled microbatch from eight to four will lower the failing allocation.
- Commit hash: `33f4237e66`.
- Job: `/power/qwen-distill-smoke-33f423` on one `GB200x4` node.
- Result: the run failed on the same 3.91 GiB allocation at the same point after loading all teacher shards. Microbatching did not change the allocation.
- Interpretation: the peak is independent of batch activations. The distillation entry point retains its original concrete random 32B `initial_model` after `Trainer.initial_state` creates a mixed-precision state and while the checkpoint teacher is loaded.
- Next action: pass a lazy model factory into the trainer so the original concrete teacher is not retained, keep microbatch size four, and repeat the smoke.

### 2026-07-26 17:30 - Lazy initialization clears the device smoke

- Hypothesis: constructing the 32B initialization only inside `Trainer.initial_state` will release it before the checkpoint teacher is loaded and remove the batch-independent 3.91 GiB allocation.
- Commit hash: `f8672e5abd`.
- Job: `/power/qwen-distill-smoke-f8672e` on one `GB200x4` node.
- Config: exact forward KL, sequence length 2,048, effective batch 8, microbatch 4, 12 steps.
- Result: the task completed all 12 steps, logged finite training loss, and saved `step-11`. The allocator OOM disappeared. Validation metrics were nonfinite because padded zero-weight positions used multiplication rather than explicit masking.
- Interpretation: the online 32B-to-0.6B system fits one `GB200x4` node. Evaluation must mask ignored positions before accumulation so undefined losses on padding cannot contaminate validation aggregates.
- Next action: land the evaluator regression fix and repeat the smoke, requiring finite validation loss and a resumable checkpoint before starting the paired screen.

### 2026-07-26 17:46 - Nonfinite validation isolated to fused unreduced NLL

- Hypothesis: nonfinite validation came from multiplying undefined padded losses by zero in the evaluator.
- Commit hash: `c3903335cb`.
- Job: `/power/qwen-distill-smoke-c39033` on one `GB200x4` node.
- Result: explicit masking did not change validation. The run again completed and checkpointed; step-11 distillation loss was `9.1567`, student gradient and parameter norms were finite, and steady-state throughput reached 27,515 tokens/s. Only the unreduced fused hard-label NLL was nonfinite.
- Interpretation: the student is not corrupted. Replace the validation-only fused linear cross-entropy with the stable identity `logsumexp(logits) - logits[target]`, using the same full-logit shape already validated by online KL.
- Next action: test and device-smoke the stable validation NLL, then run an explicit checkpoint continuation before launching the screen.

### 2026-07-26 17:51 - Systems smoke passes

- Hypothesis: materialized float32 validation logits avoid the nonfinite unreduced fused cross-entropy result without changing the hard-label NLL definition.
- Commit hash: `8dd74b043b`.
- Job: `/power/qwen-distill-smoke-8dd74b` on one `GB200x4` node.
- Result: all 12 training steps and every evaluation completed. Validation NLL decreased monotonically at the observed points from `11.210` to `8.679`; the final checkpoint committed at step 11.
- Interpretation: online Qwen3-32B-to-0.6B KL, model sharding, microbatch accumulation, held-out NLL, and checkpoint serialization are device-valid. Both hard-label controls now select the same materialized NLL implementation.
- Next action: launch the 18 paired 100M-token screen runs at batch priority and monitor each arm through its first evaluation and terminal checkpoint.

### 2026-07-26 18:04 - Screen startup recovery

- Hypothesis: the six failures in the first screen are configuration-boundary failures and can be retried without invalidating the 12 cells already training.
- Commit hash: pending snapshot.
- Job: `/power/qwen-distill-screen-792acc` on `cw-us-east-08a`, batch priority.
- Result: 12 online-distillation cells entered training. `QD-C0` and `QD-C2`, both seeds, failed before step 0 because checkpoint vocabulary padding attempted to convert a regional tokenizer directory back into a Hugging Face tokenizer. `QD-003`, both seeds, failed before step 0 because TAID rejects gradient accumulation.
- Interpretation: size the standard model vocabulary from the checkpoint configuration without changing the tokenizer. Run TAID with a full batch so its controller receives one loss update per optimizer update. Keep microbatch size four for the other online objectives.
- Next action: snapshot the fixes and launch the six-cell `screen-retry` stage at version `2026.07.26.9`; continue monitoring the original 12 cells.

### 2026-07-26 18:12 - Hard-label training loss recovery

- Hypothesis: the materialized float32 next-token loss that fixed validation will also avoid the fused training-kernel NaN without changing the hard-label objective.
- Commit hash: pending snapshot.
- Job: `/power/qwen-distill-screen-retry-3ef219` on `cw-us-east-08a`, batch priority.
- Result: both TAID cells completed full-batch updates at approximately 30,000 tokens/s. All four hard-label controls passed tokenizer and vocabulary setup, then failed their first optimizer update with `Loss is NaN`.
- Interpretation: the failure is confined to the fused linear cross-entropy path shared by training and the earlier nonfinite validation path. Add an explicit training-loss implementation and use weighted materialized float32 NLL for `QD-C0` and `QD-C2`.
- Next action: test the scalar weighted loss, snapshot it, and launch only the four CE controls as `screen-ce-retry` at version `2026.07.26.10`.
