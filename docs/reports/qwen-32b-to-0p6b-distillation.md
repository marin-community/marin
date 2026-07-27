# Distilling Qwen3 32B into 0.6B

## Summary

We tested five ways to transfer a 32B Qwen teacher into a 0.6B student, using four controls to separate the effects of
teacher supervision and initialization. The screen covered 100 million tokens with two seeds per arm. We promoted only
factorized teacher initialization to a 1.8 billion-token extension because it was the only research treatment that beat
scratch forward KL in both paired seeds and passed the zero-shot regression gate.

Initialization mattered, but the source and training horizon mattered more than the screen suggested. Factorized
teacher initialization improved mean NLL over scratch forward KL by 1.13% at 225 million tokens and 1.47% at 450 million
tokens. The advantage then narrowed and reversed: at the fixed 1.8 billion-token endpoint, factorization finished 0.39%
worse, with both paired seeds worse. Starting from the official Qwen3 0.6B Base checkpoint and training with forward KL
was the strongest extended arm, reaching mean NLL 2.5579, 3.63% below scratch forward KL. We recommend official Base plus
forward KL when that checkpoint provenance is acceptable, and plain forward KL from scratch otherwise. Factorization is
useful only when the budget is short enough to value its transient head start.

## Scope

The requested `Qwen3.5-32B` teacher and 0.6B student are not an available dense Qwen3.5 pair. We therefore used the
architecture-compatible [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B) teacher and
[`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) student. This substitution keeps the requested
32B-to-0.6B compression ratio and permits exact teacher-to-student Qwen weight mappings, but the results should not be
treated as measurements of Qwen3.5.

The model revisions are pinned to `9216db5781bf21249d130ec9da846c4624c16137` for Qwen3 32B,
`da87bfb` for Qwen3 0.6B Base, and `1cfa9a7` for the Qwen3 4B capacity-gap arm.

The study asks two questions:

1. Can teacher-derived initialization reduce the tokens needed for a scratch 0.6B student to reach a fixed validation
   loss?
2. Do intermediate objectives or a smaller teacher improve enough to offset their systems cost?

The experiment code, issue, and run group are:

- [`experiments/qwen_distillation.py`](https://github.com/marin-community/marin/blob/main/experiments/qwen_distillation.py)
- [Experiment issue #7656](https://github.com/marin-community/marin/issues/7656)
- [W&B group](https://wandb.ai/marin-community/marin/groups/QD-qwen32b-to-0p6b)

## Methods

All students use the Qwen3 0.6B architecture: 28 layers, hidden width 1,024, MLP width 3,072, 16 query heads, and 8 KV
heads. The 32B teacher has 64 layers and hidden width 5,120. Except for the 4B capacity-gap arm, online objectives use the
same frozen 32B teacher and exact full-vocabulary forward KL.

| ID | Method | Initialization | Training objective |
|---|---|---|---|
| QD-C0 | Scratch hard labels | Random | Next-token NLL |
| QD-C1 | Scratch logit KD | Random | Forward KL from 32B |
| QD-C2 | Base hard labels | Official 0.6B Base | Next-token NLL |
| QD-C3 | Base logit KD | Official 0.6B Base | Forward KL from 32B |
| QD-001 | Per-layer activation KD | Random | 90% forward KL + 10% projected hidden loss |
| QD-002 | Factorized initialization | Rank-512 teacher factorization | Forward KL from 32B |
| QD-003 | TAID | Random | Adaptive student-to-teacher interpolation |
| QD-004 | Structured initialization | Teacher weight-saliency selection | Forward KL from 32B |
| QD-005 | Capacity-gap control | Random | Forward KL from 4B |

### Per-layer activation distillation

QD-001 captures all 28 student residual streams and maps them to 28 evenly spaced points in the teacher's 64 blocks. A
learned linear projector maps the student width from 1,024 to the teacher width of 5,120. The hidden objective is mean
cosine distance after RMS normalization, scored at the same token positions as the language-modeling objective. This
avoids requiring equal hidden widths while retaining layer-by-layer supervision.

### Factorized initialization

QD-002 first maps every student layer, embedding coordinate, attention head, and MLP coordinate to evenly spaced teacher
coordinates. It then applies a rank-512 randomized low-rank approximation to each student-shaped embedding, attention, and
MLP matrix. Each reconstructed row is rescaled to the source RMS. This is a teacher-derived initialization, not a
low-rank student architecture: training updates the ordinary dense 0.6B model.

### Additional research treatments

QD-003 uses [TAID](https://arxiv.org/abs/2501.16937), beginning from a target closer to the student's current logits and
adapting toward the teacher during training. The hypothesis was that a gentler target would reduce the 53× capacity gap.

QD-004 is a structured alternative to factorization. It chooses teacher embedding and MLP coordinates by mean squared
weight energy, preserves complete grouped-query attention heads, and selects evenly spaced depth coordinates. This tests
whether retaining high-energy teacher subspaces is more useful than compressing every selected weight.

QD-005 uses the official Qwen3 4B checkpoint as the teacher. This is a direct capacity-gap control motivated by
[teacher-assistant distillation](https://arxiv.org/abs/1902.03393): if the 32B target is too difficult, a closer 4B teacher
may provide a better quality-throughput tradeoff even without a second training phase.

## Training and evaluation protocol

The regional Datakit sample combines Arxiv abstracts, Python code, NuminaMath, Spanish FinePDFs, Nemotron-CC,
Nemotron-SFT, HPLT, and SWE-Rebench/OpenHands data. Validation uses a disjoint WikiTeam source. The extended cache contains
3.45 billion train tokens and 4.50 million validation tokens, so no 1.8B-token run repeats the cache.

The screen and extension use different-sized samples from the same source families. Their validation examples are not
identical, so absolute NLL should be compared only within a phase. Promotion and endpoint conclusions use paired runs on
the same cache.

Every run uses sequence length 2,048, global batch size 8, AdamW with peak learning rate `3e-4`, 5% warmup, decay to 10%
of peak, and weight decay 0.1. A cell occupies four GB200 GPUs with a four-way model mesh. Online objectives use a
microbatch size of four; TAID uses the full batch because its controller advances once per optimizer update.

We stop on tokens rather than a loss threshold. Loss-based stopping would favor noisy validation observations and would
give methods different amounts of data. The screen therefore runs for 6,104 updates, or 100,007,936 tokens. The extension
runs for 109,864 updates, or 1,800,011,776 tokens. Held-out NLL is measured eight times per run with 16 batches per
evaluation. Terminal NLL is the primary result; the trailing three evaluations are a stability diagnostic.

The screen promotion rule was fixed before reading results:

- mean terminal NLL at least 0.5% below QD-C1;
- both paired seeds directionally better than QD-C1;
- no paired zero-shot regression larger than the frozen sampling tolerance;
- a third seed only if the mean result was within 0.25 percentage points of the threshold.

The zero-shot suite contains ARC-Easy, HellaSwag, PIQA, and WinoGrande with no demonstrations. Frozen raw-accuracy
tolerances were 0.021, 0.010, 0.024, and 0.029 respectively, approximately two worst-case binomial standard errors.

## Screen results

### Held-out loss and throughput

| ID | Mean terminal NLL | Mean zero-shot accuracy | Tokens/s | Relative to scratch KL |
|---|---:|---:|---:|---:|
| QD-C0 | 2.6221 | 0.3807 | 97,729 | +12.46% |
| QD-C1 | 2.3315 | 0.3807 | 28,066 | control |
| QD-C2 | 2.4799 | 0.3874 | 97,571 | +6.36% |
| QD-C3 | **1.9236** | **0.3939** | 27,579 | **−17.50%** |
| QD-001 | 2.3064 | 0.3810 | 22,080 | −1.08% |
| QD-002 | 2.2979 | 0.3830 | 27,888 | −1.44% |
| QD-003 | 2.3688 | 0.3824 | 30,873 | +1.60% |
| QD-004 | 2.3622 | 0.3826 | 27,907 | +1.32% |
| QD-005 | 2.3521 | 0.3821 | 59,960 | +0.88% |

QD-002 was the only treatment to pass the promotion rule. QD-001 cleared the pooled mean threshold, but its second seed
finished worse than the paired QD-C1 seed. No result was close enough to invoke the third-seed rule.

The throughput comparison also changes how the methods should be read. Hard-label training is about 3.5× faster than
online 32B KL, although it finishes at substantially worse loss. The 4B teacher is about 2.1× faster than the 32B teacher
but does not improve quality over scratch KL. Capturing and projecting every hidden layer costs another 21% relative to
ordinary 32B KL without producing a reliable paired gain.

### Promotion zero-shot check

| Seed | ARC-Easy Δ | HellaSwag Δ | PIQA Δ | WinoGrande Δ | Macro Δ |
|---|---:|---:|---:|---:|---:|
| 0 | +0.0034 | +0.0002 | +0.0011 | +0.0118 | +0.0041 |
| 1 | −0.0046 | +0.0001 | +0.0022 | +0.0039 | +0.0004 |

The table reports QD-002 minus paired QD-C1 raw accuracy. Both seeds remained inside every frozen tolerance, so the
zero-shot gate did not overturn the NLL promotion.

## Extended results

### Held-out loss and systems cost

| ID | Mean terminal NLL | Mean trailing-three NLL | Tokens/s | Hours per cell | GPU-hours per cell |
|---|---:|---:|---:|---:|---:|
| QD-C0 | 2.9796 | 3.1631 | 95,307 | 5.98 | 23.91 |
| QD-C1 | 2.6541 | 2.7949 | 27,906 | 19.65 | 78.60 |
| QD-C2 | 2.8739 | 3.0604 | 97,092 | 6.04 | 24.15 |
| QD-C3 | **2.5579** | **2.6969** | 27,863 | 19.57 | 78.29 |
| QD-002 | 2.6644 | 2.8007 | 27,493 | 19.88 | 79.51 |

Terminal NLL is the prespecified primary result. The trailing-three statistic averages each seed's last three
evaluations and then the two seeds; it reaches the same ordering despite the 16-batch evaluations being visibly noisy.
QD-C3's seed-0 trailing statistic and QD-002's seed-0 terminal value are reconstructed from durable Iris logs after their
W&B uploaders failed, so the displayed means are rounded to the precision available in those logs.

The extension does not replicate the screening endpoint. QD-002 finished 0.39% worse than QD-C1, and both paired seeds
were worse: `2.6542` versus `2.6451` for seed 0 and `2.6747` versus `2.6631` for seed 1. It therefore fails its own
promotion criterion at 1.8 billion tokens. By contrast, QD-C3 finished 3.63% below QD-C1. Official Base initialization
also helped hard-label training: QD-C2 finished 3.55% below QD-C0.

At the first two extended evaluations, factorized initialization remained better than scratch forward KL in both seeds.
At approximately 225 million tokens, mean NLL was 3.1224 for QD-002 and 3.1579 for QD-C1, a 1.13% improvement. At
approximately 450 million tokens, the means were 2.9651 and 3.0094, a 1.47% improvement. QD-C3 remained strongest at
2.7744 by the second point. At approximately 675 million tokens, the factorized mean still led by 0.78%, but seed 1 was
0.0038 NLL worse than its paired control.

The remainder of the trajectory was non-monotonic at this evaluation resolution. Factorization led scratch KL by 0.44%
at 900 million tokens, was effectively tied at 1.125 billion, led by 1.07% at 1.35 billion, and was 1.32% worse at 1.575
billion. Both methods reached mean NLL below 3.0 by the 450 million-token evaluation, but QD-002 did so in both seeds;
QD-C1 seed 0 crossed at 675 million. This is evidence for improved early reliability, not a better converged solution.

### Terminal zero-shot evaluation

| ID | ARC-Easy | HellaSwag | PIQA | WinoGrande | Macro |
|---|---:|---:|---:|---:|---:|
| QD-C0 | 0.2795 | 0.2545 | 0.5101 | 0.4984 | 0.3856 |
| QD-C1 | 0.2959 | 0.2544 | 0.5120 | 0.4996 | 0.3905 |
| QD-C2 | 0.2866 | 0.2549 | 0.5139 | 0.4984 | 0.3884 |
| QD-C3 | **0.3001** | **0.2557** | **0.5144** | **0.5036** | **0.3934** |
| QD-002 | 0.2866 | 0.2546 | 0.5131 | 0.5020 | 0.3891 |

Each entry is the mean raw accuracy over two seeds. The suite agrees with the NLL endpoint but has little separation:
QD-C3 has the highest mean on every task and QD-002 finishes 0.0014 macro accuracy below QD-C1. Factorization's paired
delta relative to scratch KL is −0.0093 on ARC-Easy, +0.0002 on HellaSwag, +0.0011 on PIQA, and +0.0024 on WinoGrande.
Those changes remain inside the frozen sampling tolerances, so the suite does not establish a capability regression. It
also does not rescue the factorized endpoint.

## Operational notes

The study exposed three reusable boundary problems without invalidating completed comparisons:

- fused hard-label cross-entropy produced nonfinite unreduced losses for this vocabulary and mesh, while materialized
  float32 NLL was finite;
- native distilled checkpoints store the student at `model/student`, which the evaluation path now addresses explicitly;
- an optional step-10,000 Hugging Face export treated the regional `s3://` tokenizer directory as a Hub repository ID.

The four hard-label extension cells resumed from native checkpoints around steps 9,000 after the export was disabled.
Their data order and fixed token endpoint were unchanged. Evaluation uses raw task accuracy because auxiliary
probability-normalization metrics can be undefined when all answer choices receive negative infinity.

Two W&B uploaders failed late in otherwise healthy training runs. Iris continued QD-C3 seed 0 and QD-002 seed 0 through
terminal evaluation and checkpoint commit. After all children exited, the original six-cell parent and one hard-label
retry child reported controller-reconciliation failures while cleaning up pods. All ten terminal checkpoints were
already committed. The separate evaluation job adopts those immutable artifact paths directly; successful restoration is
the durability criterion, rather than the parent job's final aggregate state.

All ten adopted-checkpoint evaluations succeeded. QD-C2 seed 1 also completed despite its training parent's cleanup
failure, confirming that the hard-label retry artifacts are usable beyond the former export boundary. Several evaluation
workers dropped the optional W&B results artifact while shutting down, but raw metrics were already mirrored to finelog;
the one missing W&B summary was recovered from that durable log.

## Limitations

The Qwen3 substitution is the largest scope limitation. These runs do not measure Qwen3.5. The 32B teacher is also a
post-trained model while the student checkpoint is a Base model, so QD-C3 is an operational comparison rather than a
controlled pretraining-provenance ablation. The official student checkpoint has seen much more data than any scratch
initialization in this study.

Two seeds, four zero-shot tasks, and 16 validation batches per point are sufficient for a screening decision, not a broad
capability or convergence claim. The screen and extension use different validation examples, making cross-phase absolute
NLL comparisons invalid. Finally, exact online KL costs about 3.5 times as much wall-clock time as hard-label training on
this setup; the quality comparison should not be read as compute-matched.

## Conclusions

The experiment supports five conclusions:

1. Use the official student Base checkpoint with forward KL when its training provenance is acceptable. QD-C3 was the
   best arm in both phases and beat scratch KL by 3.63% at the extended endpoint.
2. Use plain forward KL when the student must start from scratch. Rank-512 factorized initialization is a short-horizon
   accelerator, not an endpoint improvement: its 1.47% lead at 450 million tokens became a 0.39% loss at 1.8 billion.
3. Do not use all-layer activation matching in its present form. It was 21% slower than ordinary 32B KL, added a large
   projector, and did not produce a consistent paired gain.
4. A cheaper teacher is not automatically a better assistant. The 4B teacher doubled throughput but lost on terminal
   NLL, while TAID did not close the capacity gap.
5. Fixed-token stopping was necessary to see the reversal. Promoting the factorized run on the 100 million-token screen
   alone would have overstated its value for a longer training budget.

## Future work

- Compare factorized initialization with direct official-base initialization under a controlled provenance setting. The
  official checkpoint has seen far more pretraining data, so QD-C3 is an operational upper bound rather than an
  initialization-only ablation.
- Test factorization rank and coordinate mapping. Ranks 128, 256, and 1,024 would reveal whether rank 512 is near the
  quality-cost knee; learned or activation-weighted coordinate maps may improve on even spacing.
- Distill fewer hidden layers or use layerwise adapters that are discarded after training. [TinyBERT](https://arxiv.org/abs/1909.10351)
  suggests that intermediate supervision can help, but the all-layer 1,024-to-5,120 projector is too expensive here.
- Separate the teacher-assistant idea into two stages: first distill 32B into 4B, then distill the resulting 4B model into
  0.6B. The current 4B arm measures only the smaller-teacher half of that hypothesis.
- Repeat the promoted comparison with a base, rather than post-trained, teacher and a broader evaluation suite. The
  present four-task suite and two seeds are enough for screening, not a general capability claim.

## Related work

The design draws on [Minitron](https://arxiv.org/abs/2407.14679) for structured teacher-derived compression,
[TAID](https://arxiv.org/abs/2501.16937) for adaptive interpolation, and
[Teacher Assistant Knowledge Distillation](https://arxiv.org/abs/1902.03393) for the capacity-gap hypothesis.
[TinyBERT](https://arxiv.org/abs/1909.10351) motivates intermediate-state supervision. The factorized and
weight-saliency initializers are related to low-rank and pruning approaches such as
[SliceGPT](https://arxiv.org/abs/2401.15024) and [Fisher-weighted SVD](https://arxiv.org/abs/2207.00112), although this
study trains a dense student after initialization rather than preserving a compressed parameterization.
