# Corrected fixed-prefix nested experts

Date: 2026-07-29

Status: running.

## Decision

The corrected `aug-dk` control reproduces the historical training run through
update 1,000. The matched fixed25 treatment remains within the preregistered
10% optimizer-step overhead gate, but its full-model held-out loss is worse at
the first two evaluation gates. Both arms are continuing to the 4.414B-token
endpoint before a promotion decision.

This report replaces the invalidated
[first d768 burn](nested-model-training-burnin.md). Final loss curves, runtime
tables, cost projections, checkpoint transfer results, and scale-up guidance
will be added when the running pretraining and SFT jobs complete.

## Research question

The production question is whether one sparse-MoE pretraining run can yield a
large model and one or more ordinary, extractable smaller checkpoints. For the
MoE-specific arm, a small checkpoint is a fixed prefix of the large expert
bank. A sequence assigned to E128 may route only to experts 0--127 at every
layer; E16 may route only to experts 0--15. Unrestricted sequences may route
across all 256 experts.

The preregistered proxy hypothesis is:

> Restricting 25% of whole sequences to fixed E128 and E16 prefixes yields
> usable nested checkpoints while adding less than 10% to optimizer-step time
> and keeping full-E256 Paloma within 0.10 nat of a matched control.

This is a stochastic multi-architecture objective. It does not compute two
forwards on every sequence. All rows share one model invocation, loss, and
backward; only router eligibility differs.

Under ideal balance, the schedule changes routed-expert update frequency in a
simple way. Experts 0--15 receive three times their E256-control assignment
rate, experts 16--127 receive the same rate as control, and experts 128--255
receive 75% of the control rate. The extra small-prefix training is paid for by
fewer outer-expert updates, even though every sequence still activates four
experts. The full-model loss penalty should therefore be interpreted partly as
an outer-expert exposure tradeoff, not only as generic multi-objective
interference.

## Prior evidence

The strongest demonstrated precedent is Google's MatFormer work, which jointly
trains nested feed-forward widths. Google subsequently used MatFormer to train
the E2B model inside Gemma 3n E4B and released both extracted checkpoints.
Meta's LayerSkip demonstrates a related depth-nesting objective and usable
early exits. NVIDIA's Flextron shows a post-hoc alternative: convert a
pretrained model into an elastic family with a bounded continued-training
budget. Sparse upcycling establishes the reverse schedule, expanding a dense
model into an MoE before continued pretraining.

Primary sources:

- [Gemma 3n developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)
- [MatFormer](https://arxiv.org/abs/2310.07707)
- [LayerSkip](https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding/)
- [Flextron](https://research.nvidia.com/labs/lpr/publication/cai2024flextron/)
- [Sparse Upcycling](https://arxiv.org/abs/2212.05055)

These results establish nested training as a real architecture family. They do
not establish that fixed expert-prefix nesting is free, that it transfers to a
300B--700B expert-parallel layout, or that outer experts learn complementary
concepts.

## What failed in the first burn

The first d768 control did not reproduce the known-good `aug-dk` curve. Its
launcher forced one document per 8,192-token sequence and padded the rest,
while the reference densely packs documents. The phase-0 cache ledger has a
mean document length of 2,297.7 tokens, so the altered loader exposed at most
28.05% useful target occupancy on average while still counting every padded
position as a nominal training token. This is at least a 3.57-fold
overstatement of useful-token progress before accounting for the different
long-document and cross-document-attention behavior.

That run also changed the model/router source. Its router-bias norms differed
materially from the reference, so packing cannot be isolated as the only
numerical cause. The old run answers neither the control-reproduction question
nor the nested-architecture question and is excluded from every quality and
cost conclusion here.

A separate defect in the old nested-evaluation callback could perturb training
state after evaluating alternate expert modes and, in one contaminated
trajectory, lead to non-finite loss. Same-checkpoint counterfactuals isolated
the callback as causal. The corrected implementation keeps evaluation
functional: full, E128, and E16 evaluation no longer mutates the training
state. The current fixed25 arm has crossed every three-mode evaluation gate
and resumed finite training.

## Architecture variations

The broader experiment series tested two meanings of expert subsampling before
selecting this burn:

| Variation | Schedule | Extractable checkpoint | Result |
|---|---|---|---|
| E256 control | every row uses E256 | E256 | matched baseline |
| rotating ladder25 | 25% of rows sample E128/E32/E8/E1 cosets | no single stable subset | useful regularizer; reject as a breakout design |
| rotating ladder50 | 50% of rows sample E128/E32/E8/E1 cosets | no single stable subset | more small-mode exposure; reject as a breakout design |
| fixed50 | 25% E128, 25% E16, 50% E256 | E128 and E16 prefixes | broad full-model degradation in the earlier fixed-chain gate |
| fixed25 | 12.5% E128, 12.5% E16, 75% E256 | E128 and E16 prefixes | promoted into the corrected burn |

The rotating ladder tested regularization, not true nesting: a requested size
selected different expert cosets over time, so no one compact model received
the reported exposure. Fixed-prefix routing corrects that flaw. Dense
weight-matrix masking was not promoted because a masked large GEMM retains most
of the large compute and exact small-model hidden states diverge after the
first masked layer, normally requiring another forward.

## Control reconstruction

The reference run is
[`aug-dk-d768-ev-sw2k-g4-nomtp-noconv-f1`](https://wandb.ai/marin-community/marin_moe/runs/aug-dk-d768-ev-sw2k-g4-nomtp-noconv-f1).
Its immutable source bundle has SHA-256
`adc2aad8a60b45f4a105d4d6e4134cb7fff350caa77d7e56ab23fbe66bd3479b`.
The exact-source
[`nest-burn-control-augdk-repro1000-r1`](https://wandb.ai/marin-community/marin_moe/runs/nest-burn-control-augdk-repro1000-r1)
reproduction changed only run identity and output paths.

Across updates 2--1,000, the median absolute pointwise training-loss
difference was `0.002285` nat and the 95th percentile was `0.013817` nat.
Learning rates matched exactly. At update 1,000, Paloma macro loss was
`4.224867` for the reproduction and `4.221188` for the reference, a
`+0.003679` nat difference. The preregistered control gate passed.

## Experimental setup

| Property | Value |
|---|---:|
| Hidden dimension | 768 |
| Layers | 8 |
| Query / KV heads | 6 / 1 |
| Total / active experts | 256 / 4 |
| Shared experts | 1 |
| Stored parameters | 2.039B |
| Active parameter-equivalent per token | 0.255B |
| E128 / E16 prefix parameters | 1.133B / 0.340B |
| Sliding window | 2,048 |
| Global-attention cadence | every fourth layer |
| Sequence length | 8,192 |
| Global batch | 32 |
| Tokens per update | 262,144 |
| Updates | 16,840 |
| Nominal training tokens | 4.4145B |
| Compute budget | 4.14e18 model FLOPs |
| Devices per arm | 8 H100 |
| Parallelism | full FSDP; expert axis 1 |
| Data | `aug-dk` Datakit mixture from CoreWeave S3 |

The optimizer is the d768 MoeHeuristic cell: MuonH learning rate `0.00838`,
AdamH learning rate `0.00838`, plain-Adam learning rate `0.00193`, beta1
`0.9062`, beta2 approximately `0.998`, epsilon approximately `1.03e-15`, 1%
warmup, linear decay to a 0.05 minimum ratio, and no gradient clipping.

The control routes every sequence across all 256 experts. Fixed25 routes 75%
of sequences across all experts, 12.5% only across experts 0--127, and 12.5%
only across experts 0--15. The subsets are literal fixed prefixes at every
layer and update. E256, E128, and E16 keep independent eligibility-conditioned
QB router state.

Measurement runs:

- [E256 control](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e256-4b-r2)
- [fixed25 treatment](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-fixed25-4b-r2)

Checkpoint replicas:

- [E256 checkpoint replica](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e256-4b-ckpt-r1)
- [fixed25 checkpoint replica](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-fixed25-4b-ckpt-r1)

The corrected training, SFT, and native generation source is pinned at
[`60d5dc3506`](https://github.com/marin-community/marin/commit/60d5dc3506).

## Preregistered gates

The architecture is stopped or rejected if:

- full-mode Paloma is more than 0.10 nat worse than control at two consecutive
  aligned gates;
- router capacity overflow remains above 5%;
- compiled optimizer-step overhead exceeds 10% for promotion or 25% for
  immediate termination;
- loss or gradients become non-finite.

The measurement pair evaluates every 1,000 updates. Fixed25 evaluates full,
E128, and E16 modes; the control evaluates full mode. Timing excludes
compilation, data loading, checkpointing, and evaluation hooks. A separate
no-evaluation replica pair writes final checkpoints for matched SFT.

This proxy has an expert-axis size of one. The ring dispatcher retains every
assignment because its single-rank receive capacity equals all top-k
assignments; cross-rank capacity overflow is therefore structurally zero and
is not emitted as a metric. The 5% gate becomes active when the design is
tested with expert parallelism. A prefix layout that places E16 on too few
ranks can overflow even when the router itself is balanced within E16.

## Interim results

| Update | Tokens | E256 full Paloma | fixed25 full | Delta | fixed25 E128 | fixed25 E16 |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.262B | 4.219130 | 4.241795 | +0.022665 | 4.274152 | 4.399118 |
| 2,000 | 0.524B | 3.961045 | 3.992682 | +0.031638 | 4.029015 | 4.173859 |
| 3,000 | 0.786B | 3.868898 | 3.894513 | +0.025615 | 3.937864 | 4.076207 |
| 4,000 | 1.049B | 3.790510 | 3.815166 | +0.024656 | 3.856960 | 4.011194 |
| 5,000 | 1.311B | 3.732107 | 3.762341 | +0.030234 | 3.809299 | 3.951459 |

Through common update 5,100, fixed25 adds 1.19% to median compiled
optimizer-step time: 456.276 ms for control and 461.684 ms for fixed25.
Across the five aligned evaluation gates, the median full-mode Paloma delta is
`+0.025615` nat.
Fixed25's three-mode evaluation takes longer than the control's one-mode
evaluation; that instrumentation cost is excluded from the architecture
surcharge.

![Training loss while the corrected burn is running.](assets/nested-model-training-corrected-augdk-loss.png)

![Full and nested Paloma while the corrected burn is running.](assets/nested-model-training-corrected-augdk-paloma.png)

![Compiled optimizer-step duration while the corrected burn is running.](assets/nested-model-training-corrected-augdk-step-time.png)

## Runtime and cost model

The steady-state forecast uses:

`optimizer hours = target tokens / 262,144 × median compiled step seconds / 3,600`.

It excludes compilation, checkpointing, and research-only multi-mode
evaluation. Those fixed costs matter for this short burn but amortize in a
production run.

| Target tokens | E256 wall hours | fixed25 wall hours | fixed25 surcharge | E256 GPU-hours | fixed25 GPU-hours |
|---:|---:|---:|---:|---:|---:|
| 10B | 4.835 | 4.892 | 0.057 | 38.68 | 39.14 |
| 100B | 48.349 | 48.922 | 0.573 | 386.79 | 391.37 |
| 1T | 483.487 | 489.218 | 5.730 | 3,867.90 | 3,913.74 |

With fixed top-4 routing, independently training a smaller expert bank costs
approximately another full active-model run: reducing stored experts changes
checkpoint size much more than active FLOPs per token. Co-training therefore
replaces roughly 100% additional optimizer compute with the measured 1.19%
surcharge in this one-node topology. This is a systems-cost result, not yet a
claim that E128 or E16 matches an independently compute-optimal model.
Training both E128 and E16 independently would cost roughly two additional
active-model runs unless one of those runs also used a nesting objective.

## Viability at 300B--700B

The point estimate is economically viable: a 1.19% training surcharge is far
below the 10% promotion threshold and much cheaper than a separate active-top-4
run. The quality endpoint and expert-parallel replication remain gating
evidence.

This proxy uses full FSDP across one eight-H100 node and no expert-parallel
axis. A 300B--700B model will shard experts across nodes. A contiguous prefix
can then concentrate all E16 or E128 traffic on a fraction of ranks, turning a
nearly free eligibility mask into an expensive load and capacity problem. A
production design should distribute each nested bank across the topology,
collocate core and outer experts where possible, and measure overflow and
all-to-all time at the intended experts-per-rank ratio. A dropless or ragged
dispatcher is preferable to hiding overload behind extra fixed capacity.

The next d768 architecture arm should correct the known update-frequency
imbalance without another forward. Experts 128--255 are selected at 75% of
their control rate, so multiplying their routed-expert gradients by `4/3`
would restore the control expectation while retaining extra updates for the
inner prefixes. This will increase gradient variance and does not restore
missing token diversity, but it directly tests whether outer-expert
undertraining explains the observed full-mode penalty.

Expert-count nesting mainly reduces stored parameters and serving memory.
With the same top-4, expert width, backbone, and depth, the extracted model
does not receive a proportional reduction in active inference FLOPs. If the
300B option must also decode substantially faster than the 700B model, expert
prefix nesting should be combined later with a separately validated width,
depth, or top-k mechanism. Combining all axes in the first hero-run test would
make each exact submodel too rare and failures difficult to attribute.

The scale-up rule is:

- below 10% measured end-to-end surcharge, including any method-specific
  routing capacity, the approach remains economically interesting;
- near 50%, training a separate small model provides a cleaner objective and
  checkpoint;
- better standardized loss at equal model FLOPs would justify promotion, but
  parity within a small loss margin can still be useful if the extracted
  checkpoint avoids an otherwise separate run.

## Post-training protocol

The checkpoint replicas feed two 1,000-update completion-masked SFT stages:
WildChat 385.7k followed by Nemotron science reasoning. Both transformed
datasets are already in CoreWeave S3. Each stage uses sequence length 8,192,
batch 32, and eight H100s, for 262.1M nominal tokens. The fresh AdamH optimizer
uses learning rate `5e-5`, beta1 `0.9`, beta2 `0.95`, epsilon `1e-8`, 3%
warmup, cosine decay to a 0.1 minimum ratio, and gradient clipping at 1.0.

The E256 control remains unrestricted during SFT. Fixed25 retains the
75%/12.5%/12.5% E256/E128/E16 schedule so the breakout prefixes receive
completion-masked instruction updates. Its SFT training loss is therefore a
mixed-mode objective and is not a pure full-E256 comparison.

Final native-JAX generation evaluation uses 64 pinned-revision GSM8K examples
and eight deterministic instruction-adherence cases. Both the control and
fixed25 checkpoints are evaluated under E256, E128, and E16 eligibility. The
control's restricted modes are counterfactual arbitrary prefixes; comparing
them with fixed25 distinguishes useful nested training from quality that an
untrained prefix already had.

## Remaining work

The promotion decision waits for the full 4.414B-token curve. The final report
will include aligned Paloma and uncheatable results, per-domain comparisons,
step-time distributions and charged runtime, 10B--1T cost projections, a
matched SFT transfer check, and an explicit assessment of which conclusions
can transfer to a 300B--700B expert-parallel topology.
