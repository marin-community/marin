# Single-prefix nested experts at 10B tokens

Date: 2026-07-29

Status: running.

## Question

The first corrected fixed-prefix burn trained E128 and E16 modes together on
25% of sequences. It was mechanically cheap but required an estimated 11.9%
more wall time to reach the control's full-model Paloma loss. This experiment
asks two narrower questions:

1. How much of that quality tax comes from E128 versus E16 restriction?
2. Can restricting only a rotating subset of layers preserve useful prefix
   training while reducing the full-model tax?

The primary hypothesis is that E128-only nesting costs less than E16-only
nesting because it concentrates routed-expert updates much less. The
enhancement hypothesis is that exact rotating layerwise restriction improves
the full-model-versus-prefix Pareto frontier relative to restricting every
layer.

## Prior evidence

The strongest demonstrated precedent is Google's MatFormer work. It jointly
trains nested feed-forward widths, and Google subsequently used the method to
train and release Gemma 3n E2B inside E4B. Meta's LayerSkip demonstrates a
related depth-nesting objective and usable early exits. NVIDIA's Flextron is a
post-hoc alternative that converts a pretrained model into an elastic family
with bounded continued training. Google DeepMind's Mixture of Nested Experts
demonstrates nested expert widths in vision, while sparse upcycling validates
the reverse small-to-large schedule.

Primary sources:

- [Gemma 3n developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)
- [MatFormer](https://arxiv.org/abs/2310.07707)
- [LayerSkip](https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding/)
- [Flextron](https://research.nvidia.com/labs/lpr/publication/cai2024flextron/)
- [Mixture of Nested Experts](https://deepmind.google/research/publications/108549/)
- [Sparse Upcycling](https://arxiv.org/abs/2212.05055)

These results establish nested training as a real architecture family. They do
not establish that fixed expert-prefix language-model training is free, that
outer experts learn complementary concepts, or that a one-node proxy transfers
directly to a 300B--700B expert-parallel system.

## Experiment-series context

| Variation | Meaning | Decision |
|---|---|---|
| Rotating ladder25 / ladder50 | E128, E32, E8, and E1 use changing expert cosets | useful regularization test; reject as an extractable model |
| Fixed-chain50 | 25% E128, 25% E16, 50% E256 | reject for broad full-model degradation |
| Corrected combined fixed25 | 12.5% E128, 12.5% E16, 75% E256 | mechanically cheap; `+0.0313` terminal Paloma and `+11.86%` time to equivalent loss |
| Single-prefix naive | 25% one fixed E128 or E16 prefix | current causal-isolation arms |
| Single-prefix layerwise | same sequences, rotating two of eight restricted layers | current low-intensity enhancement arms |

The first Datakit d768 burn used a pack-one loader and changed model/router
source, so it is excluded. The corrected control reproduces the historical
`aug-dk` run through update 1,000 with a `0.002285` median absolute pointwise
training-loss difference and a `+0.003679` Paloma difference.

## Setup

All five arms use the reproduced `aug-dk` training contract:

| Property | Value |
|---|---:|
| Hidden dimension / layers | 768 / 8 |
| Query / KV heads | 6 / 1 |
| Routed / active / shared experts | 256 / 4 / 1 |
| Stored parameters | 2.039B |
| E128 / E16 prefix parameters | 1.133B / 0.340B |
| Active parameter-equivalent per token | 0.255B |
| Sequence length / global batch | 8,192 / 32 |
| Tokens per update | 262,144 |
| Updates / nominal tokens | 38,147 / 9.9997B |
| Devices | 8 H100 |
| Parallelism | full FSDP; expert axis 1 |
| Data | augmented Datakit mixture from CoreWeave S3 |

The 10B-token `MoeHeuristic` cell uses MuonH learning rate `0.0060668502`,
plain-Adam learning rate `0.0014000424`, beta1 `0.9062`, beta2 `0.998001`,
epsilon `1.8898444e-15`, 1% warmup, linear decay to a 0.05 minimum ratio, and
no gradient clipping. The source is pinned at
[`613e570564`](https://github.com/marin-community/marin/commit/613e570564).

The reproduced Datakit loader densely packs documents while blocking
cross-document attention. Its 168 top-level training components expand to 200
physical CoreWeave S3 caches. The second mixture phase begins at update 29,184
and 7.65B nominal tokens; final plots mark this boundary.

## Arms

| Arm | Restricted sequences | Restricted layers on those sequences | Nested layer-events |
|---|---:|---:|---:|
| E256 control | 0% | 0 / 8 | 0% |
| E128 naive | 25% | 8 / 8 to experts 0--127 | 25% |
| E16 naive | 25% | 8 / 8 to experts 0--15 | 25% |
| E128 layerwise | 25% | rotating 2 / 8 to experts 0--127 | 6.25% |
| E16 layerwise | 25% | rotating 2 / 8 to experts 0--15 | 6.25% |

The layerwise schedule is deterministic rather than eight independent
Bernoulli draws. A Bernoulli 0.25 schedule has the same expectation, but
approximately 10% of selected sequences would restrict no layer at all and
the number of restricted layers would have variance 1.5. The implemented
schedule restricts exactly two layers and rotates among `{0,4}`, `{3,7}`,
`{2,6}`, and `{1,5}`. Every layer therefore receives the same nested exposure
without introducing per-sequence count variance.

Naive restricted rows follow the exact trajectory available after extracting
the prefix. Layerwise rows do not: no training forward restricts all eight
layers simultaneously. Their full-prefix evaluation therefore measures
whether separately trained layer restrictions compose into a usable compact
model, which is why post-SFT generation remains a required gate.

Under balanced routing, the expected assignment frequencies relative to the
E256 control are:

| Arm | Prefix experts | Remaining experts |
|---|---:|---:|
| E128 naive | 1.25x | 0.75x |
| E16 naive | 4.75x | 0.75x |
| E128 layerwise | 1.0625x | 0.9375x |
| E16 layerwise | 1.9375x | 0.9375x |

This difference makes the single-prefix comparison diagnostic. If E16 is the
main source of the earlier penalty, E128-only should approach the control
curve even before changing the schedule.

## Evaluation

Every 1,000 updates, each arm evaluates full E256 mode and its relevant fixed
prefix. The control additionally evaluates E128 and E16 counterfactuals from
the same checkpoint. The primary quality metric is Paloma macro loss;
uncheatable macro loss and paired domain deltas are secondary checks.
Optimizer-step timing excludes compilation, data loading, checkpointing, and
multi-mode evaluation hooks.

The preregistered gates are:

- 1B tokens: finite training, exact routing fractions, less than 5%
  optimizer-step overhead, and full Paloma within 0.10 nat of control;
- 4.4145B tokens: the same full-model bound and a trained prefix better than
  the corresponding untrained control prefix;
- 10B tokens: rank full quality, prefix quality, domain deltas, mechanical
  overhead, and fitted time to equivalent full-model loss;
- post-training: carry the control and at most two non-dominated treatments
  through the fixed WildChat/Nemotron SFT and generation evaluation.

## Other schedules worth testing

The current layerwise arm is the cheapest enhancement because it changes only
eligibility. Several stronger designs are plausible:

1. **Balanced complement routing.** For E128, allocate 25% of rows to
   experts 0--127, 25% to experts 128--255, and 50% to full routing. This
   gives every expert exactly its control assignment rate while retaining
   2.5B E128-prefix tokens in a 10B run. E16 is less accommodating: exact
   balance requires 15 complement events for every E16 event, so 25% E16
   exposure is mathematically incompatible with balanced expert updates.
2. **Compensated full-mode routing.** On unrestricted E128-arm rows, bias the
   router toward the outer bank so full-plus-prefix assignment totals match
   control. The required full-row split is one third inner and two thirds
   outer. E16 at 25% cannot be compensated this way because its restricted
   rows alone already overexpose the inner experts.
3. **Progressive layer closure.** Begin with two restricted layers, increase
   to four, then train all eight during a short terminal phase. This preserves
   most of the layerwise balance advantage while explicitly adapting the
   jointly extracted prefix before breakout.
4. **Frequency-aware expert updates.** Scale inner- and outer-bank expert
   gradients or optimizer updates by their expected sampling frequency. This
   attacks update-magnitude imbalance directly, but it needs careful treatment
   under Muon and does not restore missing outer-expert examples.
5. **Hierarchical expert residuals.** Parameterize full experts as a shared
   E16 or E128 base plus expert-specific residuals. Full-mode tokens then
   train the extractable base without forcing all of their routing into the
   small bank. This is a larger architecture change, but it most directly
   expresses the desired rule that outer capacity should learn residual
   concepts.
6. **Full-to-prefix distillation.** Add a router-logit or expert-output
   distillation loss on nested rows. Router-only distillation is cheap;
   computing full and restricted expert outputs is stronger but adds a partial
   second forward and must earn its systems cost.
7. **Route-conditioned low-rank deltas.** Let nested rows update small
   adapters on prefix experts and shared blocks while full rows use the base
   weights. The adapters can be folded into the extracted checkpoint at
   breakout. This weakens the identical-weights constraint but could isolate
   the compact objective for much less than a second forward.

The first three can be tested without changing the expert parameterization.
Balanced complements are the clearest next E128 gate if the current E128
naive arm shows useful prefix gains but retains a measurable full-model tax.

## Post-training plan

The E256 control and at most two non-dominated treatments will use
weights-only initialization with a fresh optimizer. Each model receives 1,000
updates on WildChat followed by 1,000 updates on the canonical Nemotron
science-reasoning mixture at sequence length 8,192 and batch 32. SFT uses a
`5e-5` MuonH/Adam learning rate, cosine decay, 3% warmup, no weight decay, and
gradient norm 1.0. Treatment checkpoints preserve their pretraining
eligibility schedule unless an explicitly labeled full-routing SFT is run.

The bounded native evaluation measures full and prefix Paloma and
uncheatable loss, greedy exact match on 64 GSM8K prompts, and eight
format-following instruction cases. These small generation sets are smoke
tests for agentic usability, not benchmark-grade capability estimates.
Post-training source and model-shape validation are pinned at
[`9e9a44a4fb`](https://github.com/marin-community/marin/commit/9e9a44a4fb).

## Results

All arms pass the preregistered 1B-token gate:

| Arm | Full E256 Paloma | Full delta | Trained prefix | Prefix Paloma | Prefix gain vs control |
|---|---:|---:|---|---:|---:|
| E256 control | 3.772742 | -- | E128 | 4.023812 | -- |
| E256 control | 3.772742 | -- | E16 | 4.751665 | -- |
| E128 naive | 3.782698 | +0.009955 | E128 | 3.815397 | -0.208415 |
| E16 naive | 3.805686 | +0.032944 | E16 | 3.958464 | -0.793201 |
| E128 layerwise | 3.776149 | +0.003407 | E128 | 3.911731 | -0.112081 |
| E16 layerwise | 3.773253 | +0.000511 | E16 | 4.394279 | -0.357386 |

These values are at update 4,000 and 1.049B tokens. E128-only is gentler than
E16-only, and layerwise restriction substantially reduces the full-model
penalty while retaining part of the prefix gain. Across the four aligned
evaluations, median full-mode penalties are `+0.010667` for E128 naive,
`+0.032957` for E16 naive, `+0.004664` for E128 layerwise, and `+0.005327` for
E16 layerwise. No arm is near the `+0.10` stopping bound.

The isolation is already informative relative to the completed combined
fixed25 arm. E128-only's full penalty is approximately `+0.010`, while
E16-only is approximately `+0.033`, close to the combined arm's terminal
`+0.031`. This is evidence that the E16 objective and its concentrated inner
updates caused most of the earlier quality tax; the matched 4.4145B gate will
test whether that attribution persists.

Through common update 4,289, median compiled-step overhead is `+0.50%` for
E128 naive, `+0.38%` for E16 naive, `+0.44%` for E128 layerwise, and `+0.53%`
for E16 layerwise. A log-linear local slope model anchored at the observed
update-4,000 losses estimates full-Paloma time-to-equivalent penalties of
`+2.87%`, `+8.46%`, `+1.26%`, and `+0.65%`, respectively. Four points are too
few for a scaling conclusion; the anchored calculation is retained to show
how the estimate evolves through 10B.

The control evaluates three routing modes and has a 76-second median
evaluation hook. Each treatment evaluates two modes and has a 52--53-second
median hook. This research instrumentation is excluded from the architecture
cost.

## Cost versus control

Every arm computes router logits over E256 and dispatches top-4 experts. The
nested eligibility mask therefore does not reduce model FLOPs; it changes
which experts receive those FLOPs. The standardized optimizer-only forecast
is:

| Arm | Median step | 10B optimizer hours | 10B H100-hours | Surcharge |
|---|---:|---:|---:|---:|
| E256 control | 463.697 ms | 4.914 | 39.31 | -- |
| E128 naive | 466.021 ms | 4.938 | 39.51 | +0.50% |
| E16 naive | 465.479 ms | 4.932 | 39.46 | +0.38% |
| E128 layerwise | 465.756 ms | 4.935 | 39.48 | +0.44% |
| E16 layerwise | 466.170 ms | 4.940 | 39.52 | +0.53% |

This result is already precise enough to reject a material same-topology
kernel surcharge. Quality-adjusted cost remains the deciding measurement.

The two-checkpoint baseline is much more expensive. Analytic model FLOPs are
approximately 357.7M per token for E256 and 356.2M for standalone E128, so
training both independently costs about 1.996x one E256 run. At Gate 1, the
quality-adjusted E128-naive estimate is 1.029x. This is not yet an
apples-to-apples replacement claim: the current control provides an untrained
same-checkpoint prefix, not an independently compute-optimal E128 run.

## Scaling interpretation

E128 is the closer proxy for nesting a roughly 300B-class expert bank inside a
700B-class bank. It reduces stored routed-expert capacity but preserves the
attention backbone, shared expert, expert width, and top-k. The extracted
model therefore saves parameter memory and can use a smaller
expert-parallel footprint; it does not receive a proportional reduction in
per-token FLOPs. Nesting hidden width, depth, or top-k would be required for a
compute-smaller extraction.

The d768 parameter decomposition is approximately 0.227B shared parameters
plus 7.08M parameters per routed expert. Preserving that ratio at 700B maps an
E128 prefix to about 389B and an E92 prefix to about 301B. A
topology-friendlier E96 prefix would be about 311B. The production analogue is
therefore likely E96 inside E256; E128 is a conservative half-bank proxy and
E16 a routing stress test. At 25% sequence restriction, E96 naive would expose
prefix experts at 1.417x control frequency; the two-of-eight-layer schedule
would reduce that to 1.104x.

The one-node result does not validate the production dispatcher. Under ideal
balance, E128 naive requires about 1.25x control capacity on prefix experts
and E128 layerwise 1.0625x. E16 naive requires 4.75x and E16 layerwise 1.9375x.
At 300B--700B scale, the prefix must be striped across enough expert-parallel
ranks or replicated; placing E16 on one sixteenth of ranks would otherwise
turn a promising quality curve into capacity overflow and all-to-all
imbalance.

An architecture is economically viable if its final quality-adjusted cost is
below about 10%. The Gate 1 point estimates put both layerwise arms and E128
naive inside that range, but the 10B endpoint and an expert-parallel
replication remain required evidence.

![Interim full and prefix Paloma curves.](assets/nested-model-training-single-prefix-10b-paloma.png)

![Interim mixed-mode training loss.](assets/nested-model-training-single-prefix-10b-loss.png)

![Interim compiled optimizer-step timing.](assets/nested-model-training-single-prefix-10b-step-time.png)

![Rolling anchored time to equivalent full-mode Paloma loss.](assets/nested-model-training-single-prefix-10b-time-to-equivalent.png)

Final results will be added after the preregistered 10B endpoint and selected
post-training runs complete.

- [E256 control](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e256-10b-r1)
- [E128 naive](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e128-naive25-10b-r1)
- [E16 naive](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e16-naive25-10b-r1)
- [E128 layerwise](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e128-layer25-10b-r1)
- [E16 layerwise](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e16-layer25-10b-r1)
