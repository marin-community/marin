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

## Setup

All five arms use the reproduced `aug-dk` training contract:

| Property | Value |
|---|---:|
| Hidden dimension / layers | 768 / 8 |
| Query / KV heads | 6 / 1 |
| Routed / active / shared experts | 256 / 4 / 1 |
| Stored parameters | 2.039B |
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

The first three can be tested without changing the expert parameterization.
Balanced complements are the clearest next E128 gate if the current E128
naive arm shows useful prefix gains but retains a measurable full-model tax.

## Results

The first aligned evaluation is directionally consistent with both
hypotheses:

| Arm | Full E256 Paloma | Full delta | Trained prefix | Prefix Paloma | Prefix gain vs control |
|---|---:|---:|---|---:|---:|
| E256 control | 4.352927 | -- | E128 | 4.496725 | -- |
| E256 control | 4.352927 | -- | E16 | 4.957031 | -- |
| E128 naive | 4.369598 | +0.016671 | E128 | 4.390685 | -0.106040 |
| E16 naive | 4.390925 | +0.037998 | E16 | 4.505941 | -0.451090 |
| E128 layerwise | 4.358848 | +0.005920 | E128 | 4.439291 | -0.057435 |
| E16 layerwise | 4.360086 | +0.007159 | E16 | 4.760953 | -0.196078 |

These values are at update 1,000 and 0.262B tokens, before the preregistered
1B-token gate. E128-only is gentler than E16-only, and layerwise restriction
substantially reduces the full-model penalty while retaining part of the
prefix gain. At this gate every naive treatment domain is worse than control
in full mode; the layerwise arms are worse on 15 of 16 Paloma domains. No arm
is near the `+0.10` stopping bound.

Through the short common timing horizon at update 1,054, median compiled-step
overhead is `+0.73%` for E128 naive, `+1.17%` for E16 naive, `+1.11%` for E128
layerwise, and `+1.22%` for E16 layerwise. These timing estimates will be
replaced with full-run block-bootstrap intervals.

![Interim full and prefix Paloma curves.](assets/nested-model-training-single-prefix-10b-paloma.png)

![Interim mixed-mode training loss.](assets/nested-model-training-single-prefix-10b-loss.png)

![Interim compiled optimizer-step timing.](assets/nested-model-training-single-prefix-10b-step-time.png)

Final results will be added after the preregistered 10B endpoint and selected
post-training runs complete.

- [E256 control](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e256-10b-r1)
- [E128 naive](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e128-naive25-10b-r1)
- [E16 naive](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e16-naive25-10b-r1)
- [E128 layerwise](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e128-layer25-10b-r1)
- [E16 layerwise](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e16-layer25-10b-r1)
