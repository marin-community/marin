# Can one MoE pretraining run yield both a 300B and 700B model?

Issue: [#652](https://github.com/marin-community/marin/issues/652)

Experiment series: `NEST-MOE`

Date: 2026-07-27

## Summary

This study tests whether a large MoE can contain a useful smaller MoE without
paying for a second training forward. The smaller model is a fixed,
evenly-sharded subset of the routed expert bank. Some training sequences can
route only to that subset; the remaining sequences use the full bank. A branch
assignment is held fixed across every layer, so the restricted sequences
follow exactly the trajectory available after extracting the smaller
checkpoint.

The experiment compares a conventional 256-expert model, a conventional
128-expert model, and 256-expert models that restrict 25% or 50% of sequences
to an extractable 128-expert subset. The four arms have identical dense
backbones, data order, optimizer, active experts per token, training tokens,
and hardware.

<!-- NESTED_RESULTS_SUMMARY -->

The scope of the conclusion matters. Expert-bank nesting changes total
parameters and checkpoint memory. With the same expert width and top-k, it does
not make the extracted model a 2.3× cheaper decoder. If the 300B and 700B
choices differ mainly in routed-expert count, this experiment addresses the
choice directly. If they differ in active width, depth, or top-k, a later
MatFormer- or LayerSkip-style stage is required.

## What the existing evidence says

The strongest direct precedent is Google's MatFormer work. MatFormer jointly
trains nested feed-forward widths and reported 582M–850M decoder submodels that
matched or improved on independently trained counterparts. This is also a
shipped result rather than only a paper result: Google says Gemma 3n trained
the E2B model embedded inside E4B and released both extracted checkpoints.
Gemma 3n can additionally compose intermediate models by slicing feed-forward
widths and skipping layers.

Meta's LayerSkip is the strongest depth-specific precedent. It applies
increasing layer dropout and early-exit losses during training, producing
usable intermediate exits and self-speculative decoding. It does not directly
produce a conventional smaller standalone transformer, but it is a plausible
second axis if expert-bank nesting succeeds.

NVIDIA's Flextron demonstrates a post-hoc fallback. It converts pretrained
models to nested elastic networks using 7.63% of their original pretraining
tokens. This is attractive if simultaneous nesting damages the large model,
although it does not make a small checkpoint available during the original
pretraining run.

Google DeepMind's Mixture of Nested Experts demonstrates nested expert widths
in vision and reports equivalent accuracy at more than 2× lower inference
compute. Sparse upcycling demonstrates the reverse schedule: train a smaller
dense model, then expand it to an MoE. Both support the mechanism, but neither
is direct evidence for expert-subset nesting in a frontier language MoE.

Primary sources:

- [Gemma 3n developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)
- [MatFormer](https://arxiv.org/abs/2310.07707)
- [LayerSkip](https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding/)
- [Flextron](https://research.nvidia.com/labs/lpr/publication/cai2024flextron/)
- [Mixture of Nested Experts](https://deepmind.google/research/publications/108549/)
- [Sparse Upcycling](https://arxiv.org/abs/2212.05055)

These results support co-training as a research direction. They do not
establish that a nested language MoE is free, that its outer experts
automatically specialize in novel concepts, or that results transfer to
300B–700B total parameters.

## Ranked follow-on architectures

The four-arm experiment isolates one axis. If it passes, the next tests should
remain gated rather than becoming a combinatorial sweep.

1. **MatFormer FFN-width nesting.** Statically slice the shared expert and each
   routed expert to two intermediate widths. Sample one width per sequence or
   microbatch and use a sandwich schedule that always includes the smallest
   and largest variants over a short accumulation window. This can reduce
   active FLOPs in the extracted model, unlike expert-count nesting. It also
   changes every routed expert and is more invasive to optimized kernels.
2. **Depth nesting.** Apply LayerSkip-style increasing layer dropout and losses
   at fixed early exits. This offers a direct decoder-speed lever and can
   support self-speculative decoding. A standalone breakout needs a fixed
   retained-depth topology and an output head trained at that exit.
3. **Small-first sparse upcycling.** Train the 300B model conventionally,
   duplicate or perturb experts to create the 700B bank, then continue
   training. This protects small-model quality and is the safest fallback if
   simultaneous training undertrains the prefix or full model. It does not
   provide the 700B checkpoint during the first stage.
4. **Elastic active-expert count.** Alternate top-k values while sharing the
   same bank. This directly changes active compute but changes routing
   capacity, communication, and load-balancing behavior. It should be tested
   only after fixed-top-k subset nesting is understood.
5. **Post-hoc elastic conversion.** Use a Flextron-like recovery phase from a
   completed large checkpoint. This is operationally attractive when the large
   run cannot accept architectural risk, but its cost is paid after
   pretraining.

Dense width, depth, top-k, and expert-count nesting should not be composed in
the first large run. Sampling axes independently can produce subnetworks that
were rarely or never optimized together, while evaluating every combination
would approach the cost of training multiple models.

## Why expert-bank nesting is the first test

A masked dense matrix is not automatically cheaper. Multiplying a full
activation or weight by a mask retains the large GEMM unless the compiler
lowers a static slice. More importantly, a small loss and a large loss follow
different hidden-state trajectories after their first differing layer. Two
exact losses therefore normally require two forwards.

MoE routing offers a cheaper experiment. The full bank has \(E_L\) experts and
the small bank is a fixed subset of \(E_S\) experts. For each packed sequence,
we choose one branch:

- `small`: mask router logits outside the fixed subset at every layer;
- `full`: leave all router logits eligible.

Small and full sequences share one batch, one forward, one loss, and one
backward. Both branches keep top-4 routing, so active expert FLOPs per token
remain approximately fixed. The treatment adds a router eligibility mask and
branch-aware statistics, not a second model evaluation.

The subset uses even-numbered experts. With four experts per
expert-parallel rank, every rank owns two subset experts and two outer experts.
A contiguous E128 prefix of an E256 bank would reside on only half of the
64-way expert mesh and confound the model comparison with a communication
artifact. Extraction gathers the even experts and compacts them into a
contiguous E128 bank.

This design can encourage the outer bank to cover residual demand because only
full-branch sequences can reach it. Routing balance is evidence about
utilization, not semantic specialization; held-out loss is required before
making a capability claim.

## Preregistered decision

The primary hypothesis was fixed before the scientific runs:

> At common training tokens, the 25%-restricted E256 model retains at least
> 90% of E256-control throughput, finishes within +0.010 Paloma macro loss in
> full mode, yields an extracted E128 model within +0.030 of the standalone
> E128 control, and keeps core and outer assignment CV within +0.10 of the
> E256 control.

The 50% arm tests the expected tradeoff: more training for the extracted model
at greater risk to the full model. A result above 20% measured training
overhead or a full-model loss gap above 0.02 rejects the method for the
intended run. A result below 10% overhead that passes the loss gates promotes
the method to a larger preregistered replication, not directly to the hero
run.

The complete preregistration, including retry rules and the breakout gate, is
in
[`2026-07-26_nested-model-training-preregistration.md`](../../.agents/projects/2026-07-26_nested-model-training-preregistration.md).

## Experimental setup

### Model

| Property | Value |
|---|---:|
| Hidden dimension | 1,280 |
| Layers | 13 |
| Attention heads | 10 |
| Sequence length | 8,192 |
| Routed expert intermediate dimension | 640 |
| Shared expert intermediate dimension | 1,280 |
| Routed experts selected per token | 4 |
| Large routed experts | 256 |
| Small routed experts | 128 |
| Large total parameters | 8.64B |
| Small total parameters | 4.54B |
| Approximate active parameters, excluding LM head | 0.6B |

All arms use the same shared dense expert, attention, embeddings, hidden width,
depth, expert width, and top-k. Routed experts use Marin's ring
implementation, QB load balancing, sigmoid combine weights, and a 64-way
expert axis.

### Data and optimization

Training uses the pinned SlimPajama-6B cache with the Llama 3.1 tokenizer.
Validation uses the pinned Paloma domain caches. Every example is represented
as one fixed-shape packed document. The four arms use seed 0, the same
block-shuffle configuration, the same optimizer heuristic, a global batch of
256 sequences, and 2,097,152 tokens per optimizer step.

Each arm uses 16 four-GB200 nodes on `cw-us-east-08a`, or 64 GPUs. The runs
were submitted through the main Marin Iris controller at batch priority and
federated as whole jobs. The four arms ran concurrently:

| ID | Routed experts | Restricted rows | Role |
|---|---:|---:|---|
| `NEST-MOE-001` | 256 | 0% | large control |
| `NEST-MOE-002` | 128 | 0% | small control |
| `NEST-MOE-003` | 256 | 25% | primary nested arm |
| `NEST-MOE-004` | 256 | 50% | prefix-quality tradeoff |

### Attention backend amendment

The preregistered model treatments did not vary attention. Before the
scientific run, the GB200 FA4/CuTe forward kernel compiled and returned finite
output, but its backward kernel hung after device dispatch. Matching the
upstream backward tiling did not resolve the hang. Iris reported zero
preemptions and zero task failures while the process remained resident.

Every arm was therefore amended to Levanter's reference attention backend.
The model, data, optimizer, routing treatment, token count, and hardware
remained fixed. Relative loss comparisons remain valid. Reference-attention
wall time is not a production estimate for FA4, so the cost section separates
analytic model FLOPs from backend-specific GPU-hours.

The dependency and kernel investigation is recorded in
[`2026-07-26-nested-moe-fa4-cute.md`](../../.agents/ops/2026-07-26-nested-moe-fa4-cute.md).

## Evaluation

The primary endpoint is Paloma macro validation loss at a common final token
count. Each E256 checkpoint is evaluated in full mode and after compacting the
fixed E128 subset. The standalone E128 control is evaluated once. Secondary
metrics are training loss, validation loss, tokens/s, step time, GPU-hours,
capacity overflow, routing entropy, core/outer assignment counts, assignment
CV, and gradient norms.

Comparisons are reported at equal tokens, analytic model FLOPs, GPU-hours, and
wall time. There is one seed per arm. Paloma domain variation and consistency
across checkpoints are used as uncertainty checks; tokens from one run are not
treated as independent model replications.

The 20-step gate requires three finite steady-state steps, capacity overflow at
or below 1%, nested throughput at least 85% of the large control, and a
successful checkpoint save. The final promotion threshold is stricter at 90%
throughput.

## Results

### Training loss

<!-- NESTED_TRAINING_LOSS_FIGURE -->

### Held-out loss

<!-- NESTED_EVAL_RESULTS -->

### Routing and stability

<!-- NESTED_ROUTING_RESULTS -->

### Throughput and cost

<!-- NESTED_COST_RESULTS -->

The E256 control and both nested E256 arms have the same static model shape and
the same four active routed experts per token. Their analytic model-FLOP ratio
is therefore 1.000, excluding the negligible branch-mask bookkeeping. The
scientific question is whether measured step time remains close to 1.000 and
whether the shared optimization objective changes either checkpoint's loss.

Training two independent models costs the sum of the E256 and E128 GPU-hours.
Nested cost is the nested arm's GPU-hours plus any direct E128 breakout
cooldown. The headline overhead is:

\[
\frac{\text{nested GPU-hours} + \text{breakout GPU-hours}}
     {\text{large-control GPU-hours}} - 1.
\]

Below 10% is viable if both loss gates pass. Between 10% and 20% needs a clear
loss benefit. Above 20% does not promote, and 50% is economically close to
training the smaller model separately.

## Breakout cooldown

<!-- NESTED_BREAKOUT_RESULTS -->

At any checkpoint, the even expert subset can be gathered into an E128
checkpoint and continued with ordinary direct training. The preregistration
allows up to 10% of Gate 2 per-arm FLOPs for this cooldown, with evaluations at
2%, 5%, and 10%. Cooldown compute is charged to the nested method.

## SFT and agentic transfer

<!-- NESTED_SFT_RESULTS -->

Marin contains instruction data and SFT pipelines, including WildChat,
SmolTalk, OpenHermes, Tulu, and Nemotron-derived mixtures. This stage is
conditional: it must use the same pinned mixture and common schedule for the
large control, promoted full model, small control, and promoted extracted
model. SFT loss and a fixed agentic/code smoke are transfer checks, not
evidence of general agentic capability.

## Implications for a 300B–700B run

<!-- NESTED_SCALEUP_DECISION -->

If the intended 300B and 700B models differ mainly in routed-expert count,
expert-subset nesting is mechanically well matched to the decision:

- it produces a fixed 300B-total checkpoint throughout pretraining;
- it can use approximately the 700B model's active FLOPs rather than a second
  300B forward;
- an interleaved subset can preserve expert-parallel balance;
- the extracted model reduces checkpoint and serving memory.

The constraints are equally important:

- the extracted model is not 2.3× cheaper per token when top-k, expert width,
  dense width, and depth are unchanged;
- outer experts see only full-branch sequences and need routing/gradient
  diagnostics at scale;
- a proxy pass requires replication at a larger size and at least two seeds;
- production overhead must be remeasured with a working FA4 backward kernel;
- dense-width or depth nesting should be tested separately before composing
  multiple elastic axes.

A sensible scaled plan is to retain the 25% sequence schedule only if it passes
both loss gates, repeat it at an intermediate size with the intended
expert-parallel topology, and add a small number of fixed breakout checkpoints.
Do not add simultaneous dense-width, depth, top-k, and expert-count nesting to
the first hero run: the interaction would make a failure difficult to
interpret.

## Reproducibility

The launcher is
[`experiments/grug/moe/launch_nested_experts.py`](../../experiments/grug/moe/launch_nested_experts.py).
The implementation and tests are linked from issue
[#652](https://github.com/marin-community/marin/issues/652). W&B runs and exact
Iris job identifiers are listed with the final tables.

Claude Fable review was requested twice before launch. Both bounded local
review processes remained alive without returning review content, and the Loom
launch endpoint returned HTTP 405. No review findings were received; the
interleaved-subset correction came from the subsequent code-grounded
implementation review.
