# grug-paper-rep

Dense replication of arXiv 2609.19107 ("How Model Growth, Recursion, and
Boundary Operators Influence Scaling Exponents") at the paper's base size,
under the paper's own hyperparameters. Copied from `experiments/grug/base/`;
see that directory for the template. This is the honest-replication arm of the
effort: the MoE-vehicle ablation in `experiments/grug/moe_boundary/` showed
most of the paper's recipe knobs do not exist in that template, so this
variant implements the paper's optimizer and init scheme directly.

## What is replicated

The paper's base-size experiment (Table 6): d8 (depth 8, width 1024, ~210M
params, vocab 50,304 GPT-2), 1B tokens, batch 524,288, 2048-token sequences,
GPT-2 tokenizer on FineWeb. Three arms:

| Arm | Model | Recipe | Paper target (val loss) |
| --- | --- | --- | --- |
| `vanilla` | plain transformer | Vanilla (Table 5) | 3.3279 |
| `op1` | Operator-1 (K=1) | Operator-1 (Table 5) | 3.3057 |
| `op1-vanilla-recipe` | Operator-1 (K=1) | Vanilla (Table 5) | 3.3135 |

The third arm is the paper's recipe-transfer probe: Operator-1 still beats
Vanilla under the transferred recipe if the paper's base-size claim holds.

## Model (`model.py`)

Paper A.1 architecture: pre-norm decoder-only transformer, full RoPE, SwiGLU
MLPs (d_ff = 3·width), GQA-compatible attention with non-parametric QK-norm
before RoPE, no biases, no learned norm gains, an RMSNorm after the token
embedding and before the LM head. Init scheme (Table 5): token embedding
~ N(0, WTE²); attention/MLP input matrices ~ U(-UIS, UIS); output projections
(w_o, MLP down, LM head) zero-initialized. The residual multiplier (RM) scales
attention output and MLP down projections at forward time; the output
multiplier (OM) scales the LM head — the paper's F(αΘ) as forward-time
scalings of stored weights.

The boundary operator is the paper's Operator-1 (K=1), identical to
`experiments/grug/moe_boundary/`: prelude/core/coda split per Table 2 (d8 →
2/3/3), core entry state `α·e`, coda entry state `rms_norm(h) + α·e`, α = 1.
With `boundary_operator=False` the model is the paper's Vanilla.

## Optimizer (`optimizer.py` + `recipes.py`)

Three-group optimizer registered as `paper_muon`, matching the paper's
setup: Muon (modded-nanogpt scaling, momentum 0.95, 5 Newton-Schulz steps) for
all matrix parameters at GLR; AdamW for the token embedding at GLR·ELRM and
the LM head at GLR·HLRM. The LR schedule is the paper's: linear warmup for
WU steps, then a linear warmdown over the last WDR fraction of training to
zero; all three groups share the schedule shape.

Recipe values (Table 5) live in `recipes.py` (`VANILLA_RECIPE`,
`OPERATOR1_RECIPE`).

Interpretation choices (documented deviations):

- **WD** (weight decay) is applied as decoupled AdamW decay on the embed/head
  groups only; the Muon group trains without weight decay. The paper gives a
  single WD knob without specifying which optimizer it belongs to.
- **Muon internals** (momentum, nesterov, Newton-Schulz steps) are not
  specified in the paper and use the modded-nanogpt defaults.
- **FineWeb** is the 10BT uniformly-sampled slice, not the full corpus (the
  runs consume 1B of its ~10B tokens); validation is the first ~400k docs of
  one held-out full-corpus crawl file, which the 10BT sample (uniformly drawn
  from the full corpus) overlaps by a fraction of a percent of docs.
- **Sequence packing**: the tokenize pipeline concatenates documents to fill
  2048-token sequences (standard Levanter packing); the paper's attention
  masking across document boundaries is not documented.

## Run

Data materializes once (the tokenized caches are shared by all arms; each arm
consumes the same 1B tokens with the same seed, so arms differ only in
model/recipe):

```bash
source ~/.envvars.local
.venv/bin/iris --cluster=marin job run --no-wait \
    -e WANDB_API_KEY "$WANDB_API_KEY" \
    -- python -m experiments.grug.paper_rep.launch \
        --version dev --run --arm vanilla
```

`--arm` ∈ {`vanilla`, `op1`, `op1-vanilla-recipe`} or `data` (materialize the
caches without training). Training lands on a preemptible v4-16 (the v4-8
pools were degraded at launch time; 16 devices divide the batch evenly and
leave the global batch and data order unchanged); each arm is well under an
hour. Evals (`GrugEvalConfig`) run every 500 steps on the held-out FineWeb
file with the current (non-EMA) weights.

## Verdict (2026-09-20)

All three arms ran to completion (1907 steps, seed 0, preemptible v4-16,
shared caches). Final `eval/loss` on the held-out FineWeb file:

| Arm | Ours | Paper (Table 6) |
| --- | --- | --- |
| Vanilla | **3.2661** | 3.3279 |
| Operator-1 (own recipe) | **3.4738** | 3.3057 |
| Operator-1 (vanilla recipe) | **3.2561** | 3.3135 |

The transfer probe replicates (Operator-1 under the vanilla recipe is
−0.010 better than vanilla; the paper has −0.014): the boundary-operator
architecture itself is neutral-to-slightly-positive at this scale. The
central claim does not replicate: the Operator-1 own recipe is +0.208
WORSE than vanilla instead of −0.022 better, behind from the first eval
and never recovering — a hyperparameter-configuration effect, not a
collapse. This is consistent with the Phase-0 MoE ablation's negative
result.

Full analysis: https://github.com/marin-community/marin/issues/9292
W&B runs: `paper_rep_d8_vanilla`, `paper_rep_d8_op1`,
`paper_rep_d8_op1_vanilla_recipe` (project `marin`).

## Screening sweep: what drives the own-recipe penalty?

The full-length verdict above shows the Operator-1 own recipe is +0.208
worse than the same model under the vanilla recipe, behind from the first
eval and partially healing — an early-training injury. The screen flips one
suspect cluster at a time between the two Table 5 columns, always on the
Operator-1 model, at 1000 steps with 250-step evals (screening runs get
their own coherent schedule, so they are comparable to each other but not
to the full-length arms):

| Arm | Recipe | Tests |
| --- | --- | --- |
| `screen-base` | Vanilla | good-config reference |
| `screen-own` | Operator-1 | bad-config reference |
| `screen-wu0` | Vanilla + WU=0 | does killing warmup alone break the good config? |
| `screen-init-own` | Vanilla + own WTE/UIS/RM/OM | does the init/scale cluster alone break it? |
| `screen-opt-own` | Vanilla + own ELRM/HLRM/WD/WDR/β2 | does the optimizer cluster alone break it? |
| `screen-own-wu40` | Operator-1 + WU=40 | does restoring warmup alone repair the bad config? |

`screen-wu0` and `screen-own-wu40` test the top suspect (no warmup at full
GLR) in both directions. Submit with the same command as the main arms,
e.g. `--arm screen-wu0`. Results land in W&B as
`paper_rep_d8_screen_<name>`.

### First-stage results (2026-09-21)

All arms ran on one preemptible v4-32 (the v4-16 pool was in boot backoff
that day; same zone and global batch, so arms stay comparable to each
other). Eval loss per step:

| Arm | @250 | @500 | @750 | @1000 | Δ@1000 vs base |
| --- | --- | --- | --- | --- | --- |
| `screen-base` | 4.103 | 3.721 | 3.512 | 3.406 | — |
| `screen-own` | 4.514 | 4.147 | 3.892 | 3.774 | +0.368 |
| `screen-wu0` | 4.042 | 3.697 | 3.497 | 3.393 | −0.013 |
| `screen-init-own` | 4.607 | 4.174 | 3.850 | 3.715 | **+0.309** |
| `screen-opt-own` | 4.100 | 3.688 | 3.511 | 3.423 | +0.017 |
| `screen-own-wu40` | 4.542 | 4.148 | 3.874 | 3.757 | +0.351 |

Attribution: the penalty is driven by the model-side init cluster
(WTE 0.113, UIS 0.354, RM 0.5, OM 1 vs vanilla's 0.007, 0.063, 0.25, 0.5):
it alone reproduces 84% of the own-recipe gap and is worse than the full
own recipe at the first eval. Warmup is neutral in both directions (wu0
is not worse than base; restoring warmup does not repair the own recipe),
and the optimizer cluster (ELRM/HLRM/WD/WDR/β2) is neutral on its own.
The remaining +0.06 is an interaction: the optimizer knobs add a little
damage when combined with the init cluster.

A second-stage split (`screen-init-embed`: WTE/UIS only;
`screen-init-depth`: RM/OM only) refines the attribution to individual
knobs.
