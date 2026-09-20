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

## Verdict

Compare each arm's final eval loss against the paper targets in the table
above. The replication succeeds if the ordering and approximate gaps match:
Operator-1 (own recipe) < Operator-1 (vanilla recipe) < Vanilla, with gaps of
roughly the paper's magnitudes (−0.022 and −0.014). Report in the experiment
issue with W&B links.
