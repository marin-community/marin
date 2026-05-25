# exp31: protein iso-FLOP sweep — design

Tracks [Open-Athena/MarinFold#31](https://github.com/Open-Athena/MarinFold/issues/31).
Sibling to exp11 (data-mix), exp29 (arch), and exp30 (LR schedule).

## Goal

Iso-FLOP sweep over three training-FLOP budgets — **3e17, 1e18, 3e18** — on the
protein-docs `contacts-and-distances-v1-5x` corpus, Qwen3 architecture, WSD LR
schedule. One run per (budget, hidden) point satisfying the v5p-8 + batch + LR
+ data-budget constraints.

## Identity

- File: `experiments/protein/exp31_isoflop_sweep.py`, branch `eac/plm-exp31`.
- Run-name prefix: `prot-exp31-iso`. Wandb group: `exp31-isoflop`. `VERSION = "v1"`.
- Sweep root: `gs://marin-us-east5/sweeps/prot-exp31-isoflop/run_isoflop_sweep-{VERSION}`.

## Data

- **Train:** existing `protein-docs-cd` cache. No re-tokenize.
- **Eval:** existing `protein-docs-cd-val` cache, masked (`distance_bin_only_loss_weight`)
  — matches the training loss.
- Tokenizer `PROTEIN_TOKENIZER` pinned at `@83f597d88e9b`, vocab 2840.
- `train_weights = {cd: 1.0, cd-val: 0.0}`; `num_validation_sequences = {}` (no IID carve).

## Model grid (Qwen3 only)

`hidden ∈ range(256, 1281, 128) = {256, 384, 512, 640, 768, 896, 1024, 1152, 1280}`.

- `head_dim = 64`, `intermediate = 4 * hidden`, `num_heads = num_kv_heads = hidden / 64`.
- `num_layers = round(hidden / (64 + log2(hidden) * 4 - 8))` (exp2101 formula).
- Defaults: rope, QK-norm.

(hidden → layers): 256→3, 384→4, 512→6, 640→7, 768→8, 896→9, 1024→11, 1152→12, 1280→13.
Params: ~5M to ~340M.

## Iso-FLOP solver

`BUDGETS` are **training** FLOPs (Chinchilla `C = 6ND`). Levanter's `lm_flops_per_token`
is forward-only, so we multiply by `FWD_TO_TRAIN_FLOPS = 3`.

Per (budget, hidden):

1. `batch_exact = budget / (3 * lm_flops_per_token * STEPS_PER_RUN * SEQ_LEN)`.
2. `batch = round_to_power_of_two_ceil(batch_exact)`.
3. Halve `batch` **only** while `lr > LR_MAX`. Configs outside `[BATCH_MIN, BATCH_MAX]`
   drop in step 4 — not force-halved onto the v5p-8 cap.
4. Skip if `batch < BATCH_MIN=8` or `batch > BATCH_MAX=128`.
5. Recompute `train_steps = round(budget / (flops_per_step))`.
6. Skip if `|achieved - budget| / budget > FLOP_TOLERANCE=0.01`.
7. Skip if `params > 2B` (safety; never fires at h ≤ 1280).
8. Skip if `train_tokens > 5 * cd-train` (safety; never fires at current budgets).

## LR / β₂

```
lr(b, h) = LR_CONSTANT * sqrt(b) / h   # LR_CONSTANT from lr=3.5e-4 @ batch=128, hidden=2048
beta2(b) = BETA2_BASE ** (b / 128)     # BETA2_BASE = 0.95
```

## Trainer

| field | value |
|---|---|
| `train_batch_size` | per-trial `batch` |
| `num_train_steps` | per-trial `train_steps` |
| `learning_rate` | `versioned(lr)` |
| `beta2` | `scaled_beta2(batch)` |
| `weight_decay` / `warmup` / `lr_schedule` / `decay` | `0.01` / `0.1` / `"linear"` / `0.2` (WSD) |
| `train_seq_len` | 8192 |
| `steps_per_eval` | `num_train_steps` (1 eval, at end) |
| `max_eval_batches` | `None` (no limit) |
| `steps_per_export` | `None` (no permanent checkpoints) |
| `per_device_parallelism` | `-1` |
| `data_seed` | 31 |

`WatchConfig(watch_targets=[], interval=0)` patched in. Default temp-checkpoint cadence.

## Resources

`v5p-8` default; `TPU` env override. Region picked up from the iris CLI.

## Run name

`prot-exp31-iso-F{budget}-P{params_fmt}-T{tokens_fmt}-{VERSION}`

Example: `prot-exp31-iso-F1e18-P28.1M-T2.150B-v1`. Shape details (hidden, layers,
batch, lr) stay in the wandb config artifact + tags.

## Wandb tags

```
protein, exp31, isoflop, qwen3
budget        budget_exact
params        params_exact
tokens        tokens_exact
batch
lr            lr_exact
beta2
```

## Worker / launcher

Single `main()`. `NUM_WORKERS` (default 4) Fray workers, each takes `targets[rank::N]`
and calls `claim_and_run`. Selection: `RUNS` (CSV substring filter), `PREVIEW=yes`
(print table, no submit), `LIST_RUNS=yes` (print resolved trial names — pipe-friendly
for the per-trial bash for-loop in the module docstring).

## Preview

At `STEPS_PER_RUN=4000`: 20/27 survivors (7/7/6 across the three budgets). The
budget span (10x) and batch span ([8,128]=16x) are close enough that a single S
covers most cells; small-h × high-budget and large-h × low-budget corners drop.
