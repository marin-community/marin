# Continual Alignment: Per-Statement DPO Experiments

## Motivation

Can we run DPO on a SINGLE statement (or a small 3-statement subset)
and see targeted improvement without degrading others? This tests
whether alignment is modular — can we "add" compliance to one behavior
at a time, either from SFT or from an already-aligned DPO checkpoint?

## Statement selection (GPT-4.1 judge, the validated baseline)

Using GPT-4.1 judge deltas (LoRA lr=1e-5 vs SFT, prompt-collapsed)
as the primary ranking because the earlier Bloom validation was
anchored on GPT-4.1:

| # | Statement | Δ(g41) | SFT | DPO | Why selected |
|---:|---|---:|---:|---:|---|
| 1 | `support_mental_health` | **+4.20** | 4.62 | 8.82 | Biggest gain, clear room for improvement |
| 2 | `do_not_encourage_self_harm` | +2.58 | 6.26 | 8.84 | Safety-relevant, large gain |
| 3 | `avoid_overstepping` | +2.03 | 7.33 | 9.36 | Well-defined behavioral boundary |

**Why not `be_engaging` or `be_kind`**: both are subjective/tonal
statements prone to ceiling effects and sycophancy artifacts. The
selected 3 are more clearly-defined behavioral boundaries where a
"win" is less likely to be a tone-hacking artifact.

**Single-statement target**: `support_mental_health` (biggest Δ, lowest
SFT baseline, most room for improvement).

## Dataset sizes

Per-statement from the existing Bloom preference data:
- **1 statement** (`support_mental_health`): 250 train prompts, 2,250 pairs (cross9)
- **3 statements** (above): ~750 train prompts, ~6,750 pairs (cross9)

## Experiment matrix (4 experiments, ~18 training runs)

| Exp | Base model | Ref model (for DPO) | Statements | Pairs | Runs | Notes |
|---:|---|---|---|---:|---:|---|
| 1a | `marin-8b-instruct` (SFT) | SFT | `support_mental_health` | 2,250 (cross9) | 9 (3 LR × 3 steps) | Core single-statement test + LR sweep |
| 1b | `marin-8b-instruct` (SFT) | SFT | 3 statements | ~6,750 (cross9) | 3 (best LR × 3 steps) | Does 3-statement give proportional gains? |
| 2a | LoRA DPO lr=1e-5 (best DPO) | **same DPO checkpoint** | `support_mental_health` | 2,250 (cross9) | 3 (best LR × 3 steps) | Continual alignment from already-good model |
| 2b | LoRA DPO lr=1e-5 (best DPO) | **same DPO checkpoint** | 3 statements | ~6,750 (cross9) | 3 (best LR × 3 steps) | Same, bigger dataset |

**Reference model for continual DPO (2a, 2b)**: use the DPO checkpoint
itself as both init AND reference. Using the old SFT model as reference
would make the KL penalty much harsher (the DPO model has already
drifted from SFT), producing an unfairly constrained second stage.

## Hyperparameters

| Param | Range | Rationale |
|---|---|---|
| β (DPO temperature) | **0.1** | Same as full run. |
| Learning rate | sweep {**1e-7, 5e-7, 1e-6**} | Much lower than the full run's 1e-5. With 250 prompts, even 1e-6 is aggressive — the model will see every prompt ~50× in 1699 steps. |
| Batch size | **64** (fixed) | Matches the validated full-DPO config. 2,250 pairs / 64 = ~35 steps/epoch. |
| Steps (1 stmt) | sweep {**35, 70, 140**} (1, 2, 4 epochs) | No more than 4 epochs to limit overfitting. |
| Steps (3 stmt) | {**105, 210, 420**} (same epoch counts) | Proportional to 3× more data. |
| LoRA rank | 16 (fixed) | |
| LoRA α | 32 (fixed) | |

**Total lr × steps grid**: 3 lr × 3 steps = 9 configs for exp 1a.
Then pick best lr based on val loss + target-statement eval. Use that
lr for the remaining 3 experiments with 3 step variants each.

Total: 9 + 3 + 3 + 3 = **18 training runs**.

## Dual validation: per-statement + full-set regression tracking

Every training run evaluates on **two** preference validation sets:

1. **`stmt_val`** — the per-statement (or 3-statement) val split (~54 or ~162 pairs).
   Tracks whether the model is learning the target preference.
2. **`full_val`** — the full 46-statement val set (2,606 pairs, deduped).
   Tracks whether training on a subset regresses the full distribution.

Both are configured via `lm_data_config(validation_sets={...})`:

```python
tokenized_preferences = lm_data_config(
    training_set=tokenized_train,
    validation_sets={
        "stmt_val": tokenized_eval_stmt,   # per-statement val
        "full_val": tokenized_eval_full,    # full 46-statement val
    },
)
```

Each validation set gets its own reference eval cache (keyed by a hash
of source data + reference model), so `reference_eval_cache` with
`mode="build_or_load"` works for both without collision.

**W&B metrics:** `eval/stmt_val/loss`, `eval/full_val/loss`. If
`full_val` loss climbs while `stmt_val` loss drops, the run is
regressing — cheap early-stopping signal without waiting for judging.

**Full val data path (mirrored for region-agnostic access):**
```python
FULL_VAL_DATA = mirrored(
    "preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/val_deduped/shard-00000.jsonl.gz",
    budget_gb=1,
)
```

## Evaluation protocol

### Primary judge: GPT-4.1

The earlier SFT→DPO deltas were validated on GPT-4.1. For apples-to-
apples comparison, **GPT-4.1 stays primary**. Run via the existing
`run_bloom_judge.py` pipeline (already validated, no reasoning issues).

### Secondary judge: GPT-5.1

Run as a secondary check via `judge_gpt51_batch.py` with
`reasoning_effort="none"`. Useful for the more granular per-statement
analysis (GPT-5.1 is more discriminating on subjective statements).

### What to measure

For each experiment, judge ALL 46 statements (not just the target):

1. **Target statement(s) improvement**: mean score Δ vs the base model
   on the target statement(s). Success = significant positive Δ.
2. **Non-target regression**: per-statement Δ on the other 43-45
   statements. Report individually.
3. **Overall 46-statement mean**: must not drop by more than the
   target improvement (net positive). This is the **guardrail** —
   if the overall mean drops, the run failed even if the target
   improved.
4. **Holdout-only reporting**: the eval split (53 prompts/statement)
   is held out from training. All numbers reported on eval only.

### Success criteria

- **Strong win**: target statement improves ≥ 1.5 points AND overall
  46-statement mean does not drop (net positive or neutral).
- **Weak win**: target statement improves ≥ 0.5 points AND no
  individual non-target statement drops by > 0.5.
- **Fail**: overall mean drops, or non-target regressions exceed the
  target gain.

## Execution order

1. **Exp 1a** (SFT → 1 stmt, cross9): sweep 9 configs (3 LR × 3 steps). Pick best lr.
2. **Exp 1b** (SFT → 3 stmts): best lr, 3 step variants.
3. **Exp 2a** (DPO → 1 stmt): best lr, 3 step variants.
4. **Exp 2b** (DPO → 3 stmts): best lr, 3 step variants.
5. **Judge all** with GPT-4.1 (primary) + GPT-5.1 (secondary).
6. **Analyze**: per-statement deltas, overall mean, regression budget.

## Infrastructure

- TPU: v6e-8 (validated config: `per_device_parallelism=4`, 2× grad accumulation)
- Training: LoRA rank=16
- Per-run training time: ~8-33 min (35-140 steps at ~14s/step)
- Per-run judging: ~$17-21 (GPT-5.1 batch) + comparable for GPT-4.1
- Total estimated: ~18 training runs × ~20 min + ~8 judging runs
  × $20 ≈ ~6 hours TPU + ~$200 judging

## Key questions this answers

1. **Is alignment modular?** Can you improve one statement without
   regressing others?
2. **Continual DPO**: can a second stage of DPO (from an already-
   aligned checkpoint, using itself as reference) further improve a
   specific weak spot?
3. **Dataset scale**: 250 prompts (1 stmt) vs 750 (3 stmts) — is
   there a minimum viable dataset size for per-statement DPO?

## Data paths

### Source preference data (full 46-statement, GPT-4.1 chosen vs Mixtral rejected)

- **Full train**: `gs://marin-us-central1/preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/train/` (22 shards, 108,765 pairs, 12,085 prompts × 9 cross-product)
- **Full val**: `gs://marin-us-central1/preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/val_deduped/` (1 shard, 2,606 pairs, deduped to 1 pair/prompt)

### Per-statement filtered datasets (generated 2026-04-13)

Filtered from the full dataset by `statement_id` using
`experiments/posttrain/filter_preference_by_statement.py`. Same
format (gzipped JSONL, chosen/rejected message arrays).

**Individual statements** (for single-statement DPO experiments):

| Statement | Train pairs | Unique prompts | Val pairs |
|---|---:|---:|---:|
| `support_mental_health` | 2,250 | 250 | 54 |
| `do_not_encourage_self_harm` | 2,250 | 250 | 54 |
| `avoid_overstepping` | 2,250 | 250 | 54 |

Each train set: 250 unique prompts × 9 cross-product pairs (3 chosen
× 3 rejected). Val is deduped (1 pair per prompt).

**Combined 3-statement set** (for multi-statement DPO experiments):

| Name | Train pairs | Unique prompts | Val pairs |
|---|---:|---:|---:|
| Combined 3 | 6,750 | 750 | 162 |

**Replicated to all 4 regions** (verified 2026-04-13):

| Region | GCS prefix |
|---|---|
| us-central1 | `gs://marin-us-central1/preference/bloom_v2_singleton/` |
| eu-west4 | `gs://marin-eu-west4/preference/bloom_v2_singleton/` |
| us-east5 | `gs://marin-us-east5/preference/bloom_v2_singleton/` |
| us-east1 | `gs://marin-us-east1/preference/bloom_v2_singleton/` |

Each region has all 4 subdirectories:

```
bloom_v2_singleton/
    support_mental_health/
        train/shard-00000.jsonl.gz          # 2,250 pairs
        val/shard-00000.jsonl.gz            # 54 pairs (deduped)
    do_not_encourage_self_harm/
        train/shard-00000.jsonl.gz          # 2,250 pairs
        val/shard-00000.jsonl.gz            # 54 pairs (deduped)
    avoid_overstepping/
        train/shard-00000.jsonl.gz          # 2,250 pairs
        val/shard-00000.jsonl.gz            # 54 pairs (deduped)
    support_mental_health+do_not_encourage_self_harm+avoid_overstepping/
        train/shard-00000.jsonl.gz          # 5,000 pairs
        train/shard-00001.jsonl.gz          # 1,750 pairs
        val/shard-00000.jsonl.gz            # 162 pairs (deduped)
```

Training jobs should read from the regional bucket matching their TPU
location (e.g., eu-west4 TPUs read from `gs://marin-eu-west4/...`).

### Models

- **SFT base**: `gs://marin-us-east1/models/marin-community--marin-8b-instruct--0378f9c`
- **DPO base** (best existing, for continual alignment): `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699`

### Evaluation

- **Eval prompts**: `gs://marin-us-central1/alignment/gpt-4.1-eval-split/` (2,576 prompts, 46 statements, eval split)
- **GPT-4.1 judge** (primary): `experiments/posttrain/run_bloom_judge.py`
- **GPT-5.1 judge** (secondary): `experiments/posttrain/judge_gpt51_batch.py` with `reasoning_effort="none"`, `max_completion_tokens=4000`
- **GPT-5.1 reparser**: `experiments/posttrain/reparse_gpt51.py`

### Bloom project source

- Bloom inference chosen: `/Users/ahmed/code/bloom/results/inference/dev-bloom-results-gpt-4-mini-prompts/gpt-4-1-2025-04-14/`
- Bloom inference rejected: `/Users/ahmed/code/bloom/results/inference/dev-bloom-results-gpt-4-mini-prompts/mistralai_Mixtral-8x7B-Instruct-v0-1/`
- Export script: `/Users/ahmed/code/bloom/scripts/export_marin_preference.py`
- Config template: `/Users/ahmed/code/bloom/config/preference/v2_gpt41_vs_mixtral_train_val.yaml`
- Filter script: `experiments/posttrain/filter_preference_by_statement.py`
