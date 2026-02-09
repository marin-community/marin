# LoRA DPO Training in Levanter

LoRA DPO combines Low-Rank Adaptation with Direct Preference Optimization.
The key insight: with LoRA, the base model (without adapters) naturally serves
as the reference model, eliminating the need for a second full model copy and
saving ~50% of model parameter memory compared to standard DPO.

## How It Works

In standard DPO, you load two full copies of the model — a trainable policy and
a frozen reference. With LoRA DPO:

- **Policy model** = base model + LoRA adapters (trainable)
- **Reference model** = base model without LoRA adapters (frozen, implicit)

The `unwrap_lora_modules()` function strips LoRA adapters at each forward pass
to produce reference log-probabilities. `jax.lax.stop_gradient` prevents
backpropagation through the reference path, so only LoRA parameters receive
gradients.

## Data Format

Same as standard DPO — see [DPO-Training.md](DPO-Training.md#data-format).
Preference datasets must use the `preference_chat` format with `chosen` and
`rejected` chat transcripts.

## Configuration

LoRA DPO uses `LoraDpoConfig`. Unlike `TrainDpoConfig`, there is no
`reference_model_path` — the reference is the base model, implicit.

```yaml
data:
  tokenizer: marin-community/marin-tokenizer
  shuffle: true
  components:
    my_prefs:
      source:
        type: url
        train_urls:
          - gs://path/to/preference/data/*.jsonl.gz
      format:
        type: preference_chat
        slice_strategy: raise

train_seq_len: 4096

trainer:
  num_train_steps: 2000
  train_batch_size: 128
  mp: p=f32,c=bfloat16

optimizer:
  # LoRA needs ~10x higher LR than full fine-tuning (Thinking Machines 2025).
  # Standard DPO full-FT uses 5e-7; LoRA DPO should start at 5e-6.
  learning_rate: 5e-6
  warmup: 0.1
  lr_schedule: cosine
  max_grad_norm: 1.0

lora:
  r: 64             # Rank: 64 for medium datasets, 128-256 for large
  alpha: 64.0       # Alpha = rank is the safe default
  dropout: 0.0      # Increase to 0.05-0.1 if overfitting
  zero_init_b: true # CRITICAL for DPO — see B Matrix Initialization below
  target_modules: null  # null = all linear modules (recommended)

initialize_from_hf: meta-llama/Llama-3.1-8B
beta: 0.1
validation_split_fraction: 0.1
```

### Key Fields

- **`initialize_from_hf`** (required): HuggingFace model to load and apply
  LoRA to. This serves as both the policy base and (implicitly) the reference.
- **`lora.r`**: LoRA rank. See [Choosing Rank](#choosing-rank) below.
- **`lora.alpha`**: Scaling factor. Use `alpha = r` as default.
- **`lora.zero_init_b`**: Zero-initialize the LoRA B matrix. **Must be `true`
  for DPO** — see [B Matrix Initialization](#b-matrix-initialization) below.
  The script will override this to `true` and warn if not set.
- **`lora.target_modules`**: Which linear layers to LoRA-ize. `null` means all
  linear modules (recommended). Can also be a list of module names or a regex.
- **`beta`**: DPO temperature. Same as standard DPO.
- **`peft_save_path`**: Save PEFT-compatible adapter checkpoints.
- **`merged_hf_save_path`**: Save fully-merged HuggingFace checkpoints.

### Checkpoint Saving

LoRA DPO supports two checkpoint formats:

```yaml
peft_save_path: gs://bucket/checkpoints/peft       # Adapter-only (small)
merged_hf_save_path: gs://bucket/checkpoints/merged # Full merged model
hf_save_steps: 1000
```

- **PEFT checkpoints** save only the LoRA adapter weights. Small and fast.
  Load with `peft.PeftModel.from_pretrained()`.
- **Merged checkpoints** fold LoRA weights into the base model and save a
  standard HuggingFace checkpoint. Larger but ready for direct inference.

## Running

```bash
python -m levanter.main.lora_dpo --config_path config/lora_dpo_ultrafeedback_llama3_8b.yaml
```

## Choosing Rank

From the best practices report (`lora-dpo-best-practices.md`):

| Dataset Size | Recommended Rank | Notes |
|-------------|-----------------|-------|
| < 5K pairs | 16-64 | Lower capacity to avoid overfitting |
| 5K-50K pairs | 64-128 | Good balance of capacity and efficiency |
| 50K+ pairs | 128-256 | Use rsLoRA scaling for ranks > 64 |
| RL-based (online DPO) | 1-16 | Low information per episode |

Start with rank 64. If reward margins plateau, increase. If training loss
drops below 0.2 quickly, decrease to reduce overfitting.

## Learning Rate

This is the most critical hyperparameter. DPO requires much lower learning
rates than SFT, but LoRA needs higher rates than full fine-tuning. These
roughly cancel:

| Setting | Recommended LR |
|---------|---------------|
| SFT + LoRA | 2e-4 |
| **DPO + LoRA** | **5e-6** |
| DPO + full FT | 5e-7 |

Tuning strategy:
1. Start at 5e-6
2. If reward margins diverge/oscillate: reduce to 1e-6 or 5e-7
3. If reward margins are flat after 100+ steps: increase to 1e-5
4. Never exceed 5e-5

## B Matrix Initialization

**This is the most critical setting for LoRA-DPO.** The LoRA B matrix must be
zero-initialized so the adapter starts as identity: `W + (alpha/r) * 0 * A = W`.
This ensures the policy model exactly matches the reference model at step 0.

Without zero initialization, the policy immediately diverges from the reference,
producing catastrophically wrong log-probability margins (e.g., -597 instead of
near-zero) and a loss that starts at ~12 instead of ~0.69. Training never
recovers.

The `lora_dpo.py` script enforces this automatically: if `lora.zero_init_b` is
not set to `true`, it will override it and log a warning. Always set it
explicitly in your config:

```yaml
lora:
  zero_init_b: true
```

For SFT, random B initialization may work fine since there is no reference model
to diverge from. The default is `false` for backward compatibility with existing
SFT configs.

## Known Limitations

### lm_head is excluded from LoRA by default

**TODO**: The `lm_head` layer is excluded from LoRA targeting by default. This
is a Levanter code limitation, not a best-practices recommendation — research
suggests applying LoRA to all weight matrices is beneficial (Thinking Machines
2025).

The issue: every model class in Levanter (Llama, Qwen, Gemma, Mixtral, OLMo,
etc.) implements `get_lm_head()` as `self.lm_head.weight`. When `lm_head`
becomes a `LoraLinear`, it no longer has a `.weight` attribute (it has
`.wrapped.weight` instead), causing an `AttributeError`.

The proper fix is to update `get_lm_head()` (and `resize_vocab()`) in all
model classes to handle `LoraLinear`. Until then, `lm_head` is excluded via
`LoraConfig.exclude_modules` (defaults to `["lm_head"]`). To explicitly
include it after fixing the model classes, set `exclude_modules: []`.

### No rsLoRA support yet

Levanter's `LoraConfig` does not support rank-stabilized LoRA (rsLoRA), which
uses `alpha/sqrt(r)` scaling instead of `alpha/r`. rsLoRA is particularly
effective for ranks > 64. See `lora-dpo-best-practices.md` Section 9.

## Metrics

Same metrics as standard DPO, plus LoRA-specific logging:

| Metric | Description |
|--------|-------------|
| `dpo_loss` | The DPO training loss |
| `dpo_margin_policy` | Policy log-prob margin (chosen - rejected) |
| `dpo_margin_ref` | Reference log-prob margin (chosen - rejected) |
| `dpo_accuracy` | Fraction where policy margin > reference margin |
| `dpo_chosen_reward` | Implicit reward for chosen responses |
| `dpo_rejected_reward` | Implicit reward for rejected responses |
| `parameter_count` | Total model parameters |
| `trainable_parameter_count` | LoRA parameters only |
| `fraction_trainable` | Ratio of trainable to total parameters |

## Comparison with Standard DPO

| | Standard DPO | LoRA DPO |
|---|---|---|
| Model copies | 2 (policy + reference) | 1 (LoRA + implicit reference) |
| Memory | ~2x model size | ~1x model size + LoRA overhead |
| Trainable params | All policy params | Only LoRA params |
| Config field | `reference_model_path` | Not needed |
| Script | `train_dpo.py` | `lora_dpo.py` |
