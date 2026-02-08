# DPO Training in Levanter

Direct Preference Optimization (DPO) fine-tunes a language model to prefer
chosen responses over rejected ones. Levanter supports DPO through a dedicated
training script and preference data format.

## Data Format

Preference datasets must use the `preference_chat` format. Each example contains
two chat transcripts — `chosen` and `rejected` — as lists of
`{"role": ..., "content": ...}` messages.

Example dataset row:

```json
{
  "chosen": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"}
  ],
  "rejected": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "22"}
  ]
}
```

Tokenization uses the model's chat template to format both transcripts.
User turns are masked from the training loss by default.

## Configuration

DPO training uses `TrainDpoConfig`, which extends the standard trainer config
with DPO-specific fields:

```yaml
data:
  # Standard LmDataConfig pointing to preference_chat-formatted caches
  tokenizer: "meta-llama/Llama-3.1-8B-Instruct"
  configs:
    ultrafeedback:
      format:
        type: preference_chat
      train_urls:
        - "gs://path/to/tokenized/train"
      validation_urls:
        - "gs://path/to/tokenized/val"

model:
  type: llama
  # Model config fields...

trainer:
  num_train_steps: 2000
  train_batch_size: 128
  mp: p=f32,c=bfloat16

# DPO-specific fields
reference_model_path: "meta-llama/Llama-3.1-8B-Instruct"
reference_is_hf: true     # true if reference_model_path is a HuggingFace ID
beta: 0.1                 # Regularization strength

initialize_from_hf: "meta-llama/Llama-3.1-8B-Instruct"  # Policy initialization

optimizer:
  learning_rate: 5e-7
  warmup: 0.1
  lr_schedule: linear

validation_split_fraction: 0.1  # Auto-split from training data; null to disable
```

### Key Fields

- **`reference_model_path`** (required): The frozen reference model. Typically
  the same pretrained model used to initialize the policy.
- **`beta`**: Controls how much the policy can deviate from the reference.
  Smaller values (0.01) allow more deviation; larger values (0.5) keep the
  policy closer to the reference.
- **`reference_is_hf`**: Set to `true` when `reference_model_path` is a
  HuggingFace model ID. Set to `false` for a Levanter checkpoint path.
- **`validation_split_fraction`**: Automatically holds out a fraction of the
  training data for validation. Set to `null` to use separately configured
  validation sets.

## Running

```bash
python -m levanter.main.train_dpo --config_path my_dpo_config.yaml
```

## Architecture

DPO training wraps two copies of the model in a `DpoModel`:

- **Policy model** — trainable, updated by the optimizer.
- **Reference model** — frozen, loaded fresh each run (not saved in
  checkpoints).

The DPO loss encourages the policy to assign higher log-probability margins to
chosen responses than the reference does:

```
loss = softplus(-beta * ((log_pi_chosen - log_pi_rejected) - (log_ref_chosen - log_ref_rejected)))
```

Only the policy model's parameters are saved in training checkpoints. The
reference model is reloaded from `reference_model_path` on every
start/resume.

## Metrics

The following metrics are logged during training:

| Metric | Description |
|--------|-------------|
| `dpo_loss` | The DPO training loss |
| `dpo_margin_policy` | Policy log-prob margin (chosen - rejected) |
| `dpo_margin_ref` | Reference log-prob margin (chosen - rejected) |
| `dpo_accuracy` | Fraction of examples where policy margin > reference margin |
| `dpo_chosen_reward` | Implicit reward for chosen responses |
| `dpo_rejected_reward` | Implicit reward for rejected responses |
