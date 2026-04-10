# DPO Training in Levanter

Direct Preference Optimization (DPO) fine-tunes a language model to prefer
chosen responses over rejected ones. Levanter supports both standard DPO and
LoRA-DPO through the same `train_dpo.py` entrypoint and preference data format.

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
adapter:
  type: none

reference:
  type: separate
  model_path: "meta-llama/Llama-3.1-8B-Instruct"
  is_hf: true

beta: 0.1                 # Regularization strength

initialize_from_hf: "meta-llama/Llama-3.1-8B-Instruct"  # Policy initialization

optimizer:
  learning_rate: 5e-7
  warmup: 0.1
  lr_schedule: linear

validation_split_fraction: 0.1  # Auto-split from training data; null to disable
```

### Key Fields

- **`adapter`**: How the policy model is adapted before training.
  Standard full-parameter DPO uses `adapter.type: none`.
- **`reference`**: How the frozen reference log-probabilities are obtained.
  Standard DPO uses `reference.type: separate` and points at a frozen model.
- **`beta`**: Controls how much the policy can deviate from the reference.
  Smaller values (0.01) allow more deviation; larger values (0.5) keep the
  policy closer to the reference.
- **`validation_split_fraction`**: Automatically holds out a fraction of the
  training data for validation. Set to `null` to use separately configured
  validation sets.

For LoRA-DPO, flip only the adapter/reference blocks:

```yaml
adapter:
  type: lora
  r: 64
  alpha: 64.0
  dropout: 0.0
  zero_init_b: true
  target_modules: null  # null = all linear modules (recommended)

reference:
  type: adapter_base

optimizer:
  # LoRA needs ~10x higher LR than full fine-tuning.
  # Standard DPO full-FT uses 5e-7; LoRA DPO should start at 5e-6.
  learning_rate: 5e-6
```

With `adapter_base`, the base model (without LoRA adapters) serves as the
frozen reference, eliminating the second model copy and roughly halving
model-parameter memory.

#### B Matrix Initialization

**`zero_init_b: true` is required for LoRA-DPO.** Zero-initializing the B
matrix makes the adapter start as identity (`W + alpha/r * 0 * A = W`), so
the policy exactly matches the reference at step 0. Without this, the policy
immediately diverges, producing catastrophically wrong log-probability margins
and a loss that starts at ~12 instead of ~0.69. `train_dpo.py` rejects
`zero_init_b: false` for LoRA-DPO configs.

#### LoRA Checkpoint Saving

LoRA DPO supports two checkpoint formats:

```yaml
peft_save_path: gs://bucket/checkpoints/peft       # Adapter-only (small)
merged_hf_save_path: gs://bucket/checkpoints/merged # Full merged model
hf_save_steps: 1000
```

- **PEFT checkpoints** save only LoRA adapter weights. Load with
  `peft.PeftModel.from_pretrained()`.
- **Merged checkpoints** fold LoRA weights into the base model and save a
  standard HuggingFace checkpoint ready for direct inference.

### Reference Eval Cache

Set `reference_eval_cache.mode: build_or_load` to precompute validation-set
reference log-probs before training starts, write them to a durable sidecar
cache, and reuse them on later resumes or reruns:

```yaml
reference_eval_cache:
  mode: build_or_load
  # Optional override. By default Levanter writes a hashed cache under a
  # sibling `reference_logprobs/` directory next to the validation cache.
  cache_dir: "gs://my-bucket/dpo/reference_eval"
```

This cache is eval-only. Training still computes reference log-probs in the
normal way. The first run pays a one-time build cost; later runs load the
completed cache and skip the reference forward passes during validation. If a
job is preempted mid-build, the unfinished cache is ignored and rebuilt on the
next start.

### Generation Stop Tokens

Chat models typically use a turn-boundary token (e.g. `<|eot_id|>`, ID 128009)
to end assistant responses, while the tokenizer's `eos_token` remains the
pre-training document boundary (e.g. `<|end_of_text|>`, ID 128001). Inference
tools like vLLM use `eos_token_id` from `config.json` to decide when to stop,
so they will miss the chat stop token unless told otherwise.

Set `hf_generation_eos_token_ids` to write a `generation_config.json` alongside
each HF checkpoint with the correct stop tokens:

```yaml
hf_generation_eos_token_ids: [128001, 128009]  # <|end_of_text|> + <|eot_id|>
```

The tokenizer's `eos_token_id` is auto-added if not in the list. This field
defaults to `null` (no `generation_config.json` written), preserving backward
compatibility with pretraining checkpoints.

To determine the right stop token for your model's chat template:

```python
tokens = tokenizer.apply_chat_template(
    [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
    tokenize=True,
)
# The last token is the chat stop token (e.g. 128009 for Llama 3)
```

## Running

```bash
python -m levanter.main.train_dpo --config_path my_dpo_config.yaml
```

## Architecture

The runtime model shape depends on the configured reference path:

- **`reference.type: separate`** keeps `DpoModel(policy, reference)` in
  trainer state so the frozen reference is passed into the train step as an
  explicit input. Only the policy side is trainable/saveable.
- **`reference.type: adapter_base`** keeps only the adapted policy model in
  trainer state and derives the reference view from the adapter-free base
  model inside the step.

The DPO loss encourages the policy to assign higher log-probability margins to
chosen responses than the reference does:

```
loss = softplus(-beta * ((log_pi_chosen - log_pi_rejected) - (log_ref_chosen - log_ref_rejected)))
```

Only the policy parameters are saved in training checkpoints. When
`reference.type: separate` is used, the frozen reference weights are reloaded
from the configured path on every start/resume.

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
