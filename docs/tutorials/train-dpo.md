# Training with DPO (Direct Preference Optimization)

## Prerequisites

- [Installation](installation.md) — Set up the Marin environment.
- [Train an LM](train-an-lm.md) — Understand how training pipelines work.
- [Executor 101](executor-101.md) — Understand the executor framework.

## Overview

DPO fine-tunes a language model so that it prefers "chosen" responses over
"rejected" ones, using a reference model to regularize the update. This tutorial
walks through setting up a DPO run end-to-end in Marin.

## Required Imports

```python
from experiments.defaults import default_dpo, default_tokenize
from experiments.llama import llama_3_1_8b
from experiments.simple_dpo_config import SimpleDPOConfig
from levanter.data.text import PreferenceChatLmDatasetFormat
from marin.execution.executor import executor_main
from marin.resources.resource_configs import v5p_32
```

## Tokenizing Preference Data

Preference datasets contain pairs of conversations: a preferred ("chosen")
response and a dispreferred ("rejected") response. Use
`PreferenceChatLmDatasetFormat` to tokenize them:

```python
tokenized_train = default_tokenize(
    "my_preference_data/train",
    "HuggingFaceH4/ultrafeedback_binarized",
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    format=PreferenceChatLmDatasetFormat(),
)

tokenized_val = default_tokenize(
    "my_preference_data/val",
    "HuggingFaceH4/ultrafeedback_binarized",
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    format=PreferenceChatLmDatasetFormat(),
    is_validation=True,
)
```

The `PreferenceChatLmDatasetFormat` expects each example to have `chosen` and
`rejected` fields containing chat-formatted message lists. User turns are masked
from the loss by default.

## Configuring the DPO Run

```python
model_config = llama_3_1_8b()

dpo_config = SimpleDPOConfig(
    resources=v5p_32,
    train_batch_size=128,
    num_train_steps=2000,
    learning_rate=5e-7,
    beta=0.01,                    # Controls preference strength
    lr_schedule="linear",
    warmup=0.1,
    cooldown=0.1,
    model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    steps_per_eval=500,
    steps_per_checkpoint=500,
    steps_per_hf_export=500,
)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `beta` | Regularization strength. Lower values (0.01–0.05) give the model more freedom to deviate from the reference. Higher values (0.1–0.5) keep it closer. |
| `model_name_or_path` | HuggingFace model to initialize the policy from. Also used as the reference model unless `reference_model_path` is set separately. |
| `reference_model_path` | Path to the reference model. Defaults to `model_name_or_path`. |
| `validation_split_fraction` | Fraction of training data to hold out for validation (default 0.1). Set to `None` to use a separate validation set. |

## Creating the Training Pipeline

```python
dpo_step = default_dpo(
    name="dpo-llama-3.1-8b-ultrafeedback",
    tokenized=tokenized_train,
    model_config=model_config,
    dpo_config=dpo_config,
    tags=["dpo", "ultrafeedback", "llama3"],
)
```

## Running the Experiment

```python
if __name__ == "__main__":
    executor_main(dpo_step)
```

Submit the job:

```bash
uv run lib/marin/src/marin/run/ray_run.py --no_wait \
    --env_vars WANDB_API_KEY=${WANDB_API_KEY} \
    -- python experiments/my_dpo_experiment.py
```

## Example

See [`experiments/dpo_ultrafeedback.py`](https://github.com/marin-community/marin/blob/main/experiments/dpo_ultrafeedback.py)
for a complete working example training Llama 3.1 8B on UltraFeedback.
