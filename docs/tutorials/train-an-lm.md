# Training a Language Model

## Prerequisites

Before we start training, make sure you have gone through:

- Basic [installation](installation.md)
- Set up your [local GPU](local-gpu.md)
- You need to make sure that your Ray GPU cluster is configured.

This guide explains how to train a language model using Marin, with our examples reproducing the
[DCLM](https://arxiv.org/pdf/2406.11794) 7B/1x and 1B/1x baselines.

## Required Imports

Start by importing the necessary modules:

```python
# Import a tokenized dataset configuration from options available in Marin
from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3

# Import training utilities and configuration classes
from experiments.defaults import SimpleTrainConfig, default_train

# Import model architecture definitions
from levanter.models.llama import LlamaConfig

# Import the executor framework for running experiments
from marin.execution.executor import executor_main

# Import logging utilities
import logging
```

- [`dclm_mixture_config_llama3`](https://github.com/marin-community/marin/blob/25c0f04438d0875e36a4627a5742b8b5a94c5ada/experiments/dclm/tokenize_dclm.py#L50): A predefined dataset configuration for the DCLM mixture, this can be replaced with any tokenized dataset in Marin of the `lm_mixture_data_config` type (e.g. [Dolma](https://github.com/marin-community/marin/blob/main/experiments/dolma/exp442_dolma.py) or [Nemotron](https://github.com/marin-community/marin/blob/main/experiments/exp934_hq_vs_pt.py))
- [`SimpleTrainConfig`][experiments.simple_train_config.SimpleTrainConfig]
- [`default_train`][experiments.defaults.default_train]: A utility function that creates a training pipeline
- [`LlamaConfig`][levanter.models.llama.LlamaConfig]: A dataclass that defines the model architecture from [Levanter](https://github.com/stanford-crfm/levanter)
- [`executor_main`][marin.execution.executor.executor_main]: The main entry point for the Marin executor framework

## Setting Up the Model Configuration

Define your model architecture by creating a configuration object:

```python
model_config = LlamaConfig(
    seq_len=2048,           # Maximum sequence length for context processing
    hidden_dim=2048,        # Dimension of hidden representations
    intermediate_dim=8192,  # Dimension of feedforward layers
    num_heads=16,           # Number of attention heads
    num_layers=24,          # Number of transformer layers
)
```

You can also use pre-defined model configurations from [`experiments.llama`](https://www.github.com/marin-community/marin/blob/main/experiments/llama.py) for common model sizes.

## Defining Training Parameters

Set up your training configuration by calculating the number of training steps and defining hyperparameters:

=== "GPU"
    ```python
    # Calculate training steps based on desired token count
    NUM_TRAIN_TOKENS = int(30e9)  # Example: 30 billion tokens
    BATCH_SIZE = 256
    SEQ_LEN = 2048
    NUM_TRAIN_STEPS = NUM_TRAIN_TOKENS // (BATCH_SIZE * SEQ_LEN)

    training_config = SimpleTrainConfig(
        resources=GpuConfig(gpu_count=4),           # Hardware configuration: 4 GPUs
        train_batch_size=BATCH_SIZE,                # Sequences processed per step
        num_train_steps=NUM_TRAIN_STEPS,            # Total optimization steps
        learning_rate=3e-3,                         # Peak learning rate
        weight_decay=0.033,                         # L2 regularization
        min_lr_ratio=0.1,                           # Minimum learning rate ratio (for decay)
        warmup=5000,                                # Steps for learning rate warmup
        z_loss_weight=1e-4,                         # Optional stabilization technique
    )
    ```
=== "TPU"
    ```python
    # Calculate training steps based on desired token count
    NUM_TRAIN_TOKENS = int(30e9)  # Example: 30 billion tokens
    BATCH_SIZE = 256
    SEQ_LEN = 2048
    NUM_TRAIN_STEPS = NUM_TRAIN_TOKENS // (BATCH_SIZE * SEQ_LEN)

    training_config = SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type="v4-128"),  # Hardware configuration: 128 v4 TPU cores
        train_batch_size=BATCH_SIZE,                # Sequences processed per step
        num_train_steps=NUM_TRAIN_STEPS,            # Total optimization steps
        learning_rate=3e-3,                         # Peak learning rate
        weight_decay=0.033,                         # L2 regularization
        min_lr_ratio=0.1,                           # Minimum learning rate ratio (for decay)
        warmup=5000,                                # Steps for learning rate warmup
        z_loss_weight=1e-4,                         # Optional stabilization technique
    )
    ```

## Creating the Training Pipeline

Connect your model configuration, training parameters, and dataset to create a training pipeline:

```python
# Create the training pipeline
model = default_train(
    name="${YOUR_MODEL_NAME}",              # Unique identifier for this training run
    tokenized=dclm_mixture_config_llama3,   # Dataset configuration
    model_config=model_config,              # Model architecture
    train_config=training_config,           # Training hyperparameters
    tags=["${YOUR_TAG1}", "${YOUR_TAG2}"],  # Tags for experiment tracking
    eval_harness_tasks = [EvalTaskConfig("mmlu", 0, task_alias="mmlu_0shot"), EvalTaskConfig("mmlu", 5, task_alias="mmlu_5shot")] # Evaluation Tasks to run on the checkpoint
)

# Set up the experiment execution
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("Starting language model training experiment")
    executor_main(
        steps=[model],  # The training pipeline is a step in the experiment
        description="Language model training experiment",
    )
```

The `default_train` function creates a training pipeline that:

1. Loads and processes the dataset

2. Initializes the model according to the configuration

3. Executes the training loop with the specified hyperparameters

4. Handles distributed training across your hardware

5. Manages checkpointing and logging

## Launching the Training Job

To train the model with experiment tracking:

```bash
python marin/run/ray_run.py --env_vars WANDB_API_KEY ${YOUR_WANDB_API_KEY} -- python experiments/${YOUR_EXPERIMENT_SCRIPT}.py
```

Following Marin's guidelines, name your experiment script `experiments/exp{GITHUB_ISSUE_NUMBER}_{DESCRIPTOR}.py`, where `GITHUB_ISSUE_NUMBER` is the issue number for your experiment and `DESCRIPTOR` is a brief description.

## Monitoring Training

Monitor your training progress through:

- **Experiment tracking tools**: If using WandB, you'll see real-time metrics and visualizations logged to the "Marin" project under your default W&B organization (if you have access).
The run will be named based on the name you provided.

## Example Implementations

For a complete example of training a DCLM 1B/1x model, see the implementation in:

- Code: [experiments/howto/exp1077_reproduce_dclm_1b1x.py](https://github.com/marin-community/marin/blob/main/experiments/howto/exp1077_reproduce_dclm_1b1x.py)
- WandB: [Dashboard](https://wandb.ai/marin-community/marin/runs/dclm_1b_1x_how_to-58c8f0)

This trains on the DCLM baseline mix with the same config as described in the original DCLM paper for 1X the compute optimal number of tokens!

For a larger scale example of training a DCLM 7B/1x model, see the implementation in:

- Code: [experiments/howto/exp1078_reproduce_dclm_7b1x.py](https://github.com/marin-community/marin/blob/main/experiments/howto/exp1078_reproduce_dclm_7b1x.py)
- WandB: [Dashboard](https://wandb.ai/marin-community/marin/runs/dclm_7b_1x_how_to-fefaab)
