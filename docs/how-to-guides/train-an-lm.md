# How to Train a Language Model

## Prerequisites

Before following the language model training guide, make sure to familiarize yourself with:

- **Environment Setup**: [README.md § Setup](https://github.com/stanford-crfm/marin/blob/main/README.md#setup)
- **Core Concepts**: [docs/explanation/concepts.md](https://github.com/stanford-crfm/marin/blob/main/docs/explanation/concepts.md)
- **Executor Framework**: [docs/explanation/executor.md](https://github.com/stanford-crfm/marin/blob/main/docs/explanation/executor.md)
- **Experiment System**: [docs/explanation/experiments.md](https://github.com/stanford-crfm/marin/blob/main/docs/explanation/experiments.md)
- **Language Model Pipeline**: [docs/lm/overview.md](https://github.com/stanford-crfm/marin/blob/main/docs/lm/overview.md)

This guide explains how to train a language model using Marin.

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

- `dclm_mixture_config_llama3`: A predefined dataset configuration for the DCLM mixture, this can be replaced with any tokenized dataset in Marin of the `lm_mixture_data_config` type (e.g. [Dolma](../../experiments/dolma/exp442_dolma.py) or [Nemotron](../../experiments/exp934_hq_vs_pt.py))
- `SimpleTrainConfig`: A dataclass for organizing training hyperparameters
- `default_train`: A utility function that creates a training pipeline
- `LlamaConfig`: A dataclass that defines the model architecture from [Levanter](https://github.com/stanford-crfm/levanter)
- `executor_main`: The main entry point for the Marin executor framework

## Setting Up the Model Configuration

Define your model architecture by creating a configuration object:

```python
model_config = LlamaConfig(
    seq_len=2048,           # Maximum sequence length for context processing
    hidden_dim=2048,        # Dimension of hidden representations
    intermediate_dim=8192,  # Dimension of feedforward layers
    num_heads=16,           # Number of attention heads
    num_kv_heads=16,        # Number of key/value heads
    num_layers=24,          # Number of transformer layers
    use_flash_attention=True, # Efficient attention implementation
)
```

## Defining Training Parameters

Set up your training configuration by calculating the number of training steps and defining hyperparameters:

```python
# Calculate training steps based on desired token count
NUM_TRAIN_TOKENS = int(30e9)  # Example: 30 billion tokens
BATCH_SIZE = 256
SEQ_LEN = 2048
NUM_TRAIN_STEPS = NUM_TRAIN_TOKENS // (BATCH_SIZE * SEQ_LEN)

training_config = SimpleTrainConfig(
    accelerator_type="v4-128",  # Hardware configuration
    train_batch_size=BATCH_SIZE,  # Sequences processed per step
    num_train_steps=NUM_TRAIN_STEPS,  # Total optimization steps
    learning_rate=3e-3,     # Initial learning rate
    weight_decay=0.033,     # L2 regularization
    min_lr_ratio=0.1,       # Minimum learning rate ratio (for decay)
    warmup=5000,            # Steps for learning rate warmup
    z_loss_weight=1e-4,     # Optional stabilization technique
)
```

## Creating the Training Pipeline

Connect your model configuration, training parameters, and dataset to create a training pipeline:

```python
# Create the training pipeline
model = default_train(
    name="your_model_name",  # Unique identifier for this training run
    tokenized=dclm_mixture_config_llama3,  # Dataset configuration
    model_config=model_config,  # Model architecture
    train_config=training_config,  # Training hyperparameters
    tags=["YOUR_TAGS"],  # Tags for experiment tracking
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
python marin/run/ray_run.py --env_vars WANDB_API_KEY YOUR_WANDB_API_KEY -- python experiments/exp123_your_model.py
```

Following Marin's guidelines, name your experiment script `experiments/exp{GITHUB_ISSUE_NUMBER}_{DESCRIPTOR}.py`, where `GITHUB_ISSUE_NUMBER` is the issue number for your experiment and `DESCRIPTOR` is a brief description.

## Monitoring Training

Monitor your training progress through:

- **Experiment tracking tools**: If using Weights & Biases, you'll see real-time metrics and visualizations logged to the "Marin" project under your default W&B organization. The run will be named based on the name you provided, plus a version hash to keep track of the state of Marin upstream of your configuration.

## Example Implementations

For a complete example of training a DCLM 1B/1x model, see the implementation in:

[experiments/howto/exp1077_reproduce_dclm_1b1x.py](../../experiments/howto/exp1077_reproduce_dclm_1b1x.py)

For a larger scale example of training a DCLM 7B/1x model, see the implementation in:

[experiments/howto/exp1078_reproduce_dclm_7b1x.py](../../experiments/howto/exp1078_reproduce_dclm_7b1x.py)
