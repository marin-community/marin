# Import a tokenized dataset configuration from options available in Marin
from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3

# Import training utilities and configuration classes
from experiments.defaults import SimpleTrainConfig, default_train

# Import model architecture definitions
from levanter.models.llama import LlamaConfig

# Import the executor framework for running experiments
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

# Import logging utilities
import logging

from marin.resources import TpuPodConfig

# https://github.com/marin-community/marin/blob/main/experiments/llama.py
model_config = LlamaConfig(
    seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=7168,
    num_heads=16,
    num_kv_heads=8,
    num_layers=16,
)

# Calculate training steps based on desired token count
NUM_TRAIN_TOKENS = int(30e9)
# Example: 30 billion tokens
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
