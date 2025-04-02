"""
Sample speedrun script demonstrating how to configure and run a model within compute constraints.
"""
from experiments.speedrun.speedrun import ComputeBudget, SpeedrunConfig, default_speedrun
from experiments.simple_train_config import SimpleTrainConfig
from levanter.models.llama import LlamaConfig
from marin.execution.executor import executor_main
from experiments.pretraining_datasets import fineweb

def create_tiny_model_config():
    """Creates a tiny LLaMA model configuration for demonstration."""
    return LlamaConfig(
        d_model=512,
        n_heads=8,
        n_layers=8,
        vocab_size=32000,
        max_sequence_length=2048,
    )

def create_training_config():
    """Creates a basic training configuration."""
    return SimpleTrainConfig(
        tpu_type="v4-8",
        train_batch_size=32,
        num_train_steps=1000,
        learning_rate=1e-4,
        warmup=0.1,
    )

# create speedrun configuration
speedrun_config = SpeedrunConfig(
    compute_budget=ComputeBudget.TINY,
    model_config=create_tiny_model_config(),
    train_config=create_training_config(),
    tokenized_dataset=fineweb,
)

# run training using default_speedrun
train_step = default_speedrun(
    name="speedrun/speedrun_tiny_llama",
    config=speedrun_config,
    tags=["sample_run", "tiny_llama"]
)

if __name__ == "__main__":
    # execute the training step
    executor_main(steps=[train_step])
