"""
Sample speedrun script demonstrating how to configure and run a model within compute constraints.
"""

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3
from experiments.llama import llama_150m
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.speedrun import ComputeBudget, SpeedrunConfig, default_speedrun
from marin.execution.executor import executor_main


def create_tiny_model_config():
    """Creates a tiny LLaMA model configuration for demonstration."""
    return llama_150m


def create_training_config():
    """Creates a basic training configuration."""
    return SimpleTrainConfig(
        tpu_type="v4-128",
        train_batch_size=512,
        num_train_steps=6000,  # 512 * 1024 * 6000 = ~3B tokens (3.1457B tokens)
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=2000,
        steps_per_task_eval=2000,
    )


# create speedrun configuration
speedrun_config = SpeedrunConfig(
    compute_budget=ComputeBudget.SMALL,
    model_config=create_tiny_model_config(),
    train_config=create_training_config(),
    tokenized_dataset=dclm_mixture_config_llama3,
)

# run training using default_speedrun
train_step = default_speedrun(
    name="speedrun/150M_llama_dclm_mix_Apr2",
    config=speedrun_config,
)

if __name__ == "__main__":
    # execute the training step
    executor_main(steps=[train_step])
