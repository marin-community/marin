"""
This script runs a suite of scaling laws on smaller model configurations.
Based on the model specifications from https://github.com/stanford-crfm/marin/issues/879
"""

import dataclasses
from typing import Sequence

from levanter.models.llama import LlamaConfig
from experiments.defaults import default_scaling_law_pred, default_train
from experiments.dolma.exp442_dolma import dolma_llama3_tokenized
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
from experiments.tootsie.exp600_tootsie import dclm_mixture_config_llama3
from marin.execution.executor import executor_main, ExecutorStep, InputName
from experiments.simple_train_config import SimpleTrainConfig

# Training configuration based on paper hyperparameters
SMALLER_LADDER_TRAIN_CONFIG = SimpleTrainConfig(
    tpu_type="v4-128",
    node_count=1,
    train_batch_size=512,
    learning_rate=1e-3, # Will be replaced per model
    weight_decay=0.1,
    num_train_steps=1000,  # Will be replaced per model
    warmup=50,
    decay=0.9,
    lr_schedule="cosine",
    ema_beta=0.95,
    steps_per_eval=500,
    steps_per_task_eval=500,
)

def create_smaller_ladder_suite(
    sweep_name: str,
    tokenized: InputName | ExecutorStep,
    tags: Sequence[str] = (),
    lr_base: float = 2.5,
    max_lr: float = 5e-3,
    training_config = SMALLER_LADDER_TRAIN_CONFIG,
) -> Sequence[ExecutorStep]:
    """
    Creates a suite of smaller model configurations based on specified architecture.
    
    Model configurations:
    - 12M: 5 layers, 448 hidden dim, 7 attention heads
    - 17M: 7 layers, 448 hidden dim, 7 attention heads
    - 25M: 8 layers, 512 hidden dim, 8 attention heads
    - 35M: 9 layers, 576 hidden dim, 9 attention heads
    - 50M: 10 layers, 640 hidden dim, 10 attention heads
    - 70M: 12 layers, 704 hidden dim, 11 attention heads
    - 100M: 14 layers, 768 hidden dim, 12 attention heads
    - 200M: 18 layers, 960 hidden dim, 15 attention heads
    """
    
    # Model configurations with max steps and checkpoint intervals
    configs = [
        # (num_layers, hidden_dim, num_heads, max_steps, checkpoint_interval)
        (5, 448, 7, 2000, 100),      # Checkpoints at: 100,200,300,400,500,...,2000
        (7, 448, 7, 3000, 100),      # Checkpoints at: 100,200,300,400,500,...,3000
        (8, 512, 8, 5000, 250),     # Checkpoints at: 250,500,750,1000,...,5000
        (9, 576, 9, 6500, 200),     # Checkpoints at: 200,400,600,800,...,6500
        (10, 640, 10, 10000, 250),   # Checkpoints at: 250,500,750,1000,...,10000
        (12, 704, 11, 13000, 250),    # Checkpoints at: 250,500,750,1000,...,13000
        (14, 768, 12, 20000, 250),   # Checkpoints at: 250,500,750,1000,...,20000
        (18, 960, 15, 40000, 500),    # Checkpoints at: 500,1000,1500,...,40000
    ]
    
    steps = []
    for num_layers, hidden_dim, num_heads, max_steps, checkpoint_interval in configs:
        # Create model config for each architecture
        model_config = LlamaConfig(
            seq_len=2048,
            hidden_dim=hidden_dim,
            intermediate_dim=hidden_dim * 4,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            num_layers=num_layers,
        )

        # lreff = lrbase/(width × √depth)
        lr = min(lr_base / (hidden_dim * (num_layers ** 0.5)), max_lr)
        training_config = dataclasses.replace(training_config, learning_rate=lr)
        
        config = dataclasses.replace(
            training_config,
            num_train_steps=max_steps,
            learning_rate=lr,
            steps_per_eval=checkpoint_interval,  # Evaluate at checkpoint steps
            steps_per_task_eval=checkpoint_interval  # Run task eval at checkpoint steps
        )
        
        # Create training step for this configuration
        step = default_train(
            name=f"{sweep_name}-{hidden_dim}-{num_layers}l-lr{lr:.0e}",
            tokenized=tokenized,
            model_config=model_config,
            train_config=config,
            eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
            tags=[*tags, "smaller_ladders"]
        )
        steps.append(step)

    return steps


# Create the smaller ladder suite
smaller_suite = create_smaller_ladder_suite(
    sweep_name="scaling-law-suite-smaller-dolma-v1",
    tokenized=dolma_llama3_tokenized,
    tags=["scaling_laws"],
)

# smaller_suite_scaling_laws_pred = default_scaling_law_pred(
#     ladder_runs=smaller_suite,
#     pred_run="llama-8b-tootsie-0.001-19ad63",
#     task_losses=(
#         "eval/paloma/c4_en/bpb",
#         "eval/bpb",
#         "eval/loss",
#         "eval/paloma/c4_en/loss",
#     ),
#     task_accuracies=CORE_TASKS,
# )


if __name__ == "__main__":
    executor_main(
        steps=[
            *smaller_suite,
            # smaller_suite_scaling_laws_pred,
        ],
        description="suite for scaling laws on smaller model configurations",
    )
