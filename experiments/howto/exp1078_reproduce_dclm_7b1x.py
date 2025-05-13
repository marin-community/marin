"""
How To: Replicating DCLM 7B/1x in Marin
Link: https://arxiv.org/pdf/2406.11794

Reproduces the DCLM baseline for the 7B/1X (Chinchilla Compute Optimal Model for 7B) competition pool.

Author: Will Held

Example usage:
  # Run the training job with wandb logging enabled
  python marin/run/ray_run.py --env_vars WANDB_API_KEY YOUR_WANDB_API_KEY -- python experiments/howto/dclm_7b1x.py
"""

from levanter.models.llama import LlamaConfig

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3
from experiments.defaults import SimpleTrainConfig, default_train
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig

# Define the LlamaConfig for a 7B parameter model
# This follows the 7B-1x competition scale in the DCLM benchmark
# As described in the DCLM paper, each scale specifies the number of model parameters
# and a Chinchilla multiplier that corresponds to compute-optimal allocation
# [Reference: DCLM paper, Table 1 - Competition Scales]
SEQ_LEN = 2048
BATCH_SIZE = 2048
llama_7b_dclm = LlamaConfig(
    seq_len=SEQ_LEN,  # Maximum sequence length for processing context
    hidden_dim=4096,  # Dimension of hidden representations
    intermediate_dim=11008,  # Dimension of feedforward layers
    num_heads=32,  # Number of attention heads
    num_kv_heads=32,  # Number of key/value heads
    num_layers=32,  # Number of transformer layers
    use_flash_attention=True,  # Optimization for faster attention computation
)

NUM_TRAIN_TOKENS = int(140e9)  # 140 billion tokens

# Calculate the number of training steps based on batch size and sequence length
# This determines how many optimization steps will occur during training to reach the desired number of tokens
NUM_TRAIN_STEPS = NUM_TRAIN_TOKENS // (BATCH_SIZE * SEQ_LEN)

# Define training configuration with hyperparameters
# https://github.com/mlfoundations/dclm/blob/main/training/configs/7b_1x_fast_2e-3_lr_5e-6_zloss.json
training_config = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type="v4-128", node_count=4),
    train_batch_size=BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=2e-3,
    weight_decay=0.05,
    min_lr_ratio=0.1,
    warmup=5000,
    z_loss_weight=5e-6,
)

# Create the training pipeline for the DCLM mixture model
dclm_mixture_model = default_train(
    name="dclm_7b_1x_how_to",
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_7b_dclm,
    train_config=training_config,
    tags=["HOWTOS", "DCLM_7B_1X"],  # Tags for experiment tracking
)

# Main execution block
if __name__ == "__main__":
    executor_main(
        steps=[dclm_mixture_model],
        description="A How-To Which Reproduces the DCLM 7B/1X Baseline for the competition pool.",
    )
