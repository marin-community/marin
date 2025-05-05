"""
How To: Replicating DCLM 1B/1x in Marin
Link: https://arxiv.org/pdf/2406.11794

Reproduces the DCLM baseline for the 1B/1X (Chinchilla Compute Optimal Model for 1.4B) competition pool.

Author: Will Held

Example usage:
  # Run the training job with wandb logging enabled
  python marin/run/ray_run.py --env_vars WANDB_API_KEY YOUR_WANDB_API_KEY -- python experiments/howto/dclm_1b1x.py
"""

from levanter.models.llama import LlamaConfig

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3
from experiments.defaults import SimpleTrainConfig, default_train
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig

# Define the LlamaConfig for a 1.4B parameter model
# This follows the 1B-1x competition scale in the DCLM benchmark
# As described in the DCLM paper, each scale specifies the number of model parameters
# and a Chinchilla multiplier that corresponds to compute-optimal allocation
# [Reference: DCLM paper, Table 1 - Competition Scales]
SEQ_LEN = 2048
BATCH_SIZE = 256
llama_1_4b_dclm = LlamaConfig(
    seq_len=SEQ_LEN,  # Maximum sequence length for processing context
    hidden_dim=2048,  # Dimension of hidden representations
    intermediate_dim=8192,  # Dimension of feedforward layers (4x hidden_dim)
    num_heads=16,  # Number of attention heads
    num_kv_heads=16,  # Number of key/value heads (equal to num_heads = no grouped-query attention)
    num_layers=24,  # Number of transformer layers
    use_flash_attention=True,
)

NUM_TRAIN_TOKENS = int(28.8e9)  # 28.8 billion tokens

# Calculate the number of training steps based on batch size and sequence length
# This determines how many optimization steps will occur during training to reach the desired number of tokens
NUM_TRAIN_STEPS = NUM_TRAIN_TOKENS // (BATCH_SIZE * SEQ_LEN)  # 256 is the batch size, 2048 is the sequence length

# Define training configuration with hyperparameters
# https://github.com/mlfoundations/dclm/blob/main/training/configs/1b_1x_fast.json
training_config = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type="v4-128"),  # Hardware configuration: 128 v4 TPU cores, can be swapped for GpuConfig
    train_batch_size=BATCH_SIZE,  # Number of sequences processed per step
    num_train_steps=NUM_TRAIN_STEPS,  # Total training steps
    learning_rate=3e-3,  # Initial learning rate
    weight_decay=0.033,  # L2 regularization parameter to prevent overfitting
    min_lr_ratio=0.1,  # Minimum learning rate as fraction of initial (for cosine decay)
    warmup=5000,  # Number of steps for learning rate warmup
    z_loss_weight=1e-4,  # Stabilization technique to prevent extreme logits
)

# Create the training pipeline for the DCLM mixture model
dclm_mixture_model = default_train(
    name="dclm_1b_1x_how_to",
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_1_4b_dclm,
    train_config=training_config,
    tags=["HOWTOS", "DCLM_1B_1X"],  # Tags for experiment tracking
)

# Main execution block
if __name__ == "__main__":
    executor_main(
        steps=[dclm_mixture_model],
        description="A How-To Which Reproduces the DCLM 1B/1X Baseline for the competition pool.",
    )
