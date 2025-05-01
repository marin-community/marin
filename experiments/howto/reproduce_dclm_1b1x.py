"""
How To: Replicating DCLM 1B/1x in Marin
Link: https://arxiv.org/pdf/2406.11794

Reproduces the DCLM baseline for the 1B/1X (Chinchilla Compute Optimal Model for 1.4B) competition pool.

Author: Will Held
"""

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3
from experiments.defaults import SimpleTrainConfig, default_train
from experiments.llama import LlamaConfig
from marin.execution.executor import executor_main

llama_1_4b_dclm = LlamaConfig(
    seq_len=2048,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=16,
    num_kv_heads=16,
    num_layers=24,
    use_flash_attention=True,
)

NUM_TRAIN_TOKENS = int(28.8e9)  # 28.8 billion tokens
NUM_TRAIN_STEPS = NUM_TRAIN_TOKENS // (256 * 2048)  # 256 is the batch size, 2048 is the sequence length

training_config = SimpleTrainConfig(
    tpu_type="v4-128",
    train_batch_size=256,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=3e-3,
    weight_decay=0.033,
    min_lr_ratio=0.1,
    warmup=5000,
    z_loss_weight=1e-4,
)

dclm_mixture_model = default_train(
    name="dclm_1b_1x_how_to",
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_1_4b_dclm,
    train_config=training_config,
    tags=["HOWTOS"],
)

if __name__ == "__main__":
    executor_main(
        steps=[dclm_mixture_model],
        description="A How-To Which Reproduces the DCLM 1B/1X Baseline for the competition pool.",
    )
