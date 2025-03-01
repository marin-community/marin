"""
Train 1.4B models on standard datasets (e.g., SlimPajama) using multislice.
https://github.com/stanford-crfm/marin/issues/146
"""

from experiments.defaults import default_train
from experiments.exp72_baselines import slimpajama_tokenized
from experiments.llama import llama_1_4b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main

llama_1_4b_multislice_train_config = SimpleTrainConfig(
    tpu_type="v4-128",
    node_count=2,
    train_batch_size=1024,
    num_train_steps=10000,  # 4096 * 1024 * 10000 = 42B tokens
    learning_rate=3e-4,
    weight_decay=0.1,
)

slimpajama_model = default_train(
    name="cathy-pjama-12",
    tokenized=slimpajama_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_multislice_train_config,
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[slimpajama_model],
        description="Train 1.4B models on FineWebEdu.",
    )
