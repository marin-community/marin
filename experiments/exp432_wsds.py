"""
Train 1.4B models on standard datasets (e.g., SlimPajama) using WSD-S.
https://github.com/stanford-crfm/marin/issues/432
"""

from experiments.defaults import default_train
from experiments.exp72_baselines import fineweb_edu_tokenized
from experiments.llama import llama_1_4b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main

llama_1_4b_wsds_train_config = SimpleTrainConfig(
    tpu_type="v4-128",
    node_count=2,
    train_batch_size=1024,
    num_train_steps=10000,  # 4096 * 1024 * 10000 = 42B tokens
    learning_rate=3e-4,
    weight_decay=0.1,
    cycles=[2e3, 4e3, 6e3, 8e3],  # 5 cycles with 2000 steps/cycle
    warmup=0.05,  # 5%  * 2000 = 100 steps
    decay=0.1,  # 10% * 2000 = 200 steps
    lr_schedule="inv",
    steps_per_eval=2000,
)

fineweb_edu_model = default_train(
    name="fineweb-edu-1.4b-wsds",
    tokenized=fineweb_edu_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_wsds_train_config,
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[fineweb_edu_model],
        description="Train 1.4B models on FineWebEdu using WSD-S.",
    )
