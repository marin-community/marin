from experiments.dclm.exp433_dclm_run import dclm_mixture_config
from experiments.defaults import default_train
from experiments.llama import llama_8b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main

llama_8b_train_config = SimpleTrainConfig(
    tpu_type="v5litepod-256",
    node_count=4,
    train_batch_size=1024,
    num_train_steps=1_000_000_000,  # using wsd-s so this doesn't really matter
    # these hypers from Table 12 in https://arxiv.org/html/2406.11794v1#A6
    learning_rate=2e-3,
    weight_decay=0.05,
    # WSD-S
    cycle_length=5000,
    steps_per_eval=5000,
    steps_per_export=20000,
    warmup=1000,  # initial warmup
    # TODO: do we need rewarmup
    decay=0.1,  # 10% of 5000 = 500 steps
    lr_schedule="inv",
)

llama_8b_tootsie = default_train(
    name="llama-8b-tootsie",
    tokenized=dclm_mixture_config,
    model_config=llama_8b,
    train_config=llama_8b_train_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            llama_8b_tootsie,
        ],
        description="Train 8B model on DCLM using WSD-S.",
    )
