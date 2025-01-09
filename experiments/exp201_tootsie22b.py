"""
These are larger versions of @dlwh's "YOLO"/vibes run described in https://github.com/stanford-crfm/marin/issues/600.

The idea is to train models continuously updating the mixture, data, and anything else. With WSD-S,
there's no "middle" or "end" of the run, there's just the run. So we'll just train for a long time, updating as we go.

We call it "tootsie" because tootsie rolls are famously made by folding in the previous batch of tootsie roll into the
next batch, so we're folding in the previous mixture into the next mixture.

For now, we're training on DCLM's best mix, but that will change.
"""

from experiments.defaults import default_train
from experiments.exp600_tootsie import dclm_mixture_config_llama3
from experiments.llama import llama_13b, llama_70b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main

llama_22b_train_config = SimpleTrainConfig(
    tpu_type="v6e-256",
    node_count=2,
    train_batch_size=1024,
    num_train_steps=1_000_000,  # using wsd-s so this doesn't really matter
    learning_rate=3e-4,
    weight_decay=0.05,
    # WSD-S
    cycle_length=10000,
    steps_per_eval=10000,
    steps_per_export=20000,
    warmup=1000,  # initial warmup
    # TODO: do we need rewarmup
    decay=0.1,  # 10% of 10000 = 500 steps
    lr_schedule="inv",
)


llama_70b_train_config = SimpleTrainConfig(
    tpu_type="v6e-256",
    node_count=4,
    train_batch_size=1024,
    num_train_steps=1_000_000,  # using wsd-s so this doesn't really matter
    learning_rate=3e-5,
    weight_decay=0.05,
    # WSD-S
    cycle_length=10000,
    steps_per_eval=10000,
    steps_per_export=20000,
    warmup=1000,  # initial warmup
    # TODO: do we need rewarmup
    decay=0.1,  # 10% of 10000 = 500 steps
    lr_schedule="inv",
)

llama_13b_tootsie = default_train(
    name="llama-13b-tootsie-dummy-testing",
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_13b,
    train_config=llama_22b_train_config,
    tags=["llama", "13b", "wsd-s", "exp201", "tootsie"],
    eval_harness_tasks=[],
    use_default_evaluation=False,
)

llama_70b_tootsie = default_train(
    name="llama-70b-tootsie-dummy-testing",
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_70b,
    train_config=llama_70b_train_config,
    tags=["llama", "70b", "wsd-s", "exp201", "tootsie"],
    eval_harness_tasks=[],
    use_default_evaluation=False,
)


if __name__ == "__main__":
    executor_main(
        steps=[
            llama_13b_tootsie,
            llama_70b_tootsie,
        ],
        description="Train some models on DCLM using WSD-S.",
    )
