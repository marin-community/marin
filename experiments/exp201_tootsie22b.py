"""
This is the 22b version of @dlwh's "YOLO"/vibes run described in https://github.com/stanford-crfm/marin/issues/600.

The idea is to train a 8B model continuously updating the mixture, data, and anything else. With WSD-S,
there's no "middle" or "end" of the run, there's just the run. So we'll just train for a long time, updating as we go.

We call it "tootsie" because tootsie rolls are famously made by folding in the previous batch of tootsie roll into the
next batch, so we're folding in the previous mixture into the next mixture.

For now, we're training on DCLM's best mix, but that will change.
"""

from levanter.models.llama import LlamaConfig

from experiments.defaults import default_train
from experiments.exp600_tootsie import dclm_mixture_config_llama3
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main

llama_22b = LlamaConfig(
    seq_len=4096,
    hidden_dim=6144,
    intermediate_dim=16384,
    num_heads=48,
    num_kv_heads=16,
    num_layers=56,
)

llama_22b_train_config = SimpleTrainConfig(
    tpu_type="v6e-256",
    node_count=4,
    train_batch_size=1024,
    num_train_steps=1_000_000,  # using wsd-s so this doesn't really matter
    # these hypers from Table 12 in https://arxiv.org/html/2406.11794v1#A6
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

llama_22b_tootsie = default_train(
    name="llama-22b-tootsie",
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_22b,
    train_config=llama_22b_train_config,
    tags=["llama", "22b", "wsd-s", "exp201", "tootsie"],
    eval_harness_tasks=[],
    use_default_evaluation=False,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            llama_22b_tootsie,
        ],
        description="Train 8B model on DCLM using WSD-S.",
    )
