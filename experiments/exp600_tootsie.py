import dataclasses

from experiments.dclm.exp433_dclm_run import DCLM_MIXTURE_WEIGHTS
from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_8b
from experiments.pretraining_datasets import dclm_baseline, proofpile_2, starcoderdata
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

dclm_baseline_tokenized = default_tokenize(
    name="dclm_baseline",
    dataset=dclm_baseline,
    tokenizer=llama3_tokenizer,
)

starcoderdata_tokenized = default_tokenize(
    name="starcoderdata", dataset=starcoderdata, tokenizer=llama3_tokenizer, text_key="content"
)

proofpile_2_tokenized = default_tokenize(
    name="proofpile_2",
    dataset=proofpile_2,
    tokenizer=llama3_tokenizer,
)

dclm_components = {
    "dclm_baseline": dclm_baseline_tokenized,
    "starcoderdata": starcoderdata_tokenized,
    "proofpile_2": proofpile_2_tokenized,
}

dclm_mixture_config = lm_mixture_data_config(components=dclm_components, weights=DCLM_MIXTURE_WEIGHTS)

llama_8b_train_config = SimpleTrainConfig(
    tpu_type="v5litepod-256",
    node_count=2,
    train_batch_size=1024,
    num_train_steps=1_000_000,  # using wsd-s so this doesn't really matter
    # these hypers from Table 12 in https://arxiv.org/html/2406.11794v1#A6
    learning_rate=1e-3,  # we get divergence with 2e-3
    weight_decay=0.05,
    # WSD-S
    cycle_length=10000,
    steps_per_eval=10000,
    steps_per_export=20000,
    warmup=1000,  # initial warmup
    # TODO: do we need rewarmup
    decay=0.1,  # 10% of 5000 = 500 steps
    lr_schedule="inv",
)

llama_8b_tootsie = dataclasses.replace(
    default_train(
        name="llama-8b-tootsie-0.001",
        tokenized=dclm_mixture_config,
        model_config=llama_8b,
        train_config=llama_8b_train_config,
        tags=["llama", "8b", "wsd-s", "exp600"],
    ),
    override_output_path="checkpoints/llama-8b-tootsie-0.001-19ad63",
)

llama_1b_simple_train_config = SimpleTrainConfig(
    tpu_type="v5litepod-256",
    node_count=1,
    train_batch_size=1024,
    num_train_steps=1_000,  # using wsd-s so this doesn't really matter
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

llama_8b_simple = default_train(
    name="llama-1b-simple",
    tokenized=dclm_mixture_config,
    model_config=llama_1_4b,
    train_config=llama_1b_simple_train_config,
    tags=["llama", "1b", "wsd-s", "exp600"],
    eval_harness_tasks=[],
    use_default_evaluation=False,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            llama_8b_tootsie,
            llama_8b_simple,
        ],
        description="Train 8B model on DCLM using WSD-S.",
    )
