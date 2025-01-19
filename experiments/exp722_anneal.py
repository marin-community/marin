import os

from pretraining_datasets import dclm_baseline, fineweb_edu

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_8b
from experiments.midtraining_datasets import finemath_3_plus
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

num_train_steps_for_1T = 953674  # 4096 * 256 * 953674 = 1T tokens
llama_max_seq_len = 4096
train_batch_size = 256

llama_8b_train_config = SimpleTrainConfig(
    tpu_type="v4-128",
    node_count=2,
    train_batch_size=train_batch_size,  # NOTE(Chris): tune this?
    num_train_steps=num_train_steps_for_1T,  # 4096 * 256 * 953674 = 1T tokens
    learning_rate=2e-3,
    weight_decay=0.05,
    steps_per_export=10000,
)

# NOTE(Chris): Fineweb had not been downloaded yet and it's like 15T tokens
# Just test out with dclm baseline for now?
dclm_baseline_tokenized = (
    default_tokenize(
        name="dclm_baseline",
        dataset=dclm_baseline,
        tokenizer=llama3_tokenizer,  # NOTE(chris): some reason llama3_tokenizer not working
    ),
)
model = default_train(
    name="llama-8b-dclm-baseline-1T",
    tokenized=dclm_baseline_tokenized,
    model_config=llama_8b,
    train_config=llama_8b_train_config,
)

num_anneal_training_tokens = 50_000_000_000  # 50B tokens

anneal_percent = num_anneal_training_tokens / num_train_steps_for_1T
annealed_train_config = llama_8b_train_config.copy()

# NOTE(chris): 760000 steps is roughly 80% of the way to 1 trillion tokens which is divisible by 10000
annealed_train_config.initialize_from_checkpoint_path = os.path.join(model.name, "checkpoints/step-760000")
annealed_train_config.lr_schedule = "linear"
annealed_train_config.num_train_steps = num_anneal_training_tokens / (train_batch_size * llama_max_seq_len)


finemath_3_plus_tokenized = default_tokenize(
    name="finemath_3_plus",
    dataset=finemath_3_plus,
    tokenizer=llama3_tokenizer,
)

anneal_stage_data_mix = lm_mixture_data_config(
    components={"high-quality-web-text": fineweb_edu, "target-dataset": finemath_3_plus_tokenized},
    weights={
        "high-quality-web-text": 0.70,
        "target-dataset": 0.30,
    },
)
# TODO(chris): Make naming unique for this
annealed_model = default_train(
    name="llama-8b-anneal-finemath",
    tokenized=anneal_stage_data_mix,
    model_config=llama_8b,
    train_config=annealed_train_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            model,
            annealed_model,
        ],
        description="Train 8B model on DCLM using WSD-S.",
    )
