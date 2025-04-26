from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_sft, default_tokenize
from experiments.instruction_datasets import get_instruction_dataset
from experiments.llama import llama_8b
from experiments.simple_sft_config import SimpleSFTConfig
from marin.execution.executor import executor_main

# Get instruction dataset
instruction_dataset = get_instruction_dataset("allenai/tulu-v2-sft-mixture-olmo-4096")

# Number of tokens in the SFT dataset below
NUM_TRAIN_TOKENS = 150849275
# number of epochs over the dataset set to reproduce Olmo SFT
NUM_TRAIN_STEPS = NUM_TRAIN_TOKENS // (128 * 4096) * 3  # 3 epochs

# Add tokenization step
tokenize_step = default_tokenize(
    name="olmo702024_sft_4096_3eps",
    dataset=instruction_dataset / "**/*.jsonl.gz",
    tokenizer="stanford-crfm/marin-olmo2-tokenizer",
    format=ChatLmDatasetFormat(),
)


train_step = default_sft(
    name="checkpoints/olmo7_072024_sft_4096_3eps",
    tokenized=tokenize_step,
    model_config=llama_8b,
    sft_config=SimpleSFTConfig(
        train_batch_size=128,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=2e-6,  #  2x10^-6
        tpu_type="v4-128",
        tokenizer="EleutherAI/gpt-neox-20b",
        model_name_or_path="gs://levanter-checkpoints/marin/olmoish7b_v4_1024_0627/dlwh_7b0627/step-510000/",
        max_seq_len=4096,
        seed=0,
        weight_decay=0.0,
        warmup=0.03,
        cooldown=0.0,
        lr_schedule="linear",
        min_lr_ratio=0.0,
        steps_per_hf_export=500,
        max_grad_norm=None,
    ),
)


if __name__ == "__main__":
    executor_main(steps=[tokenize_step, train_step])
