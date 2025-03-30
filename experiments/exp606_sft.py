from instruction_datasets import get_instruction_dataset

from experiments.defaults import default_sft
from experiments.llama import llama3_tokenizer, llama_8b
from experiments.simple_sft_config import SimpleSFTConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize.tokenize import TokenizeConfig, levanter_tokenize_sft

# Get instruction dataset
tulu_3_dataset = get_instruction_dataset("allenai/tulu-3-sft-mixture")

# Number of tokens is 670,426,314
NUM_TRAIN_TOKENS = 670426314
# number of epochs over the dataset set to reproduce Olmo SFT v2
# or Tulu 3 starting from Llama 3.1 8B. This script
# is used to reproduce the Tulu 3 SFT model.
# Link: https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B
NUM_TRAIN_STEPS = NUM_TRAIN_TOKENS // (128 * 4096) * 3  # 2 epochs

# Create tokenization step for Tulu-3 dataset
tulu3_llama_tokenize_step = ExecutorStep(
    name="tokenized/tulu_sft_v3_llama3_tokenizer",
    fn=levanter_tokenize_sft,
    config=TokenizeConfig(
        train_paths=[output_path_of(tulu_3_dataset, "**/*.jsonl.gz")],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=llama3_tokenizer,
        input_field="user",
        output_field="assistant",
    ),
    description="Tokenize chat SFT data",
)

tulu_sft_config = SimpleSFTConfig(
    train_batch_size=128,
    num_train_steps=NUM_TRAIN_STEPS,  # Adjust as needed.
    learning_rate=5e-6,
    tpu_type="v4-128",
    tokenizer=llama3_tokenizer,
    model_name_or_path="meta-llama/Llama-3.1-8B",
    max_seq_len=4096,
    seed=1,
)

# Configure SFT training
sft_step = default_sft(
    name="tulu3_llama3_sft", tokenized=tulu3_llama_tokenize_step, model_config=llama_8b, sft_config=tulu_sft_config
)

if __name__ == "__main__":
    executor_main(steps=[tulu3_llama_tokenize_step, sft_step])
