from experiments.defaults import default_sft, default_tokenize
from experiments.exp964_custom_chat_tokenizer import llama3_instruct_chat_format
from experiments.instruction_datasets import get_instruction_dataset
from experiments.llama import llama3_instruct_tokenizer, llama_8b
from experiments.simple_sft_config import SimpleSFTConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig

# Get instruction dataset
nemotron_dataset = get_instruction_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset-v1", splits=["math", "safety"])

# Number of tokens is 670,426,314
NUM_TRAIN_TOKENS = 670426314
# number of epochs over the dataset set to reproduce Olmo SFT v2
# or Tulu 3 starting from Llama 3.1 8B. This script
# is used to reproduce the Tulu 3 SFT model.
# Link: https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B
NUM_TRAIN_STEPS = NUM_TRAIN_TOKENS // (128 * 4096) * 3  # 2 epochs

# Create tokenization step for Nemotron dataset
nemotron_llama_tokenize_step = default_tokenize(
    name="nemotron_sft_v1_llama3_instruct_tokenizer",
    dataset=nemotron_dataset / "**/*.jsonl.gz",
    tokenizer=llama3_instruct_tokenizer,
    format=llama3_instruct_chat_format,
)

nemotron_sft_config = SimpleSFTConfig(
    train_batch_size=128,
    num_train_steps=NUM_TRAIN_STEPS,  # Adjust as needed.
    learning_rate=5e-6,
    resources=TpuPodConfig(tpu_type="v4-128", slice_count=1),
    tokenizer=llama3_instruct_tokenizer,
    model_name_or_path="meta-llama/Llama-3.1-8B",
    max_seq_len=4096,
    seed=1,
)

# Configure SFT training
nemotron_sft_step = default_sft(
    name="nemotron_llama3_sft", tokenized=nemotron_llama_tokenize_step, model_config=llama_8b, sft_config=nemotron_sft_config
)

if __name__ == "__main__":
    executor_main(steps=[nemotron_llama_tokenize_step, nemotron_sft_step])
