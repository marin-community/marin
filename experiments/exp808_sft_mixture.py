"""
An experiment to run SFT on a high quality subset of instruction/reasoning datasets.

The default below trains on all the datasets weighted by document count and sets
the number of training steps to epoch three times. If you wish to fine-tune on a
different number of datasets you will need to change the number of training steps
accordingly.
"""

from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_sft, default_tokenize
from experiments.exp964_custom_chat_tokenizer import marin_tokenizer
from experiments.instruction_datasets import get_instruction_dataset
from experiments.llama import llama_8b
from experiments.simple_sft_config import SimpleSFTConfig
from marin.execution.executor import ExecutorStep, executor_main
from marin.processing.tokenize import lm_mixture_data_config


def create_tokenization_step(dataset_name: str) -> ExecutorStep:
    """
    Creates a tokenization ExecutorStep for a given dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'TIGER-Lab/AceCode-89K', 'HuggingFaceTB/smoltalk')

    Returns:
        ExecutorStep configured for tokenizing the specified dataset
    """
    # Get the dataset with only train split
    dataset = get_instruction_dataset(dataset_name, splits=["train"])

    # Get the last part of the path and clean it up
    short_name = dataset_name.split("/")[-1].lower().replace("-", "_")
    return default_tokenize(
        name=f"{short_name}_marin_tokenizer",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=marin_tokenizer,
        format=ChatLmDatasetFormat(),
    )


# Dataset configurations
DATASETS = {
    "acecode_89k": "TIGER-Lab/AceCode-89K",
    "smoltalk": "HuggingFaceTB/smoltalk",
    "verifiable_math_problems": "PrimeIntellect/verifiable-math-problems",
    "dolphin_r1_nonreasoning": "cognitivecomputations/dolphin-r1-nonreasoning",
    "dolphin_r1_reasoning": "cognitivecomputations/dolphin-r1-reasoning",
    "bespoke_stratos_17k": "bespokelabs/Bespoke-Stratos-17k",
    "openthoughts_114k_math": "open-r1/OpenThoughts-114k-math",
    "tulu_3_sft_mixture": "allenai/tulu-3-sft-mixture",
    "natural_reasoning": "facebook/natural_reasoning",
}

NUM_TRAIN_STEPS = 19086  # 3 Epochs over all datasets above

# Create tokenization steps for multiple datasets
tokenized_datasets = {short_name: create_tokenization_step(hf_name) for short_name, hf_name in DATASETS.items()}

# Dataset weights set with the naive baseline of the number of documents per dataset
mixture_weights = {
    "tulu_3_sft_mixture": 939343,
    "openthoughts_114k_math": 89120,
    "verifiable_math_problems": 777457,
    "acecode_89k": 87149,
    "smoltalk": 1043917,
    "natural_reasoning": 1145824,
    "dolphin_r1_nonreasoning": 214318,
    "dolphin_r1_reasoning": 585418,
    "bespoke_stratos_17k": 16710,
}


assert set(tokenized_datasets.keys()) == set(mixture_weights.keys())

# Define an SFT config appropriate for mixture training
mixture_sft_config = SimpleSFTConfig(
    train_batch_size=128,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=5e-6,
    tpu_type="v4-128",
    tokenizer=marin_tokenizer,
    model_name_or_path="meta-llama/Llama-3.1-8B",
    max_seq_len=4096,
    seed=0,
)

mixture_config = lm_mixture_data_config(
    tokenized_datasets,
    mixture_weights,
    shuffle=True,
    missing_weights_are_validation=True,
)

# Configure mixture-based SFT training
training_step = default_sft(
    name="llama3.1_mixture_total-redux",
    tokenized=mixture_config,
    model_config=llama_8b,
    sft_config=mixture_sft_config,
    tags=["llama", "mixture"],
)


if __name__ == "__main__":
    # Run all steps
    executor_main(steps=[*list(tokenized_datasets.values()), training_step])
