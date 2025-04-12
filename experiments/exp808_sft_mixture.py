"""
An experiment to run SFT on a high quality subset of instruction/reasoning datasets.

The default below trains on all the datasets weighted by document count and sets
the number of training steps to epoch three times. If you wish to fine-tune on a
different number of datasets you will need to change the number of training steps
accordingly.
"""

from levanter.data.text import SupervisedUrlSourceConfig

from experiments.defaults import default_sft
from experiments.instruction_datasets import get_instruction_dataset
from experiments.llama import llama_8b
from experiments.simple_sft_config import SimpleSFTConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize.tokenize import TokenizeConfig, levanter_tokenize_sft


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
    return ExecutorStep(
        name=f"tokenized/{short_name}_llama3_instruct_tokenizer",
        fn=levanter_tokenize_sft,
        config=TokenizeConfig(
            train_paths=[output_path_of(dataset, "**/*.jsonl.gz")],
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer="meta-llama/Llama-3.1-8B-Instruct",
            input_field="user",
            output_field="assistant",
        ),
        description="Tokenize SFT data",
    )


# Dataset configurations
DATASETS = [
    "TIGER-Lab/AceCode-89K",
    "HuggingFaceTB/smoltalk",
    "PrimeIntellect/verifiable-math-problems",
    "cognitivecomputations/dolphin-r1-nonreasoning",
    "cognitivecomputations/dolphin-r1-reasoning",
    "bespokelabs/Bespoke-Stratos-17k",
    "open-r1/OpenThoughts-114k-math",
    "allenai/tulu-3-sft-mixture",
    "facebook/natural_reasoning",
]

NUM_TRAIN_STEPS = 19086  # 3 Epochs over all datasets above

# Create tokenization steps for multiple datasets
tokenization_steps = [create_tokenization_step(dataset_name) for dataset_name in DATASETS]

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

# Create dataset configs with weights
supervised_data = {}
for step in tokenization_steps:
    short_name = step.name.split("/")[-1].replace("_llama3_instruct_tokenizer", "")
    supervised_data[short_name] = SupervisedUrlSourceConfig(
        cache_dir=output_path_of(step),
        train_urls=[output_path_of(step, "**/*.jsonl.gz")],
        input_field="user",
        output_field="assistant",
    )

# Define an SFT config appropriate for mixture training
mixture_sft_config = SimpleSFTConfig(
    train_batch_size=128,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=5e-6,
    tpu_type="v4-128",
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    model_name_or_path="meta-llama/Llama-3.1-8B",
    max_seq_len=4096,
    seed=0,
)

# Configure mixture-based SFT training
training_step = default_sft(
    name="llama3.1_mixture_total",
    tokenized=supervised_data,
    model_config=llama_8b,
    sft_config=mixture_sft_config,
    mixture_weights=mixture_weights,
    tags=["dolma", "llama", "mixture"],
)


if __name__ == "__main__":
    # Run all steps
    executor_main(steps=[*tokenization_steps, training_step])
