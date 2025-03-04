from instruction_datasets import get_instruction_dataset

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

if __name__ == "__main__":
    # Create tokenization steps for all datasets
    tokenization_steps = [create_tokenization_step(dataset_name) for dataset_name in DATASETS]
    executor_main(steps=tokenization_steps)
