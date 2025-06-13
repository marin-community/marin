from instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    download_dataset_step,
    transform_dataset_step,
    get_instruction_dataset,
)

from levanter.data.text import ChatLmDatasetFormat
from experiments.defaults import default_tokenize
from experiments.marin_models import marin_tokenizer

from marin.execution.executor import (
    executor_main,
    ExecutorStep,
)

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

def main():
    return


if __name__ == "__main__":
    dataset_names = [
        #"open-thoughts/OpenThoughts3-1.2M",
        "nvidia/Llama-Nemotron-Post-Training-Dataset-v1-SFT",
    ]
    all_steps = []
    for dataset_name in dataset_names:
        # Download the dataset
        config = INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_name]
        downloaded_dataset = download_dataset_step(config)
        all_steps.append(downloaded_dataset)
        # Transform the dataset
        transformed_dataset = transform_dataset_step(config, downloaded_dataset)
        all_steps.append(transformed_dataset)

    executor_main(steps=all_steps)
