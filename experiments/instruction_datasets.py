"""
Instruction datasets are downloaded from Hugging Face and transformed into OpenAI messages
format which can be used for SFT.

How to add a new instruction dataset:
1. Add the dataset config to INSTRUCTION_DATASET_NAME_TO_CONFIG
2. Register an adapter for the dataset in operations/transform/conversation/adapters.py

How to retrieve an instruction dataset:
1. Use the function `get_instruction_dataset` with the HF repo id.
"""

from dataclasses import dataclass

from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from operations.download.huggingface.download import DownloadConfig, download
from operations.transform.conversation.transform_conversation import TransformSFTDatasetConfig, transform_dataset


@dataclass(frozen=True)
class InstructionDatasetConfig:
    """Config to download and transform an instruction dataset.

    Args:
        hf_dataset_id: The Hugging Face repo id of the dataset.
        revision: The revision of the dataset to download. A 7-character commit hash.
        wait_for_completion: Whether to wait for the dataset to be downloaded, usually True.
        metadata_columns: The columns to extract from the dataset. Check the dataset's schema for available columns.
        filetype: The filetype of the dataset; check the dataset's files on Hugging Face for the correct filetype.
    """

    hf_dataset_id: str
    revision: str
    wait_for_completion: bool
    metadata_columns: list[str]
    filetype: str


def download_dataset_step(dataset: InstructionDatasetConfig) -> ExecutorStep:
    dataset_name = dataset.hf_dataset_id.replace("/", "--")
    download_step = ExecutorStep(
        name=f"raw/{dataset_name}",
        fn=download,
        config=DownloadConfig(
            hf_dataset_id=dataset.hf_dataset_id,
            revision=versioned(dataset.revision),
            gcs_output_path=this_output_path(),
            wait_for_completion=dataset.wait_for_completion,
        ),
    )

    return download_step


def transform_dataset_step(dataset: InstructionDatasetConfig, download_step: ExecutorStep) -> ExecutorStep:
    dataset_name = dataset.hf_dataset_id.replace("/", "--")
    download_data = output_path_of(
        download_step,
        f"{dataset.revision}/huggingface.co/datasets/{dataset.hf_dataset_id}/resolve/{dataset.revision}",
    )
    transform_step = ExecutorStep(
        name=f"documents/{dataset_name}",
        fn=transform_dataset,
        config=TransformSFTDatasetConfig(
            input_path=download_data,
            output_path=this_output_path(),
            shard_size=versioned(5000),
            metadata_columns=versioned(dataset.metadata_columns),
            filetype=dataset.filetype,
            source=dataset.hf_dataset_id,
        ),
    )

    return transform_step


def get_instruction_dataset(hf_dataset_id: str) -> ExecutorStep:
    assert hf_dataset_id in INSTRUCTION_DATASET_NAME_TO_CONFIG, f"Unknown instruction dataset: {hf_dataset_id}"
    return INSTRUCTION_DATASET_NAME_TO_TRANSFORMED_DATASET_STEP[hf_dataset_id]


INSTRUCTION_DATASET_NAME_TO_CONFIG = {
    "meta-math/MetaMathQA": InstructionDatasetConfig(
        hf_dataset_id="meta-math/MetaMathQA",
        revision="aa4f34d",
        wait_for_completion=True,
        metadata_columns=["type"],
        filetype="json",
    ),
    "allenai/tulu-v2-sft-mixture": InstructionDatasetConfig(
        hf_dataset_id="allenai/tulu-v2-sft-mixture",
        revision="6248b17",
        wait_for_completion=True,
        metadata_columns=["dataset", "id"],
        filetype="parquet",
    ),
    "openbmb/UltraInteract_sft": InstructionDatasetConfig(
        hf_dataset_id="openbmb/UltraInteract_sft",
        revision="2b102e4",
        wait_for_completion=True,
        metadata_columns=["task", "dataset"],
        filetype="parquet",
    ),
    "teknium/OpenHermes-2.5": InstructionDatasetConfig(
        hf_dataset_id="teknium/OpenHermes-2.5",
        revision="b820378",
        wait_for_completion=True,
        metadata_columns=["id", "category", "source"],
        filetype="json",
    ),
}

INSTRUCTION_DATASET_NAME_TO_TRANSFORMED_DATASET_STEP = {}

if __name__ == "__main__":
    all_steps = []
    for config in INSTRUCTION_DATASET_NAME_TO_CONFIG.values():
        downloaded_dataset = download_dataset_step(config)
        all_steps.append(downloaded_dataset)
        transformed_dataset = transform_dataset_step(config, downloaded_dataset)
        all_steps.append(transformed_dataset)
        INSTRUCTION_DATASET_NAME_TO_TRANSFORMED_DATASET_STEP[config.hf_dataset_id] = transformed_dataset

    executor_main(steps=all_steps)
