"""
Instruction datasets are downloaded from Hugging Face and transformed into OpenAI messages
format which can be used for SFT.

How to add a new instruction dataset:
1. Add the dataset config to INSTRUCTION_DATASET_NAME_TO_CONFIG
2. Register an adapter for the dataset in operations/transform/conversation/adapters.py

How to retrieve an instruction dataset:
1. Use the function `get_instruction_dataset` with the HF repo id.
"""

from dataclasses import dataclass, field

from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from operations.download.huggingface.download import DownloadConfig
from operations.download.huggingface.download_hf import download_hf
from operations.transform.conversation.transform_conversation import (
    TransformSFTDatasetConfig,
    transform_dataset,
    transform_hf_dataset,
)


@dataclass(frozen=True)
class InstructionDatasetConfig:
    """Config to download and transform an instruction dataset.

    Args:
        hf_dataset_id: The Hugging Face repo id of the dataset.
        revision: The revision of the dataset to download. A 7-character commit hash.
        wait_for_completion: Whether to wait for the dataset to be downloaded, usually True.
        metadata_columns: The columns to extract from the dataset. Check the dataset's schema for available columns.
        filetype: The filetype of the dataset; check the dataset's files on Hugging Face for the correct filetype.
        subsets: Data subsets (from HuggingFace config) to use. Empty list indicates to use all/default subset(s).
        legacy: True uses the Marin function as dataloader. False uses the `datasets` package as dataloader.
    """

    hf_dataset_id: str
    revision: str
    wait_for_completion: bool
    metadata_columns: list[str]
    filetype: str
    subsets: list[str] = field(default_factory=lambda: [])
    legacy: bool = False


INSTRUCTION_DATASET_NAME_TO_CONFIG = {
    "meta-math/MetaMathQA": InstructionDatasetConfig(
        hf_dataset_id="meta-math/MetaMathQA",
        revision="aa4f34d",
        wait_for_completion=True,
        metadata_columns=["type"],
        filetype="json",
        legacy=True,
    ),
    "allenai/tulu-v2-sft-mixture": InstructionDatasetConfig(
        hf_dataset_id="allenai/tulu-v2-sft-mixture",
        revision="6248b17",
        wait_for_completion=True,
        metadata_columns=["dataset", "id"],
        filetype="parquet",
        legacy=True,
    ),
    "openbmb/UltraInteract_sft": InstructionDatasetConfig(
        hf_dataset_id="openbmb/UltraInteract_sft",
        revision="2b102e4",
        wait_for_completion=True,
        metadata_columns=["task", "dataset"],
        filetype="parquet",
        legacy=True,
    ),
    "teknium/OpenHermes-2.5": InstructionDatasetConfig(
        hf_dataset_id="teknium/OpenHermes-2.5",
        revision="b820378",
        wait_for_completion=True,
        metadata_columns=["id", "category", "source"],
        filetype="json",
        legacy=True,
    ),
    "allenai/tulu-v2-sft-mixture-olmo-4096": InstructionDatasetConfig(
        hf_dataset_id="allenai/tulu-v2-sft-mixture-olmo-4096",
        revision="7a7c388",  # The revision hash shown in the image
        wait_for_completion=True,
        metadata_columns=["dataset", "id"],  # Keeping these metadata columns
        filetype="jsonl",  # Corrected from parquet to jsonl based on the file extension
        legacy=True,
    ),
    "allenai/tulu-3-sft-mixture": InstructionDatasetConfig(
        hf_dataset_id="allenai/tulu-3-sft-mixture",
        revision="55e9fd6",  # The revision hash shown in the image
        wait_for_completion=True,
        metadata_columns=["dataset", "id"],  # Keeping these metadata columns
        filetype="parquet",
        legacy=True,
    ),
    "TIGER-Lab/AceCode-89K": InstructionDatasetConfig(
        hf_dataset_id="TIGER-Lab/AceCode-89K",
        revision="0361e95",
        wait_for_completion=True,
        metadata_columns=["id", "source"],
        filetype="parquet",
        legacy=True,
    ),
    "cognitivecomputations/dolphin-r1": InstructionDatasetConfig(
        hf_dataset_id="cognitivecomputations/dolphin-r1",
        subsets=["nonreasoning", "reasoning-deepseek", "reasoning-flash"],
        revision="f6ac651",  # The revision hash shown in the image
        wait_for_completion=True,
        metadata_columns=["score", "refusal", "compliance_rating", "overall_quality"],  # Keeping these metadata columns
        filetype="jsonl",
    ),
    "open-r1/OpenThoughts-114k-math": InstructionDatasetConfig(
        hf_dataset_id="open-r1/OpenThoughts-114k-math",
        revision="2db609d",  # The revision hash shown in the image
        wait_for_completion=True,
        metadata_columns=["system", "source", "generated_token_count", "correct"],  # Keeping these metadata columns
        filetype="parquet",
    ),
}


def get_directory_friendly_dataset_name(hf_dataset_id: str) -> str:
    dataset_name = hf_dataset_id.replace("/", "--")
    dataset_name = dataset_name.replace(".", "-")
    dataset_name = dataset_name.replace("#", "-")
    return dataset_name


def download_dataset_step(dataset: InstructionDatasetConfig) -> ExecutorStep:
    """ExecutorStep for downloading of data from external source to GCP"""
    dataset_name = get_directory_friendly_dataset_name(dataset.hf_dataset_id)
    download_step = ExecutorStep(
        name=f"raw/{dataset_name}",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id=dataset.hf_dataset_id,
            revision=versioned(dataset.revision),
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
        ),
        override_output_path=f"raw/{dataset_name}",
    )

    return download_step


def transform_dataset_step(dataset_cfg: InstructionDatasetConfig, download_step: ExecutorStep) -> ExecutorStep:
    """ExecutorStep that preprocesses and shards the input dataset.

    ===========================================================================
    dataset_cfg: {
        ...
        "hf_dataset_id": "cognitivecomputations/dolphin-r1",
        "subsets": ["reasoning-flash"],
        ...
    }
    output_path_of(download_step) --> gs://.../raw/dolphin-r1-[hash]

    Expected files written: [
        gs://.../dolphin-r1-reasoning-flash-[hash]/shard_00001.json.gz,
        ...
        gs://.../dolphin-r1-reasoning-flash-[hash]/shard_00055.json.gz,
    ]
    ===========================================================================
    """
    dataset_name = get_directory_friendly_dataset_name(dataset_cfg.hf_dataset_id)
    download_data_path = output_path_of(download_step)

    # Transform the dataset
    if dataset_cfg.legacy:
        # Uses the Marin function
        transform_fn = transform_dataset
    else:
        # Uses the new tranform function that calls `datasets` package
        transform_fn = transform_hf_dataset

    transform_step = ExecutorStep(
        name=f"documents/{dataset_name}",
        fn=transform_fn,
        config=TransformSFTDatasetConfig(
            input_path=download_data_path,
            output_path=this_output_path(),
            shard_size=versioned(5000),
            metadata_columns=versioned(dataset_cfg.metadata_columns),
            filetype=dataset_cfg.filetype,
            source=dataset_cfg.hf_dataset_id,
            subsets=dataset_cfg.subsets,
        ),
    )

    return transform_step


def get_instruction_dataset(hf_dataset_id: str) -> ExecutorStep:
    assert hf_dataset_id in INSTRUCTION_DATASET_NAME_TO_CONFIG, f"Unknown instruction dataset: {hf_dataset_id}"
    download_step = download_dataset_step(INSTRUCTION_DATASET_NAME_TO_CONFIG[hf_dataset_id])
    transform_step = transform_dataset_step(INSTRUCTION_DATASET_NAME_TO_CONFIG[hf_dataset_id], download_step)
    return transform_step


if __name__ == "__main__":
    all_steps = []
    for config in INSTRUCTION_DATASET_NAME_TO_CONFIG.values():
        downloaded_dataset = download_dataset_step(config)
        all_steps.append(downloaded_dataset)
        transformed_dataset = transform_dataset_step(config, downloaded_dataset)
        all_steps.append(transformed_dataset)

    executor_main(steps=all_steps)
