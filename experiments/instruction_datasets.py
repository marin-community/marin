"""
Instruction datasets are downloaded from Hugging Face and transformed into OpenAI messages
format which can be used for SFT.

How to add a new instruction dataset:
1. Add the dataset config to INSTRUCTION_DATASET_NAME_TO_CONFIG
2. Register an adapter for the dataset in operations/transform/conversation/adapters.py

How to retrieve an instruction dataset:
1. Use the function `get_instruction_dataset` with the HF repo id.

Current datasets:
1. meta-math/MetaMathQA
2. allenai/tulu-v2-sft-mixture
3. openbmb/UltraInteract_sft
4. teknium/OpenHermes-2.5
5. allenai/tulu-v2-sft-mixture-olmo-4096
6. allenai/tulu-3-sft-mixture
7. TIGER-Lab/AceCode-89K
8. cognitivecomputations/dolphin-r1
9. open-r1/OpenThoughts-114k-math
10. bespokelabs/Bespoke-Stratos-17k
11. HuggingFaceTB/smoltalk
12. PrimeIntellect/verifiable-math-problems
"""

import hashlib
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
        splits: Data splits (e.g., `train`, `validation`) to use. Empty list indicates to use all splits.
                Defaults to `train` only
        legacy: True uses the Marin function as dataloader. False uses the `datasets` package as dataloader.
    """

    hf_dataset_id: str
    revision: str
    wait_for_completion: bool
    metadata_columns: list[str]
    filetype: str
    subsets: list[str] = field(default_factory=lambda: [])
    splits: list[str] = field(default_factory=lambda: ["train"])
    legacy: bool = False


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
    "allenai/tulu-v2-sft-mixture-olmo-4096": InstructionDatasetConfig(
        hf_dataset_id="allenai/tulu-v2-sft-mixture-olmo-4096",
        revision="7a7c388",  # The revision hash shown in the image
        wait_for_completion=True,
        metadata_columns=["dataset", "id"],  # Keeping these metadata columns
        filetype="jsonl",  # Corrected from parquet to jsonl based on the file extension
    ),
    "allenai/tulu-3-sft-mixture": InstructionDatasetConfig(
        hf_dataset_id="allenai/tulu-3-sft-mixture",
        revision="55e9fd6",  # The revision hash shown in the image
        wait_for_completion=True,
        metadata_columns=["dataset", "id"],  # Keeping these metadata columns
        filetype="parquet",
    ),
    "TIGER-Lab/AceCode-89K": InstructionDatasetConfig(
        hf_dataset_id="TIGER-Lab/AceCode-89K",
        revision="0361e95",
        wait_for_completion=True,
        metadata_columns=["id", "source"],
        filetype="parquet",
    ),
    "cognitivecomputations/dolphin-r1": InstructionDatasetConfig(
        hf_dataset_id="cognitivecomputations/dolphin-r1",
        subsets=["nonreasoning", "reasoning-deepseek", "reasoning-flash"],
        revision="f6ac651",  # The revision hash shown in the image
        wait_for_completion=True,
        metadata_columns=["score", "refusal", "compliance_rating", "overall_quality"],
        splits=["train"],
        filetype="jsonl",
    ),
    "open-r1/OpenThoughts-114k-math": InstructionDatasetConfig(
        hf_dataset_id="open-r1/OpenThoughts-114k-math",
        revision="2db609d",  # The revision hash shown in the image
        wait_for_completion=True,
        metadata_columns=["system", "source", "generated_token_count", "correct"],
        filetype="parquet",
    ),
    "bespokelabs/Bespoke-Stratos-17k": InstructionDatasetConfig(
        hf_dataset_id="bespokelabs/Bespoke-Stratos-17k",
        revision="9e9adba",  # The revision hash shown in the image
        wait_for_completion=True,
        filetype="parquet",
        metadata_columns=[],
    ),
    "HuggingFaceTB/smoltalk": InstructionDatasetConfig(
        hf_dataset_id="HuggingFaceTB/smoltalk",
        revision="2c849df",  # The revision hash shown in the image
        wait_for_completion=True,
        metadata_columns=["source"],  # Keeping these metadata columns
        subsets=["all"],
        filetype="parquet",
    ),
    "PrimeIntellect/verifiable-math-problems": InstructionDatasetConfig(
        hf_dataset_id="PrimeIntellect/verifiable-math-problems",
        revision="2ad7c92",  # The revision hash shown in the image
        wait_for_completion=True,
        metadata_columns=["source", "task_type", "problem_id"],  # Keeping these metadata columns
        filetype="parquet",
    ),
    "facebook/natural_reasoning": InstructionDatasetConfig(
        hf_dataset_id="facebook/natural_reasoning",
        revision="main",  # You might want to pin this to a specific commit hash later
        wait_for_completion=True,
        metadata_columns=["reference_answer"],  # Including reference_answer as metadata
        filetype="jsonl",  # The dataset appears to be in parquet format
        splits=["train"],  # Default to train split
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
        override_output_path=f"raw/{dataset_name}-{dataset.revision}",
    )

    return download_step


def transform_dataset_step(dataset_cfg: InstructionDatasetConfig, download_step: ExecutorStep) -> ExecutorStep:
    """ExecutorStep that preprocesses and shards the input dataset.

    ===========================================================================
    dataset_cfg: {
        ...
        "hf_dataset_id": "cognitivecomputations/dolphin-r1",
        "subsets": ["reasoning-flash"],
        "splits": ['train', 'validation'],
        ...
    }
    output_path_of(download_step) --> gs://.../raw/dolphin-r1-[hash]

    Expected files written: [
        gs://.../dolphin-r1-reasoning-flash-train-[hash]/shard_00001.json.gz,
        ...
        gs://.../dolphin-r1-reasoning-flash-train-[hash]/shard_00055.json.gz,
        gs://.../dolphin-r1-reasoning-flash-validation-[hash]/shard_00001.json.gz,
        ...
        gs://.../dolphin-r1-reasoning-flash-validation-[hash]/shard_00023.json.gz,
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

    config_str = f"{dataset_name}-\
        {dataset_cfg.revision}\
        -{sorted(dataset_cfg.subsets)}\
        -{sorted(dataset_cfg.splits)}"
    hashed_config_str = hashlib.md5(config_str.encode()).hexdigest()[:6]

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
            splits=dataset_cfg.splits,
        ),
        override_output_path=f"documents/{dataset_name}-{dataset_cfg.revision}-{hashed_config_str}",
    )

    return transform_step


def get_instruction_dataset(
    hf_dataset_id: str, splits: list[str] = field(default_factory=lambda: ["train"])
) -> ExecutorStep:
    # Check that config exists
    assert hf_dataset_id in INSTRUCTION_DATASET_NAME_TO_CONFIG, f"Unknown instruction dataset: {hf_dataset_id}"

    # Create a new configuration instance with the desired split.
    original_config = INSTRUCTION_DATASET_NAME_TO_CONFIG[hf_dataset_id]
    config = InstructionDatasetConfig(
        **{k: v for k, v in original_config.__dict__.items() if k != "splits"}, splits=splits
    )

    download_step = download_dataset_step(config)
    transform_step = transform_dataset_step(config, download_step)
    return transform_step


if __name__ == "__main__":
    all_steps = []
    for config in INSTRUCTION_DATASET_NAME_TO_CONFIG.values():
        downloaded_dataset = download_dataset_step(config)
        all_steps.append(downloaded_dataset)
        transformed_dataset = transform_dataset_step(config, downloaded_dataset)
        all_steps.append(transformed_dataset)

    executor_main(steps=all_steps)
