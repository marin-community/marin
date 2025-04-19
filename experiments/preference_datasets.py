from experiments.instruction_datasets import get_directory_friendly_dataset_name
from operations.download.huggingface.download import DownloadConfig
from operations.download.huggingface.download_hf import download_hf
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)

from operations.transform.conversation.transform_conversation import (
    TransformSFTDatasetConfig,
    transform_hf_dataset,
    TransformPreferenceDatasetConfig,
)

"""
Current preference datasets:
1. HuggingFaceH4/ultrafeedback_binarized
"""

@dataclass(frozen=True)
class PreferenceDatasetConfig:
    """Config to download and transform a preference dataset.

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
        adapter_name: Name of the adapter. None indicates that the adapter name is the same as the `hf_dataset_id`.
    """

    hf_dataset_id: str
    revision: str
    wait_for_completion: bool
    metadata_columns: list[str]
    filetype: str
    subsets: list[str] = field(default_factory=lambda: [])
    splits: list[str] = field(default_factory=lambda: ["train"])
    legacy: bool = False
    adapter_name: str = None


PREFERENCE_DATASET_TO_CONFIG = {
    "HuggingFaceH4/ultrafeedback_binarized": PreferenceDatasetConfig(
        hf_dataset_id="HuggingFaceH4/ultrafeedback_binarized",
        revision="3949bf5",
        wait_for_completion=True,
        metadata_columns=[],
        splits=["train_prefs"],
        filetype="parquet",
    ),
}

def download_dataset_step(dataset: PreferenceDatasetConfig) -> ExecutorStep:
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


def transform_dataset_step(dataset_cfg: PreferenceDatasetConfig, download_step: ExecutorStep) -> ExecutorStep:
    """ExecutorStep that preprocesses and shards the preference dataset (DPO, RLHF, etc)."""
    adapter_name = dataset_cfg.adapter_name if dataset_cfg.adapter_name is not None else dataset_cfg.hf_dataset_id
    dataset_name = get_directory_friendly_dataset_name(adapter_name)
    download_data_path = output_path_of(download_step)

    config_str = f"{dataset_name}-\
        {dataset_cfg.revision}\
        -{sorted(dataset_cfg.subsets)}\
        -{sorted(dataset_cfg.splits)}"
    hashed_config_str = hashlib.md5(config_str.encode()).hexdigest()[:6]

    transform_step = ExecutorStep(
        name=f"documents/{dataset_name}",
        fn=transform_hf_dataset,
        config=TransformPreferenceDatasetConfig(
            input_path=download_data_path,
            output_path=this_output_path(),
            shard_size=versioned(5000),
            metadata_columns=versioned(dataset_cfg.metadata_columns),
            filetype=dataset_cfg.filetype,
            source=dataset_cfg.hf_dataset_id,
            subsets=dataset_cfg.subsets,
            splits=dataset_cfg.splits,
            adapter_name=adapter_name,
        ),
        override_output_path=f"documents/{dataset_name}-{dataset_cfg.revision}-{hashed_config_str}",
    )

    return transform_step


if __name__ == "__main__":
    all_steps = []
    for config in PREFERENCE_DATASET_TO_CONFIG.values():
        downloaded_dataset = download_dataset_step(config)
        all_steps.append(downloaded_dataset)
        transformed_dataset = transform_dataset_step(config, downloaded_dataset)
        all_steps.append(transformed_dataset)

    executor_main(steps=all_steps)