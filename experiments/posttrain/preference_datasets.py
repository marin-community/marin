# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preference datasets are downloaded from Hugging Face and transformed into OpenAI messages
format with chosen/rejected pairs which can be used for DPO, RLHF, etc.

How to add a new preference dataset:
1. Add the dataset config to PREFERENCE_DATASET_NAME_TO_CONFIG
2. Register an adapter for the dataset in marin/transform/conversation/preference_data_adapters.py

How to retrieve a preference dataset:
1. Use the function `get_preference_dataset` with the HF repo id.

Current datasets:
1. HuggingFaceH4/ultrafeedback_binarized
   (only train_prefs split included to avoid accidentally training on test_prefs)
2. allenai/olmo-2-1124-7b-preference-mix
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from marin.download.huggingface.download_hf import DownloadConfig
from marin.download.huggingface.download_hf import download_hf as _download_hf
from marin.execution import step, deferred, output
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    StepRef,
    versioned,
)
from marin.transform.conversation.transform_preference_data import (
    TransformPreferenceDatasetConfig,
)
from marin.transform.conversation.transform_preference_data import (
    transform_hf_preference_dataset as _transform_hf_preference_dataset,
)

download_hf = deferred(_download_hf)
transform_hf_preference_dataset = deferred(_transform_hf_preference_dataset)


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
        adapter_name: Name of the adapter. None indicates that the adapter name is the same as the `hf_dataset_id`.
    """

    hf_dataset_id: str
    revision: str
    wait_for_completion: bool
    metadata_columns: list[str]
    filetype: str
    subsets: list[str] = field(default_factory=lambda: [])
    splits: list[str] = field(default_factory=lambda: ["train"])
    adapter_name: str = None


PREFERENCE_DATASET_NAME_TO_CONFIG = {
    "HuggingFaceH4/ultrafeedback_binarized": PreferenceDatasetConfig(
        hf_dataset_id="HuggingFaceH4/ultrafeedback_binarized",
        revision="3949bf5",
        wait_for_completion=True,
        metadata_columns=["prompt", "score_chosen", "score_rejected"],
        filetype="parquet",
        splits=["train_prefs"],
    ),
    "allenai/olmo-2-1124-7b-preference-mix": PreferenceDatasetConfig(
        hf_dataset_id="allenai/olmo-2-1124-7b-preference-mix",
        revision="316c96f",
        wait_for_completion=True,
        metadata_columns=["prompt", "chosen_rating", "rejected_rating"],
        filetype="parquet",
        splits=["train"],
    ),
}


def get_directory_friendly_dataset_name(hf_dataset_id: str) -> str:
    dataset_name = hf_dataset_id.replace("/", "--")
    dataset_name = dataset_name.replace(".", "-")
    dataset_name = dataset_name.replace("#", "-")
    return dataset_name


@step(name="raw")
def download_preference_dataset_step(dataset: PreferenceDatasetConfig):
    """ExecutorStep for downloading preference data from external source to GCP"""
    dataset_name = get_directory_friendly_dataset_name(dataset.hf_dataset_id)
    return download_hf(DownloadConfig(
        hf_dataset_id=dataset.hf_dataset_id,
        revision=versioned(dataset.revision),
        gcs_output_path=output(),
        wait_for_completion=True,
    ))


@step(name="preference")
def transform_preference_dataset_step(dataset_cfg: PreferenceDatasetConfig, download_step: ExecutorStep):
    """ExecutorStep that preprocesses and shards the preference dataset.

    ===========================================================================
    dataset_cfg: {
        ...
        "hf_dataset_id": "HuggingFaceH4/ultrafeedback_binarized",
        "subsets": [],
        "splits": ['train'],
        ...
    }
    output_path_of(download_step) --> gs://.../raw/HuggingFaceH4--ultrafeedback_binarized-[revision_number]

    Expected files written: [
        gs://.../HuggingFaceH4--ultrafeedback_binarized__[revision_number]_[hash]/train/shard_00001.jsonl.gz,
        ...
        gs://.../HuggingFaceH4--ultrafeedback_binarized__[revision_number]_[hash]/train/shard_00055.jsonl.gz,
    ]
    ===========================================================================
    """
    adapter_name = dataset_cfg.adapter_name if dataset_cfg.adapter_name is not None else dataset_cfg.hf_dataset_id
    dataset_name = get_directory_friendly_dataset_name(adapter_name)

    config_str = f"{dataset_name}-\
        {dataset_cfg.revision}\
        -{sorted(dataset_cfg.subsets)}\
        -{sorted(dataset_cfg.splits)}"
    hashed_config_str = hashlib.md5(config_str.encode()).hexdigest()[:6]

    return transform_hf_preference_dataset(TransformPreferenceDatasetConfig(
        input_path=download_step,
        output_path=output(),
        shard_size=versioned(5000),
        metadata_columns=versioned(dataset_cfg.metadata_columns),
        filetype=dataset_cfg.filetype,
        source=dataset_cfg.hf_dataset_id,
        subsets=dataset_cfg.subsets,
        splits=dataset_cfg.splits,
        adapter_name=adapter_name,
    ))


def get_preference_dataset(hf_dataset_id: str, splits: Sequence[str] = ("train",)) -> ExecutorStep:
    """Get a preference dataset by creating download and transform steps."""
    # Check that config exists
    assert hf_dataset_id in PREFERENCE_DATASET_NAME_TO_CONFIG, f"Unknown preference dataset: {hf_dataset_id}"

    # Create a new configuration instance with the desired split.
    original_config = PREFERENCE_DATASET_NAME_TO_CONFIG[hf_dataset_id]
    config = PreferenceDatasetConfig(
        **{k: v for k, v in original_config.__dict__.items() if k != "splits"}, splits=splits
    )

    download_step = download_preference_dataset_step(config)
    transform_step = transform_preference_dataset_step(config, download_step)
    return transform_step


@step(name="posttrain/preference_datasets/all")
def prepare_all_preference_datasets():
    """Entry point that prepares all preference datasets."""
    for config in PREFERENCE_DATASET_NAME_TO_CONFIG.values():
        downloaded_dataset = download_preference_dataset_step(config)
        transform_preference_dataset_step(config, downloaded_dataset)


if __name__ == "__main__":
    executor_main(
        steps=[prepare_all_preference_datasets()],
        description="Prepare all preference datasets for DPO/RLHF training"
    )
