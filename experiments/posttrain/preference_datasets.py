# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
   (train_prefs and test_prefs splits included; keep them separate in downstream training)
2. allenai/olmo-2-1124-7b-preference-mix
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from marin.execution.artifact import Artifact, Dataset
from marin.execution.lazy import Lazy, derived
from marin.experiment.data import hf_download
from marin.transform.conversation.transform_preference_data import (
    TransformPreferenceDatasetConfig,
    transform_hf_preference_dataset,
)


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
        splits=["train_prefs", "test_prefs"],
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


def download_preference_dataset_step(dataset: PreferenceDatasetConfig) -> Lazy[Dataset]:
    """Lazy download handle for a preference dataset from HuggingFace."""
    dataset_name = get_directory_friendly_dataset_name(dataset.hf_dataset_id)
    return hf_download(
        f"raw/{dataset_name}",
        hf_id=dataset.hf_dataset_id,
        revision=dataset.revision,
        pin=f"raw/{dataset_name}-{dataset.revision}",
    )


def transform_preference_dataset_step(
    dataset_cfg: PreferenceDatasetConfig, download_step: Lazy[Dataset]
) -> Lazy[Artifact]:
    """Lazy handle that preprocesses and shards the preference dataset.

    ===========================================================================
    dataset_cfg: {
        ...
        "hf_dataset_id": "HuggingFaceH4/ultrafeedback_binarized",
        "subsets": [],
        "splits": ['train'],
        ...
    }
    download_step path --> gs://.../raw/HuggingFaceH4--ultrafeedback_binarized-[revision_number]

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

    pin = f"preference/{dataset_name}-{dataset_cfg.revision}-{hashed_config_str}"

    def _build_config(ctx, _cfg=dataset_cfg, _dl=download_step, _name=adapter_name):
        return TransformPreferenceDatasetConfig(
            input_path=ctx.path(_dl),
            output_path=ctx.out,
            shard_size=5000,
            metadata_columns=_cfg.metadata_columns,
            filetype=_cfg.filetype,
            source=_cfg.hf_dataset_id,
            subsets=_cfg.subsets,
            splits=_cfg.splits,
            adapter_name=_name,
        )

    return derived(
        f"preference/{dataset_name}",
        fn=transform_hf_preference_dataset,
        build_config=_build_config,
        deps=(download_step,),
        pin=pin,
    )


def get_preference_dataset(hf_dataset_id: str, splits: Sequence[str] = ("train",)) -> Lazy[Artifact]:
    """Lazy handle for a preference dataset by HF id, optionally overriding splits."""
    assert hf_dataset_id in PREFERENCE_DATASET_NAME_TO_CONFIG, f"Unknown preference dataset: {hf_dataset_id}"

    original_config = PREFERENCE_DATASET_NAME_TO_CONFIG[hf_dataset_id]
    config = PreferenceDatasetConfig(
        **{k: v for k, v in original_config.__dict__.items() if k != "splits"}, splits=splits
    )

    download_step = download_preference_dataset_step(config)
    transform_step = transform_preference_dataset_step(config, download_step)
    return transform_step
