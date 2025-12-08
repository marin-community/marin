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
Instruction datasets are downloaded from Hugging Face and transformed into OpenAI messages
format which can be used for SFT.

How to add a new instruction dataset:
1. Add the dataset config to INSTRUCTION_DATASET_NAME_TO_CONFIG
2. Register an adapter for the dataset in marin/transform/conversation/adapters.py

How to retrieve an instruction dataset:
1. Use the function `get_instruction_dataset` with the HF repo id.

[TBI] = To be implemented

Current datasets:
1. GeneralReasoning/GeneralThought-195K-modelanswer
2. GeneralReasoning/GeneralThought-195K-modelreasoning
3. meta-math/MetaMathQA
4. allenai/tulu-v2-sft-mixture
5. openbmb/UltraInteract_sft
6. teknium/OpenHermes-2.5
7. allenai/tulu-v2-sft-mixture-olmo-4096
8. allenai/tulu-3-sft-mixture
9. TIGER-Lab/AceCode-89K
10. cognitivecomputations/dolphin-r1-nonreasoning
11. cognitivecomputations/dolphin-r1-reasoning
12. open-r1/OpenThoughts-114k-math
13. bespokelabs/Bespoke-Stratos-17k
14. HuggingFaceTB/smoltalk
15. PrimeIntellect/verifiable-math-problems
16. sherryy/tulu-3-sft-personas-instruction-following-expanded
17. facebook/natural_reasoning
18. open-thoughts/OpenThoughts3-1.2M
19. nvidia/Llama-Nemotron-Post-Training-Dataset-v1
20. [TBI] HuggingFaceH4/no_robots
21. [TBI] m-a-p/CodeFeedback-Filtered-Instruction
22. [TBI] nvidia/Daring-Anteater
23. [TBI] HuggingFaceH4/ultrafeedback_binarized
"""

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from urllib.parse import urlparse

import ray
from google.cloud import storage

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from marin.download.huggingface.download import DownloadConfig
from marin.download.huggingface.download_hf import download_hf
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.transform.conversation.conversation_to_dolma import (
    ConversationToDolmaConfig,
    convert_conversation_to_dolma,
)
from marin.transform.conversation.transform_conversation import (
    TransformSFTDatasetConfig,
    create_shard_output_directory,
    get_shard_dir,
    transform_and_write_batch,
    transform_hf_dataset,
)

logger = logging.getLogger("ray")


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
        adapter_name: Name of the adapter. None indicates that the adapater name is the same as the `hf_dataset_id`.
    """

    hf_dataset_id: str
    revision: str
    wait_for_completion: bool
    metadata_columns: list[str]
    filetype: str
    subsets: list[str] = field(default_factory=lambda: [])
    splits: list[str] = field(default_factory=lambda: ["train"])
    adapter_name: str = None
    use_large_dataset_transform: bool = False


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
    "cognitivecomputations/dolphin-r1-nonreasoning": InstructionDatasetConfig(
        hf_dataset_id="cognitivecomputations/dolphin-r1",
        subsets=["nonreasoning"],  # "reasoning-deepseek" & "reasoning-flash" are omitted
        revision="f6ac651",  # The revision hash shown in the image
        wait_for_completion=True,
        metadata_columns=["score", "refusal", "compliance_rating", "overall_quality"],
        splits=["train"],
        filetype="jsonl",
        adapter_name="cognitivecomputations/dolphin-r1-nonreasoning",
    ),
    "cognitivecomputations/dolphin-r1-reasoning": InstructionDatasetConfig(
        hf_dataset_id="cognitivecomputations/dolphin-r1",
        subsets=["reasoning-deepseek", "reasoning-flash"],
        revision="f6ac651",  # The revision hash shown in the image
        wait_for_completion=True,
        metadata_columns=["score", "refusal", "compliance_rating", "overall_quality"],
        splits=["train"],
        filetype="jsonl",
        adapter_name="cognitivecomputations/dolphin-r1-reasoning",
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
    "sherryy/tulu-3-sft-personas-instruction-following-expanded": InstructionDatasetConfig(
        hf_dataset_id="sherryy/tulu-3-sft-personas-instruction-following-expanded",
        revision="79ab2c4",  # The revision hash shown in the image
        wait_for_completion=True,
        metadata_columns=["dataset", "id"],  # Keeping these metadata columns
        filetype="parquet",
    ),
    "facebook/natural_reasoning": InstructionDatasetConfig(
        hf_dataset_id="facebook/natural_reasoning",
        revision="99eea5d",
        wait_for_completion=True,
        metadata_columns=["reference_answer"],  # Including reference_answer as metadata
        filetype="jsonl",  # The dataset appears to be in parquet format
        splits=["train"],  # Default to train split
    ),
    "GeneralReasoning/GeneralThought-195K-modelanswer": InstructionDatasetConfig(
        hf_dataset_id="GeneralReasoning/GeneralThought-195K",
        revision="64f7cb8",
        wait_for_completion=True,
        metadata_columns=["question_id", "question_url", "reference_answer", "model_name", "question_source", "task"],
        filetype="jsonl",  # The dataset appears to be in parquet format
        splits=["train"],  # Default to train split
        adapter_name="GeneralReasoning/GeneralThought-195K-modelanswer",
    ),
    "GeneralReasoning/GeneralThought-195K-modelreasoning": InstructionDatasetConfig(
        hf_dataset_id="GeneralReasoning/GeneralThought-195K",
        revision="64f7cb8",
        wait_for_completion=True,
        metadata_columns=["question_id", "question_url", "reference_answer", "model_name", "question_source", "task"],
        filetype="jsonl",  # The dataset appears to be in parquet format
        splits=["train"],  # Default to train split
        adapter_name="GeneralReasoning/GeneralThought-195K-modelreasoning",
    ),
    "open-thoughts/OpenThoughts3-1.2M": InstructionDatasetConfig(
        hf_dataset_id="open-thoughts/OpenThoughts3-1.2M",
        revision="61bcf9d",  # The revision hash shown in the image
        wait_for_completion=True,
        metadata_columns=["difficulty", "source", "domain"],
        filetype="parquet",
    ),
    "nvidia/Llama-Nemotron-Post-Training-Dataset-v1-SFT": InstructionDatasetConfig(
        hf_dataset_id="nvidia/Llama-Nemotron-Post-Training-Dataset",
        revision="ab2a40d",
        wait_for_completion=True,
        filetype="jsonl",
        splits=["chat", "code", "math", "science", "safety"],
        subsets=["SFT"],
        metadata_columns=["category", "license", "generator"],
        adapter_name="nvidia/Llama-Nemotron-Post-Training-Dataset-v1-SFT",
        use_large_dataset_transform=True,
    ),
}

for split_name in SMOLTALK2_SPLITS:
    dataset_key = f"HuggingFaceTB/smoltalk2/{split_name}"
    INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_key] = InstructionDatasetConfig(
        name=f"HuggingFaceTB/smoltalk2/{split_name}",
        hf_dataset_id="HuggingFaceTB/smoltalk2",
        revision="fc6cc21",
        adapter=multi_turn_adapter(metadata_remap={"chat_template_kwargs": "chat_template_kwargs"}),
        metadata_columns=[],
        subsets=["SFT"],
        metadata_columns=["category", "license", "generator"],
        adapter_name="nvidia/Llama-Nemotron-Post-Training-Dataset-v1-SFT",
        use_large_dataset_transform=True,
    ),
    "nvidia/Nemotron-Post-Training-Dataset-v2-SFT": InstructionDatasetConfig(
        hf_dataset_id="nvidia/Nemotron-Post-Training-Dataset-v2",
        revision="5c89e01",
        wait_for_completion=True,
        filetype="parquet",
        splits=["chat", "code", "math", "stem"],
        subsets=["SFT"],
        metadata_columns=["category", "license", "generator", "reasoning"],
        adapter_name="nvidia/Nemotron-Post-Training-Dataset-v2-SFT",
        use_large_dataset_transform=False,
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
    output_path_of(download_step) --> gs://.../raw/dolphin-r1-[revision_number]-[hash]

    Expected files written: [
        gs://.../dolphin_r1__[revision_number]_[hash]/reasoning_flash/train/shard_00001.json.gz,
        ...
        gs://.../dolphin_r1__[revision_number]_[hash]/reasoning_flash/train/shard_00055.json.gz,
        gs://.../dolphin_r1__[revision_number]_[hash]/reasoning_flash/validation/shard_00001.json.gz,
        ...
        gs://.../dolphin_r1__[revision_number]_[hash]/reasoning_flash/validation/shard_00023.json.gz,
    ]
    ===========================================================================
    """
    adapter_name = dataset_cfg.adapter_name if dataset_cfg.adapter_name is not None else dataset_cfg.hf_dataset_id
    dataset_name = get_directory_friendly_dataset_name(adapter_name)
    download_data_path = output_path_of(download_step)

    config_str = f"{dataset_name}-\
        {dataset_cfg.revision}\
        -{sorted(dataset_cfg.subsets)}\
        -{sorted(dataset_cfg.splits)}"
    hashed_config_str = hashlib.md5(config_str.encode()).hexdigest()[:6]

    transform_config = TransformSFTDatasetConfig(
        input_path=download_data_path,
        output_path=this_output_path(),
        shard_size=versioned(5000),
        metadata_columns=versioned(dataset_cfg.metadata_columns),
        filetype=dataset_cfg.filetype,
        source=dataset_cfg.hf_dataset_id,
        subsets=dataset_cfg.subsets,
        splits=dataset_cfg.splits,
        adapter_name=adapter_name,
    )

    if dataset_cfg.use_large_dataset_transform:
        transform_fn = transform_large_dataset
    else:
        transform_fn = transform_hf_dataset

    transform_step = ExecutorStep(
        name=f"documents/{dataset_name}",
        fn=transform_fn,
        config=transform_config,
        override_output_path=f"documents/{dataset_name}-{dataset_cfg.revision}-{hashed_config_str}",
    )

    return transform_step


########### Nemotron SFT ###########
def list_jsonl_files_in_gcs(bucket_name: str, gcs_directory_path: str) -> list[str]:
    """
    List all .jsonl files in a GCS directory.

    Args:
        bucket_name (str): The name of the GCS bucket.
        gcs_directory_path (str): The path to the directory in GCS (excluding the bucket name).

    Returns:
        list[str]: List of full GCS paths to .jsonl files.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # List all the blobs (files) with the specified prefix
    blobs = bucket.list_blobs(prefix=gcs_directory_path)

    jsonl_files = []
    for blob in blobs:
        if blob.name.endswith(".jsonl") and "provenance.json" not in blob.name:
            jsonl_files.append(blob.name)

    return jsonl_files


def download_single_file_from_gcs(bucket_name: str, gcs_file_path: str, local_file_path: str) -> None:
    """
    Download a single file from GCS to a local path.

    Args:
        bucket_name (str): The name of the GCS bucket.
        gcs_file_path (str): The path to the file in GCS (excluding the bucket name).
        local_file_path (str): The local file path where the file will be saved.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_file_path)

    # Create local directory if it doesn't exist
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

    # Download the blob to the local file path
    blob.download_to_filename(local_file_path)
    logger.info(f"Downloaded gs://{bucket_name}/{gcs_file_path} to {local_file_path}")


def transform_large_dataset(cfg: TransformSFTDatasetConfig):
    """We need a custom transform function because Nemotron is too large (~140GB in total).
    Downloading the entire dataset to disk can fill up the disk and cause disk failure.
    Even downloading splits will cause failure (code split is 50GB+)

    This approach:
    1. Lists all .jsonl files in the GCS directory
    2. Downloads each file individually
    3. Processes each file immediately
    4. Deletes the file after processing to save disk space
    """
    assert len(cfg.subsets) == 1, "This script only supports the SFT subset"
    assert len(cfg.splits) > 0, "Nemotron requires splits to be specified"

    # parse gs://my-bucket/path/to/mmlu into "my-bucket", "path/to/mmlu", and "mmlu"
    parsed_url = urlparse(cfg.input_path)
    bucket = parsed_url.netloc
    gcp_path = parsed_url.path.lstrip("/")
    temp_dir = os.path.join("tmp", "large_dataset_processing")

    # Ensure temp directory exists
    os.makedirs(temp_dir, exist_ok=True)

    # 3. For each subset...
    write_ops = []
    try:
        for subset in cfg.subsets:
            for split in cfg.splits:
                # List all .jsonl files in the GCS directory
                jsonl_files = list_jsonl_files_in_gcs(bucket, f"{gcp_path}/{subset}/{split}")
                # Should be gs://nemotron-x/SFT/code/code_v1.jsonl, gs://nemotron-x/SFT/code/code_v1.1.jsonl, etc.
                # For chat: gs://nemotron-x/SFT/chat/chat.jsonl

                for gcs_file_path in jsonl_files:
                    # Extract filename for local path
                    filename = os.path.basename(gcs_file_path)
                    local_file_path = os.path.join(temp_dir, filename)

                    try:
                        # Download the single file
                        logger.info(f"Downloading file: {gcs_file_path}")
                        download_single_file_from_gcs(bucket, gcs_file_path, local_file_path)

                        # Create GCP target directory
                        subset_output_path = get_shard_dir(cfg.output_path, subset, split)
                        if len(jsonl_files) > 1:
                            # Extract version suffix from filename (e.g., "chat_v1.1" from "chat_v1.1.jsonl")
                            suffix = filename.replace(".jsonl", "").replace(".", "_").strip()
                            # Make the new output path be e.g. nemotron-x/SFT/code/code_v1.1
                            # For chat, it will remain as nemotron-x/SFT/chat
                            subset_output_path += "/" + suffix
                        output_path = create_shard_output_directory(subset_output_path)

                        # Process the downloaded file
                        with open(local_file_path, "r") as f:
                            batch = []
                            shard_idx = 0
                            for line in f:
                                try:
                                    row = json.loads(line)
                                    # Validate required fields
                                    if "input" not in row or "output" not in row:
                                        logger.error(f"Missing required fields: {row}")
                                        raise ValueError(f"Skipping row - missing required fields: {row}")

                                    # Convert input to string if it's a list
                                    if isinstance(row["input"], list):
                                        row["input"] = "\n".join(str(x) for x in row["input"])
                                    elif not isinstance(row["input"], str):
                                        row["input"] = str(row["input"])

                                    # Ensure output is a string
                                    if not isinstance(row["output"], str):
                                        row["output"] = str(row["output"])

                                    # Ensure metadata fields exist
                                    for col in cfg.metadata_columns:
                                        if col not in row:
                                            row[col] = ""  # Set empty string for missing metadata

                                    batch.append(row)

                                    # When batch reaches shard size, process and write it
                                    if len(batch) >= cfg.shard_size:
                                        # Queue the batch for writing
                                        write_ops.append(
                                            transform_and_write_batch.remote(
                                                batch.copy(),  # need .copy() or else ray will fail
                                                shard_idx,
                                                output_path,
                                                cfg,
                                            )
                                        )
                                        # Clear batch and increment shard index
                                        batch = []
                                        shard_idx += 1

                                except json.JSONDecodeError as e:
                                    logger.error(f"Error decoding JSON from line: {e}")
                                    raise e
                                except Exception as e:
                                    logger.error(f"Error processing row: {e}")
                                    raise e

                            # Write any remaining rows in the final batch
                            if batch:
                                write_ops.append(
                                    transform_and_write_batch.remote(
                                        batch.copy(),  # need .copy() or else ray will fail
                                        shard_idx,
                                        output_path,
                                        cfg,
                                    )
                                )

                        logger.info(f"Processed file: {local_file_path}")

                    except Exception as e:
                        logger.error(f"Error processing file {gcs_file_path}: {e}")
                        raise e
                    finally:
                        # Always clean up the local file to save disk space
                        if os.path.exists(local_file_path):
                            os.remove(local_file_path)
                            logger.info(f"Deleted local file: {local_file_path}")
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            logger.info(f"Cleaning up temp directory: {temp_dir}")
            shutil.rmtree(temp_dir)

    # Wait for all write operations to complete
    ray.get(write_ops)
    return cfg.output_path


def get_instruction_dataset(hf_dataset_id: str, splits: Sequence[str] = ("train",)) -> ExecutorStep:
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


tulu_3_in_dolma = ExecutorStep(
    name="dolma/tulu_3_in_dolma",
    fn=convert_conversation_to_dolma,
    config=ConversationToDolmaConfig(output_path_of(get_instruction_dataset("allenai/tulu-3-sft-mixture"))),
)


# levanter treats validation and  training as separate so we tokenize twice. Not ideal, but fine here.
tulu3_flat_llama_tokenized_as_validation = default_tokenize(
    "tulu_sft", tulu_3_in_dolma, tokenizer=llama3_tokenizer, is_validation=True
).with_output_path("tokenized/tulu_sft-1bb7d4")
"""
"flat" here means that we interpolated all the chat messages into a single string per doc
"""

tulu3_flat_llama_tokenized_as_train = default_tokenize(
    "tulu_sft", tulu_3_in_dolma, tokenizer=llama3_tokenizer, is_validation=False
).with_output_path("tokenized/tulu_sft-349fb7/")


if __name__ == "__main__":
    all_steps = []
    for config in INSTRUCTION_DATASET_NAME_TO_CONFIG.values():
        downloaded_dataset = download_dataset_step(config)
        all_steps.append(downloaded_dataset)
        transformed_dataset = transform_dataset_step(config, downloaded_dataset)
        all_steps.append(transformed_dataset)

    executor_main(steps=all_steps)
