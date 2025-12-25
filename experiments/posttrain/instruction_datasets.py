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
Instruction datasets are streamed from Hugging Face and transformed into OpenAI messages
format which can be used for SFT.

How to add a new instruction dataset:
1. Add the dataset config to INSTRUCTION_DATASET_NAME_TO_CONFIG
2. Provide a TransformAdapter in that config entry (no separate registration required)

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
18. HuggingFaceTB/smoltalk2
19. nvidia/Nemotron-Post-Training-Dataset-v1
20. nvidia/Nemotron-Post-Training-Dataset-v2
21. HuggingFaceH4/no_robots
"""

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
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
from marin.transform.conversation.adapters import InputDatasetFormat, TransformAdapter
from marin.transform.conversation.transform_conversation import (
    TransformSFTDatasetConfig,
    create_shard_output_directory,
    get_shard_dir,
    transform_and_write_batch,
    transform_hf_dataset,
)

SMOLTALK2_SPLITS = [
    "LongAlign_64k_Qwen3_32B_yarn_131k_think",
    "OpenThoughts3_1.2M_think",
    "aya_dataset_Qwen3_32B_think",
    "multi_turn_reasoning_if_think",
    "s1k_1.1_think",
    "smolagents_toolcalling_traces_think",
    "smoltalk_everyday_convs_reasoning_Qwen3_32B_think",
    "smoltalk_multilingual8_Qwen3_32B_think",
    "smoltalk_systemchats_Qwen3_32B_think",
    "table_gpt_Qwen3_32B_think",
    "LongAlign_64k_context_lang_annotated_lang_6_no_think",
    "Mixture_of_Thoughts_science_no_think",
    "OpenHermes_2.5_no_think",
    "OpenThoughts3_1.2M_no_think_no_think",
    "hermes_function_calling_v1_no_think",
    "smoltalk_multilingual_8languages_lang_5_no_think",
    "smoltalk_smollm3_everyday_conversations_no_think",
    "smoltalk_smollm3_explore_instruct_rewriting_no_think",
    "smoltalk_smollm3_smol_magpie_ultra_no_think",
    "smoltalk_smollm3_smol_rewrite_no_think",
    "smoltalk_smollm3_smol_summarize_no_think",
    "smoltalk_smollm3_systemchats_30k_no_think",
    "table_gpt_no_think",
    "tulu_3_sft_personas_instruction_following_no_think",
    "xlam_traces_no_think",
]

NEMOTRON_V2_SPLITS = [
    "stem",
    "chat",
    "math",
    "code",
    "multilingual_ja",
    "multilingual_de",
    "multilingual_it",
    "multilingual_es",
    "multilingual_fr",
]

NEMOTRON_V1_SPLITS = ["chat", "code", "math", "stem", "tool_calling"]


@dataclass(frozen=True)
class InstructionDatasetConfig:
    """Config to download and transform an instruction dataset.

    Args:
        hf_dataset_id: The Hugging Face repo id of the dataset.
        revision: The revision of the dataset to download. A 7-character commit hash.
        adapter: Adapter that converts rows from this dataset to OpenAI chat format.
        metadata_columns: The columns to extract from the dataset. Check the dataset's schema for available columns.
        subsets: Data subsets (from HuggingFace config) to use. Empty list indicates to use all/default subset(s).
        splits: Data splits (e.g., `train`, `validation`) to use. Empty list indicates to use all splits.
                Defaults to `train` only
        name: Optional friendly name for the dataset; defaults to `hf_dataset_id`.
        max_parallelism: Max number of parallel data processing tasks. Reduce if needed to avoid HF rate limits.
    """

    hf_dataset_id: str
    revision: str
    adapter: TransformAdapter
    metadata_columns: list[str]
    name: str | None = None
    subsets: list[str] = field(default_factory=lambda: [])
    splits: list[str] = field(default_factory=lambda: ["train"])
    max_parallelism: int | None = 32  # 32 works for free users; set to None to use default behavior (full parallelism)


def multi_turn_adapter(
    conversation_column: str = "messages",
    role_key: str = "role",
    user_value: str = "user",
    assistant_value: str = "assistant",
    system_value: str = "system",
    content_key: str = "content",
    metadata_remap: dict[str, str] | None = None,
    replacements: dict[str, str] | None = None,
    extra_metadata_fn=None,
) -> TransformAdapter:
    return TransformAdapter(
        dataset_format=InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN,
        conversation_column=conversation_column,
        role_key=role_key,
        user_value=user_value,
        assistant_value=assistant_value,
        system_value=system_value,
        content_key=content_key,
        metadata_remap=metadata_remap or {},
        replacements=replacements,
        extra_metadata_fn=extra_metadata_fn,
    )


def instruction_response_adapter(
    *,
    instruction_column: str,
    response_column: str,
    content_key: str = "",
    filter_on_key: str = "",
    metadata_remap: dict[str, str] | None = None,
    replacements: dict[str, str] | None = None,
    extra_metadata_fn=None,
) -> TransformAdapter:
    return TransformAdapter(
        dataset_format=InputDatasetFormat.INSTRUCTION_RESPONSE,
        instruction_column=instruction_column,
        response_column=response_column,
        content_key=content_key,
        filter_on_key=filter_on_key,
        metadata_remap=metadata_remap or {},
        replacements=replacements,
        extra_metadata_fn=extra_metadata_fn,
    )


def instruct_column_response_adapter(
    instruction_column: str,
    response_column: str,
    content_key: str,
    metadata_remap: dict[str, str] | None = None,
    replacements: dict[str, str] | None = None,
    extra_metadata_fn=None,
) -> TransformAdapter:
    return TransformAdapter(
        dataset_format=InputDatasetFormat.INSTRUCT_COLUMN_RESPONSE,
        instruction_column=instruction_column,
        response_column=response_column,
        content_key=content_key,
        metadata_remap=metadata_remap or {},
        replacements=replacements,
        extra_metadata_fn=extra_metadata_fn,
    )


def instruct_msg_response_adapter(
    *,
    instruction_column: str,
    response_column: str,
    role_key: str,
    user_value: str,
    assistant_value: str,
    system_value: str,
    content_key: str,
    metadata_remap: dict[str, str] | None = None,
    replacements: dict[str, str] | None = None,
    extra_metadata_fn=None,
) -> TransformAdapter:
    return TransformAdapter(
        dataset_format=InputDatasetFormat.INSTRUCT_MSG_RESPONSE,
        instruction_column=instruction_column,
        response_column=response_column,
        role_key=role_key,
        user_value=user_value,
        assistant_value=assistant_value,
        system_value=system_value,
        content_key=content_key,
        metadata_remap=metadata_remap or {},
        replacements=replacements,
        extra_metadata_fn=extra_metadata_fn,
    )


@dataclass
class ReasoningToChatKwargs:
    """Callable metadata helper to toggle thinking mode based on the "reasoning" column."""

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        value = row.get("reasoning")
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"on", "true", "1"}:
                return {"chat_template_kwargs": {"enable_thinking": True}}
            if lowered in {"off", "false", "0"}:
                return {"chat_template_kwargs": {"enable_thinking": False}}
        if isinstance(value, bool):
            return {"chat_template_kwargs": {"enable_thinking": value}}
        return {}


reasoning_to_chat_kwargs = ReasoningToChatKwargs()


INSTRUCTION_DATASET_NAME_TO_CONFIG = {
    "meta-math/MetaMathQA": InstructionDatasetConfig(
        hf_dataset_id="meta-math/MetaMathQA",
        revision="aa4f34d",
        adapter=instruction_response_adapter(
            instruction_column="query",
            response_column="response",
        ),
        metadata_columns=["type"],
        name="meta-math/MetaMathQA",
    ),
    "allenai/tulu-v2-sft-mixture": InstructionDatasetConfig(
        hf_dataset_id="allenai/tulu-v2-sft-mixture",
        revision="6248b17",
        adapter=multi_turn_adapter(),
        metadata_columns=["dataset", "id"],
        name="allenai/tulu-v2-sft-mixture",
    ),
    "openbmb/UltraInteract_sft": InstructionDatasetConfig(
        hf_dataset_id="openbmb/UltraInteract_sft",
        revision="2b102e4",
        adapter=instruction_response_adapter(
            instruction_column="instruction",
            response_column="response",
        ),
        metadata_columns=["task", "dataset"],
        name="openbmb/UltraInteract_sft",
    ),
    "teknium/OpenHermes-2.5": InstructionDatasetConfig(
        hf_dataset_id="teknium/OpenHermes-2.5",
        revision="b820378",
        adapter=multi_turn_adapter(
            conversation_column="conversations",
            role_key="from",
            user_value="human",
            assistant_value="gpt",
            system_value="system",
            content_key="value",
        ),
        metadata_columns=["id", "category", "source"],
        name="teknium/OpenHermes-2.5",
    ),
    "allenai/tulu-v2-sft-mixture-olmo-4096": InstructionDatasetConfig(
        hf_dataset_id="allenai/tulu-v2-sft-mixture-olmo-4096",
        revision="7a7c388",
        adapter=multi_turn_adapter(),
        metadata_columns=["dataset", "id"],
        name="allenai/tulu-v2-sft-mixture-olmo-4096",
    ),
    "allenai/tulu-3-sft-mixture": InstructionDatasetConfig(
        hf_dataset_id="allenai/tulu-3-sft-mixture",
        revision="55e9fd6",
        adapter=multi_turn_adapter(),
        metadata_columns=["dataset", "id"],
        name="allenai/tulu-3-sft-mixture",
    ),
    "TIGER-Lab/AceCode-89K": InstructionDatasetConfig(
        hf_dataset_id="TIGER-Lab/AceCode-89K",
        revision="13216309a9f6cb40b60cb1a9750071efeac414ad",
        adapter=instruction_response_adapter(
            instruction_column="question",
            response_column="inferences",
            content_key="completion",
            filter_on_key="pass_rate",
        ),
        metadata_columns=["id", "source"],
        name="TIGER-Lab/AceCode-89K",
    ),
    "cognitivecomputations/dolphin-r1-nonreasoning": InstructionDatasetConfig(
        hf_dataset_id="cognitivecomputations/dolphin-r1",
        revision="f6ac651",
        adapter=multi_turn_adapter(),
        metadata_columns=["score", "refusal", "compliance_rating", "overall_quality"],
        name="cognitivecomputations/dolphin-r1-nonreasoning",
        subsets=["nonreasoning"],
        splits=["train"],
    ),
    "cognitivecomputations/dolphin-r1-reasoning": InstructionDatasetConfig(
        hf_dataset_id="cognitivecomputations/dolphin-r1",
        revision="f6ac651",
        adapter=instruct_msg_response_adapter(
            instruction_column="messages",
            response_column="answer",
            role_key="role",
            user_value="user",
            assistant_value="assistant",
            system_value="system",
            content_key="content",
        ),
        metadata_columns=["score", "refusal", "compliance_rating", "overall_quality"],
        name="cognitivecomputations/dolphin-r1-reasoning",
        subsets=["reasoning-deepseek", "reasoning-flash"],
        splits=["train"],
    ),
    "open-r1/OpenThoughts-114k-math": InstructionDatasetConfig(
        hf_dataset_id="open-r1/OpenThoughts-114k-math",
        revision="2db609d",
        adapter=multi_turn_adapter(),
        metadata_columns=["system", "source", "generated_token_count", "correct"],
        name="open-r1/OpenThoughts-114k-math",
    ),
    "bespokelabs/Bespoke-Stratos-17k": InstructionDatasetConfig(
        hf_dataset_id="bespokelabs/Bespoke-Stratos-17k",
        revision="9e9adba",
        adapter=TransformAdapter(
            dataset_format=InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN,
            instruction_column="system",
            conversation_column="conversations",
            role_key="from",
            user_value="user",
            assistant_value="assistant",
            content_key="value",
        ),
        metadata_columns=[],
        name="bespokelabs/Bespoke-Stratos-17k",
    ),
    "HuggingFaceTB/smoltalk": InstructionDatasetConfig(
        hf_dataset_id="HuggingFaceTB/smoltalk",
        revision="2c849df",
        adapter=multi_turn_adapter(metadata_remap={"chat_template_kwargs": "chat_template_kwargs"}),
        metadata_columns=["source"],
        name="HuggingFaceTB/smoltalk",
        subsets=["all"],
    ),
    "HuggingFaceH4/no_robots": InstructionDatasetConfig(
        hf_dataset_id="HuggingFaceH4/no_robots",
        revision="e6f9a4a",
        adapter=multi_turn_adapter(),
        metadata_columns=["category", "prompt_id"],
        name="HuggingFaceH4/no_robots",
        splits=["train"],
    ),
    "PrimeIntellect/verifiable-math-problems": InstructionDatasetConfig(
        hf_dataset_id="PrimeIntellect/verifiable-math-problems",
        revision="2ad7c92",
        adapter=instruction_response_adapter(
            instruction_column="prompt",
            response_column="gold_standard_solution",
        ),
        metadata_columns=["source", "task_type", "problem_id"],
        name="PrimeIntellect/verifiable-math-problems",
    ),
    "sherryy/tulu-3-sft-personas-instruction-following-expanded": InstructionDatasetConfig(
        hf_dataset_id="sherryy/tulu-3-sft-personas-instruction-following-expanded",
        revision="79ab2c4",
        adapter=multi_turn_adapter(),
        metadata_columns=["dataset", "id"],
        name="sherryy/tulu-3-sft-personas-instruction-following-expanded",
    ),
    "facebook/natural_reasoning": InstructionDatasetConfig(
        hf_dataset_id="facebook/natural_reasoning",
        revision="99eea5d",
        adapter=instruct_column_response_adapter(
            instruction_column="question",
            response_column="responses",
            content_key="response",
        ),
        metadata_columns=["reference_answer"],
        name="facebook/natural_reasoning",
        splits=["train"],
    ),
    "GeneralReasoning/GeneralThought-195K-modelanswer": InstructionDatasetConfig(
        hf_dataset_id="GeneralReasoning/GeneralThought-195K",
        revision="64f7cb8",
        adapter=instruction_response_adapter(
            instruction_column="question",
            response_column="model_answer",
        ),
        metadata_columns=[
            "question_id",
            "question_url",
            "reference_answer",
            "model_name",
            "question_source",
            "task",
        ],
        name="GeneralReasoning/GeneralThought-195K-modelanswer",
        splits=["train"],
    ),
    "GeneralReasoning/GeneralThought-195K-modelreasoning": InstructionDatasetConfig(
        hf_dataset_id="GeneralReasoning/GeneralThought-195K",
        revision="64f7cb8",
        adapter=instruction_response_adapter(
            instruction_column="question",
            response_column="model_reasoning",
        ),
        metadata_columns=[
            "question_id",
            "question_url",
            "reference_answer",
            "model_name",
            "question_source",
            "task",
        ],
        name="GeneralReasoning/GeneralThought-195K-modelreasoning",
        splits=["train"],
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
        splits=[split_name],
    )

for split_name in NEMOTRON_V2_SPLITS:
    dataset_key = f"nvidia/Nemotron-Post-Training-Dataset-v2/{split_name}"
    INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_key] = InstructionDatasetConfig(
        name=dataset_key,
        hf_dataset_id="nvidia/Nemotron-Post-Training-Dataset-v2",
        revision="5c89e01",
        adapter=multi_turn_adapter(extra_metadata_fn=reasoning_to_chat_kwargs),
        metadata_columns=["category", "generator", "license"],
        splits=[split_name],
    )

for split_name in NEMOTRON_V1_SPLITS:
    dataset_key = f"nvidia/Nemotron-Post-Training-Dataset-v1/{split_name}"
    INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_key] = InstructionDatasetConfig(
        name=dataset_key,
        hf_dataset_id="nvidia/Nemotron-Post-Training-Dataset-v1",
        revision="74e23eb",
        adapter=multi_turn_adapter(extra_metadata_fn=reasoning_to_chat_kwargs),
        metadata_columns=["category", "generator", "license", "metadata", "version"],
        splits=[split_name],
    )


def get_directory_friendly_dataset_name(hf_dataset_id: str) -> str:
    dataset_name = hf_dataset_id.replace("/", "--")
    dataset_name = dataset_name.replace(".", "-")
    dataset_name = dataset_name.replace("#", "-")
    return dataset_name


def transform_dataset_step(dataset_cfg: InstructionDatasetConfig) -> ExecutorStep:
    """ExecutorStep that preprocesses the input dataset into a canonicalized format for SFT training."""
    adapter = dataset_cfg.adapter
    output_name = dataset_cfg.name if dataset_cfg.name is not None else dataset_cfg.hf_dataset_id
    dataset_name = get_directory_friendly_dataset_name(output_name)

    adapter_dict = dataclasses.asdict(adapter)
    adapter_dict["dataset_format"] = adapter_dict["dataset_format"].value

    def canonicalize(value):
        if isinstance(value, dict):
            return {k: canonicalize(v) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return [canonicalize(x) for x in value]
        if callable(value):
            return f"{value.__module__}.{value.__qualname__}"
        return value

    adapter_signature = canonicalize(adapter_dict)
    adapter_signature_str = json.dumps(adapter_signature, sort_keys=True)

    config_str = f"{dataset_name}-\
        {dataset_cfg.revision}\
        -{sorted(dataset_cfg.subsets)}\
        -{sorted(dataset_cfg.splits)}\
        -{adapter_signature_str}"
    hashed_config_str = hashlib.md5(config_str.encode()).hexdigest()[:6]

    transform_step = ExecutorStep(
        name=f"documents/{output_name}",
        fn=transform_hf_dataset,
        config=TransformSFTDatasetConfig(
            source=versioned(dataset_cfg.hf_dataset_id),
            revision=versioned(dataset_cfg.revision),
            output_path=this_output_path(),
            metadata_columns=versioned(dataset_cfg.metadata_columns),
            adapter=versioned(adapter),
            subsets=versioned(dataset_cfg.subsets),
            splits=versioned(dataset_cfg.splits),
            max_parallelism=dataset_cfg.max_parallelism,
        ),
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


def get_instruction_dataset(hf_dataset_id: str, splits: Sequence[str] | None = None) -> ExecutorStep:
    # Check that config exists
    assert hf_dataset_id in INSTRUCTION_DATASET_NAME_TO_CONFIG, f"Unknown instruction dataset: {hf_dataset_id}"

    # Create a new configuration instance with the desired split.
    original_config = INSTRUCTION_DATASET_NAME_TO_CONFIG[hf_dataset_id]
    if splits is None:
        splits = original_config.splits
    config = InstructionDatasetConfig(
        **{k: v for k, v in original_config.__dict__.items() if k != "splits"}, splits=splits
    )

    return transform_dataset_step(config)


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
        transformed_dataset = transform_dataset_step(config)
        all_steps.append(transformed_dataset)

    executor_main(steps=all_steps)
