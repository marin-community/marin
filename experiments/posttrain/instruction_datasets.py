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
    StepRef,
    versioned,
)
from marin.transform.conversation.conversation_to_dolma import (
    ConversationToDolmaConfig,
    convert_conversation_to_dolma,
)
from marin.transform.conversation.adapters import InputDatasetFormat, TransformAdapter
from marin.transform.conversation.transform_conversation import (
    TransformSFTDatasetConfig,
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
    # nvidia/OpenMathReasoning - CoT split (Chain of Thought reasoning)
    "nvidia/OpenMathReasoning/cot": InstructionDatasetConfig(
        hf_dataset_id="nvidia/OpenMathReasoning",
        revision="d3d0866",
        adapter=instruction_response_adapter(
            instruction_column="problem",
            response_column="generated_solution",
        ),
        metadata_columns=[
            "expected_answer",
            "problem_source",
            "problem_type",
            "generation_model",
            "inference_mode",
            "pass_rate_72b_tir",
            "used_in_kaggle",
        ],
        name="nvidia/OpenMathReasoning/cot",
        splits=["cot"],
    ),
    # nvidia/OpenMathReasoning - TIR split (Tool-Integrated Reasoning)
    "nvidia/OpenMathReasoning/tir": InstructionDatasetConfig(
        hf_dataset_id="nvidia/OpenMathReasoning",
        revision="d3d0866",
        adapter=instruction_response_adapter(
            instruction_column="problem",
            response_column="generated_solution",
        ),
        metadata_columns=[
            "expected_answer",
            "problem_source",
            "problem_type",
            "generation_model",
            "inference_mode",
            "pass_rate_72b_tir",
            "used_in_kaggle",
        ],
        name="nvidia/OpenMathReasoning/tir",
        splits=["tir"],
    ),
    # nvidia/OpenMathReasoning - genselect split (curated subset)
    "nvidia/OpenMathReasoning/genselect": InstructionDatasetConfig(
        hf_dataset_id="nvidia/OpenMathReasoning",
        revision="d3d0866",
        adapter=instruction_response_adapter(
            instruction_column="problem",
            response_column="generated_solution",
        ),
        metadata_columns=[
            "expected_answer",
            "problem_source",
            "problem_type",
            "generation_model",
            "inference_mode",
            "pass_rate_72b_tir",
            "used_in_kaggle",
        ],
        name="nvidia/OpenMathReasoning/genselect",
        splits=["genselect"],
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
            output_path=StepRef(_step=None),
            metadata_columns=versioned(dataset_cfg.metadata_columns),
            adapter=versioned(adapter),
            subsets=versioned(dataset_cfg.subsets),
            splits=versioned(dataset_cfg.splits),
            max_parallelism=dataset_cfg.max_parallelism,
        ),
        override_output_path=f"documents/{dataset_name}-{dataset_cfg.revision}-{hashed_config_str}",
    )

    return transform_step


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
    config=ConversationToDolmaConfig(get_instruction_dataset("allenai/tulu-3-sft-mixture")),
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
