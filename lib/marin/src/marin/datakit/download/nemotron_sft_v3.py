# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron post-training SFT exports as structured Harmony chat sources."""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType

import pyarrow as pa
from openai_harmony import Message

import msgspec
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.context import ZephyrContext
from zephyr.readers import open_file

from marin.datakit.chat_normalize import CHAT_SCHEMA, validate_chat_messages, validate_tool_definitions, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.nemotron_chat_prompts import SEED_DATASET_REVISIONS, restore_chat_prompts
from marin.datakit.download.rollout_transforms import load_parquet_batched, openai_chat_document
from marin.execution.step_spec import StepSpec

TRANSFORM_VERSION = "2026.09.17.chat-v1"
MAX_CONSECUTIVE_IDENTICAL_LINES = 256


@dataclass(frozen=True)
class NemotronSFTPartition:
    config: str
    split: str
    file_glob: str
    skipped_jsonl_lines: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class NemotronSFTRepository:
    hf_dataset_id: str
    revision: str
    partitions: Mapping[str, NemotronSFTPartition]

    def __post_init__(self):
        object.__setattr__(self, "partitions", MappingProxyType(dict(self.partitions)))


_MULTILINGUAL_V1_LANGUAGES = ("de", "es", "fr", "it", "ja", "zh")
_MULTILINGUAL_V2_LANGUAGES = ("hi", "ja", "ko", "pt")
_ARC_AGI_CONFIGS = (
    "small_reasoning_and_tools",
    "small_reasoning_no_tools",
    "small_tools_no_reasoning",
    "small_no_reasoning_no_tools",
    "large_reasoning_and_tools",
    "large_reasoning_no_tools",
    "large_tools_no_reasoning",
    "large_no_reasoning_no_tools",
)

NEMOTRON_SFT_V3_REPOSITORIES: Mapping[str, NemotronSFTRepository] = MappingProxyType(
    {
        "agentic_v1": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-Agentic-v1",
            revision="650d590978ca35c8f1ecea2faf136e5fac421b62",
            partitions={
                split: NemotronSFTPartition(config="default", split=split, file_glob=f"data/{split}.jsonl")
                for split in ("interactive_agent", "tool_calling")
            },
        ),
        "agentic_v2": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SFT-Agentic-v2",
            revision="49e79a3be5ab8cf7511a12958b95cfd6408cd8db",
            partitions={
                "interactive_agent": NemotronSFTPartition(
                    config="default",
                    split="interactive_agent",
                    file_glob="data/interactive_agent.jsonl",
                ),
                "search": NemotronSFTPartition(
                    config="default",
                    split="search",
                    file_glob="data/search.jsonl",
                ),
                "tool_calling": NemotronSFTPartition(
                    config="default",
                    split="tool_calling",
                    file_glob="data/tool_calling.jsonl",
                    skipped_jsonl_lines=frozenset({1095}),
                ),
            },
        ),
        "opencode_v1": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SFT-OpenCode-v1",
            revision="556d5237acff203f3e1a0be49428634c3606cda2",
            partitions={
                split: NemotronSFTPartition(
                    config="default",
                    split=split,
                    file_glob=f"{split}/data.jsonl",
                )
                for split in (
                    "bash_only_tool_skills",
                    "bash_only_tool",
                    "general",
                    "question_tool",
                    "agent_skills",
                    "agent_skills_question_tool",
                )
            },
        ),
        "swe_v2": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SFT-SWE-v2",
            revision="bd151f3f2d89c4804dda0083d912bd9f6a0a9fb7",
            partitions={
                "agentless": NemotronSFTPartition(
                    config="default",
                    split="agentless",
                    file_glob="data/agentless.jsonl",
                ),
                "openhands_swe": NemotronSFTPartition(
                    config="default",
                    split="openhands_swe",
                    file_glob="data/swe.jsonl",
                ),
            },
        ),
        "safety_v1": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SFT-Safety-v1",
            revision="913fd7c803a9378dab0ce4fef80297ce115781f6",
            partitions={"train": NemotronSFTPartition(config="default", split="train", file_glob="data/train.jsonl")},
        ),
        "competitive_programming_v2": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SFT-Competitive-Programming-v2",
            revision="778afc98a9e027e10b3cd78020c120e93e142ef2",
            partitions={
                "exercism": NemotronSFTPartition(
                    config="default",
                    split="exercism",
                    file_glob="data/exercism.jsonl",
                ),
                "text_to_sql": NemotronSFTPartition(
                    config="default",
                    split="text_to_sql",
                    file_glob="data/text_to_sql.jsonl",
                ),
                "competitive_coding_cpp": NemotronSFTPartition(
                    config="default",
                    split="competitive_coding_cpp",
                    file_glob="data/competitive_programming_cpp_*.jsonl",
                ),
                "competitive_coding_python": NemotronSFTPartition(
                    config="default",
                    split="competitive_coding_python",
                    file_glob="data/competitive_programming_python_*.jsonl",
                ),
            },
        ),
        "instruction_following_chat_v2": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SFT-Instruction-Following-Chat-v2",
            revision="1a9454ed054b8544503ab8d8c0a519d141a44c5b",
            partitions={
                split: NemotronSFTPartition(
                    config="default",
                    split=split,
                    file_glob=f"data/{split}.jsonl",
                )
                for split in ("reasoning_off", "reasoning_on")
            },
        ),
        "multilingual_v1": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SFT-Multilingual-v1",
            revision="22c86505762a7c595abee309d720084351c9f4ba",
            partitions={
                f"{domain}_{language}": NemotronSFTPartition(
                    config="default",
                    split=f"{domain}_{language}",
                    file_glob=(
                        f"data/super-v3_{domain}_{language}_translated"
                        f"{'_postedit' if domain == 'stem' else ''}_final.jsonl"
                    ),
                )
                for domain in ("code", "math", "stem")
                for language in _MULTILINGUAL_V1_LANGUAGES
            },
        ),
        "arc_agi_v1": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SFT-ARC-AGI-v1",
            revision="92837449e198007b76830b75508cc6946795cd11",
            partitions={
                config: NemotronSFTPartition(
                    config=config,
                    split="train",
                    file_glob=f"data/{config}/*.jsonl",
                )
                for config in _ARC_AGI_CONFIGS
            },
        ),
        "cuda_v1": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SFT-CUDA-v1",
            revision="1a06167a6e1e90d928094184173898cbb9bf42de",
            partitions={"train": NemotronSFTPartition(config="default", split="train", file_glob="data/train.jsonl")},
        ),
        "instruction_following_chat_v3": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SFT-Instruction-Following-Chat-v3",
            revision="be3b3e04ef605ac9d3f8f35b9d5a632f4a3a3402",
            partitions={
                "chat": NemotronSFTPartition(config="default", split="chat", file_glob="data/chat.jsonl"),
                "instruction_following": NemotronSFTPartition(
                    config="default",
                    split="instruction_following",
                    file_glob="data/instruction_following.jsonl",
                ),
            },
        ),
        "math_v4": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SFT-Math-v4",
            revision="a94e56aeddcf6e75d28c8bd210f40fa62309288d",
            partitions={"train": NemotronSFTPartition(config="default", split="train", file_glob="data/train.jsonl")},
        ),
        "math_v2": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-Math-v2",
            revision="8e793210e175b6406c752a870f585f62de98c0d3",
            partitions={
                "high": NemotronSFTPartition(config="default", split="high", file_glob="data/high_part*.parquet"),
                "medium": NemotronSFTPartition(config="default", split="medium", file_glob="data/medium.parquet"),
                "low": NemotronSFTPartition(config="default", split="low", file_glob="data/low.parquet"),
            },
        ),
        "math_proofs_v1": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-Math-Proofs-v1",
            revision="97229c590831adfe96202f5cd071d444d535bf91",
            partitions={"lean": NemotronSFTPartition(config="default", split="lean", file_glob="data/lean.jsonl")},
        ),
        "math_proofs_v2": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-Math-Proofs-v2",
            revision="7665d7f1d006fd89aa852a9dab8060c60b63f814",
            partitions={"train": NemotronSFTPartition(config="default", split="train", file_glob="data/train.jsonl")},
        ),
        "multilingual_v2": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SFT-Multilingual-v2",
            revision="971a252224b75414b1b67c55dbe0446d8b6606a0",
            partitions={
                f"{domain}_{language}": NemotronSFTPartition(
                    config="default",
                    split=f"{domain}_{language}",
                    file_glob=(
                        f"ultra-v3_{domain}_{language}_translated{'_postedit' if domain == 'stem' else ''}_final.jsonl"
                    ),
                )
                for domain in ("code", "math", "stem")
                for language in _MULTILINGUAL_V2_LANGUAGES
            },
        ),
        "safety_v2": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SFT-Safety-v2",
            revision="8a40a63c9a1a340874b874f980953be53bff0a07",
            partitions={"train": NemotronSFTPartition(config="default", split="train", file_glob="data/train.jsonl")},
        ),
        "science_v2": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SFT-Science-v2",
            revision="6536a5021222a94126968e8c92f29ee47fc8a7df",
            partitions={
                config: NemotronSFTPartition(config=config, split="train", file_glob=f"{config}.jsonl")
                for config in ("rqa", "so", "syn_mcq", "vendor")
            },
        ),
        "finance_v1": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SpecializedDomains-Finance-v1",
            revision="5a21b106168facb96ced11b883c2a9b4788ee939",
            partitions={"train": NemotronSFTPartition(config="default", split="train", file_glob="data/train.jsonl")},
        ),
        "swe_v1": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SWE-v1",
            revision="0fe17a965b297a9c943a59050a14c42d5f0083ce",
            partitions={
                "r2e_gym": NemotronSFTPartition(config="default", split="r2e_gym", file_glob="data/r2e_gym.jsonl")
            },
        ),
        "math_v3": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SFT-Math-v3",
            revision="ff4439c1073c87e006ab7ee5f1e5e28c4790dab3",
            partitions={"train": NemotronSFTPartition(config="default", split="train", file_glob="data/train.jsonl")},
        ),
    }
)


SOURCE_CHAT_SCHEMA = pa.schema([*CHAT_SCHEMA, pa.field("source_train_turns", pa.list_(pa.bool_()))])


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _has_pathological_line_repetition(text: str) -> bool:
    previous = None
    run_length = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            previous = None
            run_length = 0
            continue
        if line == previous:
            run_length += 1
        else:
            previous = line
            run_length = 1
        if run_length > MAX_CONSECUTIVE_IDENTICAL_LINES:
            return True
    return False


def _parse_messages(value: object) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("messages must be a list")
    messages: list[dict] = []
    for message in value:
        if isinstance(message, str):
            message = json.loads(message)
        if not isinstance(message, dict):
            raise ValueError("Each message must be a JSON object")
        parsed = dict(message)
        if isinstance(parsed.get("tool_calls"), str):
            parsed["tool_calls"] = json.loads(parsed["tool_calls"])
        content = parsed.get("content")
        if content is not None and not isinstance(content, str):
            parsed["content"] = _canonical_json(content)
        messages.append(parsed)
    return messages


def _tool_definitions(value: object) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("tools must be a list")
    definitions: list[dict] = []
    for tool in value:
        if not isinstance(tool, dict):
            raise ValueError("Tool definitions must be objects")
        if "id" in tool and "inputSchema" in tool:
            input_schema = tool["inputSchema"]
            if not isinstance(input_schema, dict):
                raise ValueError("OpenCode inputSchema must be an object")
            parameters = input_schema.get("jsonSchema", input_schema)
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["id"],
                        "description": tool.get("description", ""),
                        "parameters": parameters,
                    },
                }
            )
        else:
            definitions.append(tool)
    return definitions


def _template_kwargs(row: dict) -> dict:
    value = row.get("chat_template_kwargs") or {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("chat_template_kwargs must be an object")
    kwargs = dict(value)
    if "thinking" in kwargs:
        kwargs["enable_thinking"] = kwargs.pop("thinking")
    tools = _tool_definitions(row.get("tools"))
    if tools:
        kwargs["tools"] = tools
    return kwargs


def row_to_chat_doc(row: dict, *, family: str, partition_name: str) -> list[dict]:
    """Convert a self-contained source row to validated Harmony chat."""
    repository = NEMOTRON_SFT_V3_REPOSITORIES[family]
    counter = f"nemotron_sft_v3/{family}/{partition_name}/chat"
    try:
        messages = _parse_messages(row.get("messages"))
        if not messages:
            counters.pipeline.update_counter(f"{counter}/empty_messages_filtered", 1)
            return []
        if family == "instruction_following_chat_v3" and partition_name == "chat":
            first_user = next((message for message in messages if message.get("role") == "user"), None)
            if first_user is None or not first_user.get("content"):
                counters.pipeline.update_counter(f"{counter}/withheld_prompt_filtered", 1)
                return []
        source_text = "\n".join(
            text
            for message in messages
            for text in (message.get("content"), message.get("reasoning_content"))
            if isinstance(text, str)
        )
        if _has_pathological_line_repetition(source_text):
            counters.pipeline.update_counter(f"{counter}/pathological_repetition_filtered", 1)
            return []
        kwargs = _template_kwargs(row)
        document = openai_chat_document(
            messages,
            repository.hf_dataset_id,
            source_id=row.get("uuid"),
            chat_template_kwargs=kwargs,
        )
        normalized_messages = [Message.from_dict(message) for message in document["messages"]]
        validate_chat_messages(normalized_messages)
        validate_tool_definitions(kwargs.get("tools", []), normalized_messages)
        metadata = row.get("metadata")
        source_train_turns = metadata.get("train_turns") if isinstance(metadata, Mapping) else None
        if source_train_turns is not None and (
            not isinstance(source_train_turns, list)
            or len(source_train_turns) != len(messages)
            or any(not isinstance(value, bool) for value in source_train_turns)
        ):
            raise ValueError("Source train_turns must contain one boolean per source message")
        document["source_train_turns"] = source_train_turns
    except (UnicodeError, ValueError) as error:
        counters.pipeline.update_counter(f"{counter}/quarantined/{type(error).__name__}", 1)
        return []
    counters.pipeline.update_counter(f"{counter}/kept", 1)
    return [document]


def load_jsonl_with_skips(source: str, skipped_lines: frozenset[int]) -> Iterator[dict]:
    """Load JSONL while skipping exact malformed lines in an immutable source revision."""
    decoder = msgspec.json.Decoder()
    with open_file(source, "rt") as f:
        for line_number, line in enumerate(f, start=1):
            if line_number in skipped_lines:
                counters.pipeline.update_counter("nemotron_sft_v3/skipped_jsonl_line", 1)
                continue
            line = line.strip()
            if not line:
                continue
            record = decoder.decode(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} in {source} must be a JSON object")
            counters.pipeline.update_counter(counters.RECORDS_IN, 1)
            yield record


@cache
def download_nemotron_sft_v3_step(family: str) -> StepSpec:
    """Create one pinned raw-file download shared by a repository's partitions."""
    repository = NEMOTRON_SFT_V3_REPOSITORIES[family]
    return download_hf_step(
        f"raw/nemotron_sft_v3/{family}",
        hf_dataset_id=repository.hf_dataset_id,
        revision=repository.revision,
        hf_urls_glob=sorted({partition.file_glob for partition in repository.partitions.values()}),
    )


def _transform_chat(input_path: str, output_path: str, *, family: str, partition_name: str) -> None:
    partition = NEMOTRON_SFT_V3_REPOSITORIES[family].partitions[partition_name]
    files = Dataset.from_files(prefix_join(input_path, partition.file_glob))
    if partition.file_glob.endswith(".parquet"):
        rows = files.flat_map(load_parquet_batched)
    elif partition.skipped_jsonl_lines:
        rows = files.flat_map(lambda path: load_jsonl_with_skips(path, partition.skipped_jsonl_lines))
    else:
        rows = files.load_jsonl()

    pipeline = rows.flat_map(lambda row: row_to_chat_doc(row, family=family, partition_name=partition_name)).write_parquet(
        prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"), schema=SOURCE_CHAT_SCHEMA, skip_existing=True
    )
    ZephyrContext(name=f"nemotron-sft-chat-{family}-{partition_name}", resources=ResourceConfig(cpu=1, ram="16g")).execute(pipeline)


def _processed_chat_step(download: StepSpec, *, family: str, partition_name: str) -> StepSpec:
    return StepSpec(
        name=f"processed-chat/nemotron_sft_v3/{family}/{partition_name}",
        deps=[download],
        fn=lambda output_path: _transform_chat(download.output_path, output_path, family=family, partition_name=partition_name),
        hash_attrs={"family": family, "partition": partition_name, "version": TRANSFORM_VERSION},
    )


def _restored_chat_step(download: StepSpec) -> StepSpec:
    return StepSpec(
        name="restored-chat/nemotron_sft_v3/instruction_following_chat_v3",
        deps=[download],
        fn=lambda output_path: restore_chat_prompts(download.output_path, output_path),
        hash_attrs={"version": "2026.09.17", "seed_revisions": dict(SEED_DATASET_REVISIONS)},
    )


@cache
def nemotron_sft_v3_chat_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return a pinned download, chat transform, and normalization chain per partition."""
    steps: dict[str, tuple[StepSpec, ...]] = {}
    for family, repository in NEMOTRON_SFT_V3_REPOSITORIES.items():
        download = download_nemotron_sft_v3_step(family)
        for partition_name in repository.partitions:
            inputs = (download,)
            if family == "instruction_following_chat_v3" and partition_name == "chat":
                inputs = download, _restored_chat_step(download)
            processed = _processed_chat_step(inputs[-1], family=family, partition_name=partition_name)
            normalized = normalize_chat_step(
                name=f"normalized-chat/nemotron_sft_v3/{family}/{partition_name}",
                download=processed,
                output_schema=SOURCE_CHAT_SCHEMA,
            )
            steps[f"nemotron_sft_v3/{family}/{partition_name}"] = *inputs, processed, normalized
    return steps
