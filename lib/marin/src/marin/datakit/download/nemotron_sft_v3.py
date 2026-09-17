# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron post-training SFT exports as structured Harmony chat sources."""

import json
from collections.abc import Iterator, Mapping
from functools import cache
from types import MappingProxyType

import msgspec
import pyarrow as pa
from fray.types import ResourceConfig
from openai_harmony import Message
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import open_file

from marin.datakit.chat_normalize import (
    CHAT_SCHEMA,
    normalize_chat_step,
    validate_chat_messages,
    validate_tool_definitions,
)
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.nemotron_chat_prompts import SEED_DATASET_REVISIONS, restore_chat_prompts
from marin.datakit.download.rollout_transforms import load_parquet_batched, openai_chat_document
from marin.execution.step_spec import StepSpec

TRANSFORM_VERSION = "2026.09.17.chat-v4"
MAX_CONSECUTIVE_IDENTICAL_LINES = 256
_SKIPPED_JSONL_LINES = MappingProxyType({("agentic_v2", "tool_calling"): frozenset({1095})})
_RESTORED_CHAT_FAMILY = "instruction_following_chat_v3"
_RESTORED_CHAT_PARTITION = "chat"
_SWE_V2_FAMILY = "swe_v2"
_SWE_V2_AGENTLESS_PARTITION = "agentless"
# The source response uses separate reasoning_content, so this wire-format request conflicts with Marin's template.
_SWE_V2_AGENTLESS_FORMAT_SUFFIX = (
    "\n\nOutput format requirement: Please put your reasoning tokens in a separate code block, starting with <think> "
    "and ending with </think>, and the solution tokens in a separate code block, starting with <solution> "
    "and ending with </solution>."
)

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

# Each entry holds the HF dataset ID, pinned revision, and partition file globs.
NEMOTRON_SFT_V3_REPOSITORIES: Mapping[str, tuple[str, str, Mapping[str, str]]] = MappingProxyType(
    {
        family: (hf_dataset_id, revision, MappingProxyType(partitions))
        for family, (hf_dataset_id, revision, partitions) in {
            "agentic_v1": (
                "nvidia/Nemotron-Agentic-v1",
                "650d590978ca35c8f1ecea2faf136e5fac421b62",
                {split: f"data/{split}.jsonl" for split in ("interactive_agent", "tool_calling")},
            ),
            "agentic_v2": (
                "nvidia/Nemotron-SFT-Agentic-v2",
                "49e79a3be5ab8cf7511a12958b95cfd6408cd8db",
                {
                    "interactive_agent": "data/interactive_agent.jsonl",
                    "search": "data/search.jsonl",
                    "tool_calling": "data/tool_calling.jsonl",
                },
            ),
            "opencode_v1": (
                "nvidia/Nemotron-SFT-OpenCode-v1",
                "556d5237acff203f3e1a0be49428634c3606cda2",
                {
                    split: f"{split}/data.jsonl"
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
            _SWE_V2_FAMILY: (
                "nvidia/Nemotron-SFT-SWE-v2",
                "bd151f3f2d89c4804dda0083d912bd9f6a0a9fb7",
                {_SWE_V2_AGENTLESS_PARTITION: "data/agentless.jsonl", "openhands_swe": "data/swe.jsonl"},
            ),
            "safety_v1": (
                "nvidia/Nemotron-SFT-Safety-v1",
                "913fd7c803a9378dab0ce4fef80297ce115781f6",
                {"train": "data/train.jsonl"},
            ),
            "competitive_programming_v2": (
                "nvidia/Nemotron-SFT-Competitive-Programming-v2",
                "778afc98a9e027e10b3cd78020c120e93e142ef2",
                {
                    "exercism": "data/exercism.jsonl",
                    "text_to_sql": "data/text_to_sql.jsonl",
                    "competitive_coding_cpp": "data/competitive_programming_cpp_*.jsonl",
                    "competitive_coding_python": "data/competitive_programming_python_*.jsonl",
                },
            ),
            "instruction_following_chat_v2": (
                "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2",
                "1a9454ed054b8544503ab8d8c0a519d141a44c5b",
                {split: f"data/{split}.jsonl" for split in ("reasoning_off", "reasoning_on")},
            ),
            "multilingual_v1": (
                "nvidia/Nemotron-SFT-Multilingual-v1",
                "22c86505762a7c595abee309d720084351c9f4ba",
                {
                    f"{domain}_{language}": (
                        f"data/super-v3_{domain}_{language}_translated"
                        f"{'_postedit' if domain == 'stem' else ''}_final.jsonl"
                    )
                    for domain in ("code", "math", "stem")
                    for language in _MULTILINGUAL_V1_LANGUAGES
                },
            ),
            "arc_agi_v1": (
                "nvidia/Nemotron-SFT-ARC-AGI-v1",
                "92837449e198007b76830b75508cc6946795cd11",
                {config: f"data/{config}/*.jsonl" for config in _ARC_AGI_CONFIGS},
            ),
            "cuda_v1": (
                "nvidia/Nemotron-SFT-CUDA-v1",
                "1a06167a6e1e90d928094184173898cbb9bf42de",
                {"train": "data/train.jsonl"},
            ),
            _RESTORED_CHAT_FAMILY: (
                "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3",
                "be3b3e04ef605ac9d3f8f35b9d5a632f4a3a3402",
                {
                    _RESTORED_CHAT_PARTITION: "data/chat.jsonl",
                    "instruction_following": "data/instruction_following.jsonl",
                },
            ),
            "math_v4": (
                "nvidia/Nemotron-SFT-Math-v4",
                "a94e56aeddcf6e75d28c8bd210f40fa62309288d",
                {"train": "data/train.jsonl"},
            ),
            "math_v2": (
                "nvidia/Nemotron-Math-v2",
                "8e793210e175b6406c752a870f585f62de98c0d3",
                {"high": "data/high_part*.parquet", "medium": "data/medium.parquet", "low": "data/low.parquet"},
            ),
            "math_proofs_v1": (
                "nvidia/Nemotron-Math-Proofs-v1",
                "97229c590831adfe96202f5cd071d444d535bf91",
                {"lean": "data/lean.jsonl"},
            ),
            "math_proofs_v2": (
                "nvidia/Nemotron-Math-Proofs-v2",
                "7665d7f1d006fd89aa852a9dab8060c60b63f814",
                {"train": "data/train.jsonl"},
            ),
            "multilingual_v2": (
                "nvidia/Nemotron-SFT-Multilingual-v2",
                "971a252224b75414b1b67c55dbe0446d8b6606a0",
                {
                    f"{domain}_{language}": (
                        f"ultra-v3_{domain}_{language}_translated{('_postedit' if domain == 'stem' else '')}_final.jsonl"
                    )
                    for domain in ("code", "math", "stem")
                    for language in _MULTILINGUAL_V2_LANGUAGES
                },
            ),
            "safety_v2": (
                "nvidia/Nemotron-SFT-Safety-v2",
                "8a40a63c9a1a340874b874f980953be53bff0a07",
                {"train": "data/train.jsonl"},
            ),
            "science_v2": (
                "nvidia/Nemotron-SFT-Science-v2",
                "6536a5021222a94126968e8c92f29ee47fc8a7df",
                {config: f"{config}.jsonl" for config in ("rqa", "so", "syn_mcq", "vendor")},
            ),
            "finance_v1": (
                "nvidia/Nemotron-SpecializedDomains-Finance-v1",
                "5a21b106168facb96ced11b883c2a9b4788ee939",
                {"train": "data/train.jsonl"},
            ),
            "swe_v1": (
                "nvidia/Nemotron-SWE-v1",
                "0fe17a965b297a9c943a59050a14c42d5f0083ce",
                {"r2e_gym": "data/r2e_gym.jsonl"},
            ),
            "math_v3": (
                "nvidia/Nemotron-SFT-Math-v3",
                "ff4439c1073c87e006ab7ee5f1e5e28c4790dab3",
                {"train": "data/train.jsonl"},
            ),
        }.items()
    }
)


SOURCE_CHAT_SCHEMA = pa.schema([*CHAT_SCHEMA, pa.field("source_train_turns", pa.list_(pa.bool_()))])


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _restore_backslash_b(value: object) -> object:
    """Undo source JSON's accidental backspace decoding in LaTeX text."""
    if isinstance(value, str):
        return value.replace("\b", r"\b")
    if isinstance(value, list):
        return [_restore_backslash_b(item) for item in value]
    if isinstance(value, dict):
        return {key: _restore_backslash_b(item) for key, item in value.items()}
    return value


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
        parsed = {key: _restore_backslash_b(value) for key, value in parsed.items()}
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
    hf_dataset_id, _, _ = NEMOTRON_SFT_V3_REPOSITORIES[family]
    counter = f"nemotron_sft_v3/{family}/{partition_name}/chat"
    try:
        messages = _parse_messages(row.get("messages"))
        if not messages:
            counters.pipeline.update_counter(f"{counter}/empty_messages_filtered", 1)
            return []
        if family == _SWE_V2_FAMILY and partition_name == _SWE_V2_AGENTLESS_PARTITION:
            for message in messages:
                content = message.get("content")
                if (
                    message.get("role") == "user"
                    and isinstance(content, str)
                    and content.endswith(_SWE_V2_AGENTLESS_FORMAT_SUFFIX)
                ):
                    message["content"] = content.removesuffix(_SWE_V2_AGENTLESS_FORMAT_SUFFIX)
                    counters.pipeline.update_counter(f"{counter}/format_suffix_removed", 1)
        if family == _RESTORED_CHAT_FAMILY and partition_name == _RESTORED_CHAT_PARTITION:
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
            hf_dataset_id,
            source_id=row.get("uuid") or None,
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
    hf_dataset_id, revision, partitions = NEMOTRON_SFT_V3_REPOSITORIES[family]
    return download_hf_step(
        f"raw/nemotron_sft_v3/{family}",
        hf_dataset_id=hf_dataset_id,
        revision=revision,
        hf_urls_glob=sorted(set(partitions.values())),
    )


def _transform_chat(input_path: str, output_path: str, *, family: str, partition_name: str) -> None:
    _, _, partitions = NEMOTRON_SFT_V3_REPOSITORIES[family]
    file_glob = partitions[partition_name]
    files = Dataset.from_files(prefix_join(input_path, file_glob))
    if file_glob.endswith(".parquet"):
        rows = files.flat_map(load_parquet_batched)
    elif skipped_lines := _SKIPPED_JSONL_LINES.get((family, partition_name)):
        rows = files.flat_map(lambda path: load_jsonl_with_skips(path, skipped_lines))
    else:
        rows = files.load_jsonl()

    pipeline = rows.flat_map(
        lambda row: row_to_chat_doc(row, family=family, partition_name=partition_name)
    ).write_parquet(
        prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"),
        schema=SOURCE_CHAT_SCHEMA,
        skip_existing=True,
    )
    ZephyrContext(
        name=f"nemotron-sft-chat-{family}-{partition_name}", resources=ResourceConfig(cpu=1, ram="16g")
    ).execute(pipeline)


def _processed_chat_step(download: StepSpec, *, family: str, partition_name: str) -> StepSpec:
    return StepSpec(
        name=f"processed-chat/nemotron_sft_v3/{family}/{partition_name}",
        deps=[download],
        fn=lambda output_path: _transform_chat(
            download.output_path, output_path, family=family, partition_name=partition_name
        ),
        hash_attrs={"family": family, "partition": partition_name, "version": TRANSFORM_VERSION},
    )


def _restored_chat_step(download: StepSpec) -> StepSpec:
    return StepSpec(
        name=f"restored-chat/nemotron_sft_v3/{_RESTORED_CHAT_FAMILY}",
        deps=[download],
        fn=lambda output_path: restore_chat_prompts(download.output_path, output_path),
        hash_attrs={"version": "2026.09.17", "seed_revisions": dict(SEED_DATASET_REVISIONS)},
    )


@cache
def nemotron_sft_v3_chat_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return a pinned download, chat transform, and normalization chain per partition."""
    steps: dict[str, tuple[StepSpec, ...]] = {}
    for family, (_, _, partitions) in NEMOTRON_SFT_V3_REPOSITORIES.items():
        download = download_nemotron_sft_v3_step(family)
        for partition_name in partitions:
            inputs = (download,)
            if family == _RESTORED_CHAT_FAMILY and partition_name == _RESTORED_CHAT_PARTITION:
                inputs = download, _restored_chat_step(download)
            processed = _processed_chat_step(inputs[-1], family=family, partition_name=partition_name)
            normalized = normalize_chat_step(
                name=f"normalized-chat/nemotron_sft_v3/{family}/{partition_name}",
                download=processed,
                output_schema=SOURCE_CHAT_SCHEMA,
            )
            steps[f"nemotron_sft_v3/{family}/{partition_name}"] = *inputs, processed, normalized
    return steps
