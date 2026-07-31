# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron post-training v3 SFT dataset downloads and transforms."""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cache
from html import escape
from types import MappingProxyType

import msgspec
from fray.types import ResourceConfig
from rigging.filesystem import prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.readers import open_file

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import load_parquet_batched, text_document
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

TRANSFORM_VERSION = "v1"


@dataclass(frozen=True)
class NemotronSFTPartition:
    config: str
    split: str
    file_glob: str
    required_first_message_content_role: str | None = None
    skipped_jsonl_lines: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class NemotronSFTRepository:
    hf_dataset_id: str
    revision: str
    partitions: Mapping[str, NemotronSFTPartition]

    def __post_init__(self):
        object.__setattr__(self, "partitions", MappingProxyType(dict(self.partitions)))


_MULTILINGUAL_V1_LANGUAGES = ("de", "es", "fr", "it", "zh")
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
                split: NemotronSFTPartition(
                    config="default",
                    split=split,
                    file_glob=f"data/{split}.jsonl",
                )
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
        "competitive_programming_v1": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-Competitive-Programming-v1",
            revision="d6e7c6b404ed5db6e1104b41d0f80a0c7dad7bf8",
            partitions={
                "competitive_coding_cpp": NemotronSFTPartition(
                    config="default",
                    split="competitive_coding_cpp",
                    file_glob="data/competitive_coding_cpp.part_*.jsonl",
                ),
                "competitive_coding_python": NemotronSFTPartition(
                    config="default",
                    split="competitive_coding_python",
                    file_glob="data/competitive_coding_python.part_*.jsonl",
                ),
                "infinibyte": NemotronSFTPartition(
                    config="default",
                    split="infinibyte",
                    file_glob="data/infinibyte.part_*.jsonl",
                ),
            },
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
                "chat": NemotronSFTPartition(
                    config="default",
                    split="chat",
                    file_glob="data/chat.jsonl",
                    required_first_message_content_role="user",
                ),
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
                "high": NemotronSFTPartition(
                    config="default",
                    split="high",
                    file_glob="data/high_part*.parquet",
                ),
                "medium": NemotronSFTPartition(
                    config="default",
                    split="medium",
                    file_glob="data/medium.parquet",
                ),
                "low": NemotronSFTPartition(
                    config="default",
                    split="low",
                    file_glob="data/low.parquet",
                ),
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
                "r2e_gym": NemotronSFTPartition(
                    config="default",
                    split="r2e_gym",
                    file_glob="data/r2e_gym.jsonl",
                )
            },
        ),
        "swe_v3": NemotronSFTRepository(
            hf_dataset_id="nvidia/Nemotron-SFT-SWE-v3",
            revision="3f73de64c1fe928a8f538fe45ccc10c228cc4c6a",
            partitions={"train": NemotronSFTPartition(config="default", split="train", file_glob="data/*.parquet")},
        ),
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _render_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return _canonical_json(value)


def _tool_calls(value: object) -> list[object]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        raise ValueError(f"tool_calls must be a list, dict, or JSON string, got {type(value).__name__}")
    return value


def _render_message(message: dict) -> str:
    role = message.get("role") or "unknown"
    attrs = []
    for key in ("name", "tool_call_id"):
        value = message.get(key)
        if value is not None:
            attrs.append(f'{key}="{escape(str(value), quote=True)}"')
    attr_text = f" {' '.join(attrs)}" if attrs else ""
    parts = [f"<{role}{attr_text}>"]

    if "reasoning_content" in message:
        parts.extend(["<reasoning>", _render_value(message["reasoning_content"]), "</reasoning>"])
    if "content" in message:
        parts.extend(["<content>", _render_value(message["content"]), "</content>"])
    if message.get("function_call") is not None:
        parts.extend(["<function_call>", _canonical_json(message["function_call"]), "</function_call>"])
    for tool_call in _tool_calls(message.get("tool_calls")):
        parts.extend(["<tool_call>", _canonical_json(tool_call), "</tool_call>"])

    extra = {
        key: value
        for key, value in message.items()
        if key not in {"role", "name", "tool_call_id", "reasoning_content", "content", "function_call", "tool_calls"}
    }
    if extra:
        parts.extend(["<metadata>", _canonical_json(extra), "</metadata>"])
    parts.append(f"</{role}>")
    return "\n".join(parts)


def doc_to_text(row: dict) -> str:
    """Serialize one Nemotron SFT row without dropping structured chat fields."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"messages must be a list, got {type(messages).__name__}")

    metadata = {key: value for key, value in row.items() if key not in {"messages", "tools"}}
    parts = []
    if metadata:
        parts.append(f"<metadata>\n{_canonical_json(metadata)}\n</metadata>")

    tools = row.get("tools")
    if tools not in (None, "", []):
        parts.append(f"<tools>\n{_canonical_json(tools)}\n</tools>")

    parts.extend(_render_message(message) for message in messages)
    return "\n\n".join(parts)


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


def _source_identity(repository: NemotronSFTRepository, partition: NemotronSFTPartition) -> str:
    return f"hf://datasets/{repository.hf_dataset_id}@{repository.revision}/{partition.config}/{partition.split}"


def row_to_doc(row: dict, *, family: str, partition_name: str) -> list[dict]:
    """Convert a source row to a canonical document for one partition."""
    repository = NEMOTRON_SFT_V3_REPOSITORIES[family]
    partition = repository.partitions[partition_name]
    counter = f"nemotron_sft_v3/{family}/{partition_name}"
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        counters.pipeline.update_counter(f"{counter}/dropped", 1)
        return []
    required_role = partition.required_first_message_content_role
    if required_role is not None:
        first_message = next((message for message in messages if message.get("role") == required_role), None)
        if first_message is None or first_message.get("content") is None:
            counters.pipeline.update_counter(f"{counter}/dropped_missing_{required_role}_content", 1)
            return []

    text = doc_to_text(row)
    counters.pipeline.update_counter(f"{counter}/kept", 1)
    return [text_document(text, _source_identity(repository, partition))]


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


def _transform(input_path: str, output_path: str, *, family: str, partition_name: str) -> None:
    partition = NEMOTRON_SFT_V3_REPOSITORIES[family].partitions[partition_name]
    files = Dataset.from_files(prefix_join(input_path, partition.file_glob))
    if partition.file_glob.endswith(".parquet"):
        rows = files.flat_map(load_parquet_batched)
    elif partition.skipped_jsonl_lines:
        rows = files.flat_map(lambda path: load_jsonl_with_skips(path, partition.skipped_jsonl_lines))
    else:
        rows = files.load_jsonl()

    pipeline = rows.flat_map(lambda row: row_to_doc(row, family=family, partition_name=partition_name)).write_parquet(
        prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"), skip_existing=True
    )
    ctx = ZephyrContext(
        name=f"nemotron-sft-v3-{family}-{partition_name}",
        resources=ResourceConfig(cpu=1, ram="16g"),
    )
    ctx.execute(pipeline)


def _processed_step(download: StepSpec, *, family: str, partition_name: str) -> StepSpec:
    return StepSpec(
        name=f"processed/nemotron_sft_v3/{family}/{partition_name}",
        deps=[download],
        fn=lambda output_path: _transform(
            input_path=download.output_path,
            output_path=output_path,
            family=family,
            partition_name=partition_name,
        ),
        hash_attrs={
            "family": family,
            "partition": partition_name,
            "version": TRANSFORM_VERSION,
        },
    )


@cache
def nemotron_sft_v3_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return a pinned download, transform, and normalize chain for every partition."""
    steps: dict[str, tuple[StepSpec, ...]] = {}
    for family, repository in NEMOTRON_SFT_V3_REPOSITORIES.items():
        download = download_nemotron_sft_v3_step(family)
        for partition_name in repository.partitions:
            processed = _processed_step(download, family=family, partition_name=partition_name)
            name = f"nemotron_sft_v3/{family}/{partition_name}"
            normalized = normalize_step(
                name=f"normalized/{name}",
                download=processed,
            )
            steps[name] = (download, processed, normalized)
    return steps
