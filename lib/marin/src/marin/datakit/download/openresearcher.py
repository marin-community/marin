# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenResearcher/OpenResearcher-Dataset download and transform.

OpenResearcher contains browser-backed deep research trajectories. Each row is
one full conversation with nested OpenAI-style message blocks, browser tool
calls, browser observations, and a final answer. We render every trajectory as a
single text document while preserving seed, answer, and browser-use metadata.
"""

import hashlib
import html
import json
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Any

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import load_parquet_batched
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "OpenResearcher/OpenResearcher-Dataset"
HF_REVISION = "bb955d5"
HF_FULL_REVISION = "bb955d554fc4051a0fe88691b36972a235141941"
LICENSE = "mit"
TRANSFORM_VERSION = "v1"

SEED_CONFIGS = tuple(f"seed_{seed}" for seed in range(42, 58))
SEED_CONFIG_SET = frozenset(SEED_CONFIGS)

BROWSER_RECIPIENT_PREFIX = "browser."
EXACT_ANSWER_RE = re.compile(r"Exact Answer:\s*(.+?)(?:\n\s*\n|\nConfidence:|$)", re.DOTALL)
BOXED_ANSWER_RE = re.compile(r"^\\boxed\{(.+)\}$", re.DOTALL)


def stable_json(value: Any) -> str:
    """Return deterministic JSON for nested metadata fields."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def seed_from_path(path: str) -> str:
    """Extract the OpenResearcher seed config from a downloaded parquet path."""
    for part in path.split("/"):
        if part in SEED_CONFIG_SET:
            return part
    raise ValueError(f"Could not find seed config in OpenResearcher path: {path}")


def load_openresearcher_rows(path: str) -> Iterator[dict]:
    """Load raw parquet rows and attach the source seed inferred from the path."""
    source_seed = seed_from_path(path)
    for row in load_parquet_batched(path):
        yield {**row, "source_seed": source_seed}


def content_text(content: Any) -> str:
    """Extract text blocks from an OpenResearcher message content field."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _system_content_block(message: dict) -> dict:
    content = message.get("content")
    if not isinstance(content, list):
        return {}
    for block in content:
        if isinstance(block, dict) and block.get("type") == "system_content":
            return block
    return {}


def _tool_schema(messages: Sequence[dict]) -> dict:
    for message in messages:
        if message.get("role") != "system":
            continue
        block = _system_content_block(message)
        tools = block.get("tools")
        if isinstance(tools, dict) and tools:
            return tools
    return {}


def _tool_names_from_schema(tools: dict) -> list[str]:
    names = []
    for namespace, schema in tools.items():
        if not isinstance(schema, dict):
            continue
        tool_defs = schema.get("tools")
        if not isinstance(tool_defs, list):
            continue
        for tool in tool_defs:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name")
            if isinstance(name, str) and name:
                names.append(f"{namespace}.{name}")
    return sorted(names)


def compact_system_text(message: dict) -> str:
    """Render compact system metadata without inlining bulky browser schemas."""
    block = _system_content_block(message)
    if not block:
        return content_text(message.get("content")).strip()

    lines = []
    for key in ("model_identity", "knowledge_cutoff", "reasoning_effort", "conversation_start_date"):
        value = block.get(key)
        if isinstance(value, str) and value:
            lines.append(f"{key}: {value}")

    channel_config = block.get("channel_config")
    if isinstance(channel_config, dict):
        valid_channels = channel_config.get("valid_channels")
        if isinstance(valid_channels, list):
            lines.append(f"valid_channels: {', '.join(str(channel) for channel in valid_channels)}")
        if channel_config.get("channel_required") is not None:
            lines.append(f"channel_required: {channel_config['channel_required']}")

    tools = block.get("tools")
    if isinstance(tools, dict) and tools:
        tool_names = _tool_names_from_schema(tools)
        if tool_names:
            lines.append(f"available_tools: {', '.join(tool_names)}")

    return "\n".join(lines)


def _message_attrs(message: dict) -> str:
    attrs = []
    attr_values = {
        "channel": message.get("channel"),
        "to": message.get("recipient"),
        "name": message.get("name"),
        "content_type": message.get("content_type"),
    }
    for name, value in attr_values.items():
        if value is None or value == "":
            continue
        escaped = html.escape(str(value), quote=True)
        attrs.append(f'{name}="{escaped}"')
    if not attrs:
        return ""
    return " " + " ".join(attrs)


def render_message(message: dict) -> str:
    """Render one nested OpenResearcher message as a tagged transcript turn."""
    role = string_or_empty(message.get("role")) or "unknown"
    body = compact_system_text(message) if role == "system" else content_text(message.get("content")).strip()
    attrs = _message_attrs(message)
    return f"<{role}{attrs}>\n{body}\n</{role}>"


def _extract_final_answer_text(messages: Sequence[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("channel") == "final":
            return content_text(message.get("content"))
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return content_text(message.get("content"))
    return ""


def extract_exact_answer(final_answer_text: str) -> str:
    match = EXACT_ANSWER_RE.search(final_answer_text)
    if match is None:
        return ""
    return match.group(1).strip()


def normalize_answer_for_match(value: Any) -> str:
    text = string_or_empty(value).strip()
    boxed = BOXED_ANSWER_RE.match(text)
    if boxed is not None:
        text = boxed.group(1).strip()

    for quote in ('"', "'", "`"):
        if len(text) >= 2 and text.startswith(quote) and text.endswith(quote):
            text = text[1:-1].strip()

    text = text.strip("*").strip()
    return re.sub(r"\s+", " ", text).casefold()


def answer_match_status(gold_answer: Any, exact_answer: str) -> str:
    normalized_gold = normalize_answer_for_match(gold_answer)
    normalized_exact = normalize_answer_for_match(exact_answer)
    if not normalized_gold or not normalized_exact:
        return "unknown"
    if normalized_gold == normalized_exact:
        return "match"
    return "mismatch"


def _message_role_counts(messages: Sequence[dict]) -> Counter[str]:
    return Counter(string_or_empty(message.get("role")) or "unknown" for message in messages)


def _browser_call_counts(messages: Sequence[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for message in messages:
        recipient = string_or_empty(message.get("recipient"))
        if message.get("role") == "assistant" and recipient.startswith(BROWSER_RECIPIENT_PREFIX):
            counts[recipient] += 1
    return counts


def _trajectory_id(row: dict, messages: Sequence[dict]) -> str:
    identity_parts = {
        "source_seed": string_or_empty(row.get("source_seed")),
        "qid": string_or_empty(row.get("qid")),
        "chunk_idx": row.get("chunk_idx"),
        "attempt": row.get("attempt"),
        "messages": messages,
    }
    digest = hashlib.sha256(stable_json(identity_parts).encode("utf-8")).hexdigest()
    seed = string_or_empty(row.get("source_seed")) or "unknown_seed"
    qid = string_or_empty(row.get("qid")) or "unknown_qid"
    chunk_idx = string_or_empty(row.get("chunk_idx")) or "unknown_chunk"
    return f"{seed}:{qid}:chunk-{chunk_idx}:{digest}"


def row_to_doc(row: dict) -> list[dict]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        counters.increment("openresearcher/dropped/no_messages")
        return []
    if not all(isinstance(message, dict) for message in messages):
        counters.increment("openresearcher/dropped/malformed_message")
        return []

    question = string_or_empty(row.get("question")).strip()
    if not question:
        counters.increment("openresearcher/dropped/no_question")
        return []

    status = string_or_empty(row.get("status"))
    if status and status != "success":
        counters.increment("openresearcher/dropped/non_success_status")
        return []

    final_answer_text = _extract_final_answer_text(messages).strip()
    exact_answer = extract_exact_answer(final_answer_text)
    answer_match = answer_match_status(row.get("answer"), exact_answer)
    if answer_match == "mismatch":
        counters.increment("openresearcher/answer_mismatch")
    elif answer_match == "unknown":
        counters.increment("openresearcher/answer_match_unknown")

    role_counts = _message_role_counts(messages)
    browser_counts = _browser_call_counts(messages)
    source_seed = string_or_empty(row.get("source_seed"))
    rendered_messages = "\n\n".join(render_message(message) for message in messages)
    metadata_header = (
        "<openresearcher_metadata>\n"
        f"source_seed: {source_seed}\n"
        f"qid: {string_or_empty(row.get('qid'))}\n"
        f"question: {question}\n"
        f"gold_answer: {string_or_empty(row.get('answer'))}\n"
        f"status: {status}\n"
        f"answer_match: {answer_match}\n"
        "</openresearcher_metadata>"
    )
    text = f"{metadata_header}\n\n{rendered_messages}".strip()
    if not text:
        counters.increment("openresearcher/dropped/empty_text")
        return []

    counters.increment("openresearcher/kept")
    return [
        {
            "id": _trajectory_id(row, messages),
            "text": text,
            "source": HF_DATASET_ID,
            "source_revision": HF_FULL_REVISION,
            "license": LICENSE,
            "source_seed": source_seed,
            "qid": string_or_empty(row.get("qid")),
            "chunk_idx": row.get("chunk_idx"),
            "attempt": row.get("attempt"),
            "question": question,
            "answer": string_or_empty(row.get("answer")),
            "exact_answer": exact_answer,
            "answer_match": answer_match,
            "final_answer_text": final_answer_text,
            "messages_json": stable_json(messages),
            "tool_schema_json": stable_json(_tool_schema(messages)),
            "message_count": len(messages),
            "system_message_count": role_counts["system"],
            "developer_message_count": role_counts["developer"],
            "user_message_count": role_counts["user"],
            "assistant_message_count": role_counts["assistant"],
            "tool_message_count": role_counts["tool"],
            "browser_search_count": browser_counts["browser.search"],
            "browser_open_count": browser_counts["browser.open"],
            "browser_find_count": browser_counts["browser.find"],
            "row_hash": hashlib.sha256(stable_json(row).encode("utf-8")).hexdigest(),
        }
    ]


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/seed_*/train-*.parquet")
        .flat_map(load_openresearcher_rows)
        .flat_map(row_to_doc)
        .reshard(64)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="openresearcher-transform", resources=ResourceConfig(cpu=1, ram="32g", disk="10g"))
    ctx.execute(pipeline)


def _step_name(base_name: str, step_suffix: str) -> str:
    if not step_suffix:
        return base_name
    return f"{base_name}-{step_suffix}"


def _validated_seed_configs(seed_configs: Sequence[str]) -> tuple[str, ...]:
    seed_configs = tuple(seed_configs)
    if not seed_configs:
        raise ValueError("seed_configs must contain at least one OpenResearcher seed config")

    unknown_seeds = sorted(set(seed_configs) - SEED_CONFIG_SET)
    if unknown_seeds:
        valid_range = f"{SEED_CONFIGS[0]} through {SEED_CONFIGS[-1]}"
        raise ValueError(f"Unknown OpenResearcher seed config(s): {unknown_seeds}. Expected {valid_range}.")

    return seed_configs


def download_openresearcher_step(
    seed_configs: Sequence[str] = SEED_CONFIGS,
    *,
    step_suffix: str = "",
) -> StepSpec:
    """Download and transform OpenResearcher train trajectories into documents."""
    seed_configs = _validated_seed_configs(seed_configs)

    hf_urls_glob = [f"{seed}/train-*.parquet" for seed in seed_configs]
    dl = download_hf_step(
        _step_name("raw/openresearcher-dataset", step_suffix),
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=hf_urls_glob,
        zephyr_max_parallelism=8,
        worker_resources=ResourceConfig(cpu=1, ram="32g", disk="10g"),
    )

    return StepSpec(
        name=_step_name("processed/openresearcher-dataset", step_suffix),
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": TRANSFORM_VERSION, "seed_configs": list(seed_configs)},
    )
