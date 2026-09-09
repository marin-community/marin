# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""open-thoughts/AgentTrove dataset download and transform.

Terminal agent rollouts (terminus-2 harness) distilled from a mix of teacher
models. Each row holds a multi-turn conversation solving a coding or CLI task.

Rollouts distilled from proprietary GPT, Claude, or Gemini teachers are dropped,
leaving the open-weight teachers. Provenance lives in three fields of differing
completeness (``original_teacher`` is populated on every row of the pinned
revision; ``model`` and ``model_provider`` are null for some streams), so a row
is dropped when any of them names an excluded family.

Transcripts carrying a task outcome are prefixed with a tag so the model can
condition on it. Only two streams record a verdict — r2egym rows hold a numeric
reward (``"1.0"``/``"0.0"``) and swesmith rows hold ``"success"``/``"timeout"``,
about 20k rows between them. Every other non-null ``result`` is a harness error
class (``AgentTimeoutError`` dominates at ~177k rows), meaning the episode ended
before anything verified it; those get a distinct tag rather than being labeled
a task failure. The ~78% of rows with no ``result`` are left unprefixed.

Each document also carries two provenance columns, both populated on every row of
the pinned revision: ``teacher`` (the model that generated the rollout, e.g.
``GLM-4.6``) and ``task_source`` (the upstream task set, e.g. ``r2egym``). Null
values normalize to the empty string so the columns stay non-nullable strings.
"""

import pyarrow as pa
from fray.types import ResourceConfig
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.opencode import INLINE_TOOL_CALL, opencode_protocol_messages, prompt_tool_definitions
from marin.datakit.download.rollout_transforms import (
    TRAJECTORY_FAILED_TAG,
    TRAJECTORY_SOLVED_TAG,
    TRAJECTORY_UNVERIFIED_TAG,
    checked_openai_chat_document,
    load_parquet_batched,
    merge_adjacent_user_messages,
    render_role_message,
    text_document,
)
from marin.datakit.download.terminus import TASK_DESCRIPTION_MARKER, terminus_protocol_messages
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

SOURCE_CHAT_SCHEMA = pa.schema(
    [
        *CHAT_SCHEMA,
        pa.field("teacher", pa.string()),
        pa.field("task_source", pa.string()),
        pa.field("result", pa.string()),
    ]
)

HF_DATASET_ID = "open-thoughts/AgentTrove"
HF_REVISION = "b395a43"

# Lowercased substrings identifying an excluded teacher family across the
# ``model``, ``model_provider``, and ``original_teacher`` fields.
EXCLUDED_TEACHER_MARKERS = frozenset({"gpt", "openai", "claude", "anthropic", "gemini"})

# GPT-OSS is OpenAI's open-weights release, so it is kept even though its
# provenance ("GPT-OSS-120B", "openai/gpt-oss-120b") matches two excluded
# markers. An allowed marker on any field keeps the row outright.
ALLOWED_TEACHER_MARKERS = frozenset({"gpt-oss"})

PROVENANCE_FIELDS = ("model", "model_provider", "original_teacher")

# ``result`` mixes conventions across streams: swesmith uses these words,
# r2egym a numeric reward, and every other value is an error class name.
RESULT_SOLVED_WORDS = frozenset({"success"})
RESULT_FAILED_WORDS = frozenset({"timeout"})


def result_to_tag(result: str | None) -> str | None:
    """Map a ``result`` value to its outcome tag, or None when the row records no outcome."""
    if not result:
        return None
    if result in RESULT_SOLVED_WORDS:
        return TRAJECTORY_SOLVED_TAG
    if result in RESULT_FAILED_WORDS:
        return TRAJECTORY_FAILED_TAG
    try:
        reward = float(result)
    except ValueError:
        return TRAJECTORY_UNVERIFIED_TAG
    return TRAJECTORY_SOLVED_TAG if reward >= 1.0 else TRAJECTORY_FAILED_TAG


def is_excluded_teacher(row: dict) -> bool:
    provenance = [(row.get(field) or "").lower() for field in PROVENANCE_FIELDS]
    if any(marker in value for value in provenance for marker in ALLOWED_TEACHER_MARKERS):
        return False
    return any(marker in value for value in provenance for marker in EXCLUDED_TEACHER_MARKERS)


def row_to_doc(row: dict) -> list[dict]:
    if is_excluded_teacher(row):
        counters.pipeline.update_counter("agenttrove/dropped_teacher", 1)
        return []

    conversations = row.get("conversations")
    if not conversations:
        counters.pipeline.update_counter("agenttrove/dropped", 1)
        return []

    tag = result_to_tag(row.get("result"))
    rendered = "\n\n".join(render_role_message(m) for m in conversations)
    text = f"{tag}\n\n{rendered}" if tag else rendered

    counters.pipeline.update_counter("agenttrove/kept", 1)
    return [
        {
            **text_document(text, HF_DATASET_ID),
            "teacher": row.get("original_teacher") or "",
            "task_source": row.get("original_source") or "",
        }
    ]


def row_to_chat_doc(row: dict) -> list[dict]:
    if is_excluded_teacher(row):
        return []
    conversations = row.get("conversations")
    if not conversations:
        return []
    if any(INLINE_TOOL_CALL.search(message.get("content") or "") for message in conversations):
        tools = prompt_tool_definitions("\n".join(message.get("content") or "" for message in conversations[:2]))
        if tools is None:
            counters.pipeline.update_counter("agenttrove/chat/missing_tool_definitions_filtered", 1)
            return []
        converted = opencode_protocol_messages(conversations, tools)
    else:
        first = conversations[0]
        content = first.get("content")
        if first.get("role") == "user" and isinstance(content, str) and TASK_DESCRIPTION_MARKER in content:
            conversations = [{**first, "content": content[content.index(TASK_DESCRIPTION_MARKER) :]}, *conversations[1:]]
        converted = terminus_protocol_messages(conversations)
    if converted is None:
        return []
    messages, metadata = converted
    merged_messages = merge_adjacent_user_messages(messages)
    if merged_count := len(messages) - len(merged_messages):
        counters.pipeline.update_counter("agenttrove/chat/adjacent_user_merged", merged_count)
    return checked_openai_chat_document(
        merged_messages,
        HF_DATASET_ID,
        counter_prefix="agenttrove/chat",
        teacher=row.get("original_teacher") or "",
        task_source=row.get("original_source") or "",
        result=row.get("result") or "",
        **metadata,
    )


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet_batched)
        .flat_map(row_to_doc)
        .reshard(64)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="agenttrove-transform", resources=ResourceConfig(cpu=1, ram="32g"))
    ctx.execute(pipeline)


def transform_chat(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet_batched)
        .flat_map(row_to_chat_doc)
        .reshard(64)
        .write_parquet(
            f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", schema=SOURCE_CHAT_SCHEMA, skip_existing=True
        )
    )
    ZephyrContext(name="agenttrove-chat-transform", resources=ResourceConfig(cpu=1, ram="32g")).execute(pipeline)


def download_agenttrove_step() -> StepSpec:
    """Download AgentTrove and transform the open-weight-teacher rollouts into documents."""
    dl = download_hf_step(
        "raw/agenttrove",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
    )

    return StepSpec(
        name="processed/agenttrove",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v1"},
    )


def agenttrove_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for agenttrove."""
    processed = download_agenttrove_step()
    return (
        processed,
        normalize_step(name="normalized/agenttrove", download=processed),
    )


def agenttrove_chat_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the structured-chat normalization chain for AgentTrove."""
    download = download_hf_step("raw/agenttrove", hf_dataset_id=HF_DATASET_ID, revision=HF_REVISION)
    processed = StepSpec(
        name="processed-chat/agenttrove",
        deps=[download],
        fn=lambda output_path: transform_chat(download.output_path, output_path),
        hash_attrs={"version": "2026.09.09.adjacent-users"},
    )
    return processed, normalize_chat_step(
        output_schema=SOURCE_CHAT_SCHEMA, name="normalized-chat/agenttrove", download=processed
    )
