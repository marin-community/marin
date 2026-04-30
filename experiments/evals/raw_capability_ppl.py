# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curated raw capability PPL slices for chat, agent traces, and reasoning QA."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from functools import lru_cache
from typing import Any

from marin.datakit.ingestion_manifest import (
    IdentityTreatment,
    IngestionPolicy,
    IngestionSourceManifest,
    MaterializedOutputMetadata,
    SampleCapConfig,
    SecretRedaction,
    StagingMetadata,
    UsagePolicy,
    write_ingestion_metadata_json,
)
from marin.evaluation.perplexity_gap import raw_text_dataset
from marin.execution.executor import ExecutorStep, this_output_path
from marin.utils import fsspec_mkdirs, fsspec_size, load_dataset_with_backoff
from rigging.filesystem import open_url

RENDERING_VERSION = "v4"
LONG_TAIL_PPL_EPIC_ISSUE = 5005
RAW_CAPABILITY_ISSUE = 4963
DEFAULT_OUTPUT_FILENAME = "data-00000-of-00001.jsonl.gz"

WILDCHAT_MAX_ROWS = 8_192
OPENHANDS_MAX_ROWS = 4_096

WILDCHAT_REVISION = "f66566ceaaeb619dd98ffb0f3bf3ce1f86775ac4"
OPENHANDS_REVISION = "35455389ab51bf5e2306bfd436ef72d0f98bf882"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
GLOBAL_MGSM_REVISION = "29087245a1a9788d4e0413f4927e994d49633bbb"

DEFAULT_CAPABILITY_TAGS = {
    "chat/wildchat": ("chat", "dialogue", "multi_turn"),
    "agent_traces/openhands_swe_rebench": ("agent_traces", "code", "tool_use"),
    "reasoning_qa/gsm8k_main": ("reasoning_qa", "math", "english"),
    "reasoning_qa/global_mgsm_en": ("reasoning_qa", "math", "multilingual"),
}


class CapabilityEvalRenderer(StrEnum):
    WILDCHAT = "wildchat"
    OPENHANDS = "openhands"
    GSM8K = "gsm8k"
    GLOBAL_MGSM = "global_mgsm"


@dataclass(frozen=True)
class CapabilityEvalDatasetConfig:
    dataset_id: str
    revision: str
    split: str
    renderer: CapabilityEvalRenderer
    dataset_name: str | None = None
    max_rows: int | None = None
    source_manifest: IngestionSourceManifest | None = None
    output_path: str = field(default_factory=this_output_path)  # type: ignore[arg-type]
    rendering_version: str = RENDERING_VERSION


def render_capability_eval_dataset(config: CapabilityEvalDatasetConfig) -> None:
    """Render a curated structured HF dataset slice into raw-text rows."""
    dataset = load_dataset_with_backoff(
        path=config.dataset_id,
        name=config.dataset_name,
        split=config.split,
        revision=config.revision,
        streaming=True,
        context=f"{config.dataset_id}:{config.dataset_name or 'default'}:{config.split}",
    )

    fsspec_mkdirs(config.output_path)
    output_file = os.path.join(config.output_path, DEFAULT_OUTPUT_FILENAME)

    emitted = 0
    with open_url(output_file, "wt", compression="gzip") as raw_text_out:
        for row in dataset:
            chat_row = _render_capability_chat_row(config, row)
            if chat_row is None:
                continue

            raw_text_row = _project_chat_row_to_raw_text(config, chat_row)
            if raw_text_row is None:
                continue

            raw_text_out.write(json.dumps(raw_text_row, sort_keys=True) + "\n")
            emitted += 1
            if config.max_rows is not None and emitted >= config.max_rows:
                break

    if config.source_manifest is not None:
        write_ingestion_metadata_json(
            manifest=config.source_manifest,
            materialized_output=MaterializedOutputMetadata(
                input_path=_input_path(config),
                output_path=config.output_path,
                output_file=output_file,
                record_count=emitted,
                bytes_written=fsspec_size(output_file),
                metadata={"renderer": config.renderer.value},
            ),
        )


def capability_raw_validation_sets() -> dict[str, Any]:
    """Curated default raw eval slices for missing capability families."""
    steps = capability_raw_steps()
    return {
        "chat/wildchat": raw_text_dataset(
            steps["chat/wildchat"].cd("data-*.jsonl.gz"),
            tags=DEFAULT_CAPABILITY_TAGS["chat/wildchat"],
        ),
        "agent_traces/openhands_swe_rebench": raw_text_dataset(
            steps["agent_traces/openhands_swe_rebench"].cd("data-*.jsonl.gz"),
            tags=DEFAULT_CAPABILITY_TAGS["agent_traces/openhands_swe_rebench"],
        ),
        "reasoning_qa/gsm8k_main": raw_text_dataset(
            steps["reasoning_qa/gsm8k_main"].cd("data-*.jsonl.gz"),
            tags=DEFAULT_CAPABILITY_TAGS["reasoning_qa/gsm8k_main"],
        ),
        "reasoning_qa/global_mgsm_en": raw_text_dataset(
            steps["reasoning_qa/global_mgsm_en"].cd("data-*.jsonl.gz"),
            tags=DEFAULT_CAPABILITY_TAGS["reasoning_qa/global_mgsm_en"],
        ),
    }


@lru_cache
def capability_raw_steps() -> dict[str, ExecutorStep]:
    return {
        "chat/wildchat": _rendered_dataset_step(
            "chat/wildchat",
            dataset_id="allenai/WildChat",
            revision=WILDCHAT_REVISION,
            split="train",
            renderer=CapabilityEvalRenderer.WILDCHAT,
            max_rows=WILDCHAT_MAX_ROWS,
            source_manifest=CAPABILITY_SOURCE_MANIFESTS["chat/wildchat"],
        ),
        "agent_traces/openhands_swe_rebench": _rendered_dataset_step(
            "agent_traces/openhands_swe_rebench",
            dataset_id="nebius/SWE-rebench-openhands-trajectories",
            revision=OPENHANDS_REVISION,
            split="train",
            renderer=CapabilityEvalRenderer.OPENHANDS,
            max_rows=OPENHANDS_MAX_ROWS,
            source_manifest=CAPABILITY_SOURCE_MANIFESTS["agent_traces/openhands_swe_rebench"],
        ),
        "reasoning_qa/gsm8k_main": _rendered_dataset_step(
            "reasoning_qa/gsm8k_main",
            dataset_id="openai/gsm8k",
            dataset_name="main",
            revision=GSM8K_REVISION,
            split="train",
            renderer=CapabilityEvalRenderer.GSM8K,
            source_manifest=CAPABILITY_SOURCE_MANIFESTS["reasoning_qa/gsm8k_main"],
        ),
        "reasoning_qa/global_mgsm_en": _rendered_dataset_step(
            "reasoning_qa/global_mgsm_en",
            dataset_id="CohereLabs/global-mgsm",
            dataset_name="en",
            revision=GLOBAL_MGSM_REVISION,
            split="test",
            renderer=CapabilityEvalRenderer.GLOBAL_MGSM,
            source_manifest=CAPABILITY_SOURCE_MANIFESTS["reasoning_qa/global_mgsm_en"],
        ),
    }


def _eval_only_policy(provenance_notes: str) -> IngestionPolicy:
    return IngestionPolicy(
        usage_policy=UsagePolicy.EVAL_ONLY,
        use_policy="Eval-only capability probe. Do not mix these held-out slices into training.",
        requires_sanitization=False,
        identity_treatment=IdentityTreatment.PRESERVE,
        secret_redaction=SecretRedaction.NONE,
        contamination_risk="high: direct eval contamination if this slice is reused in training",
        provenance_notes=provenance_notes,
    )


CAPABILITY_SOURCE_MANIFESTS: dict[str, IngestionSourceManifest] = {
    "chat/wildchat": IngestionSourceManifest(
        dataset_key="allenai/WildChat",
        slice_key="capability/chat/wildchat/train",
        source_label="wildchat:train",
        source_urls=("https://huggingface.co/datasets/allenai/WildChat",),
        source_license="ODC-By 1.0",
        source_format="huggingface_parquet_chat_transcripts",
        surface_form="multi_turn_user_assistant_chat",
        policy=_eval_only_policy("Pinned Hugging Face revision for held-out chat PPL evaluation."),
        staging=StagingMetadata(
            transform_name="render_capability_eval_dataset",
            serializer_name=CapabilityEvalRenderer.WILDCHAT.value,
            split="train",
            metadata={
                "output_filename": DEFAULT_OUTPUT_FILENAME,
                "provenance_fields": ["id", "source"],
            },
        ),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(RAW_CAPABILITY_ISSUE,),
        sample_caps=SampleCapConfig(max_examples=WILDCHAT_MAX_ROWS),
        source_metadata={"hf_revision": WILDCHAT_REVISION},
    ),
    "agent_traces/openhands_swe_rebench": IngestionSourceManifest(
        dataset_key="nebius/SWE-rebench-openhands-trajectories",
        slice_key="capability/agent_traces/openhands_swe_rebench/train",
        source_label="openhands_swe_rebench:train",
        source_urls=("https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories",),
        source_license="CC BY 4.0",
        source_format="huggingface_parquet_agent_trajectories",
        surface_form="assistant_tool_observation_patch_trace",
        policy=_eval_only_policy("Pinned Hugging Face revision for held-out agent-trace PPL evaluation."),
        staging=StagingMetadata(
            transform_name="render_capability_eval_dataset",
            serializer_name=CapabilityEvalRenderer.OPENHANDS.value,
            split="train",
            metadata={
                "output_filename": DEFAULT_OUTPUT_FILENAME,
                "provenance_fields": ["id", "source"],
            },
        ),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(RAW_CAPABILITY_ISSUE,),
        sample_caps=SampleCapConfig(max_examples=OPENHANDS_MAX_ROWS),
        source_metadata={"hf_revision": OPENHANDS_REVISION},
    ),
    "reasoning_qa/gsm8k_main": IngestionSourceManifest(
        dataset_key="openai/gsm8k",
        slice_key="capability/reasoning_qa/gsm8k_main/train",
        source_label="gsm8k:main_train",
        source_urls=("https://huggingface.co/datasets/openai/gsm8k",),
        source_license="MIT",
        source_format="huggingface_parquet_qa",
        surface_form="math_word_problem_question_answer",
        policy=_eval_only_policy("Pinned Hugging Face revision for held-out numeracy PPL evaluation."),
        staging=StagingMetadata(
            transform_name="render_capability_eval_dataset",
            serializer_name=CapabilityEvalRenderer.GSM8K.value,
            split="train",
            subset="main",
            metadata={
                "output_filename": DEFAULT_OUTPUT_FILENAME,
                "provenance_fields": ["id", "source"],
            },
        ),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(RAW_CAPABILITY_ISSUE,),
        source_metadata={"hf_revision": GSM8K_REVISION},
    ),
    "reasoning_qa/global_mgsm_en": IngestionSourceManifest(
        dataset_key="CohereLabs/global-mgsm",
        slice_key="capability/reasoning_qa/global_mgsm_en/test",
        source_label="global_mgsm:en_test",
        source_urls=("https://huggingface.co/datasets/CohereLabs/global-mgsm",),
        source_license="Mixed upstream public licenses; see original per-language sources",
        source_format="huggingface_parquet_qa",
        surface_form="multilingual_math_question_answer",
        policy=_eval_only_policy("Pinned Hugging Face revision for held-out multilingual numeracy PPL evaluation."),
        staging=StagingMetadata(
            transform_name="render_capability_eval_dataset",
            serializer_name=CapabilityEvalRenderer.GLOBAL_MGSM.value,
            split="test",
            subset="en",
            metadata={
                "output_filename": DEFAULT_OUTPUT_FILENAME,
                "provenance_fields": ["id", "source"],
            },
        ),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(RAW_CAPABILITY_ISSUE,),
        source_metadata={"hf_revision": GLOBAL_MGSM_REVISION},
    ),
}


def _rendered_dataset_step(
    name: str,
    *,
    dataset_id: str,
    revision: str,
    split: str,
    renderer: CapabilityEvalRenderer,
    dataset_name: str | None = None,
    max_rows: int | None = None,
    source_manifest: IngestionSourceManifest | None = None,
) -> ExecutorStep:
    config = CapabilityEvalDatasetConfig(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        revision=revision,
        split=split,
        renderer=renderer,
        max_rows=max_rows,
        source_manifest=source_manifest,
    )
    slug = _directory_safe_name(name)
    suffix = _config_suffix(config)
    return ExecutorStep(
        name=f"raw/capability_ppl/{name}",
        fn=render_capability_eval_dataset,
        config=config,
        override_output_path=f"raw/capability_ppl/{slug}-{suffix}",
    )


def _render_capability_chat_row(config: CapabilityEvalDatasetConfig, row: Mapping[str, Any]) -> dict[str, Any] | None:
    source_name = config.dataset_id if config.dataset_name is None else f"{config.dataset_id}/{config.dataset_name}"

    if config.renderer == CapabilityEvalRenderer.WILDCHAT:
        return _chat_like_row(
            row,
            source=source_name,
            record_id=_require_string(row, "conversation_id"),
            messages=row["conversation"],
            metadata_fields=("language", "model", "redacted", "timestamp", "toxic", "turn"),
        )
    if config.renderer == CapabilityEvalRenderer.GSM8K:
        question = _require_string(row, "question")
        answer = _require_string(row, "answer")
        return _openai_chat_row(
            source=source_name,
            record_id=_stable_hash(question, answer),
            messages=(
                {"role": "user", "content": f"Question:\n{question}"},
                {"role": "assistant", "content": f"Answer:\n{answer}"},
            ),
            metadata={"question": question, "answer": answer},
        )
    if config.renderer == CapabilityEvalRenderer.GLOBAL_MGSM:
        instruction = _require_string(row, "instruction")
        question = _require_string(row, "question")
        answer_prefix = _require_string(row, "answer_prefix")
        answer = _require_string(row, "answer")
        return _openai_chat_row(
            source=source_name,
            record_id=_stable_hash(instruction, question, answer_prefix, answer),
            messages=(
                {"role": "user", "content": f"Instruction:\n{instruction}\n\nQuestion:\n{question}"},
                {"role": "assistant", "content": f"Answer Prefix:\n{answer_prefix}\n\nAnswer:\n{answer}"},
            ),
            metadata={
                "answer": answer,
                "answer_prefix": answer_prefix,
                "instruction": instruction,
                "question": question,
            },
        )
    if config.renderer == CapabilityEvalRenderer.OPENHANDS:
        return _openhands_chat_row(row, source=source_name)

    raise ValueError(f"Unsupported renderer: {config.renderer}")


def _project_chat_row_to_raw_text(
    config: CapabilityEvalDatasetConfig,
    chat_row: Mapping[str, Any],
) -> dict[str, Any] | None:
    if config.renderer == CapabilityEvalRenderer.OPENHANDS:
        text = _project_openhands_targets(chat_row)
    else:
        messages = _require_messages(chat_row)
        text = _render_messages(messages)

    if not text:
        return None

    return {
        "id": chat_row["id"],
        "source": chat_row["source"],
        "text": text,
        "metadata": chat_row.get("metadata", {}),
    }


def _chat_like_row(
    row: Mapping[str, Any],
    *,
    source: str,
    record_id: str,
    messages: Iterable[Mapping[str, Any]],
    metadata_fields: tuple[str, ...],
) -> dict[str, Any] | None:
    normalized_messages = _normalize_messages(messages)
    if not normalized_messages:
        return None

    return _openai_chat_row(
        source=source,
        record_id=record_id,
        messages=normalized_messages,
        metadata={field: row[field] for field in metadata_fields if field in row},
    )


def _openhands_chat_row(row: Mapping[str, Any], *, source: str) -> dict[str, Any] | None:
    messages = _normalize_messages(row["trajectory"])
    if not messages:
        return None

    tools = row.get("tools")
    return _openai_chat_row(
        source=source,
        record_id=_require_string(row, "trajectory_id"),
        messages=messages,
        metadata={
            "exit_status": row.get("exit_status"),
            "gen_tests_correct": row.get("gen_tests_correct"),
            "instance_id": row.get("instance_id"),
            "model_patch": row.get("model_patch"),
            "pred_passes_gen_tests": row.get("pred_passes_gen_tests"),
            "repo": row.get("repo"),
            "resolved": row.get("resolved"),
            "tools": tools,
            "trajectory_id": row.get("trajectory_id"),
        },
        chat_template_kwargs={"tools": tools} if tools else None,
    )


def _openai_chat_row(
    *,
    source: str,
    record_id: str,
    messages: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(tz=timezone.utc).isoformat()
    row = {
        "id": record_id,
        "source": source,
        "messages": list(messages),
        "added": now,
        "created": "",
        "metadata": dict(metadata),
    }
    if chat_template_kwargs:
        row["chat_template_kwargs"] = dict(chat_template_kwargs)
    return row


def _normalize_messages(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        normalized_message: dict[str, Any] = {"role": _canonical_role_key(message.get("role"))}

        content = message.get("content")
        if isinstance(content, str):
            normalized_message["content"] = _normalize_text(content)
        elif content is None:
            normalized_message["content"] = ""
        else:
            normalized_message["content"] = content

        name = message.get("name")
        if isinstance(name, str) and name:
            normalized_message["name"] = name

        tool_call_id = message.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id:
            normalized_message["tool_call_id"] = tool_call_id

        tool_calls = message.get("tool_calls")
        if tool_calls:
            normalized_message["tool_calls"] = _normalize_tool_calls(tool_calls)

        if normalized_message.get("content") or normalized_message.get("tool_calls"):
            normalized.append(normalized_message)

    return normalized


def _normalize_tool_calls(tool_calls: Any) -> Any:
    if isinstance(tool_calls, list):
        return [_normalize_tool_calls(tool_call) for tool_call in tool_calls]
    if not isinstance(tool_calls, dict):
        return tool_calls

    normalized = dict(tool_calls)
    function = normalized.get("function")
    if isinstance(function, dict):
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                normalized["function"] = {**function, "arguments": json.loads(arguments)}
            except json.JSONDecodeError:
                normalized["function"] = {**function, "arguments": _normalize_text(arguments)}
    return normalized


def _project_openhands_targets(chat_row: Mapping[str, Any]) -> str:
    messages = _require_messages(chat_row)
    transcript = _render_messages(
        messages,
        include_roles=frozenset({"assistant"}),
        strip_tool_call_ids=True,
    )

    sections = []
    if transcript:
        sections.append(transcript)

    metadata = chat_row.get("metadata", {})
    if isinstance(metadata, Mapping):
        raw_patch = metadata.get("model_patch")
        model_patch = _normalize_text(raw_patch) if isinstance(raw_patch, str) else ""
        if model_patch:
            sections.append("Final Patch:\n" + model_patch)

    return "\n\n".join(sections)


def _render_messages(
    messages: Iterable[Mapping[str, Any]],
    *,
    include_roles: frozenset[str] | None = None,
    strip_tool_call_ids: bool = False,
) -> str:
    blocks = []
    for message in messages:
        role_key = _canonical_role_key(message.get("role"))
        if include_roles is not None and role_key not in include_roles:
            continue

        role = _render_role(role_key)
        header = role
        name = message.get("name")
        if isinstance(name, str) and name:
            header += f" [{name}]"
        tool_call_id = message.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id:
            header += f" (tool_call_id={tool_call_id})"

        parts = [f"{header}:"]
        content = message.get("content")
        content_text = _normalize_text(content) if isinstance(content, str) else ""
        if content_text:
            parts.append(content_text)

        tool_calls = message.get("tool_calls")
        if tool_calls:
            parts.append("Tool Calls:")
            if strip_tool_call_ids:
                tool_calls = _drop_tool_call_ids(tool_calls)
            parts.append(_canonical_json(tool_calls))

        block = "\n".join(part for part in parts if part).strip()
        if block:
            blocks.append(block)

    return "\n\n".join(blocks)


def _require_messages(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Expected OAI chat row to contain a list-valued 'messages' field.")
    return messages


def _render_role(raw_role: Any) -> str:
    role = _canonical_role_key(raw_role)
    if role == "assistant":
        return "Assistant"
    if role == "system":
        return "System"
    if role == "tool":
        return "Tool"
    return "User" if role == "user" else role.replace("_", " ").title()


def _canonical_role_key(raw_role: Any) -> str:
    return str(raw_role or "unknown").strip().lower()


def _drop_tool_call_ids(value: Any) -> Any:
    if isinstance(value, list):
        return [_drop_tool_call_ids(item) for item in value]
    if isinstance(value, dict):
        stripped = {key: val for key, val in value.items() if key not in {"id", "tool_call_id", "type"}}
        return {key: _strip_tool_call_metadata_value(val) for key, val in stripped.items()}
    return _strip_tool_call_metadata_value(value)


def _strip_tool_call_metadata_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_tool_call_metadata_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_strip_tool_call_metadata_value(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_normalize_json_like(value), ensure_ascii=False, indent=2, sort_keys=True)


def _normalize_json_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_json_like(val) for key, val in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize_json_like(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return _normalize_text(value)
        return _normalize_json_like(parsed)
    return value


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _require_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Expected {key!r} to be a string, got {type(value).__name__}.")
    return value


def _stable_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _config_suffix(config: CapabilityEvalDatasetConfig) -> str:
    config_dict = asdict(config)
    config_dict["renderer"] = config.renderer.value
    if config.source_manifest is not None:
        config_dict["source_manifest"] = config.source_manifest.to_dict()
    config_dict.pop("output_path", None)
    payload = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:6]


def _directory_safe_name(name: str) -> str:
    return name.replace("/", "--").replace(".", "-").replace("#", "-")


def _input_path(config: CapabilityEvalDatasetConfig) -> str:
    if config.dataset_name is None:
        return f"hf://datasets/{config.dataset_id}/{config.revision}/{config.split}"
    return f"hf://datasets/{config.dataset_id}/{config.dataset_name}/{config.revision}/{config.split}"
