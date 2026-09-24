# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate unverified synthetic conversations for a curriculum capability."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

import pyarrow as pa
from marin.datakit.chat_normalize import CHAT_SCHEMA
from marin.datakit.download.rollout_transforms import openai_chat_document
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.inference.openai_batch import CHAT_COMPLETIONS_ENDPOINT, OpenAIBatchClient
from marin.inference.structured_output import StructuredTool
from pydantic import Field, ValidationError
from rigging.filesystem.storage_path import StoragePath
from zephyr.writers import write_parquet_file

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_BULK_TOKEN_ENV, GLM_MODEL, resolve_glm_base_url
from experiments.post_training.task_curriculum.catalog_artifact import TASK_CURRICULUM, TaskCurriculumCatalogArtifact
from experiments.post_training.task_curriculum.models import CapabilitySection, CurriculumCatalog, StrictModel

logger = logging.getLogger(__name__)

CHAT_FILENAME = "chat/part-00000-of-00001.parquet"
TASKS_FILENAME = "tasks/part-00000-of-00001.parquet"
RAW_RESPONSES_FILENAME = "raw-responses.jsonl"
MANIFEST_FILENAME = "manifest.json"
TASK_SCHEMA = pa.schema(
    [
        pa.field("request_id", pa.string(), nullable=False),
        pa.field("capability_id", pa.string(), nullable=False),
        pa.field("task", pa.string()),
        pa.field("accepted", pa.bool_(), nullable=False),
        pa.field("rejection_reason", pa.string()),
    ]
)


class ConversationTurn(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class GeneratedConversation(StrictModel):
    task: str = Field(min_length=1)
    continuation: list[ConversationTurn] = Field(min_length=1, max_length=5)


CONVERSATION_TOOL = StructuredTool(
    name="submit_conversation",
    description="Submit one self-contained task and its simulated conversation.",
    output_type=GeneratedConversation,
)


@dataclass(frozen=True)
class GenerateCurriculumSFTConfig:
    catalog_path: str
    output_path: str
    capability_id: str
    requested_examples: int
    accepted_examples: int
    seed: int
    max_completion_tokens: int
    relay_job: str


def capability_packet(catalog: CurriculumCatalog, capability_id: str) -> dict[str, Any]:
    """Select one trainable capability and its subject context from the pinned catalog."""
    for entry in catalog.curricula:
        for section in entry.curriculum.sections:
            if section.id == capability_id and isinstance(section, CapabilitySection):
                return {
                    "catalog_version": catalog.catalog_version,
                    "subject": entry.curriculum.subject_name,
                    "capability_id": section.id,
                    "name": section.name,
                    "outcome": section.outcome,
                    "includes": section.includes,
                    "excludes": section.excludes,
                    "sampling_facets": [facet.model_dump(mode="json") for facet in section.sampling_facets],
                }
    raise ValueError(f"unknown curriculum capability: {capability_id}")


def generation_prompt(packet: dict[str, Any]) -> str:
    """Request a task and answer without evaluation examples or fabricated tool use."""
    return (
        "Create one distinct, self-contained task that exercises the curriculum capability below. "
        "Invent all context needed to answer it; do not rely on live facts. "
        "Then simulate a text-only user-assistant conversation that solves the task. "
        "The task is the first user message. The continuation must begin with an assistant reply, "
        "alternate user and assistant turns, and end with a complete assistant answer. "
        "Do not invent tool calls, tool observations, external sources, or citations. "
        "Use the capability boundaries and vary the task across sampling facets. "
        "Do not copy curriculum probes or benchmark examples.\n"
        f"Capability: {json.dumps(packet, ensure_ascii=False, sort_keys=True)}"
    )


def batch_request(config: GenerateCurriculumSFTConfig, packet: dict[str, Any], index: int) -> dict[str, Any]:
    body = {
        "model": GLM_MODEL,
        "messages": [
            {"role": "system", "content": "Generate one fictional task and its complete conversation."},
            {"role": "user", "content": generation_prompt(packet)},
        ],
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "temperature": 0.8,
        "seed": config.seed + index,
        "max_tokens": config.max_completion_tokens,
    }
    body.update(CONVERSATION_TOOL.request_fields())
    return {
        "custom_id": f"conversation-{index:05d}",
        "method": "POST",
        "url": CHAT_COMPLETIONS_ENDPOINT,
        "body": body,
    }


def conversation_messages(conversation: GeneratedConversation) -> list[dict[str, str]]:
    """Build a chat record from a task and a structurally complete continuation."""
    task = conversation.task.strip()
    if not task or len(conversation.continuation) % 2 == 0:
        raise ValueError("conversation must have a task and end with an assistant turn")
    messages = [{"role": "user", "content": task}]
    for index, turn in enumerate(conversation.continuation):
        expected_role = "assistant" if index % 2 == 0 else "user"
        if turn.role != expected_role or not turn.content.strip():
            raise ValueError("conversation roles must alternate and contain text")
        messages.append({"role": turn.role, "content": turn.content.strip()})
    return messages


def parse_batch(
    raw_output: str, config: GenerateCurriculumSFTConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select structurally valid, distinct conversations; retain rejection accounting."""
    responses: dict[str, dict[str, Any]] = {}
    for line in raw_output.splitlines():
        if not line.strip():
            continue
        response = json.loads(line)
        request_id = response["custom_id"]
        if request_id in responses:
            raise ValueError(f"duplicate GLM request ID: {request_id}")
        responses[request_id] = response

    expected_ids = {f"conversation-{index:05d}" for index in range(config.requested_examples)}
    if set(responses) != expected_ids:
        raise ValueError(
            f"GLM batch request IDs differ: missing={sorted(expected_ids - set(responses))}, "
            f"unexpected={sorted(set(responses) - expected_ids)}"
        )

    task_records: list[dict[str, Any]] = []
    chat_documents: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for request_id in sorted(responses):
        response = responses[request_id]
        result = response.get("response") or {}
        task: str | None = None
        document: dict[str, Any] | None = None
        reason: str | None = None
        if response.get("error") or result.get("status_code") != 200:
            reason = "request_failed"
        elif result["body"]["choices"][0].get("finish_reason") == "length":
            reason = "truncated"
        else:
            try:
                conversation = CONVERSATION_TOOL.parse(result["body"])
                task = conversation.task.strip()
                messages = conversation_messages(conversation)
                document = openai_chat_document(messages, f"curriculum-sft/{config.capability_id}", source_id=request_id)
            except (UnicodeError, ValidationError, ValueError):
                reason = "invalid_conversation"

        normalized_task = " ".join(task.lower().split()) if task is not None else None
        if reason is None and normalized_task in seen_tasks:
            reason = "duplicate_task"
        if reason is None and len(chat_documents) >= config.accepted_examples:
            reason = "surplus"
        accepted = reason is None
        if accepted:
            assert normalized_task is not None and document is not None
            seen_tasks.add(normalized_task)
            chat_documents.append(document)
        task_records.append(
            {
                "request_id": request_id,
                "capability_id": config.capability_id,
                "task": task,
                "accepted": accepted,
                "rejection_reason": reason,
            }
        )

    if len(chat_documents) != config.accepted_examples:
        raise ValueError(f"only {len(chat_documents)} synthetic conversations; need {config.accepted_examples}")
    return task_records, chat_documents


def generate_conversations(config: GenerateCurriculumSFTConfig) -> Artifact:
    """Write task audit rows, Datakit chat Parquet, and exact GLM responses."""
    catalog = TaskCurriculumCatalogArtifact(path=config.catalog_path).read_catalog()
    packet = capability_packet(catalog, config.capability_id)
    client = OpenAIBatchClient(resolve_glm_base_url(config.relay_job), os.environ[GLM_BULK_TOKEN_ENV])
    requests = [batch_request(config, packet, index) for index in range(config.requested_examples)]
    submission = client.submit(requests, f"curriculum-sft-{config.capability_id}.jsonl")
    batch_output = client.output(client.wait(submission.batch_id, 5.0))
    if batch_output.errors:
        raise RuntimeError(f"GLM batch {submission.batch_id} returned errors")
    task_records, chat_documents = parse_batch(batch_output.output, config)

    output = StoragePath(config.output_path)
    output.mkdirs()
    tasks_path = output / TASKS_FILENAME
    chat_path = output / CHAT_FILENAME
    tasks_path.parent.mkdirs()
    chat_path.parent.mkdirs()
    write_parquet_file(task_records, str(tasks_path), schema=TASK_SCHEMA)
    write_parquet_file(chat_documents, str(chat_path), schema=CHAT_SCHEMA)
    (output / RAW_RESPONSES_FILENAME).write_text(batch_output.output)
    (output / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "batch_id": submission.batch_id,
                "catalog_version": catalog.catalog_version,
                "capability_id": config.capability_id,
                "generator": GLM_MODEL,
                "requested": len(task_records),
                "accepted": len(chat_documents),
                "verified": False,
                "task_data": TASKS_FILENAME,
                "chat_data": CHAT_FILENAME,
                "raw_responses": RAW_RESPONSES_FILENAME,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("generated %s synthetic conversations for %s", len(chat_documents), config.capability_id)
    return Artifact(path=config.output_path)


def generate_curriculum_sft(
    capability_id: str,
    *,
    version: str,
    requested_examples: int,
    accepted_examples: int,
    seed: int,
    max_completion_tokens: int,
) -> ArtifactStep[Artifact]:
    """Build one GLM task-plus-conversation step for a pinned curriculum capability."""

    def build_config(ctx: StepContext) -> GenerateCurriculumSFTConfig:
        return GenerateCurriculumSFTConfig(
            catalog_path=ctx.artifact_path(TASK_CURRICULUM),
            output_path=ctx.output_path,
            capability_id=capability_id,
            requested_examples=requested_examples,
            accepted_examples=accepted_examples,
            seed=seed,
            max_completion_tokens=max_completion_tokens,
            relay_job=DEFAULT_GLM_RELAY_JOB,
        )

    return ArtifactStep(
        name=user_owned_name(f"documents/curriculum-sft/{capability_id}/generated-chat"),
        version=version,
        artifact_type=Artifact,
        run=generate_conversations,
        build_config=build_config,
        deps=(TASK_CURRICULUM,),
    )
