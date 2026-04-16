# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from rigging.filesystem import open_url

from marin.rl.integrations.openreward.client import load_openreward_client
from marin.rl.integrations.openreward.models import (
    OpenRewardPromptBlock,
    OpenRewardPromptBlockType,
    OpenRewardTaskManifest,
    OpenRewardTaskManifestEntry,
    OpenRewardToolSpec,
    SecretsMapping,
)


class OpenRewardTaskLike(Protocol):
    """Minimal task interface used by the manifest builder."""

    environment_name: str
    task_spec: Mapping[str, Any]


class OpenRewardToolLike(Protocol):
    """Minimal tool interface used by the manifest builder."""

    name: str
    description: str
    input_schema: Mapping[str, Any] | None


class OpenRewardTextBlockLike(Protocol):
    """Minimal text block interface returned by OpenReward sessions."""

    type: str
    text: str
    detail: Mapping[str, Any] | None


class OpenRewardImageBlockLike(Protocol):
    """Minimal image block interface returned by OpenReward sessions."""

    type: str
    data: str
    mimeType: str
    detail: Mapping[str, Any] | None


class OpenRewardSessionLike(Protocol):
    """Minimal synchronous session interface used by the manifest builder."""

    def __enter__(self) -> "OpenRewardSessionLike": ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None: ...

    def get_prompt(self) -> Sequence[OpenRewardTextBlockLike | OpenRewardImageBlockLike]: ...

    def list_tools(self, provider_format: None = None) -> Sequence[OpenRewardToolLike]: ...


class OpenRewardEnvironmentLike(Protocol):
    """Minimal synchronous environment interface used by the manifest builder."""

    name: str
    deployment_name: str

    def num_tasks(self, split: str) -> int: ...

    def get_task(self, split: str, index: int) -> OpenRewardTaskLike: ...

    def session(
        self,
        task: OpenRewardTaskLike | None = None,
        secrets: SecretsMapping | None = None,
        *,
        split: str | None = None,
        index: int | None = None,
    ) -> OpenRewardSessionLike: ...


def _clone_json(value: Any, field_name: str) -> Any:
    """Normalize JSON-like values into plain Python containers."""

    try:
        return json.loads(json.dumps(value))
    except TypeError as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc


def _prompt_block_from_openreward(
    block: OpenRewardTextBlockLike | OpenRewardImageBlockLike,
) -> OpenRewardPromptBlock:
    detail = None if getattr(block, "detail", None) is None else _clone_json(block.detail, "prompt block detail")

    if block.type == OpenRewardPromptBlockType.TEXT:
        return OpenRewardPromptBlock(type=OpenRewardPromptBlockType.TEXT, text=block.text, detail=detail)
    if block.type == OpenRewardPromptBlockType.IMAGE:
        return OpenRewardPromptBlock(
            type=OpenRewardPromptBlockType.IMAGE,
            data=block.data,
            mime_type=block.mimeType,
            detail=detail,
        )
    raise ValueError(f"Unsupported OpenReward prompt block type: {block.type!r}")


def _tool_spec_from_openreward(tool: OpenRewardToolLike) -> OpenRewardToolSpec:
    input_schema = None if tool.input_schema is None else _clone_json(tool.input_schema, f"tool schema {tool.name}")
    return OpenRewardToolSpec(
        name=tool.name,
        description=tool.description,
        input_schema=input_schema,
    )


def resolve_task_indices(
    total_tasks: int,
    *,
    indices: Sequence[int] | None = None,
    start: int | None = None,
    stop: int | None = None,
) -> list[int]:
    """Resolve a deterministic task subset using Python slice semantics."""

    if total_tasks < 0:
        raise ValueError(f"total_tasks must be non-negative, got {total_tasks}")
    if indices is not None and (start is not None or stop is not None):
        raise ValueError("indices cannot be combined with start/stop")

    if indices is None:
        return list(range(total_tasks))[slice(start, stop)]

    resolved: list[int] = []
    seen_indices: set[int] = set()
    for raw_index in indices:
        normalized_index = raw_index if raw_index >= 0 else total_tasks + raw_index
        if normalized_index < 0 or normalized_index >= total_tasks:
            raise ValueError(f"Task index {raw_index} is out of bounds for {total_tasks} tasks")
        if normalized_index in seen_indices:
            raise ValueError(f"Duplicate task index {raw_index} resolved to {normalized_index}")
        seen_indices.add(normalized_index)
        resolved.append(normalized_index)

    return resolved


def build_openreward_task_manifest(
    environment: OpenRewardEnvironmentLike,
    split: str,
    *,
    indices: Sequence[int] | None = None,
    start: int | None = None,
    stop: int | None = None,
    secrets: SecretsMapping | None = None,
) -> OpenRewardTaskManifest:
    """Snapshot tasks, prompts, and tools for one OpenReward split."""

    total_tasks = environment.num_tasks(split)
    selected_indices = resolve_task_indices(total_tasks, indices=indices, start=start, stop=stop)

    entries: list[OpenRewardTaskManifestEntry] = []
    environment_name: str | None = None
    for task_index in selected_indices:
        task = environment.get_task(split, task_index)
        if environment_name is None:
            environment_name = task.environment_name
        elif task.environment_name != environment_name:
            raise ValueError(
                "Expected one environment name per manifest, "
                f"got both {environment_name!r} and {task.environment_name!r}"
            )

        with environment.session(task=task, secrets=secrets) as session:
            prompt_blocks = [_prompt_block_from_openreward(block) for block in session.get_prompt()]
            tools = [_tool_spec_from_openreward(tool) for tool in session.list_tools()]

        entries.append(
            OpenRewardTaskManifestEntry(
                task_index=task_index,
                task_spec=_clone_json(task.task_spec, f"task_spec[{task_index}]"),
                prompt_blocks=prompt_blocks,
                tools=tools,
            )
        )

    return OpenRewardTaskManifest(
        deployment_name=environment.deployment_name,
        environment_name=environment_name or getattr(environment, "variant", None) or environment.name,
        split=split,
        tasks=entries,
    )


def save_openreward_task_manifest(manifest: OpenRewardTaskManifest, output_path: str) -> None:
    """Write a manifest to a JSON file."""

    with open_url(output_path, "w", encoding="utf-8") as fd:
        fd.write(manifest.model_dump_json(indent=2))


def load_openreward_task_manifest(input_path: str) -> OpenRewardTaskManifest:
    """Load a manifest from a JSON file."""

    with open_url(input_path, "rb") as fd:
        return OpenRewardTaskManifest.model_validate_json(fd.read())


def prepare_openreward_task_manifest(
    environment_name: str,
    split: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    variant: str | None = None,
    indices: Sequence[int] | None = None,
    start: int | None = None,
    stop: int | None = None,
    secrets: SecretsMapping | None = None,
) -> OpenRewardTaskManifest:
    """Build a task manifest from a live OpenReward environment."""

    openreward_client = load_openreward_client()
    with openreward_client(api_key=api_key, base_url=base_url) as client:
        environment = client.environments.get(environment_name, variant=variant)
        return build_openreward_task_manifest(
            environment,
            split,
            indices=indices,
            start=start,
            stop=stop,
            secrets=secrets,
        )
