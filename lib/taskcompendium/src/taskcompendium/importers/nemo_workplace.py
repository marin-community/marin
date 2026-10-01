# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import pinned NeMo Workplace Assistant example and dataset rows."""

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, NamedTuple

from taskcompendium.grading import structured_exact
from taskcompendium.lowering import HarborEnvironmentConfig, ToolBinding
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ProviderRequirement,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.provider_sources import ToolProviderCache
from taskcompendium.submission import ProviderState, SubmissionConvention

DATASET = "nvidia/Nemotron-RL-agent-workplace_assistant"
DATASET_REVISION = "c86a908379e0a361a573c395e175d3c1aa128e6c"
IMPORTER_REVISION = "taskcompendium-nemo-workplace-v7"
DATASET_SPLIT_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REVISION}"
DATASET_SPLIT_MAX_BYTES = 32 * 1024 * 1024
DATASET_SPLIT_ROW_COUNTS = {"train": 1255, "validation": 545}
DATASET_SPLIT_SHA256 = {
    "train": "cb64a4eca977c049f5b64a5d3346fcb560ed20cdf186b31135a9b332320c26c8",
    "validation": "6cce92f9cd8807983c4aa8f1006d5fab5e8c4f16ae28b90222739d98905df01b",
}
WorkplaceSplit = Literal["train", "validation"]
SOURCE_REVISION = "1e668906d2e69a9e8ee9aaafc60050a4025d9688"
SOURCE_EXAMPLE_PATH = "resources_servers/workplace_assistant/data/example.jsonl"
SOURCE_EXAMPLE_URL = f"https://raw.githubusercontent.com/NVIDIA-NeMo/Gym/{SOURCE_REVISION}/{SOURCE_EXAMPLE_PATH}"
SOURCE_EXAMPLE_SHA256 = "2df8a2537121aa40041b46a96683f0397cc945bf5de83062b76ca6e3eaf6296d"
SOURCE_EXAMPLE_MAX_BYTES = 128 * 1024
SOURCE_EXAMPLE_MAX_ROWS = 32
ROW_SHA256_BY_ID = {
    0: "a92e1627d61734c323071765bba22a7f25bc3ec7e096b77b0c8d4f7e2848f447",
    1: "795df366fe7dc973184dd630bbb07d107cce39bf557f9eb83bf606b776e6bfc7",
    2: "0039528c9fe951e706611ec766c504dc96197f06de9e9c191a2d808208f70552",
    3: "db067d6ae118a3bde1c14ca4dee055c6c62aa5dbb876c01346ca368e256e4f61",
    4: "ba3ee0fa9e41141699a785b22b1bfadda76eb5f09bab33d4a2285a9fd7a7a3d4",
}
ROW_SHA256 = ROW_SHA256_BY_ID[0]
PROVIDER_REPOSITORY = "https://github.com/marin-community/nemo_workplace"
PROVIDER_GIT_REVISION = "27b39001312617403021635f0492e120abc2cd34"
PROVIDER = f"python+git+{PROVIDER_REPOSITORY}@{PROVIDER_GIT_REVISION}:nemo_workplace.provider:NemoWorkplaceProvider"
PROVIDER_NAME = "workplace"
SOURCE_CATEGORIES = frozenset(
    {
        "workplace_assistant_email",
        "workplace_assistant_calendar",
        "workplace_assistant_customer_relationship_manager",
        "workplace_assistant_project_management",
        "workplace_assistant_analytics",
    }
)


class WorkplaceImport(NamedTuple):
    specification: TaskSpec
    convention: SubmissionConvention
    environment_config: HarborEnvironmentConfig


def select_rows(source: bytes) -> tuple[bytes, ...]:
    """Select all supported exact rows from the pinned upstream JSONL payload."""
    if len(source) > SOURCE_EXAMPLE_MAX_BYTES:
        raise ValueError("Workplace example source exceeds size limit")
    if hashlib.sha256(source).hexdigest() != SOURCE_EXAMPLE_SHA256:
        raise ValueError("Workplace example source does not match its pinned digest")
    lines = source.splitlines(keepends=True)
    if len(lines) > SOURCE_EXAMPLE_MAX_ROWS:
        raise ValueError("Workplace example source exceeds row limit")
    selected: dict[int, bytes] = {}
    for line in lines:
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Workplace example source requires rows with integer IDs")
        row_id = row.get("id")
        if not isinstance(row_id, int) or isinstance(row_id, bool):
            raise ValueError("Workplace example source requires rows with integer IDs")
        if row_id not in ROW_SHA256_BY_ID:
            raise ValueError(f"Unsupported Workplace source row {row_id}")
        if row_id in selected:
            raise ValueError(f"Workplace example source has duplicate row {row_id}")
        if hashlib.sha256(line).hexdigest() != ROW_SHA256_BY_ID[row_id]:
            raise ValueError(f"Workplace row {row_id} does not match its pinned raw digest")
        selected[row_id] = line
    if set(selected) != set(ROW_SHA256_BY_ID):
        raise ValueError("Workplace example source does not contain all pinned rows")
    return tuple(selected[row_id] for row_id in sorted(selected))


def select_row_zero(source: bytes) -> bytes:
    """Select row 0 for callers that need the original single-row entrypoint."""
    return select_rows(source)[0]


def select_dataset_rows(source: bytes, split: WorkplaceSplit) -> tuple[bytes, ...]:
    """Select exact rows from a pinned Hugging Face split JSONL payload."""
    if len(source) > DATASET_SPLIT_MAX_BYTES:
        raise ValueError(f"Workplace {split} split exceeds size limit")
    if hashlib.sha256(source).hexdigest() != DATASET_SPLIT_SHA256[split]:
        raise ValueError(f"Workplace {split} split does not match its pinned digest")
    lines = source.splitlines(keepends=True)
    if len(lines) != DATASET_SPLIT_ROW_COUNTS[split]:
        raise ValueError(f"Workplace {split} split does not match its pinned row count")
    selected: list[bytes] = []
    seen: set[int] = set()
    for line in lines:
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Workplace {split} split requires rows with integer IDs")
        row_id = row.get("id")
        if not isinstance(row_id, int) or isinstance(row_id, bool):
            raise ValueError(f"Workplace {split} split requires rows with integer IDs")
        if row_id in seen:
            raise ValueError(f"Workplace {split} split has duplicate row {row_id}")
        seen.add(row_id)
        selected.append(line)
    if seen != set(range(DATASET_SPLIT_ROW_COUNTS[split])):
        raise ValueError(f"Workplace {split} split IDs do not match its pinned row range")
    return tuple(selected)


def _provider_module(provider_source: Path, *, cache: ToolProviderCache) -> ModuleType:
    provider = cache.stage(PROVIDER, provider_source).factory
    return importlib.import_module(provider.__module__)


def workplace_environment_config(provider_source: Path) -> HarborEnvironmentConfig:
    """Load the Workplace tool binding from a verified provider snapshot."""
    with ToolProviderCache() as cache:
        return _environment_config(_provider_module(provider_source, cache=cache))


def _environment_config(provider: ModuleType) -> HarborEnvironmentConfig:
    """Select the source provider and its immutable tool surface."""
    return HarborEnvironmentConfig(
        tool_providers={
            PROVIDER_NAME: ToolBinding(
                action_interface=provider.ACTION_INTERFACE,
                seed_sha256=provider.SEED_SHA256,
                provider=PROVIDER,
                provider_revision=provider.PROVIDER_REVISION,
                tools_sha256=provider.TOOLS_SHA256,
                tools=tuple(item["function"]["name"] for item in provider.TOOL_DEFINITIONS),
            )
        },
    )


def _conversation(messages: Any) -> ConversationInput:
    if not isinstance(messages, list) or not messages:
        raise ValueError("Workplace source requires input messages")
    turns: list[TextMessage] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Workplace input message must be an object")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user"} or not isinstance(content, str) or not content.strip():
            raise ValueError("Workplace input requires system or user text")
        turns.append(TextMessage(role=role, content=content))
    return ConversationInput(events=tuple(turns))


def _gold(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("Workplace ground truth requires an action list")
    actions = []
    for action in value:
        if not isinstance(action, dict) or not isinstance(action.get("name"), str):
            raise ValueError("Workplace ground truth action requires a name")
        arguments = action.get("arguments")
        if not isinstance(arguments, str) or not isinstance(json.loads(arguments), dict):
            raise ValueError("Workplace ground truth action requires JSON object arguments")
        actions.append({"name": action["name"], "arguments": arguments})
    return actions


def import_row(data: bytes, provider_source: Path) -> WorkplaceImport:
    """Import the raw row after checking its digest, source tools, and seed."""
    row = json.loads(data)
    if not isinstance(row, dict):
        raise ValueError("Workplace source requires an integer row ID")
    row_id = row.get("id")
    if not isinstance(row_id, int) or isinstance(row_id, bool):
        raise ValueError("Workplace source requires an integer row ID")
    expected_digest = ROW_SHA256_BY_ID.get(row_id)
    if expected_digest is None or hashlib.sha256(data).hexdigest() != expected_digest:
        raise ValueError(f"Workplace row {row_id} does not match its pinned raw digest")
    with ToolProviderCache() as cache:
        provider = _provider_module(provider_source, cache=cache)
        if provider._seed_digest() != provider.SEED_SHA256:
            raise ValueError("Workplace seed differs from its pinned digest")
        return _import_row(row, provider, source_row=str(row_id), task_id=f"nemo-workplace-{row_id}")


def import_dataset_split(source: bytes, split: WorkplaceSplit, provider_source: Path) -> tuple[WorkplaceImport, ...]:
    """Convert every row in one pinned NeMo Workplace dataset split."""
    selected = select_dataset_rows(source, split)
    with ToolProviderCache() as cache:
        provider = _provider_module(provider_source, cache=cache)
        if provider._seed_digest() != provider.SEED_SHA256:
            raise ValueError("Workplace seed differs from its pinned digest")
        imports = []
        for data in selected:
            row = json.loads(data)
            row_id = row["id"]
            row_sha256 = hashlib.sha256(data).hexdigest()
            try:
                imports.append(
                    _import_row(
                        row,
                        provider,
                        source_row=f"{split}:{row_id}:{row_sha256}",
                        task_id=f"nemo-workplace-{split}-{row_id}",
                    )
                )
            except ValueError as exc:
                raise ValueError(f"Workplace {split} row {row_id}: {exc}") from exc
        return tuple(imports)


def _import_row(row: dict[str, Any], provider: ModuleType, *, source_row: str, task_id: str) -> WorkplaceImport:
    """Build a TaskSpec from a row already verified against a pinned source."""
    if row.get("environment_name") != "workplace_assistant" or row.get("category") not in SOURCE_CATEGORIES:
        raise ValueError("Workplace source routing metadata differs")
    request = row.get("responses_create_params")
    if not isinstance(request, dict) or set(request) != {"input", "tools", "parallel_tool_calls", "temperature"}:
        raise ValueError("Unsupported Workplace source request")
    if (
        request["parallel_tool_calls"] is not provider.REQUEST_PARALLEL_TOOL_CALLS
        or request["temperature"] != provider.REQUEST_TEMPERATURE
    ):
        raise ValueError("Unsupported Workplace source tool-call settings")
    schemas = request["tools"]
    if not isinstance(schemas, list) or schemas != provider.get_tools()["schemas"]:
        raise ValueError("Workplace source tools differ from the pinned provider")
    actions = _gold(row["ground_truth"])
    if any(action["name"] not in {schema["name"] for schema in schemas} for action in actions):
        raise ValueError("Workplace ground truth calls an unavailable tool")
    expected = json.loads(provider.expected_state_json(actions))
    specification = TaskSpec(
        id=task_id,
        context=_conversation(request["input"]),
        verifier=structured_exact(expected),
        source=Source(dataset=DATASET, revision=DATASET_REVISION, row=source_row, importer_revision=IMPORTER_REVISION),
        environment_requirements=EnvironmentRequirements(),
        tool_providers={
            PROVIDER_NAME: ProviderRequirement(
                action_interface=provider.ACTION_INTERFACE, seed_sha256=provider.SEED_SHA256
            )
        },
        answer_type=AnswerType.STATE,
    )
    return WorkplaceImport(
        specification, ProviderState(id="state", provider=PROVIDER_NAME), _environment_config(provider)
    )
