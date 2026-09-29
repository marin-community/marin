# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import only the pinned NeMo Workplace Assistant row 0."""

import hashlib
import json
from typing import Any, NamedTuple

from nemo_workplace.provider import (
    ACTION_INTERFACE,
    PROVIDER_REVISION,
    REQUEST_PARALLEL_TOOL_CALLS,
    REQUEST_TEMPERATURE,
    SEED_SHA256,
    TOOL_DEFINITIONS,
    TOOLS_SHA256,
    _seed_digest,
    expected_state_json,
)
from nemo_workplace.tools import get_tools
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
from taskcompendium.resources import ResourceVisibility, TaskResource
from taskcompendium.submission import ProviderState, SubmissionConvention

DATASET = "nvidia/Nemotron-RL-agent-workplace_assistant"
DATASET_REVISION = "c86a908379e0a361a573c395e175d3c1aa128e6c"
DATASET_CARD_SHA256 = "afb783d398188a18a54222f9f124930bbf6de051587ce3c4949684bb9da8931a"
DATASET_CARD_URL = f"https://huggingface.co/datasets/{DATASET}/blob/{DATASET_REVISION}/README.md"
IMPORTER_REVISION = "taskcompendium-nemo-workplace-row0-v4"
SOURCE_REVISION = "1e668906d2e69a9e8ee9aaafc60050a4025d9688"
SOURCE_EXAMPLE_PATH = "resources_servers/workplace_assistant/data/example.jsonl"
SOURCE_EXAMPLE_URL = f"https://raw.githubusercontent.com/NVIDIA-NeMo/Gym/{SOURCE_REVISION}/{SOURCE_EXAMPLE_PATH}"
SOURCE_EXAMPLE_SHA256 = "2df8a2537121aa40041b46a96683f0397cc945bf5de83062b76ca6e3eaf6296d"
SOURCE_EXAMPLE_MAX_BYTES = 128 * 1024
SOURCE_EXAMPLE_MAX_ROWS = 32
ROW_SHA256 = "a92e1627d61734c323071765bba22a7f25bc3ec7e096b77b0c8d4f7e2848f447"
SOURCE_ROW_PATH = "source-row.json"
SOURCE_PROVENANCE_PATH = "source-provenance.json"
PROVIDER_REPOSITORY = "https://github.com/marin-community/nemo_workplace"
PROVIDER_GIT_REVISION = "f92a1efb9bcf416b953136463b2022c5525f4b23"
PROVIDER = f"python+git+{PROVIDER_REPOSITORY}@{PROVIDER_GIT_REVISION}:nemo_workplace.provider:NemoWorkplaceProvider"
PROVIDER_NAME = "workplace"
EXAMPLE_LICENSE_URL = (
    f"https://github.com/NVIDIA-NeMo/Gym/blob/{SOURCE_REVISION}/resources_servers/workplace_assistant/README.md"
)


class WorkplaceImport(NamedTuple):
    specification: TaskSpec
    convention: SubmissionConvention
    environment_config: HarborEnvironmentConfig


def select_row_zero(source: bytes) -> bytes:
    """Select the exact row 0 bytes from the pinned upstream JSONL payload."""
    if len(source) > SOURCE_EXAMPLE_MAX_BYTES:
        raise ValueError("Workplace example source exceeds size limit")
    if hashlib.sha256(source).hexdigest() != SOURCE_EXAMPLE_SHA256:
        raise ValueError("Workplace example source does not match its pinned digest")
    lines = source.splitlines(keepends=True)
    if len(lines) > SOURCE_EXAMPLE_MAX_ROWS:
        raise ValueError("Workplace example source exceeds row limit")
    selected: bytes | None = None
    for line in lines:
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("id"), int):
            raise ValueError("Workplace example source requires rows with integer IDs")
        if row["id"] == 0:
            if selected is not None:
                raise ValueError("Workplace example source has duplicate row 0")
            selected = line
    if selected is None or hashlib.sha256(selected).hexdigest() != ROW_SHA256:
        raise ValueError("Workplace row 0 does not match its pinned raw digest")
    return selected


def workplace_environment_config() -> HarborEnvironmentConfig:
    """Select the source provider and its immutable tool surface."""
    return HarborEnvironmentConfig(
        tool_providers={
            PROVIDER_NAME: ToolBinding(
                action_interface=ACTION_INTERFACE,
                seed_sha256=SEED_SHA256,
                provider=PROVIDER,
                provider_revision=PROVIDER_REVISION,
                tools_sha256=TOOLS_SHA256,
                tools=tuple(item["function"]["name"] for item in TOOL_DEFINITIONS),
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
    if not isinstance(value, list) or not value:
        raise ValueError("Workplace ground truth requires actions")
    actions = []
    for action in value:
        if not isinstance(action, dict) or not isinstance(action.get("name"), str):
            raise ValueError("Workplace ground truth action requires a name")
        arguments = action.get("arguments")
        if not isinstance(arguments, str) or not isinstance(json.loads(arguments), dict):
            raise ValueError("Workplace ground truth action requires JSON object arguments")
        actions.append({"name": action["name"], "arguments": arguments})
    return actions


def import_row(data: bytes) -> WorkplaceImport:
    """Import the raw row after checking its digest, source tools, and seed."""
    if hashlib.sha256(data).hexdigest() != ROW_SHA256:
        raise ValueError("Workplace row 0 does not match its pinned raw digest")
    if _seed_digest() != SEED_SHA256:
        raise ValueError("Workplace seed differs from its pinned digest")
    row = json.loads(data)
    if not isinstance(row, dict) or row.get("id") != 0:
        raise ValueError("Only Workplace source row 0 is supported")
    if row.get("environment_name") != "workplace_assistant" or row.get("category") != "workplace_assistant_email":
        raise ValueError("Workplace source routing metadata differs")
    request = row.get("responses_create_params")
    if not isinstance(request, dict) or set(request) != {"input", "tools", "parallel_tool_calls", "temperature"}:
        raise ValueError("Unsupported Workplace source request")
    if (
        request["parallel_tool_calls"] is not REQUEST_PARALLEL_TOOL_CALLS
        or request["temperature"] != REQUEST_TEMPERATURE
    ):
        raise ValueError("Unsupported Workplace source tool-call settings")
    schemas = request["tools"]
    if not isinstance(schemas, list) or schemas != get_tools()["schemas"]:
        raise ValueError("Workplace source tools differ from the pinned provider")
    actions = _gold(row["ground_truth"])
    if any(action["name"] not in {schema["name"] for schema in schemas} for action in actions):
        raise ValueError("Workplace ground truth calls an unavailable tool")
    expected = json.loads(expected_state_json(actions))
    provenance = {
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "dataset_owner": "NVIDIA Corporation",
        "dataset_license": "CC-BY-4.0",
        "dataset_card_url": DATASET_CARD_URL,
        "dataset_card_sha256": DATASET_CARD_SHA256,
        "source_repository": "NVIDIA-NeMo/Gym",
        "source_revision": SOURCE_REVISION,
        "source_path": SOURCE_EXAMPLE_PATH,
        "source_file_sha256": SOURCE_EXAMPLE_SHA256,
        "source_license": "Apache-2.0",
        "source_license_url": EXAMPLE_LICENSE_URL,
        "provider_repository": PROVIDER_REPOSITORY,
        "provider_git_revision": PROVIDER_GIT_REVISION,
        "row": 0,
        "raw_sha256": ROW_SHA256,
        "seed_sha256": SEED_SHA256,
        "tools_sha256": TOOLS_SHA256,
    }
    specification = TaskSpec(
        id="nemo-workplace-0",
        context=_conversation(request["input"]),
        verifier=structured_exact(expected),
        source=Source(dataset=DATASET, revision=DATASET_REVISION, row="0", importer_revision=IMPORTER_REVISION),
        environment_requirements=EnvironmentRequirements(),
        tool_providers={PROVIDER_NAME: ProviderRequirement(action_interface=ACTION_INTERFACE, seed_sha256=SEED_SHA256)},
        answer_type=AnswerType.STATE,
        resources=(
            TaskResource(path=SOURCE_ROW_PATH, visibility=ResourceVisibility.VERIFIER, content=data.decode("utf-8")),
            TaskResource(
                path=SOURCE_PROVENANCE_PATH,
                visibility=ResourceVisibility.VERIFIER,
                content=json.dumps(provenance, sort_keys=True, separators=(",", ":")),
            ),
        ),
    )
    return WorkplaceImport(
        specification, ProviderState(id="state", provider=PROVIDER_NAME), workplace_environment_config()
    )
