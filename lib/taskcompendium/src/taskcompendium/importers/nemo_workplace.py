# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import only the pinned NeMo Workplace Assistant row 0."""

import hashlib
import json
from importlib.resources import files
from typing import Any

from taskcompendium.grading import state_match
from taskcompendium.lowering import HarborEnvironmentConfig
from taskcompendium.models import AnswerType, Source, TaskRequirements, TaskSpec
from taskcompendium.providers.nemo_workplace.provider import (
    ACTION_INTERFACE,
    PROVIDER_REVISION,
    SEED_SHA256,
    TOOL_DEFINITIONS,
    TOOLS_SHA256,
    _seed_digest,
    expected_state_json,
)
from taskcompendium.providers.nemo_workplace.tools import get_tools
from taskcompendium.resources import ResourceVisibility, TaskResource
from taskcompendium.submission import AnswerFormat, SubmissionConvention

DATASET = "nvidia/Nemotron-RL-agent-workplace_assistant"
DATASET_REVISION = "c86a908379e0a361a573c395e175d3c1aa128e6c"
IMPORTER_REVISION = "taskcompendium-nemo-workplace-row0-v1"
SOURCE_REVISION = PROVIDER_REVISION
ROW_SHA256 = "a92e1627d61734c323071765bba22a7f25bc3ec7e096b77b0c8d4f7e2848f447"
SOURCE_ROW_PATH = "source-row.json"
SOURCE_PROVENANCE_PATH = "source-provenance.json"
PROVIDER = "nemo_workplace:v1"


def provider_binding() -> HarborEnvironmentConfig:
    """Select the registered source provider and its immutable tool surface."""
    return HarborEnvironmentConfig(
        environment="stateful",
        action_interface=ACTION_INTERFACE,
        seed_sha256=SEED_SHA256,
        provider=PROVIDER,
        provider_revision=PROVIDER_REVISION,
        tools_sha256=TOOLS_SHA256,
        tools=tuple(item["function"]["name"] for item in TOOL_DEFINITIONS),
    )


def _instructions(messages: Any) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("Workplace source requires input messages")
    turns = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Workplace input message must be an object")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user"} or not isinstance(content, str) or not content.strip():
            raise ValueError("Workplace input requires system or user text")
        turns.append(f"{role.title()}:\n{content.strip()}")
    return "\n\n".join(turns)


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


def import_row(data: bytes) -> tuple[TaskSpec, SubmissionConvention, HarborEnvironmentConfig]:
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
    if request["parallel_tool_calls"] is not False or request["temperature"] != 1.0:
        raise ValueError("Unsupported Workplace source tool-call settings")
    schemas = request["tools"]
    if not isinstance(schemas, list) or schemas != get_tools()["schemas"]:
        raise ValueError("Workplace source tools differ from the pinned provider")
    actions = _gold(row["ground_truth"])
    if any(action["name"] not in {schema["name"] for schema in schemas} for action in actions):
        raise ValueError("Workplace ground truth calls an unavailable tool")
    expected = expected_state_json(actions)
    attribution = json.loads(files("taskcompendium.importers").joinpath("data/workplace-0.attribution.json").read_text())
    if attribution["dataset"] != DATASET or attribution["dataset_revision"] != DATASET_REVISION:
        raise ValueError("Workplace attribution differs from pinned dataset")
    provenance = {
        **attribution,
        "source_repository": "NVIDIA-NeMo/Gym",
        "source_revision": SOURCE_REVISION,
        "row": 0,
        "raw_sha256": ROW_SHA256,
        "seed_sha256": SEED_SHA256,
        "tools_sha256": TOOLS_SHA256,
    }
    specification = TaskSpec(
        id="nemo-workplace-0",
        instructions=_instructions(request["input"]),
        verifier=state_match(expected),
        source=Source(dataset=DATASET, revision=DATASET_REVISION, row="0", importer_revision=IMPORTER_REVISION),
        requirements=TaskRequirements(action_interfaces=(ACTION_INTERFACE,), seed_sha256=SEED_SHA256),
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
    return specification, SubmissionConvention(id="state", answer_format=AnswerFormat.STATE), provider_binding()


def load_fixture() -> tuple[TaskSpec, SubmissionConvention, HarborEnvironmentConfig]:
    """Return a runnable row 0 task from data included in the installed wheel."""
    return import_row(files("taskcompendium.importers").joinpath("data/workplace-0.json").read_bytes())
