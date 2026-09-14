# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded importer for NeMo Gym's seeded Workplace Assistant case."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec

from taskcompendium.execution import ChatWithTools, HarborTaskBinding, ProviderEnvironment, ProviderToolBinding
from taskcompendium.models import (
    ActionInterface,
    AnswerRequirements,
    AssistantFinal,
    Embedded,
    ProviderStateVerifier,
    Rejected,
    RejectionReason,
    Rendering,
    Resource,
    ResourceRole,
    Source,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpecification,
)
from taskcompendium.providers.nemo_workplace.provider import ADAPTER, SEED_SHA256
from taskcompendium.providers.nemo_workplace.tools import get_tools

DATASET = "nvidia/Nemotron-RL-agent-workplace_assistant"
DATASET_REVISION = "c86a908379e0a361a573c395e175d3c1aa128e6c"
GYM_REVISION = "1e668906d2e69a9e8ee9aaafc60050a4025d9688"
IMPORTER_REVISION = "taskcompendium-nemo-workplace-v2"
FIXTURE_NAME = "workplace-0.json"
FIXTURE_SHA256 = "a92e1627d61734c323071765bba22a7f25bc3ec7e096b77b0c8d4f7e2848f447"
INTERFACE = ActionInterface("workplace_assistant", "nemo-gym-v1", SEED_SHA256)
_SOURCE_ROW_PATH = "source-row.json"
_SOURCE_PROVENANCE_PATH = "source-provenance.json"
_PROVIDER_IMPORT_PATH = "taskcompendium.providers.nemo_workplace.provider:NemoWorkplaceEnvironment"


@dataclass(frozen=True)
class ProviderCall:
    """A private scripted provider call used by behavioral tests and local trial fixtures."""

    name: str
    arguments: str


@dataclass(frozen=True)
class WorkplaceSample:
    """One source-pinned Workplace specification and its explicit private execution choice."""

    specification: TaskSpecification
    rendering: Rendering
    binding: HarborTaskBinding
    known_good: tuple[ProviderCall, ...]
    wrong_mutation: tuple[ProviderCall, ...]
    noop: tuple[ProviderCall, ...]
    recovery: tuple[ProviderCall, ...]


def _source(row: str) -> Source:
    return Source(DATASET, DATASET_REVISION, row, IMPORTER_REVISION)


def _instructions(request: dict[str, Any]) -> str:
    messages = request.get("input")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Workplace row needs source input messages")
    turns: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Workplace source message must be an object")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user"} or not isinstance(content, str) or not content.strip():
            raise ValueError("Workplace source message must have a system or user text payload")
        turns.append(f"{role.title()}:\n{content.strip()}")
    return "\n\n".join(turns)


def _gold_calls(value: Any) -> tuple[ProviderCall, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("Workplace ground truth must be a nonempty action list")
    calls: list[ProviderCall] = []
    for action in value:
        if (
            not isinstance(action, dict)
            or not isinstance(action.get("name"), str)
            or not isinstance(action.get("arguments"), str)
        ):
            raise ValueError("Workplace ground truth actions require names and JSON-string arguments")
        arguments = json.loads(action["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError("Workplace tool arguments must decode to an object")
        calls.append(ProviderCall(action["name"], action["arguments"]))
    return tuple(calls)


def _tool_names(value: Any) -> set[str]:
    if not isinstance(value, list) or len(value) != 27:
        raise ValueError("Workplace source must advertise all 27 tools")
    names = {tool.get("name") for tool in value if isinstance(tool, dict) and isinstance(tool.get("name"), str)}
    if len(names) != len(value):
        raise ValueError("Workplace tool names must be unique nonempty strings")
    provided = {schema["name"] for schema in get_tools()["schemas"]}
    if names != provided:
        raise ValueError("Workplace source tools do not match the shared provider")
    return names


def provider_binding() -> HarborTaskBinding:
    """Bind any Workplace source row to the shared seeded provider."""
    return HarborTaskBinding(
        ProviderEnvironment(
            INTERFACE,
            _PROVIDER_IMPORT_PATH,
            {"interface": msgspec.to_builtins(INTERFACE), "seed_sha256": SEED_SHA256},
        ),
        ChatWithTools((ProviderToolBinding(INTERFACE.name),)),
    )


def import_row(data: bytes) -> TaskSpecification | Rejected:
    """Import id 0 without exposing its seed, source row, or target actions publicly."""
    source = _source("0")
    if hashlib.sha256(data).hexdigest() != FIXTURE_SHA256:
        return Rejected(source, RejectionReason.UNRECOVERABLE_SOURCE, "source row does not match pinned raw digest")
    try:
        row = json.loads(data)
        if not isinstance(row, dict) or row.get("id") != 0:
            raise ValueError("bounded Workplace importer requires source row id 0")
        request = row["responses_create_params"]
        if not isinstance(request, dict):
            raise ValueError("Workplace responses parameters must be an object")
        instructions = _instructions(request)
        schemas = request.get("tools")
        if not isinstance(schemas, list) or len(schemas) != 27:
            raise ValueError("pinned Workplace source must advertise all 27 tools")
        gold = _gold_calls(row["ground_truth"])
        if not isinstance(row.get("category"), str) or not isinstance(row.get("environment_name"), str):
            raise ValueError("Workplace routing metadata is missing")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return Rejected(source, RejectionReason.UNRECOVERABLE_SOURCE, str(error))
    return TaskSpecification(
        id="nemo/workplace/0",
        requirements=TaskRequirements(action_interfaces=(INTERFACE,)),
        resources=(Resource(_SOURCE_ROW_PATH, (ResourceRole.VERIFIER,), Embedded(data)),),
        metadata=TaskMetadata(source, competencies=("stateful_tool_use",), task_shape="stateful_domain"),
        steps=(
            StepSpecification(
                instructions=instructions,
                verifier=ProviderStateVerifier(
                    INTERFACE,
                    ADAPTER,
                    {"ground_truth": [{"name": call.name, "arguments": call.arguments} for call in gold]},
                ),
                answer_requirements=AnswerRequirements("text"),
            ),
        ),
    )


def import_hub_row(data: bytes, *, split: str, offset: int) -> TaskSpecification | Rejected:
    """Import a pinned Workplace Hub row into the shared seeded provider."""
    source = _source(str(offset))
    try:
        row = json.loads(data)
        if not isinstance(row, dict) or row.get("id") != offset:
            raise ValueError("Workplace Hub row id must match its sampled offset")
        request = row["responses_create_params"]
        if not isinstance(request, dict):
            raise ValueError("Workplace responses parameters must be an object")
        instructions = _instructions(request)
        tool_names = _tool_names(request.get("tools"))
        gold = _gold_calls(row["ground_truth"])
        if not all(call.name in tool_names for call in gold):
            raise ValueError("Workplace ground truth must use source-advertised tools")
        if not isinstance(row.get("category"), str) or not isinstance(row.get("environment_name"), str):
            raise ValueError("Workplace routing metadata is missing")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return Rejected(source, RejectionReason.UNRECOVERABLE_SOURCE, str(error))
    provenance = {
        "dataset": DATASET,
        "revision": DATASET_REVISION,
        "split": split,
        "offset": str(offset),
    }
    return TaskSpecification(
        id=f"nemo/workplace/{offset}",
        requirements=TaskRequirements(action_interfaces=(INTERFACE,)),
        resources=(
            Resource(_SOURCE_ROW_PATH, (ResourceRole.VERIFIER,), Embedded(data)),
            Resource(
                _SOURCE_PROVENANCE_PATH,
                (ResourceRole.VERIFIER,),
                Embedded(json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()),
            ),
        ),
        metadata=TaskMetadata(source, competencies=("stateful_tool_use",), task_shape="stateful_domain"),
        steps=(
            StepSpecification(
                instructions=instructions,
                verifier=ProviderStateVerifier(
                    INTERFACE,
                    ADAPTER,
                    {"ground_truth": [{"name": call.name, "arguments": call.arguments} for call in gold]},
                ),
                answer_requirements=AnswerRequirements("text"),
            ),
        ),
    )


def build_sample(fixture_root: Path) -> WorkplaceSample:
    """Load the source-pinned sample with provider configuration and private behavioral attempts."""
    data = (fixture_root / FIXTURE_NAME).read_bytes()
    specification = import_row(data)
    if isinstance(specification, Rejected):
        raise ValueError(specification.detail)
    good = _gold_calls(json.loads(data)["ground_truth"])
    return WorkplaceSample(
        specification=specification,
        rendering=Rendering("nemo-workplace-chat", AssistantFinal()),
        binding=provider_binding(),
        known_good=good,
        wrong_mutation=(
            ProviderCall(
                "email_reply_email",
                '{"email_id":"00000057","body":"Thanks for the update - I will not follow up."}',
            ),
        ),
        noop=(ProviderCall("email_get_email_information_by_id", '{"email_id":"00000057","field":"subject"}'),),
        recovery=(
            ProviderCall("email_reply_email", '{"email_id":"00000057","unknown":"x"}'),
            *good,
        ),
    )
