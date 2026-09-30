# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declared container surfaces at the TaskSpec export boundary."""

import hashlib
import json

import pytest

from taskcompendium.grading import exact_answer, structured_exact
from taskcompendium.lowering import HarborEnvironmentConfig, ToolBinding, lower_to_harbor
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FunctionDefinition,
    ProviderRequirement,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.submission import AnswerCall, PlainText, ProviderState

from .conftest import container_runtime


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _external_services(tmp_path, monkeypatch, *, names=("lookup_a", "lookup_b")):
    bindings = []
    for service, name in zip(("a", "b"), names, strict=True):
        definitions = ({"type": "function", "function": {"name": name, "parameters": {"type": "object"}}},)
        bindings.append(
            ToolBinding(
                action_interface=f"{service}:v1",
                seed_sha256=service * 64,
                provider_revision=f"{service}-v1",
                runtime=container_runtime(f"{service}:v1", service * 64, f"{service}-v1", definitions),
                tools=(name,),
                tools_sha256=_digest(definitions),
                tool_definitions=definitions,
                state_available=False,
            )
        )
    return tuple(bindings)


def _specification() -> TaskSpec:
    return TaskSpec(
        id="two-services",
        context=ConversationInput(events=(TextMessage(role="user", content="Answer using both services."),)),
        environment_requirements=EnvironmentRequirements(),
        tool_providers={
            "a": ProviderRequirement(action_interface="a:v1", seed_sha256="a" * 64),
            "b": ProviderRequirement(action_interface="b:v1", seed_sha256="b" * 64),
        },
        answer_type=AnswerType.TEXT,
        verifier=exact_answer("done"),
        source=Source(dataset="test", revision="1", row="0", importer_revision="1"),
    )


def test_provider_tool_collisions_reject_export(tmp_path, monkeypatch):
    bindings = _external_services(tmp_path, monkeypatch, names=("lookup", "lookup"))
    with pytest.raises(ValueError, match="Tool names must be unique"):
        HarborEnvironmentConfig(tool_providers=dict(zip(("a", "b"), bindings, strict=True)))

    terminal_collision = _external_services(tmp_path, monkeypatch, names=("lookup", "submit_answer"))
    environment_config = HarborEnvironmentConfig(tool_providers=dict(zip(("a", "b"), terminal_collision, strict=True)))
    convention = AnswerCall(id="answer-call")
    with pytest.raises(ValueError, match="Submission function names collide"):
        lower_to_harbor(_specification(), convention, environment_config, tmp_path / "task")

    plain_specification = _specification().model_copy(
        update={"final_tools": (FunctionDefinition(name="lookup", parameters={"type": "object"}),)}
    )
    plain = PlainText(id="plain")
    with pytest.raises(ValueError, match="Submission function names collide"):
        lower_to_harbor(plain_specification, plain, environment_config, tmp_path / "plain-task")


@pytest.mark.parametrize(
    ("changed", "error"),
    (({"seed_sha256": "c" * 64}, "seed differs"), ({"action_interface": "other:v1"}, "interface differs")),
)
def test_provider_requirement_mismatch_rejects_export_before_task_files(tmp_path, monkeypatch, changed, error):
    bindings = _external_services(tmp_path, monkeypatch)
    wrong = bindings[1].model_copy(update=changed)
    environment_config = HarborEnvironmentConfig(tool_providers={"a": bindings[0], "b": wrong})
    convention = PlainText(id="plain")
    with pytest.raises(ValueError, match=error):
        lower_to_harbor(_specification(), convention, environment_config, tmp_path / "task")
    assert not (tmp_path / "task").exists()


def test_state_grader_cannot_target_an_unbound_service(tmp_path, monkeypatch):
    bindings = _external_services(tmp_path, monkeypatch)
    environment_config = HarborEnvironmentConfig(tool_providers=dict(zip(("a", "b"), bindings, strict=True)))
    specification = _specification().model_copy(
        update={"answer_type": AnswerType.STATE, "verifier": structured_exact({})}
    )
    convention = ProviderState(id="state", provider="missing")
    with pytest.raises(ValueError, match="State convention names a missing provider"):
        lower_to_harbor(specification, convention, environment_config, tmp_path / "task")


def test_state_provider_must_expose_canonical_snapshot(tmp_path, monkeypatch):
    bindings = _external_services(tmp_path, monkeypatch)
    environment_config = HarborEnvironmentConfig(tool_providers=dict(zip(("a", "b"), bindings, strict=True)))
    specification = _specification().model_copy(
        update={"answer_type": AnswerType.STATE, "verifier": structured_exact({})}
    )
    convention = ProviderState(id="state", provider="b")

    with pytest.raises(ValueError, match="canonical provider state"):
        lower_to_harbor(specification, convention, environment_config, tmp_path / "task")


def test_host_chat_rejects_workspace_requirements_with_tool_providers(tmp_path, monkeypatch):
    bindings = _external_services(tmp_path, monkeypatch)
    environment_config = HarborEnvironmentConfig(tool_providers={"a": bindings[0], "b": bindings[1]})
    convention = PlainText(id="plain")

    with pytest.raises(ValueError, match="workspace capability"):
        lower_to_harbor(
            _specification().model_copy(
                update={"environment_requirements": EnvironmentRequirements(capabilities=("shell",))}
            ),
            convention,
            environment_config,
            tmp_path / "capability",
        )


def test_container_export_contains_manifest_without_provider_code(tmp_path, monkeypatch):
    bindings = _external_services(tmp_path, monkeypatch)
    config = HarborEnvironmentConfig(tool_providers=dict(zip(("a", "b"), bindings, strict=True)))
    task = lower_to_harbor(_specification(), PlainText(id="plain"), config, tmp_path / "task")
    exported = json.loads((task / "environment_config.json").read_text())
    assert exported["tool_providers"]["a"]["runtime"]["image"] == bindings[0].runtime.image
    assert list((task / "environment").iterdir()) == []
