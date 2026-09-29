# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provider selection at the boundary between private tasks and Harbor export."""

import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType

import pytest

from taskcompendium.grading import exact_answer, structured_exact
from taskcompendium.harbor.runner import ChatLaunch, run_trial
from taskcompendium.lowering import HarborEnvironmentConfig, ToolBinding, compatible_lowerings, lower_to_harbor
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FinalTools,
    FunctionDefinition,
    ProviderRequirement,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.provider_sources import SOURCE_MANIFEST, validate_staged_git_provider
from taskcompendium.resources import ResourceVisibility, TaskResource
from taskcompendium.submission import AnswerCall, PlainText, ProviderState


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _external_services(
    tmp_path,
    monkeypatch,
    *,
    names: tuple[str, str] = ("lookup_a", "lookup_b"),
):
    """Provide two independently pinned importable services outside TaskCompendium."""
    module_name = f"external_tool_services_{_digest(names)[:8]}"
    definitions = tuple(
        ({"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}},)
        for name in names
    )
    source = [
        "class ServiceA:",
        "    ACTION_INTERFACE = 'a:v1'",
        "    SEED_SHA256 = 'a' * 64",
        "    PROVIDER_REVISION = 'external-a-v1'",
        f"    TOOL_DEFINITIONS = {definitions[0]!r}",
        "    def __init__(self, *, seed_sha256, action_interface): pass",
        "    async def native_tool_definitions(self): return list(self.TOOL_DEFINITIONS)",
        "    async def dispatch_action(self, name, arguments, call_id): return '{}'",
        "",
        "class ServiceB:",
        "    ACTION_INTERFACE = 'b:v1'",
        "    SEED_SHA256 = 'b' * 64",
        "    PROVIDER_REVISION = 'external-b-v1'",
        f"    TOOL_DEFINITIONS = {definitions[1]!r}",
        "    def __init__(self, *, seed_sha256, action_interface): pass",
        "    async def native_tool_definitions(self): return list(self.TOOL_DEFINITIONS)",
        "    async def dispatch_action(self, name, arguments, call_id): return '{}'",
    ]
    (tmp_path / f"{module_name}.py").write_text("\n".join(source) + "\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    bindings = tuple(
        ToolBinding(
            action_interface=f"{service}:v1",
            seed_sha256=service * 64,
            provider=f"python:{module_name}:Service{service.upper()}",
            provider_revision=f"external-{service}-v1",
            tools=(name,),
            tools_sha256=_digest(definition),
        )
        for service, name, definition in zip(("a", "b"), names, definitions, strict=True)
    )
    return bindings


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


def _git_provider(tmp_path: Path) -> tuple[ToolBinding, Path]:
    module = f"external_{hashlib.sha256(str(tmp_path).encode()).hexdigest()[:8]}"
    checkout = tmp_path / "provider-checkout"
    package = checkout / "src" / module
    package.mkdir(parents=True)
    definitions = ({"type": "function", "function": {"name": "lookup_a", "parameters": {"type": "object"}}},)
    (package / "__init__.py").write_text(
        "class ServiceA:\n"
        "    ACTION_INTERFACE = 'a:v1'\n"
        "    SEED_SHA256 = 'a' * 64\n"
        "    PROVIDER_REVISION = 'external-a-v1'\n"
        f"    TOOL_DEFINITIONS = {definitions!r}\n"
        "    def __init__(self, *, seed_sha256, action_interface):\n"
        "        assert seed_sha256 == self.SEED_SHA256 and action_interface == self.ACTION_INTERFACE\n"
        "    async def native_tool_definitions(self): return list(self.TOOL_DEFINITIONS)\n"
        "    async def dispatch_action(self, name, arguments, call_id): return '{}'\n"
    )
    (checkout / "LICENSE").write_text("Apache-2.0\n")
    subprocess.run(("git", "init", "-q", str(checkout)), check=True)
    subprocess.run(
        ("git", "-C", str(checkout), "remote", "add", "origin", "https://github.com/example/source.git"), check=True
    )
    subprocess.run(("git", "-C", str(checkout), "add", "."), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "source",
        ),
        check=True,
    )
    commit = subprocess.check_output(("git", "-C", str(checkout), "rev-parse", "HEAD"), text=True).strip()
    binding = ToolBinding(
        action_interface="a:v1",
        seed_sha256="a" * 64,
        provider=f"python+git+https://github.com/example/source@{commit}:{module}:ServiceA",
        provider_revision="external-a-v1",
        tools=("lookup_a",),
        tools_sha256=_digest(definitions),
    )
    return binding, checkout


async def test_git_provider_exports_verified_source_snapshot_and_runs_offline(tmp_path, monkeypatch):
    binding, checkout = _git_provider(tmp_path)
    module_name = binding.provider.rsplit(":", maxsplit=2)[1]
    monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
    _, second = _external_services(tmp_path, monkeypatch)
    specification = _specification()
    environment_config = HarborEnvironmentConfig(tool_providers={"a": binding, "b": second})
    convention = PlainText(id="plain")

    assert compatible_lowerings(
        specification,
        (convention,),
        (environment_config,),
        trusted_provider_sources={"a": checkout},
    )
    task = lower_to_harbor(
        specification,
        convention,
        environment_config,
        tmp_path / "task",
        trusted_provider_sources={"a": checkout},
    )
    source = task / "environment" / "provider_sources" / "a"
    validate_staged_git_provider(binding.provider, source)
    assert (source / "LICENSE").read_text() == "Apache-2.0\n"
    assert (source / "src" / binding.provider.rsplit(":", 2)[1] / "__init__.py").is_file()

    requests = []

    def respond(request, timeout):
        payload = json.loads(request.data)
        requests.append(payload)
        if any(message.get("tool_call_id") == "git1" for message in payload["messages"]):
            message = {"role": "assistant", "content": "done"}
        else:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "git1", "type": "function", "function": {"name": "lookup_a", "arguments": "{}"}}],
            }
        return BytesIO(json.dumps({"choices": [{"message": message}]}).encode())

    monkeypatch.setattr("taskcompendium.harbor.adapter.urllib.request.urlopen", respond)
    launch = ChatLaunch(model="model", api_base="https://example.invalid")
    results = await asyncio.gather(
        run_trial(task, environment_config, launch, tmp_path / "trials", "run-1"),
        run_trial(task, environment_config, launch, tmp_path / "trials", "run-2"),
    )
    assert [result.verifier_result.rewards for result in results] == [{"reward": 1.0}] * 2
    assert [tool["function"]["name"] for tool in requests[0]["tools"]] == ["lookup_a", "lookup_b"]
    assert (
        sum(any(message.get("tool_call_id") == "git1" for message in request["messages"]) for request in requests) == 2
    )
    validate_staged_git_provider(binding.provider, source)

    (source / "LICENSE").write_text("changed\n")
    with pytest.raises(ValueError, match="source digest mismatch"):
        await run_trial(
            task,
            environment_config,
            launch,
            tmp_path / "trials",
            "tampered",
        )
    manifest_path = source / SOURCE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    changed = b"changed\n"
    license_entry = next(item for item in manifest["files"] if item["path"] == "LICENSE")
    license_entry["sha256"] = hashlib.sha256(changed).hexdigest()
    license_entry["git_blob"] = hashlib.sha1(b"blob 8\0" + changed, usedforsecurity=False).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tree differs from pinned commit"):
        await run_trial(task, environment_config, launch, tmp_path / "trials", "manifest-tampered")
    shutil.rmtree(task)
    assert not task.exists()


def test_git_provider_rejects_changed_source_before_export_or_launch(tmp_path, monkeypatch):
    binding, checkout = _git_provider(tmp_path)
    _, second = _external_services(tmp_path, monkeypatch)
    (checkout / "LICENSE").write_text("changed\n")
    with pytest.raises(ValueError, match="must be clean"):
        lower_to_harbor(
            _specification(),
            PlainText(id="plain"),
            HarborEnvironmentConfig(tool_providers={"a": binding, "b": second}),
            tmp_path / "task",
            trusted_provider_sources={"a": checkout},
        )
    assert not (tmp_path / "task").exists()


def test_external_services_export_as_one_chat_task(tmp_path, monkeypatch):
    bindings = _external_services(tmp_path, monkeypatch)
    environment_config = HarborEnvironmentConfig(tool_providers=dict(zip(("a", "b"), bindings, strict=True)))
    specification = _specification()
    convention = AnswerCall(id="answer-call")

    assert compatible_lowerings(specification, (convention,), (environment_config,))
    task_dir = lower_to_harbor(specification, convention, environment_config, tmp_path / "task")
    exported = json.loads((task_dir / "environment_config.json").read_text())
    assert list(exported["tool_providers"]) == ["a", "b"]
    assert [binding["provider"] for binding in exported["tool_providers"].values()] == [
        binding.provider for binding in bindings
    ]


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
        update={"final_tools": FinalTools(functions=(FunctionDefinition(name="lookup", parameters={"type": "object"}),))}
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
    with pytest.raises(ValueError, match="agent-visible files"):
        lower_to_harbor(
            _specification().model_copy(
                update={
                    "resources": (
                        TaskResource(path="input.txt", visibility=ResourceVisibility.AGENT, content="task input"),
                    )
                }
            ),
            convention,
            environment_config,
            tmp_path / "resource",
        )
