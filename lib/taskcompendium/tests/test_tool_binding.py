# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provider selection at the boundary between private tasks and Harbor export."""

import asyncio
import hashlib
import inspect
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType

import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from upath import UPath

from taskcompendium.grading import exact_answer, structured_exact
from taskcompendium.harbor.adapter import CompositeToolEnvironment
from taskcompendium.harbor.runner import ChatLaunch, run_trial
from taskcompendium.lowering import HarborEnvironmentConfig, ToolBinding, compatible_lowerings, lower_to_harbor
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
from taskcompendium.provider_sources import (
    SOURCE_MANIFEST,
    ToolProviderCache,
    stage_git_provider,
    validate_staged_git_provider,
)
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


def _git_provider(
    tmp_path: Path, *, module: str | None = None, revision: str = "first", failure: str | None = None
) -> tuple[ToolBinding, Path]:
    module = module or f"external_{hashlib.sha256(str(tmp_path).encode()).hexdigest()[:8]}"
    checkout = tmp_path / "provider-checkout"
    package = checkout / "src" / module
    package.mkdir(parents=True)
    definitions = ({"type": "function", "function": {"name": "lookup_a", "parameters": {"type": "object"}}},)
    (package / "__init__.py").write_text(
        "import asyncio\n"
        "from pathlib import Path\n"
        "from .counter import next_call\n"
        "class ServiceA:\n"
        "    ACTION_INTERFACE = 'a:v1'\n"
        "    SEED_SHA256 = 'a' * 64\n"
        "    PROVIDER_REVISION = 'external-a-v1'\n"
        f"    TOOL_DEFINITIONS = {definitions!r}\n"
        "    def __init__(self, *, seed_sha256, action_interface):\n"
        "        assert seed_sha256 == self.SEED_SHA256 and action_interface == self.ACTION_INTERFACE\n"
        "        self.calls = 0\n"
        "    async def native_tool_definitions(self): return list(self.TOOL_DEFINITIONS)\n"
        "    async def dispatch_action(self, name, arguments, call_id):\n"
        "        self.calls += 1\n"
        f"        return str(({revision!r}, self.calls, next_call()))\n"
        "    async def start(self):\n"
        + (
            "        raise RuntimeError('start failed')\n"
            if failure == "start"
            else "        raise asyncio.CancelledError()\n" if failure == "cancel" else "        pass\n"
        )
        + "    async def stop(self):\n"
        "        assert Path(__file__).is_file()\n"
        + ("        raise RuntimeError('stop failed')\n" if failure == "stop" else "        pass\n")
    )
    (package / "counter.py").write_text(
        "calls = 0\ndef next_call():\n    global calls\n    calls += 1\n    return calls\n"
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


@pytest.fixture
def git_provider_task(tmp_path, monkeypatch):
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
    return task, binding, environment_config


def test_git_provider_exports_verified_source_snapshot(git_provider_task):
    task, binding, _ = git_provider_task
    source = task / "environment" / "provider_sources" / "a"
    validate_staged_git_provider(binding.provider, source)
    assert (source / "LICENSE").read_text() == "Apache-2.0\n"
    assert (source / "src" / binding.provider.rsplit(":", 2)[1] / "__init__.py").is_file()


async def test_git_provider_runs_concurrent_offline_trials(git_provider_task, tmp_path, monkeypatch):
    task, _, environment_config = git_provider_task
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


@pytest.mark.parametrize("rewrite_manifest", (False, True), ids=("source", "source-and-manifest"))
async def test_git_provider_tampering_prevents_launch(git_provider_task, tmp_path, rewrite_manifest):
    task, _, environment_config = git_provider_task
    source = task / "environment" / "provider_sources" / "a"
    changed = b"changed\n"
    (source / "LICENSE").write_bytes(changed)
    if rewrite_manifest:
        manifest_path = source / SOURCE_MANIFEST
        manifest = json.loads(manifest_path.read_text())
        license_entry = next(item for item in manifest["files"] if item["path"] == "LICENSE")
        (source / "LICENSE").rename(source / "LICENSE.changed")
        license_entry["path"] = "LICENSE.changed"
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tree differs from pinned commit"):
        await run_trial(
            task,
            environment_config,
            ChatLaunch(model="model", api_base="https://example.invalid"),
            tmp_path / "trials",
            "tampered",
        )


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


@pytest.mark.parametrize(
    "changed",
    (
        {"action_interface": "a:v2"},
        {"seed_sha256": "c" * 64},
        {"provider_revision": "wrong-revision"},
        {"tools_sha256": "0" * 64},
    ),
)
def test_git_provider_surface_mismatch_excludes_candidate(tmp_path, monkeypatch, changed):
    binding, checkout = _git_provider(tmp_path)
    binding = binding.model_copy(update=changed)
    _, second = _external_services(tmp_path, monkeypatch)
    environment_config = HarborEnvironmentConfig(tool_providers={"a": binding, "b": second})
    specification = _specification().model_copy(
        update={
            "tool_providers": {
                "a": ProviderRequirement(action_interface=binding.action_interface, seed_sha256=binding.seed_sha256),
                "b": _specification().tool_providers["b"],
            }
        }
    )
    assert not compatible_lowerings(
        specification, (PlainText(id="plain"),), (environment_config,), trusted_provider_sources={"a": checkout}
    )


def test_git_provider_without_state_excludes_state_candidate(tmp_path, monkeypatch):
    binding, checkout = _git_provider(tmp_path)
    _, second = _external_services(tmp_path, monkeypatch)
    environment_config = HarborEnvironmentConfig(tool_providers={"a": binding, "b": second})
    specification = _specification().model_copy(
        update={"answer_type": AnswerType.STATE, "verifier": structured_exact({})}
    )
    assert not compatible_lowerings(
        specification,
        (ProviderState(id="state", provider="a"),),
        (environment_config,),
        trusted_provider_sources={"a": checkout},
    )


@pytest.mark.parametrize("name", ("../../../outside", "/absolute", "nested/provider", "..", r"..\outside"))
def test_escaping_provider_name_prevents_export(tmp_path, monkeypatch, name):
    binding, checkout = _git_provider(tmp_path)
    with pytest.raises(ValueError):
        environment_config = HarborEnvironmentConfig(tool_providers={name: binding})
        lower_to_harbor(
            _specification(),
            PlainText(id="plain"),
            environment_config,
            tmp_path / "task",
            trusted_provider_sources={name: checkout},
        )
    assert not (tmp_path / "task").exists()
    assert not (tmp_path / "outside").exists()


async def test_cache_isolates_revisions_and_fresh_instances(tmp_path):
    first, first_checkout = _git_provider(tmp_path / "first", module="same_package", revision="first")
    second, second_checkout = _git_provider(tmp_path / "second", module="same_package", revision="second")
    first_source, second_source = tmp_path / "first-source", tmp_path / "second-source"
    stage_git_provider(first.provider, first_checkout, first_source)
    stage_git_provider(second.provider, second_checkout, second_source)
    with ToolProviderCache() as first_cache, ToolProviderCache() as second_cache:
        first_staged = first_cache.stage(first.provider, first_source)
        second_staged = second_cache.stage(second.provider, second_source)
        first_provider = first_cache.load(
            first_staged, seed_sha256=first.seed_sha256, action_interface=first.action_interface
        )
        second_provider = second_cache.load(
            second_staged, seed_sha256=second.seed_sha256, action_interface=second.action_interface
        )
        fresh_provider = first_cache.load(
            first_staged, seed_sha256=first.seed_sha256, action_interface=first.action_interface
        )
        responses = await asyncio.gather(
            first_provider.dispatch_action("lookup_a", "{}", "first"),
            second_provider.dispatch_action("lookup_a", "{}", "second"),
        )
        assert responses == ["('first', 1, 1)", "('second', 1, 1)"]
        assert await fresh_provider.dispatch_action("lookup_a", "{}", "fresh") == "('first', 1, 2)"
        module_names = [staged.factory.__module__ for staged in (first_staged, second_staged)]
        source_files = [Path(inspect.getfile(staged.factory)) for staged in (first_staged, second_staged)]
    assert all(name not in sys.modules and name + ".counter" not in sys.modules for name in module_names)
    assert all(not path.exists() for path in source_files)
    assert (first_source / "LICENSE").is_file() and (second_source / "LICENSE").is_file()


@pytest.mark.parametrize("failure", (None, "start", "stop", "cancel"))
async def test_environment_releases_imports_after_provider_cleanup(tmp_path, failure):
    binding, checkout = _git_provider(tmp_path, failure=failure)
    environment_dir = tmp_path / "environment"
    stage_git_provider(binding.provider, checkout, environment_dir / "provider_sources" / "a")
    composite = CompositeToolEnvironment(
        environment_dir=environment_dir,
        environment_name="task",
        session_id="trial",
        trial_paths=TrialPaths(UPath(tmp_path / "trial")),
        task_env_config=EnvironmentConfig(),
        tool_providers={"a": binding.model_dump(mode="json")},
    )
    tool_provider = composite.providers["a"]
    module_name = type(tool_provider).__module__
    source_file = Path(inspect.getfile(type(tool_provider)))
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await composite.start(False)
    elif failure == "start":
        with pytest.raises(RuntimeError, match="start failed"):
            await composite.start(False)
    else:
        await composite.start(False)
        if failure == "stop":
            with pytest.raises(ExceptionGroup, match="Provider cleanup failed"):
                await composite.stop(False)
        else:
            await composite.stop(False)
    assert module_name not in sys.modules and module_name + ".counter" not in sys.modules
    assert not source_file.exists()


async def test_changed_tool_schema_fails_before_trial_or_endpoint(git_provider_task, tmp_path, monkeypatch):
    task, _, environment_config = git_provider_task
    changed = environment_config.model_copy(
        update={
            "tool_providers": {
                **environment_config.tool_providers,
                "a": environment_config.tool_providers["a"].model_copy(update={"tools_sha256": "0" * 64}),
            }
        }
    )
    (task / "environment_config.json").write_text(changed.model_dump_json())

    def unexpected_request(request, timeout):
        raise AssertionError("Invalid provider must not reach the model endpoint")

    monkeypatch.setattr("taskcompendium.harbor.adapter.urllib.request.urlopen", unexpected_request)
    with pytest.raises(ValueError, match="tool schemas differ"):
        await run_trial(
            task, changed, ChatLaunch(model="model", api_base="https://example.invalid"), tmp_path / "trials", "invalid"
        )
    assert not (tmp_path / "trials").exists()
