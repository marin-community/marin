# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A pinned Workplace row imported into a real container-backed Harbor trial."""

import asyncio
import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from taskcompendium.container_service import ContainerService, ContainerToolProvider
from taskcompendium.harbor.runner import ChatLaunch, run_trial
from taskcompendium.importers import nemo_workplace
from taskcompendium.importers.nemo_workplace import (
    PROVIDER,
    PROVIDER_GIT_REVISION,
    PROVIDER_REPOSITORY,
    import_row,
)
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import ConversationTrace, ToolResult
from taskcompendium.provider_sources import stage_git_provider
from taskcompendium.tool_provider import tool_schema_sha256

ROW = Path(__file__).parent / "fixtures/nemo/workplace-row0.jsonl"


@pytest.fixture(scope="module")
def provider_source(tmp_path_factory):
    checkout = tmp_path_factory.mktemp("workplace-source") / "source"
    subprocess.run(("git", "clone", "--quiet", PROVIDER_REPOSITORY, str(checkout)), check=True)
    subprocess.run(("git", "-C", str(checkout), "checkout", "--quiet", PROVIDER_GIT_REVISION), check=True)
    snapshot = tmp_path_factory.mktemp("workplace-snapshot") / "provider"
    stage_git_provider(PROVIDER, checkout, snapshot)
    return snapshot


@pytest.fixture(scope="module")
def workplace_runtime(request):
    image = request.config.getoption("provider_image")
    if image is None:
        raise ValueError("Docker Workplace tests require --provider-image=<repository@sha256:digest>")
    return ContainerService(image=image, command=("python", "-m", "nemo_workplace.server"))


@contextmanager
def policy(actions):
    class Endpoint(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            observations = sum(message["role"] == "tool" for message in request["messages"])
            if observations < len(actions):
                action = actions[observations]
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{observations}",
                            "type": "function",
                            "function": action,
                        }
                    ],
                }
            else:
                message = {"role": "assistant", "content": "Done."}
            payload = json.dumps({"choices": [{"message": message}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format_string, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.docker
@pytest.mark.parametrize("mode,reward", (("correct", 1.0), ("wrong", 0.0), ("noop", 0.0), ("recovery", 1.0)))
async def test_workplace_import_container_harbor_state_outcomes(
    tmp_path, provider_source, workplace_runtime, mode, reward
):
    raw = ROW.read_bytes()
    specification, convention, config = import_row(raw, provider_source, workplace_runtime)
    actions = json.loads(raw)["ground_truth"]
    if mode == "noop":
        actions = []
    elif mode == "wrong":
        actions = actions[:-1]
    elif mode == "recovery":
        actions = [{"name": actions[0]["name"], "arguments": "{}"}, *actions]
    task = lower_to_harbor(specification, convention, config, tmp_path / "task")
    with policy(actions) as endpoint:
        result = await run_trial(
            task, config, ChatLaunch(model="scripted", api_base=endpoint), tmp_path / "trials", "run"
        )
    assert result.exception_info is None
    assert result.verifier_result.rewards == {"reward": reward}
    trace = ConversationTrace.model_validate_json((tmp_path / "trials/run/agent/submission.json").read_text())
    assert [event.call_id for event in trace.events if isinstance(event, ToolResult)] == [
        f"call-{i}" for i in range(len(actions))
    ]
    provider_trace = [
        json.loads(line) for line in (tmp_path / "trials/run/agent/provider-workplace.jsonl").read_text().splitlines()
    ]
    assert [
        event["request"]["params"]["call_id"]
        for event in provider_trace
        if event.get("request", {}).get("method") == "call"
    ] == [f"call-{i}" for i in range(len(actions))]
    assert not list((task / "environment").iterdir())


@pytest.mark.docker
async def test_workplace_concurrent_and_fresh_harbor_trials_are_isolated(tmp_path, provider_source, workplace_runtime):
    raw = ROW.read_bytes()
    specification, convention, config = import_row(raw, provider_source, workplace_runtime)
    task = lower_to_harbor(specification, convention, config, tmp_path / "task")
    with policy(json.loads(raw)["ground_truth"]) as endpoint:
        launch = ChatLaunch(model="scripted", api_base=endpoint)
        results = await asyncio.gather(
            *(run_trial(task, config, launch, tmp_path / "trials", f"trial-{i}") for i in range(2))
        )
        results.append(await run_trial(task, config, launch, tmp_path / "trials", "fresh"))
    assert [result.exception_info for result in results] == [None] * 3
    assert [result.verifier_result.rewards for result in results] == [{"reward": 1.0}] * 3


@pytest.mark.docker
async def test_workplace_service_has_no_workspace_access_or_network(tmp_path, provider_source, workplace_runtime):
    _, _, config = import_row(ROW.read_bytes(), provider_source, workplace_runtime)
    binding = config.tool_providers["workplace"]
    provider = ContainerToolProvider(
        runtime=workplace_runtime,
        identity=binding.identity,
        tool_definitions=list(binding.tool_definitions),
        trace_path=tmp_path / "service.jsonl",
    )
    await provider.start()
    try:
        inspection = json.loads(subprocess.check_output(("docker", "inspect", provider.container_name)))[0]
        host = inspection["HostConfig"]
        assert host["NetworkMode"] == "none"
        assert host["Binds"] is None and not host["Privileged"]
        assert host["ReadonlyRootfs"] and host["CapDrop"] == ["ALL"]
        assert inspection["Config"]["User"] == "65532:65532"
        assert all(mount["Type"] != "bind" for mount in inspection["Mounts"])
    finally:
        await provider.stop()


def test_import_retains_provider_source_until_expected_state_is_built(tmp_path, monkeypatch, runtime_factory):
    checkout = tmp_path / "checkout"
    package = checkout / "src" / "synthetic_workplace"
    package.mkdir(parents=True)
    interface, seed, revision_name = "synthetic:v1", "a" * 64, "synthetic-v1"
    schema = {"name": "finish", "parameters": {"type": "object"}}
    definitions = [{"type": "function", "function": schema}]
    (package / "__init__.py").write_text(
        "import importlib\nimport json\nfrom pathlib import Path\n"
        f"ACTION_INTERFACE = {interface!r}\nSEED_SHA256 = {seed!r}\n"
        f"PROVIDER_REVISION = {revision_name!r}\n"
        f"TOOLS_SHA256 = {tool_schema_sha256(definitions)!r}\n"
        f"TOOL_DEFINITIONS = {definitions!r}\n"
        "REQUEST_PARALLEL_TOOL_CALLS = False\nREQUEST_TEMPERATURE = 0\n"
        "class ToolProvider: pass\n"
        "def _seed_digest(): return SEED_SHA256\n"
        f"def get_tools(): return {{'schemas': [{schema!r}]}}\n"
        "def expected_state_json(actions):\n"
        "    assert importlib.import_module(__name__) is not None\n"
        "    return json.dumps({'snapshot_present': Path(__file__).is_file()})\n"
    )
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "add", "origin", "https://github.com/example/synthetic"], check=True
    )
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "Synthetic provider",
        ],
        check=True,
    )
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    locator = f"python+git+https://github.com/example/synthetic@{revision}:synthetic_workplace:ToolProvider"
    monkeypatch.setattr(nemo_workplace, "PROVIDER", locator)
    source = tmp_path / "source"
    stage_git_provider(locator, checkout, source)
    row = json.dumps(
        {
            "id": 0,
            "environment_name": "workplace_assistant",
            "category": "workplace_assistant_calendar",
            "responses_create_params": {
                "input": [{"role": "user", "content": "Finish the task."}],
                "tools": [schema],
                "parallel_tool_calls": False,
                "temperature": 0,
            },
            "ground_truth": [],
        }
    ).encode()
    monkeypatch.setattr(nemo_workplace, "ROW_SHA256_BY_ID", {0: hashlib.sha256(row).hexdigest()})
    before = set(sys.modules)

    runtime = runtime_factory(interface, seed, revision_name, definitions)
    imported = import_row(row, source, runtime)

    assert json.loads(imported.specification.verifier.parameters_json)["expected"] == {"snapshot_present": True}
    assert not any(name.startswith("_taskcompendium_providers_") for name in set(sys.modules) - before)
