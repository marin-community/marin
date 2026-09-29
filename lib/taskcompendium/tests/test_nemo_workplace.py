# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Workplace row 0 import and mutable provider behavior."""

import asyncio
import hashlib
import json
import shutil
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from threading import Lock, Thread

import pytest

from taskcompendium.harbor.runner import AgentStrategy, ChatLaunch, run_trial
from taskcompendium.importers.nemo_workplace import DATASET_REVISION, ROW_SHA256, import_row
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.providers.nemo_workplace.provider import (
    NemoWorkplaceEnvironment,
    _seed_digest,
    expected_state_json,
)
from taskcompendium.providers.nemo_workplace.tools import get_tools
from taskcompendium.resources import ResourceVisibility

FIXTURES = Path(__file__).parent / "fixtures/nemo"
ROW = files("taskcompendium.importers").joinpath("data/workplace-0.json")
PROVENANCE = FIXTURES / "workplace-0.provenance.json"


def _environment() -> NemoWorkplaceEnvironment:
    environment = object.__new__(NemoWorkplaceEnvironment)
    environment.tool_env = get_tools()
    environment.trace = []
    return environment


def _source() -> tuple[bytes, dict]:
    data = ROW.read_bytes()
    return data, json.loads(data)


def test_workplace_import_pins_row_tool_surface_and_private_state():
    data, row = _source()
    specification, convention, binding = import_row(data)
    assert hashlib.sha256(data).hexdigest() == ROW_SHA256
    assert _seed_digest() == binding.seed_sha256
    assert specification.source.revision == DATASET_REVISION
    assert specification.answer_type.value == convention.answer_format.value == "state"
    assert len(binding.tools) == len(row["responses_create_params"]["tools"]) == 27
    assert {resource.visibility for resource in specification.resources} == {ResourceVisibility.VERIFIER}
    assert all(resource.path not in specification.instructions for resource in specification.resources)
    assert "ground_truth" not in specification.instructions
    assert "source-row.json" in {resource.path for resource in specification.resources}
    provenance = json.loads(PROVENANCE.read_text())
    assert provenance["source_fixture"]["fixture_raw_sha256"] == ROW_SHA256
    private_provenance = next(
        resource for resource in specification.resources if resource.path == "source-provenance.json"
    )
    source_record = json.loads(private_provenance.content)
    assert (source_record["dataset_license"], source_record["dataset_owner"]) == (
        "CC-BY-4.0",
        "NVIDIA Corporation",
    )


def test_workplace_import_rejects_unpinned_row_and_changed_tools(monkeypatch):
    data, row = _source()
    with pytest.raises(ValueError, match="pinned raw digest"):
        import_row(data + b" ")
    row["responses_create_params"]["tools"][0]["name"] = "wrong_tool"
    changed = json.dumps(row).encode()
    monkeypatch.setattr("taskcompendium.importers.nemo_workplace.ROW_SHA256", hashlib.sha256(changed).hexdigest())
    with pytest.raises(ValueError, match="pinned provider"):
        import_row(changed)


def test_workplace_wheel_installs_every_pinned_seed_file(tmp_path):
    """Build under the repository's CSV ignore rule, as a git dependency build does."""
    package_root = Path(__file__).parents[1]
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".gitignore").write_text("*.csv\n")
    project = checkout / "taskcompendium"
    project.mkdir()
    shutil.copy2(package_root / "pyproject.toml", project / "pyproject.toml")
    shutil.copytree(package_root / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    wheel_dir = tmp_path / "wheels"
    build = subprocess.run(
        ["uv", "build", "--offline", "--project", str(project), "--wheel", "--out-dir", str(wheel_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    wheel = next(wheel_dir.glob("taskcompendium-*.whl"))
    installed = tmp_path / "installed"
    install = subprocess.run(
        ["uv", "pip", "install", "--offline", "--no-deps", "--target", str(installed), str(wheel)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert install.returncode == 0, install.stderr

    provenance = json.loads(PROVENANCE.read_text())
    seed = installed / "taskcompendium/providers/nemo_workplace/vendor/csv_data"
    assert {path.relative_to(seed).as_posix() for path in seed.rglob("*.csv")} == set(provenance["seed"]["files"])
    for relative, record in provenance["seed"]["files"].items():
        assert hashlib.sha256((seed / relative).read_bytes()).hexdigest() == record["raw_sha256"]


async def test_workplace_success_wrong_and_noop_state():
    _, row = _source()
    gold = row["ground_truth"]
    expected = expected_state_json(gold)
    success, wrong, noop = (_environment() for _ in range(3))
    action = gold[0]
    await success.dispatch_action(action["name"], action["arguments"], "call-1")
    await wrong.dispatch_action(
        action["name"],
        '{"email_id":"00000057","body":"Thanks for the update - I will not follow up."}',
        "call-1",
    )
    await noop.dispatch_action(
        "email_get_email_information_by_id", '{"email_id":"00000057","field":"subject"}', "call-1"
    )
    assert success.grade_state(expected) == 1.0
    assert wrong.grade_state(expected) == 0.0
    assert noop.grade_state(expected) == 0.0


async def test_workplace_tool_error_recovers_and_retains_call_order():
    _, row = _source()
    action = row["ground_truth"][0]
    environment = _environment()
    error = await environment.dispatch_action(action["name"], '{"email_id":"00000057","unknown":"x"}', "call-bad")
    success = await environment.dispatch_action(action["name"], action["arguments"], "call-good")
    assert "Error executing tool" in error
    assert [entry["call_id"] for entry in environment.trace] == ["call-bad", "call-good"]
    assert [entry["output"] for entry in environment.trace] == [error, success]
    assert environment.grade_state(expected_state_json(row["ground_truth"])) == 1.0


async def test_workplace_concurrent_trials_start_from_fresh_seed():
    _, row = _source()
    action = row["ground_truth"][0]
    first, second = _environment(), _environment()
    await asyncio.gather(
        first.dispatch_action(action["name"], action["arguments"], "first"),
        second.dispatch_action(
            "email_get_email_information_by_id", '{"email_id":"00000057","field":"subject"}', "second"
        ),
    )
    expected = expected_state_json(row["ground_truth"])
    assert (first.grade_state(expected), second.grade_state(expected)) == (1.0, 0.0)


async def test_workplace_harbor_scripted_endpoint_recovers_after_tool_error(tmp_path):
    data, row = _source()
    specification, convention, binding = import_row(data)
    task_dir = lower_to_harbor(specification, convention, binding, tmp_path / "task")
    gold = row["ground_truth"][0]
    calls = [
        ('{"email_id":"00000057","unknown":"x"}', "bad"),
        (gold["arguments"], "good"),
    ]
    requests = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            if len(requests) <= len(calls):
                arguments, label = calls[len(requests) - 1]
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{label}",
                            "type": "function",
                            "function": {"name": gold["name"], "arguments": arguments},
                        }
                    ],
                }
            else:
                message = {"role": "assistant", "content": "Done."}
            body = json.dumps(
                {
                    "id": f"chatcmpl-{len(requests)}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "fixture",
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = await run_trial(
            task_dir,
            binding,
            ChatLaunch(
                model="fixture",
                api_base=f"http://127.0.0.1:{server.server_port}/v1",
                strategy=AgentStrategy.STATEFUL_TOOLS,
                max_turns=4,
            ),
            tmp_path / "trials",
            "workplace",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert len(requests) == 3
    assert len(requests[0]["tools"]) == 27
    assert requests[1]["messages"][-1]["tool_call_id"] == "call-bad"
    assert "Error executing tool" in requests[1]["messages"][-1]["content"]
    assert requests[2]["messages"][-1]["tool_call_id"] == "call-good"
    assert "successfully" in requests[2]["messages"][-1]["content"]
    assert "ground_truth" not in json.dumps(requests)
    metadata = result.agent_result.metadata
    assert len(metadata["tool_definitions"]) == 27
    assert [action["call_id"] for action in metadata["tools"]] == ["call-bad", "call-good"]
    assert [action["observation"] for action in metadata["tools"]] == [
        requests[1]["messages"][-1]["content"],
        requests[2]["messages"][-1]["content"],
    ]
    assert [message["role"] for message in metadata["all_messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]


async def test_workplace_harbor_trials_are_fresh_and_concurrent(tmp_path):
    specification, convention, binding = import_row(ROW.read_bytes())
    task_dir = lower_to_harbor(specification, convention, binding, tmp_path / "task")
    gold = json.loads(ROW.read_text())["ground_truth"][0]
    turns: dict[str, int] = {}
    lock = Lock()

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            model = payload["model"]
            with lock:
                turns[model] = turns.get(model, 0) + 1
                turn = turns[model]
            if turn == 1:
                if model.startswith("good"):
                    name, arguments = gold["name"], gold["arguments"]
                else:
                    name, arguments = "email_get_email_information_by_id", '{"email_id":"00000057","field":"subject"}'
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"{model}-call",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            else:
                message = {"role": "assistant", "content": "Done."}
            body = json.dumps(
                {
                    "id": f"chatcmpl-{model}-{turn}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def trial(model: str):
        launch = ChatLaunch(
            model=model,
            api_base=f"http://127.0.0.1:{server.server_port}/v1",
            strategy=AgentStrategy.STATEFUL_TOOLS,
            max_turns=3,
        )
        return await run_trial(task_dir, binding, launch, tmp_path / "trials", model)

    try:
        good, noop = await asyncio.gather(trial("good-1"), trial("noop"))
        good_again = await trial("good-2")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert all(result.exception_info is None for result in (good, noop, good_again))
    assert [result.verifier_result.rewards for result in (good, noop, good_again)] == [
        {"reward": 1.0},
        {"reward": 0.0},
        {"reward": 1.0},
    ]
