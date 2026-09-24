# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for dashboard tool schemas, execution, and agent workspaces."""

import asyncio
import os
import subprocess
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import marin.inference.repository_snapshot as repository_snapshot
import pytest
import requests
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.inference.chat_template_protocol import ChatTemplateProtocol, ToolCallFormat, protocol_for_chat_template
from marin.inference.dashboard_server import ServingInfo, bind_serving_socket, build_dashboard_app, serve_app_background
from marin.inference.python_tool_routes import TOOL_WORKER_TIMEOUT
from marin.inference.python_tools import python_tools_from_source
from marin.inference.repository_snapshot import RepositorySnapshotTooLarge, clone_repository_snapshot

from experiments.llama import llama3_instruct_trainable_chat_template
from experiments.sft.delphi_chat_template import DELPHI_V0_CHAT_TEMPLATE

MULTIPLY_TOOL_SOURCE = '''
def multiply(value: int, factor: int = 2) -> int:
    """Multiply a value by an optional factor."""
    print("running multiply")
    return value * factor
'''

HOST_ENVIRONMENT_TOOL_SOURCE = '''
def host_environment(name: str) -> str | None:
    """Return a simulated environment variable."""
    import os
    return os.getenv(name)
'''

CPU_EXHAUSTION_TOOL_SOURCE = """
def spin() -> int:
    while True:
        pass
"""

DASHBOARD_REQUEST_TIMEOUT = TOOL_WORKER_TIMEOUT + 5


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(["git", "-C", str(repository), *arguments], check=True, capture_output=True)
    return result.stdout.decode()


def _repository_with_history(repository: Path) -> None:
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    (repository / "README.md").write_text("first revision\n")
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=Original Author",
        "-c",
        "user.email=author@example.com",
        "commit",
        "-m",
        "initial import",
    )
    (repository / "README.md").write_text("second revision\n")
    (repository / "src").mkdir()
    (repository / "src/main.py").write_text("print('hello')\n")
    (repository / "image.bin").write_bytes(b"\xff\x00")
    (repository / "node_modules").mkdir()
    (repository / "node_modules/dependency.js").write_text("ignored\n")
    _git(repository, "add", ".")
    parent_commit = _git(repository, "rev-parse", "HEAD").strip()
    _git(repository, "update-index", "--add", "--cacheinfo", f"160000,{parent_commit},vendor")
    _git(
        repository,
        "-c",
        "user.name=Second Author",
        "-c",
        "user.email=second@example.com",
        "commit",
        "-m",
        "add program",
    )


def _clone_local_repository(source: Path, destination: Path) -> None:
    subprocess.run(
        ["git", "clone", "--quiet", "--no-checkout", "--local", "--", str(source), str(destination)],
        check=True,
        capture_output=True,
    )


def test_python_tool_source_requires_typed_functions_and_validates_arguments():
    [tool] = python_tools_from_source(MULTIPLY_TOOL_SOURCE)

    assert tool.name == "multiply"
    assert tool.validate_arguments({"value": "3"}) == {"value": 3, "factor": 2}
    definition = tool.definition()
    assert definition["type"] == "function"
    function = definition["function"]
    assert isinstance(function, dict)
    assert function["name"] == "multiply"
    assert function["description"] == "Multiply a value by an optional factor."
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    assert parameters["type"] == "object"
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    value_schema = properties["value"]
    assert isinstance(value_schema, dict)
    assert value_schema["type"] == "integer"
    factor_schema = properties["factor"]
    assert isinstance(factor_schema, dict)
    assert factor_schema["type"] == "integer"
    assert factor_schema["default"] == 2
    assert parameters["required"] == ["value"]

    with pytest.raises(ValueError):
        python_tools_from_source("def untyped(value: int):\n    return value\n")

    with pytest.raises(ValueError):
        python_tools_from_source("async def asynchronous(value: int) -> int:\n    return value\n")

    with pytest.raises(ValueError):
        python_tools_from_source("CONSTANT = 1\n")


def test_python_tool_schema_rejects_executable_defaults_without_running_them(tmp_path):
    marker = tmp_path / "executed"
    source = f"""def unsafe(value: int = open({str(marker)!r}, "w").write("executed")) -> int:
    return value
"""

    with pytest.raises(ValueError):
        python_tools_from_source(source)

    assert not marker.exists()


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        (
            MARIN_CHAT_TEMPLATE,
            ChatTemplateProtocol(
                thinking_start="<|start_think|>",
                thinking_end="<|end_think|>",
                tool_call_start="<tool_call>",
                tool_call_end="</tool_call>",
                tool_call_format=ToolCallFormat.DELIMITED,
            ),
        ),
        (
            DELPHI_V0_CHAT_TEMPLATE,
            ChatTemplateProtocol(
                thinking_start="<|start_think|>",
                thinking_end="<|end_think|>",
                tool_call_start="<|tool_call|>",
                tool_call_end="<|tool_call_end|>",
                tool_call_format=ToolCallFormat.DELIMITED,
            ),
        ),
        (llama3_instruct_trainable_chat_template, ChatTemplateProtocol(tool_call_format=ToolCallFormat.JSON)),
    ],
    ids=["datakit", "delphi", "llama-json"],
)
def test_chat_template_protocol_matches_generated_output_format(template, expected):
    assert protocol_for_chat_template(template) == expected


@pytest.fixture
def dashboard_tool_base_url() -> Iterator[str]:
    dashboard_sock = bind_serving_socket("127.0.0.1", 0)
    dashboard_port = dashboard_sock.getsockname()[1]
    info = ServingInfo(
        model="fake-model",
        backend="vllm",
        tensor_parallel_size=1,
        max_model_len=4096,
        dtype="bfloat16",
        has_chat_template=True,
        endpoint="/serve/fake",
    )
    app = build_dashboard_app(
        upstream_base_url="http://127.0.0.1:1",
        model_id="fake-model",
        info=info,
    )

    with serve_app_background(app, dashboard_sock):
        yield f"http://127.0.0.1:{dashboard_port}"


def test_dashboard_executes_typed_python_tools_in_shellsim(dashboard_tool_base_url):
    definitions = requests.post(
        f"{dashboard_tool_base_url}/tools",
        json={"source": MULTIPLY_TOOL_SOURCE},
        timeout=DASHBOARD_REQUEST_TIMEOUT,
    )
    result = requests.post(
        f"{dashboard_tool_base_url}/tools/multiply",
        json={"source": MULTIPLY_TOOL_SOURCE, "arguments": {"value": "6", "factor": 7}},
        timeout=DASHBOARD_REQUEST_TIMEOUT,
    )
    invalid = requests.post(
        f"{dashboard_tool_base_url}/tools/multiply",
        json={"source": MULTIPLY_TOOL_SOURCE, "arguments": {"unknown": 1}},
        timeout=DASHBOARD_REQUEST_TIMEOUT,
    )

    assert definitions.json()[0]["function"]["name"] == "multiply"
    assert definitions.json()[0]["function"]["parameters"]["required"] == ["value"]
    assert result.json() == 42
    assert invalid.status_code == 422


def test_dashboard_shared_chat_round_trips_through_short_id(dashboard_tool_base_url):
    snapshot = {
        "version": 1,
        "title": "Billing analysis",
        "model": "fake-model",
        "messages": [
            {"role": "user", "content": "Why was I charged twice?"},
            {
                "role": "assistant",
                "content": "The second charge should be refunded.",
                "error": None,
            },
        ],
    }

    created = requests.post(f"{dashboard_tool_base_url}/chat-shares", json=snapshot, timeout=DASHBOARD_REQUEST_TIMEOUT)
    assert created.status_code == 201
    share_id = created.json()["id"]
    fetched = requests.get(f"{dashboard_tool_base_url}/chat-shares/{share_id}", timeout=DASHBOARD_REQUEST_TIMEOUT)

    assert fetched.status_code == 200
    assert len(share_id) == 16
    assert created.headers["cache-control"] == "no-store"
    assert fetched.json() == snapshot
    assert fetched.headers["cache-control"] == "no-store"


def test_dashboard_shared_chat_rejects_payload_over_512_kib(dashboard_tool_base_url):
    oversized = requests.post(
        f"{dashboard_tool_base_url}/chat-shares",
        json={"message": "x" * (512 * 1024)},
        timeout=DASHBOARD_REQUEST_TIMEOUT,
    )

    assert oversized.status_code == 413


def test_dashboard_shellsim_cannot_read_host_environment(dashboard_tool_base_url, monkeypatch):
    monkeypatch.setenv("MARIN_SHELLSIM_HOST_SECRET", "must-not-leak")
    host_environment = requests.post(
        f"{dashboard_tool_base_url}/tools/host_environment",
        json={
            "source": HOST_ENVIRONMENT_TOOL_SOURCE,
            "arguments": {"name": "MARIN_SHELLSIM_HOST_SECRET"},
        },
        timeout=DASHBOARD_REQUEST_TIMEOUT,
    )

    assert host_environment.json() is None


def test_dashboard_shellsim_enforces_cpu_limit(dashboard_tool_base_url):
    exhausted = requests.post(
        f"{dashboard_tool_base_url}/tools/spin",
        json={"source": CPU_EXHAUSTION_TOOL_SOURCE, "arguments": {}},
        timeout=DASHBOARD_REQUEST_TIMEOUT,
    )

    assert exhausted.status_code == 422
    assert "cpu_exhausted" in exhausted.json()["details"]


def test_dashboard_shell_workspace_replays_git_branch_workflow(dashboard_tool_base_url):
    files = {
        "calculator.py": "def add(left: int, right: int) -> int:\n    return left - right\n",
        "test_calculator.py": "from calculator import add\n\nassert add(17, 24) == 41\nprint('test passed')\n",
    }
    edit = (
        "git switch -c fix; "
        "sed -i 's/left - right/left + right/' calculator.py; "
        "git add calculator.py; "
        "git commit -m 'fix calculator'"
    )
    edited = requests.post(
        f"{dashboard_tool_base_url}/shell",
        json={"files": files, "commits": [], "history": [], "command": edit},
        timeout=DASHBOARD_REQUEST_TIMEOUT,
    )
    verified = requests.post(
        f"{dashboard_tool_base_url}/shell",
        json={
            "files": files,
            "commits": [],
            "history": [edit],
            "command": (
                "git switch main; "
                "git merge fix; "
                "python3.14 test_calculator.py; "
                "git log --format='%s' --all; "
                "printf '\\nSTATUS\\n'; "
                "git status --short"
            ),
        },
        timeout=DASHBOARD_REQUEST_TIMEOUT,
    )

    assert edited.status_code == 200
    assert edited.json()["exit_code"] == 0
    assert verified.status_code == 200
    assert verified.json()["exit_code"] == 0
    assert "test passed" in verified.json()["stdout"]
    assert "fix calculator" in verified.json()["stdout"]
    assert "baseline" in verified.json()["stdout"]
    assert verified.json()["stdout"].endswith("STATUS\n")


def test_dashboard_shell_workspace_rejects_parent_paths(dashboard_tool_base_url):
    response = requests.post(
        f"{dashboard_tool_base_url}/shell",
        json={"files": {"../secret": "no"}, "commits": [], "history": [], "command": "ls"},
        timeout=DASHBOARD_REQUEST_TIMEOUT,
    )

    assert response.status_code == 400


def test_dashboard_shell_workspace_has_no_network_clone(dashboard_tool_base_url):
    response = requests.post(
        f"{dashboard_tool_base_url}/shell",
        json={
            "files": {},
            "commits": [],
            "history": [],
            "command": "git clone https://github.com/rjpower/shellsim external",
        },
        timeout=DASHBOARD_REQUEST_TIMEOUT,
    )

    assert response.status_code == 200
    assert response.json()["exit_code"] != 0


def test_dashboard_repository_import_rejects_non_github_urls(dashboard_tool_base_url):
    response = requests.post(
        f"{dashboard_tool_base_url}/shell/repository",
        json={"url": "https://example.com/owner/repository"},
        timeout=DASHBOARD_REQUEST_TIMEOUT,
    )

    assert response.status_code == 400


def test_repository_import_clones_bounded_text_history(tmp_path):
    source = tmp_path / "source"
    _repository_with_history(source)
    clone_urls: list[str] = []

    def clone_repository(url: str, destination: Path) -> None:
        clone_urls.append(url)
        _clone_local_repository(source, destination)

    snapshot = asyncio.run(
        clone_repository_snapshot(
            "https://github.com/owner/repository.git",
            clone_repository=clone_repository,
        )
    )

    assert clone_urls == ["https://github.com/owner/repository.git"]
    assert snapshot.files == {"README.md": "second revision\n", "src/main.py": "print('hello')\n"}
    assert [commit.message for commit in snapshot.commits] == ["initial import", "add program"]
    assert snapshot.commits[0].changes == {"README.md": "first revision\n"}
    assert snapshot.commits[1].changes == {
        "README.md": "second revision\n",
        "src/main.py": "print('hello')\n",
    }
    assert snapshot.skipped_files == 3
    assert not snapshot.truncated_history


def test_repository_clone_stops_when_temporary_storage_exceeds_limit(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "clone-finished"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        'for destination in "$@"; do :; done\n'
        'mkdir -p "$destination"\n'
        'head -c 2048 /dev/zero > "$destination/pack"\n'
        "sleep 1\n"
        'touch "$MARIN_CLONE_MARKER"\n'
    )
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setenv("MARIN_CLONE_MARKER", str(marker))
    monkeypatch.setattr(repository_snapshot, "MAX_REPOSITORY_CLONE_BYTES", 1024)

    with pytest.raises(RepositorySnapshotTooLarge):
        asyncio.run(clone_repository_snapshot("https://github.com/owner/repository"))

    assert not marker.exists()


def test_dashboard_shell_workspace_recreates_imported_git_history(dashboard_tool_base_url, tmp_path):
    source = tmp_path / "source"
    _repository_with_history(source)

    def clone_repository(_url: str, destination: Path) -> None:
        _clone_local_repository(source, destination)

    snapshot = asyncio.run(
        clone_repository_snapshot("https://github.com/owner/repository", clone_repository=clone_repository)
    )
    response = requests.post(
        f"{dashboard_tool_base_url}/shell",
        json={
            "files": snapshot.files,
            "commits": [asdict(commit) for commit in snapshot.commits],
            "history": [],
            "command": "git log --format='%s'; printf '\\nOLD\\n'; git show HEAD^:README.md",
        },
        timeout=DASHBOARD_REQUEST_TIMEOUT,
    )

    assert response.status_code == 200
    assert response.json()["exit_code"] == 0
    assert response.json()["stdout"] == "add program\ninitial import\n\nOLD\nfirst revision\n"
