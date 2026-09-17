# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for dashboard tool schemas, execution, and agent workspaces."""

import asyncio
import io
import zipfile
from collections.abc import Iterator

import httpx
import pytest
import requests
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.inference.chat_template_protocol import ChatTemplateProtocol, ToolCallFormat, protocol_for_chat_template
from marin.inference.dashboard_server import ServingInfo, bind_serving_socket, build_dashboard_app, serve_app_background
from marin.inference.python_tool_routes import TOOL_WORKER_TIMEOUT
from marin.inference.python_tools import python_tools_from_source
from marin.inference.repository_snapshot import fetch_repository_snapshot

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


def test_dashboard_shell_workspace_replays_commands_with_simulated_git(dashboard_tool_base_url):
    files = {
        "calculator.py": "def add(left: int, right: int) -> int:\n    return left - right\n",
        "test_calculator.py": "from calculator import add\n\nassert add(17, 24) == 41\nprint('test passed')\n",
    }
    edit = "sed -i 's/left - right/left + right/' calculator.py"
    edited = requests.post(
        f"{dashboard_tool_base_url}/shell",
        json={"files": files, "history": [], "command": edit},
        timeout=DASHBOARD_REQUEST_TIMEOUT,
    )
    verified = requests.post(
        f"{dashboard_tool_base_url}/shell",
        json={
            "files": files,
            "history": [edit],
            "command": "python3.14 test_calculator.py; git diff calculator.py",
        },
        timeout=DASHBOARD_REQUEST_TIMEOUT,
    )

    assert edited.status_code == 200
    assert edited.json()["exit_code"] == 0
    assert verified.status_code == 200
    assert verified.json()["exit_code"] == 0
    assert "test passed" in verified.json()["stdout"]
    assert "+    return left + right" in verified.json()["stdout"]


def test_dashboard_shell_workspace_rejects_parent_paths(dashboard_tool_base_url):
    response = requests.post(
        f"{dashboard_tool_base_url}/shell",
        json={"files": {"../secret": "no"}, "history": [], "command": "ls"},
        timeout=DASHBOARD_REQUEST_TIMEOUT,
    )

    assert response.status_code == 400


def test_dashboard_shell_workspace_has_no_network_clone(dashboard_tool_base_url):
    response = requests.post(
        f"{dashboard_tool_base_url}/shell",
        json={
            "files": {},
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


def test_repository_import_extracts_bounded_text_snapshot():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as repository_zip:
        repository_zip.writestr("owner-repository-ref/README.md", "# Example\n")
        repository_zip.writestr("owner-repository-ref/src/main.py", "print('hello')\n")
        repository_zip.writestr("owner-repository-ref/image.bin", b"\xff\x00")
        repository_zip.writestr("owner-repository-ref/node_modules/dependency.js", "ignored\n")

    def github_archive(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.github.com/repos/owner/repository/zipball"
        return httpx.Response(200, content=archive.getvalue())

    snapshot = asyncio.run(
        fetch_repository_snapshot(
            "https://github.com/owner/repository.git",
            transport=httpx.MockTransport(github_archive),
        )
    )

    assert snapshot.files == {"README.md": "# Example\n", "src/main.py": "print('hello')\n"}
    assert snapshot.skipped_files == 2


def test_repository_import_does_not_follow_redirects_outside_github():
    requests_seen: list[httpx.Request] = []

    def redirect(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    with pytest.raises(ValueError):
        asyncio.run(
            fetch_repository_snapshot(
                "https://github.com/owner/repository",
                transport=httpx.MockTransport(redirect),
            )
        )

    assert len(requests_seen) == 1
