# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""External image metadata and JSON-lines services at Docker I/O boundaries."""

import asyncio
import hashlib
import json
import subprocess
import sys
from enum import StrEnum

import pytest

from taskcompendium.container_service import IMAGE_LABELS, ContainerService
from taskcompendium.tool_provider import tool_schema_sha256

SERVICE = """import json, sys
config = json.loads(sys.argv[1])
state = 0
for line in sys.stdin:
    request = json.loads(line)
    method = request['method']
    fault = config['fault']
    if method == 'initialize':
        result = {**request['params'], 'tools': config['tools']}
    elif method == 'call':
        if fault == 'call_failure':
            print(json.dumps({'id': request['id'], 'error': {'message': 'provider failed'}}), flush=True)
            continue
        state += 1
        result = json.dumps({'value': state})
    elif method == 'state':
        if fault == 'state_unavailable':
            print(json.dumps({'id': request['id'], 'error': {'message': 'state unavailable'}}), flush=True)
            continue
        result = state
    else:
        raise ValueError(method)
    response_id = 'invalid' if fault == 'wrong_response_id' and method == 'state' else request['id']
    print(json.dumps({'id': response_id, 'result': result}), flush=True)
"""


class ServiceFault(StrEnum):
    NORMAL = "normal"
    STATE_UNAVAILABLE = "state_unavailable"
    WRONG_RESPONSE_ID = "wrong_response_id"
    CALL_FAILURE = "call_failure"


@pytest.fixture
def runtime_factory(monkeypatch):
    images = {}
    original = subprocess.run

    def run(arguments, **kwargs):
        if arguments[:3] == ("docker", "image", "inspect") and arguments[3] in images:
            return subprocess.CompletedProcess(
                arguments, 0, json.dumps([{"Config": {"Labels": images[arguments[3]]}}]), ""
            )
        return original(arguments, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)

    def create(interface, seed, revision, definitions, *, fault=ServiceFault.NORMAL):
        labels = {
            IMAGE_LABELS[key]: value
            for key, value in {
                "action_interface": interface,
                "seed_sha256": seed,
                "provider_revision": revision,
                "tools_sha256": tool_schema_sha256(definitions),
            }.items()
        }
        image = "test/service@sha256:" + hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest()
        images[image] = labels
        return ContainerService(image=image, command=(json.dumps({"tools": list(definitions), "fault": fault}),))

    return create


@pytest.fixture
def service_processes(tmp_path, monkeypatch, runtime_factory):
    """Replace external Docker I/O with real persistent subprocess services."""
    path = tmp_path / "service.py"
    path.write_text(SERVICE)
    processes = {}
    original = asyncio.create_subprocess_exec

    async def execute(*arguments, **kwargs):
        if arguments[:2] == ("docker", "run"):
            name = arguments[arguments.index("--name") + 1]
            image_index = next(index for index, item in enumerate(arguments) if "@sha256:" in item)
            config = arguments[image_index + 1]
            process = await original(sys.executable, "-u", str(path), config, **kwargs)
            processes[name] = process
            return process
        if arguments[:2] == ("docker", "rm"):
            process = processes.pop(arguments[-1])
            if process.returncode is None:
                process.terminate()
            await process.wait()
            return await original(sys.executable, "-c", "pass", **kwargs)
        return await original(*arguments, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", execute)
    yield processes
    assert not processes, "Trial leaked service processes"


def pytest_addoption(parser):
    parser.addoption("--provider-image", default=None, help="Already staged digest-pinned Workplace service image")
