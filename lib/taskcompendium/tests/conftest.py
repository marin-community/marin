# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An external JSON-lines service at the Docker subprocess boundary."""

import asyncio
import hashlib
import json
import subprocess
import sys

import pytest

from taskcompendium.container_service import IMAGE_LABELS, ContainerService

FAKE_IMAGES = {}

SERVICE = """import json, sys
config = json.loads(sys.argv[1])
state = 0
for line in sys.stdin:
    request = json.loads(line)
    method = request['method']
    if method == 'initialize':
        result = {**request['params'], 'tools': config['tools']}
    elif method == 'call':
        if config.get('fail_call'):
            print(json.dumps({'id': request['id'], 'error': {'message': 'provider failed'}}), flush=True)
            continue
        state += 1
        result = json.dumps({'value': state})
    elif method == 'state':
        if config.get('broken_state'):
            print(json.dumps({'id': request['id'], 'error': {'message': 'state unavailable'}}), flush=True)
            continue
        result = state
    else:
        raise ValueError(method)
    response_id = 'invalid' if config.get('wrong_id') and method == 'state' else request['id']
    print(json.dumps({'id': response_id, 'result': result}), flush=True)
"""


@pytest.fixture
def service_processes(tmp_path, monkeypatch):
    """Replace only external Docker I/O with real persistent subprocess services."""
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


def service_command(definitions, *, broken_state=False, wrong_id=False, fail_call=False):
    return (
        json.dumps(
            {"tools": list(definitions), "broken_state": broken_state, "wrong_id": wrong_id, "fail_call": fail_call}
        ),
    )


def pytest_addoption(parser):
    parser.addoption("--provider-image", default=None, help="Already staged digest-pinned Workplace service image")


@pytest.fixture(autouse=True)
def staged_service_images(monkeypatch):
    """Supply immutable image metadata only for explicitly registered test images."""
    FAKE_IMAGES.clear()
    original = subprocess.run

    def run(arguments, **kwargs):
        if arguments[:3] == ("docker", "image", "inspect") and arguments[3] in FAKE_IMAGES:
            return subprocess.CompletedProcess(
                arguments, 0, json.dumps([{"Config": {"Labels": FAKE_IMAGES[arguments[3]]}}]), ""
            )
        return original(arguments, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)


def container_runtime(interface, seed, revision, definitions, *, broken_state=False, wrong_id=False, fail_call=False):
    tools_digest = hashlib.sha256(
        json.dumps(list(definitions), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    labels = {
        IMAGE_LABELS[key]: value
        for key, value in {
            "action_interface": interface,
            "seed_sha256": seed,
            "provider_revision": revision,
            "tools_sha256": tools_digest,
        }.items()
    }
    image = "test/service@sha256:" + hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest()
    FAKE_IMAGES[image] = labels
    return ContainerService(
        image=image,
        command=service_command(definitions, broken_state=broken_state, wrong_id=wrong_id, fail_call=fail_call),
    )
