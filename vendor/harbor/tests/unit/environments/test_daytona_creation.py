"""Exercise SDK isolation through a real subprocess and a local fake service."""

import asyncio
import json
import os
from pathlib import Path

import pytest
from daytona import CreateSandboxFromImageParams, CreateSnapshotParams, Image
from daytona.common.errors import DaytonaValidationError

from harbor.environments.daytona.creation import create_in_subprocess


@pytest.fixture
def fake_daytona(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Replace only remote SDK operations; use Daytona's real environment helper.
    (tmp_path / "sitecustomize.py").write_text("""
import asyncio
import json
import os
from types import SimpleNamespace
from daytona import AsyncDaytona
from daytona._async.snapshot import AsyncSnapshotService
from daytona._utils.environment import isolated_env
from daytona.common.errors import DaytonaValidationError

original_init = AsyncDaytona.__init__
authorization = None

def init(self, config=None):
    global authorization
    original_init(self, config)
    authorization = self._api_client.default_headers["Authorization"]

AsyncDaytona.__init__ = init

async def create(self, params, **kwargs):
    port = int(os.environ["TEST_DAYTONA_PORT"])
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        with isolated_env():
            writer.write(json.dumps({"pid": os.getpid(), "authorization": authorization, "dockerfile": params.image.dockerfile(), "context": [open(c.source_path).read() for c in params.image._context_list], "labels": getattr(params, "labels", None)}).encode() + b"\\n")
            await writer.drain()
            action = await reader.readexactly(1)
        if action == b"e":
            raise DaytonaValidationError("disk full", status_code=400, headers={"Retry-After": "30"})
        print("SDK output must not corrupt the result pipe")
        return SimpleNamespace(id="created-resource")
    finally:
        writer.close()
        await writer.wait_closed()

AsyncDaytona.create = create
AsyncSnapshotService.create = create
""")
    monkeypatch.setenv(
        "PYTHONPATH", f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    )
    monkeypatch.setenv("DAYTONA_API_KEY", "fake-key")


@pytest.mark.parametrize("kind", ["snapshot", "sandbox"])
@pytest.mark.parametrize("api_key", [None, "selected-org-key"])
async def test_creation_keeps_parent_environment_and_image(
    kind, api_key, fake_daytona, monkeypatch, tmp_path
):
    connection = asyncio.get_running_loop().create_future()

    async def accept(reader, writer):
        connection.set_result((reader, writer))

    async with await asyncio.start_server(accept, "127.0.0.1", 0) as server:
        monkeypatch.setenv("TEST_DAYTONA_PORT", str(server.sockets[0].getsockname()[1]))
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "archive-key")
        (tmp_path / "payload.txt").write_text("preserved build context")
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM ubuntu:22.04\nCOPY payload.txt /payload.txt\n")
        image = Image.from_dockerfile(str(dockerfile))
        params = (
            CreateSnapshotParams(name="snapshot", image=image)
            if kind == "snapshot"
            else CreateSandboxFromImageParams(
                image=image, labels={"harbor.instance": "trial"}
            )
        )
        task = asyncio.create_task(
            create_in_subprocess(params, timeout=60, api_key=api_key)
        )
        reader, writer = await asyncio.wait_for(connection, 10)
        try:
            observed = json.loads(await reader.readline())
            assert os.environ["AWS_ACCESS_KEY_ID"] == "archive-key"
            assert os.environ["DAYTONA_API_KEY"] == "fake-key"
            assert observed["authorization"] == f"Bearer {api_key or 'fake-key'}"
            assert observed["pid"] != os.getpid()
            assert "COPY payload.txt /payload.txt" in observed["dockerfile"]
            assert observed["context"] == ["preserved build context"]
            if kind == "sandbox":
                assert observed["labels"] == {"harbor.instance": "trial"}
            writer.write(b"s")
            await writer.drain()
            assert await asyncio.wait_for(task, 10) == "created-resource"
        finally:
            writer.close()
            await writer.wait_closed()
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task


@pytest.mark.parametrize("action", ["error", "cancel"])
async def test_creation_preserves_sdk_errors_and_reaps_child(
    action, fake_daytona, monkeypatch
):
    connection = asyncio.get_running_loop().create_future()

    async def accept(reader, writer):
        connection.set_result((reader, writer))

    async with await asyncio.start_server(accept, "127.0.0.1", 0) as server:
        monkeypatch.setenv("TEST_DAYTONA_PORT", str(server.sockets[0].getsockname()[1]))
        params = CreateSandboxFromImageParams(image=Image.base("ubuntu:22.04"))
        task = asyncio.create_task(create_in_subprocess(params, timeout=60))
        reader, writer = await asyncio.wait_for(connection, 10)
        try:
            observed = json.loads(await reader.readline())
            if action == "error":
                writer.write(b"e")
                await writer.drain()
                with pytest.raises(DaytonaValidationError) as error:
                    await asyncio.wait_for(task, 10)
                assert error.value.status_code == 400
                assert error.value.headers["Retry-After"] == "30"
            else:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 10)
            with pytest.raises(ProcessLookupError):
                os.kill(observed["pid"], 0)
        finally:
            writer.close()
            await writer.wait_closed()
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
