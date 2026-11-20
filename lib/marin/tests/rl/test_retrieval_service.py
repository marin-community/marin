# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
from pathlib import Path

import pytest
from fray.current_client import set_current_client
from fray.iris_backend import FrayIrisClient
from fray.types import ResourceConfig, create_environment
from rigging.secrets import SecretBehavior
from rigging.timing import Duration
from rolloutengine.harness.retrieval_service import OpenResearcherServiceProvider
from rolloutengine.harness.shell import ToolContext
from shellbox.machine import Command, ExitReason, Result
from shellbox.process_session import SubprocessSession
from taskcompendium.models import ProviderRequirement
from taskcompendium.openresearcher import OPENRESEARCHER_ACTION_INTERFACE, OpenResearcherState

from marin.rl.retrieval_backend import RetrievalServer, retrieval_handler
from marin.rl.retrieval_service import RetrievalServiceConfig, RetrievalServiceLaunch, retrieval_service


class FixtureIndex:
    def __init__(self, text: str):
        self.content = json.dumps({"docid": "7385", "text": text}, ensure_ascii=False).encode()

    def search(self, query: str) -> list[dict]:
        return [{"docid": "7385", "score": 1.0, "snippet": query}]

    def document(self, docid: str) -> bytes:
        assert docid == "7385"
        return self.content


class ProcessMachine:
    """Run the guest protocol in real local processes at the machine I/O boundary."""

    async def open_process(self, command: Command) -> SubprocessSession:
        process = await asyncio.create_subprocess_exec(
            *command.argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        async def stop():
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

        return SubprocessSession(process, stop=stop)

    async def run(self, command: Command) -> Result:
        result = await asyncio.to_thread(
            subprocess.run, command.argv, input=command.stdin, capture_output=True, timeout=command.timeout
        )
        return Result(result.returncode, result.stdout, result.stderr, False, False, ExitReason.EXITED)

    async def download(self, source: str, target: Path) -> None:
        shutil.copyfile(source, target)

    async def upload(self, source: Path, target: str) -> None:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    async def close(self) -> None:
        pass


@pytest.mark.timeout(120)
def test_retrieval_private_relay_preserves_documents_larger_than_capture_limit():
    state = OpenResearcherState(
        implementation_revision="a" * 40,
        corpus_revision="8367703babe6e080899a48782f365c284c32b668",
        index_manifest_sha256="b" * 64,
        tokenizer_revision="c" * 40,
        top_k=5,
        snippet_max_tokens=512,
        include_get_document=True,
    )
    text = "full document Ω\n" * 1_300_000
    expected = hashlib.sha256(text.encode()).hexdigest()
    with RetrievalServer(
        ("127.0.0.1", 0), retrieval_handler(FixtureIndex(text), state.model_dump(mode="json")), 4
    ) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:

            async def check():
                provider = OpenResearcherServiceProvider(f"http://127.0.0.1:{server.server_port}")
                requirement = ProviderRequirement(
                    action_interface=OPENRESEARCHER_ACTION_INTERFACE,
                    network_access="service_only",
                    initial_state=state.model_dump(mode="json"),
                )
                session = await provider.open(ToolContext(ProcessMachine(), requirement))
                try:
                    search = json.loads((await session.call("search", {"query": "fixture query"})).content)
                    assert search == [{"docid": "7385", "score": 1.0, "snippet": "fixture query"}]
                    result = json.loads((await session.call("get_document", {"docid": "7385"})).content)
                    assert hashlib.sha256(result["text"].encode()).hexdigest() == expected
                    assert result["docid"] == "7385"
                finally:
                    await session.close()

            asyncio.run(check())
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_service_child_submission_forwards_mounted_storage_credentials(monkeypatch, tmp_path):
    for name, value in {"CW_KEY_ID": "mounted-id", "CW_KEY_SECRET": "mounted-secret"}.items():
        (tmp_path / name).write_text(value)
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "DAYTONA_RL_API_KEY").write_text("parent-only")
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(tmp_path))

    class SubmissionCaptured(Exception):
        pass

    class IrisSubmission:
        def submit(self, **request):
            environment = request["environment"]
            wire = environment.to_proto()
            assert dict(wire.secrets) == {"CW_KEY_ID": "mounted-id", "CW_KEY_SECRET": "mounted-secret"}
            assert not {"CW_KEY_ID", "CW_KEY_SECRET", "DAYTONA_RL_API_KEY"} & wire.env_vars.keys()
            assert environment.secret_behavior is SecretBehavior.ISOLATE
            assert request["ports"] == ["retrieval"]
            raise SubmissionCaptured

    launch = RetrievalServiceLaunch(
        config=RetrievalServiceConfig(
            state=OpenResearcherState(
                implementation_revision="a" * 40,
                corpus_revision="8367703babe6e080899a48782f365c284c32b668",
                index_manifest_sha256="c" * 64,
                tokenizer_revision="d" * 40,
                top_k=5,
                snippet_max_tokens=512,
                include_get_document=True,
            ),
            index_uri="s3://fixture/index",
            root="/opt/openresearcher",
            backend_python="/usr/local/bin/python",
            max_concurrent_requests=64,
            endpoint_name="retrieval",
        ),
        resources=ResourceConfig(cpu=16, ram="96GB", disk="128GB"),
        environment=create_environment(secret_behavior=SecretBehavior.ISOLATE),
        startup_timeout=60,
        lifetime=Duration.from_hours(1),
    )
    with set_current_client(FrayIrisClient.from_iris_client(IrisSubmission())):
        with pytest.raises(SubmissionCaptured), retrieval_service(launch):
            pytest.fail("Submission capture must stop before service readiness")
