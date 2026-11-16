# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from shellbox.machine import Command, ExitReason, Result
from taskcompendium.models import ProviderRequirement
from taskcompendium.openresearcher import OPENRESEARCHER_ACTION_INTERFACE, OpenResearcherState

from rolloutengine.harness.browsecomp import OpenResearcherProvider
from rolloutengine.harness.shell import ToolContext


class CapturedProcessMachine:
    """Run real fixture producers and emulate the machine's stdout capture boundary."""

    def __init__(self, environment: dict[str, str]):
        self.environment = environment
        self.downloads: list[Path] = []

    async def run(self, command: Command) -> Result:
        result = await asyncio.to_thread(
            subprocess.run,
            command.argv,
            cwd=command.cwd,
            env={**os.environ, **self.environment, **command.env},
            input=command.stdin,
            capture_output=True,
            timeout=command.timeout,
        )
        limit = command.output_limit_bytes
        return Result(
            result.returncode,
            result.stdout[:limit],
            result.stderr[:limit],
            len(result.stdout) > limit,
            len(result.stderr) > limit,
            ExitReason.EXITED,
        )

    async def upload(self, source: Path, target: str) -> None:
        shutil.copyfile(source, target)

    async def download(self, source: str, target: Path) -> None:
        self.downloads.append(Path(source))
        shutil.copyfile(source, target)

    async def close(self) -> None:
        pass


def document_machine(root: Path, payload: bytes) -> CapturedProcessMachine:
    package = root / "openresearcher"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "tool.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "request = json.load(sys.stdin)\n"
        "assert request['name'] == 'get_document'\n"
        "assert request['arguments'] == {'docid': 'fixture'}\n"
        "sys.stdout.buffer.write(Path(os.environ['DOCUMENT_FIXTURE']).read_bytes())\n"
    )
    path = root / "document.json"
    path.write_bytes(payload)
    return CapturedProcessMachine({"PYTHONPATH": str(root), "DOCUMENT_FIXTURE": str(path)})


def research_requirement() -> ProviderRequirement:
    state = OpenResearcherState(
        implementation_revision="a" * 40,
        corpus_revision="8367703babe6e080899a48782f365c284c32b668",
        index_manifest_sha256="c" * 64,
        tokenizer_revision="d" * 40,
        top_k=5,
        snippet_max_tokens=512,
        include_get_document=True,
    )
    return ProviderRequirement(
        action_interface=OPENRESEARCHER_ACTION_INTERFACE, initial_state=state.model_dump(mode="json")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("repetitions", [1, 900_000], ids=["normal", "over-16MiB"])
async def test_openresearcher_document_transfer_preserves_complete_producer_bytes(tmp_path, repetitions):
    text = "Cedar evidence with café.\n" * repetitions
    document = {"docid": "fixture", "text": text}
    machine = document_machine(tmp_path, json.dumps(document, ensure_ascii=False).encode())
    session = await OpenResearcherProvider(command_timeout=30).open(ToolContext(machine, research_requirement()))
    result = json.loads((await session.call("get_document", {"docid": "fixture"})).content)
    assert result["docid"] == document["docid"]
    assert hashlib.sha256(result["text"].encode()).digest() == hashlib.sha256(text.encode()).digest()
    assert len(result["text"].encode()) == len(text.encode())
    assert machine.downloads and all(not path.exists() for path in machine.downloads)


@pytest.mark.asyncio
async def test_openresearcher_malformed_document_aborts_and_removes_owned_file(tmp_path):
    machine = document_machine(tmp_path, b'{"text":')
    session = await OpenResearcherProvider(command_timeout=30).open(ToolContext(machine, research_requirement()))
    with pytest.raises(json.JSONDecodeError):
        await session.call("get_document", {"docid": "fixture"})
    assert machine.downloads and all(not path.exists() for path in machine.downloads)
