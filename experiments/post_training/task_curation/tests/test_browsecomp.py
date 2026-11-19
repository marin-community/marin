# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from rolloutengine.harness.browsecomp import BrowseCompProvider
from rolloutengine.harness.shell import ShellHarness, ToolContext, task_prompt
from rolloutengine.test_support import rollout_limits
from shellbox.machine import ExitReason, Machine, Result
from taskcompendium.browsecomp import BROWSECOMP_ACTION_INTERFACE
from taskcompendium.harbor.adapter import TASK_SPEC_FILE, write_harbor_task
from taskcompendium.models import ExecutionRequirements, TaskSpec
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.shell import resolve_shell_tool

from experiments.post_training.task_curation.datasets.browsecomp.tasks import convert_query

IMAGE = "ghcr.io/marin-community/rl-data-browsecomp-plus@sha256:" + "a" * 64
ROW = {"query_id": "fixture", "query": "d0Adh66ja4lhTNK1qt1jpvdGysZg", "answer": "FA=="}


def test_browsecomp_native_roundtrip_preserves_search_interface_and_private_answer(tmp_path):
    task = convert_query(ROW, IMAGE)
    native = tmp_path / "task"
    write_harbor_task(
        task,
        native,
        ExecutionRequirements(cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=300, verifier_timeout_sec=60),
    )
    restored = TaskSpec.model_validate_json((native / TASK_SPEC_FILE).read_text())
    harness = ShellHarness(
        command_timeout=None, tools={BROWSECOMP_ACTION_INTERFACE: BrowseCompProvider(command_timeout=None)}
    )
    harness.check(restored, rollout_limits())
    prompt = task_prompt(restored, resolve_shell_tool(restored, harness.shell_tool), ())
    assert [tool["function"]["name"] for tool in prompt.options["tools"]] == ["search"]
    assert "What is two plus two?" in prompt.messages[0]["content"]
    assert "Exact Answer: 4" not in json.dumps(prompt.messages)
    config = next(resource for resource in restored.resources.verifier if resource.path == "config.json")
    assert json.loads(resource_bytes(config)) == {"question": "What is two plus two?", "answer": "4"}


@pytest.mark.asyncio
async def test_browsecomp_producer_failure_is_not_a_successful_empty_result():
    task = convert_query(ROW, IMAGE)
    requirement = task.environment_requirements.tool_providers["retrieval"]
    machine = cast(
        Machine,
        SimpleNamespace(
            run=AsyncMock(
                return_value=Result(
                    exit_code=1,
                    stdout=b"[]",
                    stderr=b"Index is unavailable",
                    stdout_truncated=False,
                    stderr_truncated=False,
                    reason=ExitReason.EXITED,
                )
            )
        ),
    )
    session = await BrowseCompProvider(command_timeout=None).open(ToolContext(machine, requirement))
    with pytest.raises(RuntimeError, match="Index is unavailable"):
        await session.call("search", {"query": "anything"})


def test_native_library_diagnostics_do_not_corrupt_retrieval_json():
    program = """
import json
import os
import sys
from experiments.post_training.task_curation.datasets.browsecomp.tool import source_diagnostics
with source_diagnostics():
    os.write(1, b"native JVM diagnostic\\n")
    print("Python dependency diagnostic")
json.dump([{"docid": "fixture", "score": 1, "snippet": "result"}], sys.stdout)
"""
    result = subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == [{"docid": "fixture", "score": 1, "snippet": "result"}]
    assert "native JVM diagnostic" in result.stderr
    assert "Python dependency diagnostic" in result.stderr
