# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local QUICK-to-rollout checks; run explicitly with ``-m manual -n 0``.

Requires working bubblewrap and the repository's prepared Python environment with pytest,
Verifyit and typer. The image provider substitutes that environment for declared source images;
image provisioning is not exercised. No downloads, model requests or package installs occur.
"""

import base64
import json
import sys
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.lazy import run
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.machine import Backend, HostImage, MachineSpec
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    CommandSemantics,
    EnvironmentRequirements,
    FunctionDefinition,
    ShellToolBinding,
    TaskSpec,
)
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import Converter, ImportRejection, NormalizedTask, RawRow
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from taskcompendium.runtime.shell import ShellToolConfig
from zephyr.context import ZephyrContext
from zephyr.readers import load_parquet

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions
from experiments.post_training.task_curation.datasets.skyrl import math
from experiments.post_training.task_curation.datasets.tasktrove import code, python_tests
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.pipeline import CurationRecipe
from experiments.post_training.task_curation.tests.conversion import tasktrove_row
from experiments.post_training.task_curation.tests.test_tasktrove_grading import BROKEN_PACKAGE, TYPER_PACKAGE
from experiments.post_training.task_curation.tests.test_tasktrove_text import fixture_files
from lib.rolloutengine.tests.test_rollout import ReplayModel, engine, lowered, machine_runtime, shell_call

pytestmark = [pytest.mark.manual, pytest.mark.asyncio]
IMAGE = "fixture/prepared-python@sha256:" + "0" * 64


@dataclass
class PreparedImageFactory:
    """Use real bubblewrap machines with the already installed test dependencies."""

    local: LocalMachineFactory
    backend = Backend.DOCKER

    async def create(self, spec: MachineSpec):
        return await self.local.create(replace(spec, source=HostImage(), memory_mb=None))


@pytest.fixture
def machines():
    repo = Path(__file__).resolve().parents[4]
    environment = Path(sys.prefix)
    interpreter = Path(sys.base_prefix).parent
    return PreparedImageFactory(
        LocalMachineFactory(read_only=(repo, environment, interpreter), bin_dirs=(environment / "bin",))
    )


def convert_for_local_rollout(row: RawRow, context: ConversionContext, *, convert: Converter, tool_name: str | None):
    converted = convert(row, context)
    if isinstance(converted, ImportRejection):
        return converted
    normalized = converted if isinstance(converted, NormalizedTask) else NormalizedTask(converted, ())
    task = normalized.task
    if task.environment_requirements != EnvironmentRequirements():
        task = task.model_copy(
            update={
                "environment_requirements": task.environment_requirements.model_copy(
                    update={
                        "docker_image": IMAGE,
                        "docker_build": None,
                        "packages_lock": None,
                        "command_semantics": CommandSemantics.LINUX_PROCESS,
                    }
                )
            }
        )
    if tool_name is not None:
        task = task.model_copy(
            update={
                "interaction_tools": (
                    FunctionDefinition(
                        name=tool_name,
                        parameters={
                            "type": "object",
                            "properties": {"script": {"type": "string"}},
                            "required": ["script"],
                            "additionalProperties": False,
                        },
                    ),
                ),
                "tool_bindings": {tool_name: ShellToolBinding(command_parameter="script")},
            }
        )
    return replace(normalized, task=task)


def write_files(files):
    return " && ".join(
        f"mkdir -p {Path(path).parent} && printf %s {base64.b64encode(data).decode()} | base64 -d > {path}"
        for path, data in files.items()
    )


@pytest.mark.parametrize("case", ["math500", "tasktrove-codeforces", "tasktrove-stack_pytest"])
async def test_quick_parquet_rollout_grades_correct_and_wrong_answers(case, tmp_path, monkeypatch, machines):
    source = next(source for module in (math, code, python_tests) for source in module.sources() if source.name == case)
    recipe = cast(CurationRecipe, source.config)
    tool_name = "Bash" if case == "tasktrove-codeforces" else None
    if case == "math500":
        input_path = tmp_path / "rows.jsonl"
        input_path.write_text(json.dumps({"problem": "What is six times seven?", "answer": "42"}) + "\n")
        source_format = SourceFormat.JSONL
        candidates = [r"\boxed{42}", r"\boxed{-1}"]
    else:
        fixture = "codenet" if tool_name else "stack_pytest"
        files = fixture_files(fixture)
        if tool_name:
            files["instruction.md"] = b"Use Bash to write your solution.\n" + files["instruction.md"]
        input_path = tmp_path / "tasks.parquet"
        pq.write_table(pa.Table.from_pylist([tasktrove_row(files)]), input_path)
        source_format = SourceFormat.PARQUET
        candidates = (
            [files["solution/solve.sh"].decode(), "printf 'print(0)' > /app/solution.py"]
            if tool_name
            else [write_files(TYPER_PACKAGE), write_files(BROKEN_PACKAGE)]
        )
    recipe = replace(
        recipe,
        source=replace(recipe.source, files=(input_path.name,), format=source_format, select=None),
        convert=partial(convert_for_local_rollout, convert=recipe.convert, tool_name=tool_name),
        grader=Environment(image=IMAGE),
    )
    source = replace(source, config=recipe)
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    runtime = CampaignRuntime()
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
            runtime.activate(context),
        ):
            step = source.pipeline(
                source, PipelineOptions(SourceProcessingMode.QUICK, runtime, inputs=InputOverrides(root=str(tmp_path)))
            )
            artifact = run(step, max_concurrent=1)[0]
    finally:
        client.shutdown()
    assert artifact.result is not None
    rows = [
        row
        for shard in (StoragePath(artifact.result.outputs["normalize"]) / "*.parquet").glob()
        for row in load_parquet(str(shard))
    ]
    assert len(rows) == 1 and rows[0]["normalization_kind"] is None
    task = TaskSpec.model_validate_json(rows[0]["task_json"])
    for candidate, expected in zip(candidates, (1.0, 0.0), strict=True):
        messages = [{"role": "assistant", "content": candidate}]
        if case != "math500":
            call = shell_call(candidate)
            function = call["tool_calls"][0]["function"]
            function["name"] = tool_name or "terminal"
            function["arguments"] = json.dumps({"script" if tool_name else "cmd": candidate})
            messages = [call, {"role": "assistant", "content": "Done."}]
        model = ReplayModel(messages)
        result = await engine(model, {"local": machines}).run(
            lowered(
                task,
                machine=None if case == "math500" else machine_runtime(),
                verifier_machine=None if case == "math500" else machine_runtime(),
                verifier_timeout=60,
                shell_tool=ShellToolConfig(name="terminal", command_parameter="cmd"),
            )
        )
        assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, expected), result.grade.error
        if case != "math500":
            tools = model.requests[0].options["tools"]
            assert [tool["function"]["name"] for tool in tools] == [tool_name or "terminal"]
