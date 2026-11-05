# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Python benchmarks with private test programs run as Shellbox commands.

The source's 3-second (CruxEval: 10-second) execution alarm starts after Python
startup and NumPy preload. A forked worker has the source's one-second termination allowance.
Shellbox bounds startup separately and supplies isolation and resource limits.
"""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from verifyit.adapters import completion

from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.code import THREAD_ENVIRONMENT
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import ChatGeneration, EvaluationItem, TextGeneration
from taskcompendium.grader import GraderPackage
from taskcompendium.identity import canonical_sha256
from taskcompendium.importers.evalchemy.assets.code_protocol import function_prompt, mbpp_prompt
from taskcompendium.models import FileReward, RewardFile, RewardFileFormat, ScriptGrader, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import inline_resource

ASSETS = Path(__file__).parent / "assets"
CODE_ASSETS = (
    inline_resource("prepare.py", (ASSETS / "python_prepare.py").read_bytes()),
    inline_resource("code_protocol.py", (ASSETS / "code_protocol.py").read_bytes()),
    inline_resource("NOTICE.txt", (ASSETS / "NOTICE.txt").read_bytes()),
    inline_resource(
        "lm-evaluation-harness-MIT.txt",
        (ASSETS / "lm-evaluation-harness-MIT.txt").read_bytes(),
    ),
)
CODE_REVISION = (
    "2-"
    + canonical_sha256(
        {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                ASSETS / "code_protocol.py",
                Path(completion.__file__),
                ASSETS / "python_prepare.py",
                ASSETS / "crux_input.txt",
                ASSETS / "crux_output.txt",
            )
        }
    )[:16]
)


def read_mbpp(path: StoragePath, _context: ConversionContext):
    """Keep the source's first 500 rows and its fixed examples at positions 1, 2 and 3."""
    with path.open("rb") as file:
        rows = pq.read_table(file).to_pylist()
    examples = rows[1:4]
    for row in rows[:500]:
        yield {**row, "evaluation_prompt": mbpp_prompt(row, examples)}


def read_crux(path: StoragePath, _context: ConversionContext):
    with path.open("rt") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    for direction in ("input", "output"):
        template = (ASSETS / f"crux_{direction}.txt").read_text()
        for row in rows:
            prompt = template.replace("{input}", row["input"]).replace("{output}", row["output"])
            yield {**row, "direction": direction, "evaluation_prompt": prompt.replace("{code}", row["code"])}


def convert_python(row: RawRow, context: ConversionContext, *, suite: str) -> TaskSpec:
    problem = dict(row.data)
    prompt = (
        (problem["prompt"] if suite == "humaneval" else function_prompt(problem))
        if suite in {"humaneval", "humanevalplus"}
        else problem["evaluation_prompt"]
    )
    environment = required_grader_environment(context)
    environment = environment.model_copy(
        update={
            "setup_commands": (*environment.setup_commands, "python3 /tests/prepare.py"),
            "environment_variables": {**environment.environment_variables, **THREAD_ENVIRONMENT},
        }
    )
    config = {"suite": suite, "problem": problem, "direction": problem.get("direction", "")}
    package = GraderPackage(
        ScriptGrader(
            argv=("python3", "/app/program.py"),
            cwd="/app",
            environment=environment,
            timeout=60,
            answer_path="/app/answer.txt",
            reward=FileReward(
                files=(RewardFile(path="/app/reward.json", format=RewardFileFormat.JSON),), pass_above=0.5
            ),
        ),
        (*CODE_ASSETS, inline_resource("config.json", json.dumps(config, allow_nan=False, sort_keys=True).encode())),
    )
    task = conversation_task(row, events=(TextMessage(role="user", content=prompt),), package=package, evidence=problem)
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", f"benchmark:{suite}", "code")})


def python_item(task: TaskSpec, *, generation: ChatGeneration | TextGeneration) -> EvaluationItem:
    item = chat_item(task)
    row = source_evidence(task)
    return item.model_copy(
        update={
            "operation": generation,
            "groups": {"direction": row["direction"]} if "direction" in row else {"language": "python"},
        }
    )
