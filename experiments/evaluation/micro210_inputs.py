# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind frozen Micro210 task resources to a new judge without changing task membership."""

import json
import tomllib
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import tomli_w
from rolloutengine.models import TaskRolloutSpec
from taskcompendium.models import TaskResource, TaskSpec
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.evaluation.judge_config import JudgeConfig

RESOURCE_DIRECTORY = Path(__file__).parent


def evalchemy_judge_resources(files: dict[str, TaskResource], judge: JudgeConfig) -> dict[str, bytes]:
    """Bind the Evalchemy task's transport and model while retaining its rubric."""
    config = json.loads(resource_bytes(files["config.json"]))
    config.update(judge_model=judge.model, judge=asdict(judge))
    source = resource_bytes(files["grade.py"]).decode()
    bindings = {
        "from eval.execution.judge import JudgeBackend": (
            "from micro210_together_judge import TogetherJudgeBackend as JudgeBackend"
        ),
        "JudgeBackend(judge, session)": 'JudgeBackend(judge, session, policy=config["judge"])',
    }
    for original, replacement in bindings.items():
        if source.count(original) != 1:
            raise ValueError(f"Frozen Micro210 grade.py changed its judge binding: {original}")
        source = source.replace(original, replacement)
    return {
        "config.json": json.dumps(config).encode(),
        "grade.py": source.encode(),
        "micro210_together_judge.py": (RESOURCE_DIRECTORY / "together_judge.py").read_bytes(),
        "micro210_judge_config.py": (RESOURCE_DIRECTORY / "judge_config.py").read_bytes(),
    }


def bixbench_judge_resources(files: dict[str, TaskResource], judge: JudgeConfig) -> dict[str, bytes]:
    """Bind BixBench's JSON-schema client to the same judge and explicit deadlines."""
    config = tomllib.loads(resource_bytes(files["task.toml"]).decode())
    config["verifier"]["env"].update(
        OPENAI_API_KEY="${OPENAI_API_KEY}", OPENAI_BASE_URL=judge.base_url, MODEL_NAME=judge.model
    )
    source = resource_bytes(files["tests/llm_judge.py"]).decode()
    bindings = {
        'OpenAI(api_key=os.getenv("OPENAI_API_KEY"))': (
            f'OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout={judge.bixbench_request_timeout!r}, '
            f"max_retries={judge.bixbench_transport_attempts - 1})"
        ),
        "max_tokens=1024,": (
            f"max_tokens={judge.bixbench_max_tokens}, temperature={judge.bixbench_temperature!r}, "
            f"top_p={judge.top_p!r}, "
            f"reasoning_effort={str(judge.reasoning_effort)!r},"
        ),
    }
    for original, replacement in bindings.items():
        if source.count(original) != 1:
            raise ValueError(f"Frozen BixBench client changed its judge binding: {original}")
        source = source.replace(original, replacement)
    return {"task.toml": tomli_w.dumps(config).encode(), "tests/llm_judge.py": source.encode()}


def task_with_judge(task: TaskSpec, judge: JudgeConfig) -> TaskSpec:
    """Replace only judge configuration and transport resources in a task package."""
    files = {resource.path: resource for resource in task.resources.verifier}
    replacements = {}
    if "config.json" in files:
        replacements.update(evalchemy_judge_resources(files, judge))
    if "bixbench" in task.id:
        replacements.update(bixbench_judge_resources(files, judge))
    if not replacements:
        return task
    resources = tuple(resource for resource in task.resources.verifier if resource.path not in replacements) + tuple(
        inline_resource(path, content) for path, content in replacements.items()
    )
    return task.model_copy(update={"resources": task.resources.model_copy(update={"verifier": resources})})


def rebind_judge_inputs(source: str, destination: str, judge: JudgeConfig) -> None:
    """Write a derived table with the same rows, prompts, answers and rollout settings."""
    table = pq.read_table(source)
    rows = []
    for row in table.to_pylist():
        rollout = TaskRolloutSpec.model_validate_json(row["task_rollout_spec"])
        task = task_with_judge(rollout.task, judge)
        rollout = rollout.model_copy(update={"task": task})
        rows.append({**row, "task_spec": task.model_dump_json(), "task_rollout_spec": rollout.model_dump_json()})
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), destination)
