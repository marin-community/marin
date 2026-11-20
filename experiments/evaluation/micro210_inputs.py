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
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.evaluation.judge_config import JudgeConfig

RESOURCE_DIRECTORY = Path(__file__).parent


def task_with_judge(task: TaskSpec, judge: JudgeConfig) -> TaskSpec:
    """Replace only judge configuration and transport resources in a task package."""
    files = {resource.path: resource for resource in task.resources.verifier}
    replacements = {}
    if "config.json" in files:
        config = json.loads(resource_bytes(files["config.json"]))
        config.update(judge_model=judge.model, judge=asdict(judge))
        replacements["config.json"] = json.dumps(config).encode()
        source = resource_bytes(files["grade.py"]).decode()
        original = "from eval.execution.judge import JudgeBackend"
        if original not in source:
            raise ValueError("Frozen Micro210 grade.py no longer imports the expected judge transport")
        source = source.replace(original, "from micro210_together_judge import TogetherJudgeBackend as JudgeBackend")
        source = source.replace("JudgeBackend(judge, session)", 'JudgeBackend(judge, session, policy=config["judge"])')
        replacements["grade.py"] = source.encode()
        replacements["micro210_together_judge.py"] = (RESOURCE_DIRECTORY / "together_judge.py").read_bytes()
        replacements["micro210_judge_config.py"] = (RESOURCE_DIRECTORY / "judge_config.py").read_bytes()
    if "bixbench" in task.id:
        config = tomllib.loads(resource_bytes(files["task.toml"]).decode())
        config["verifier"]["env"].update(
            OPENAI_API_KEY="${OPENAI_API_KEY}", OPENAI_BASE_URL=judge.base_url, MODEL_NAME=judge.model
        )
        replacements["task.toml"] = tomli_w.dumps(config).encode()
        source = resource_bytes(files["tests/llm_judge.py"]).decode()
        source = source.replace(
            'OpenAI(api_key=os.getenv("OPENAI_API_KEY"))',
            f'OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout={judge.bixbench_request_timeout!r}, '
            f"max_retries={judge.bixbench_transport_attempts - 1})",
        )
        source = source.replace(
            "max_tokens=1024,",
            f"max_tokens={judge.bixbench_max_tokens}, temperature={judge.temperature!r}, top_p={judge.top_p!r}, "
            f"reasoning_effort={str(judge.reasoning_effort)!r},",
        )
        replacements["tests/llm_judge.py"] = source.encode()
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
