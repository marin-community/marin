# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind frozen Micro210 task resources to a new judge without changing task membership."""

import hashlib
import json
import random
import re
import tomllib
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import tomli_w
from rolloutengine.models import TaskRolloutSpec
from taskcompendium.models import TaskResource, TaskSpec
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.evaluation.judge_config import JudgeConfig
from experiments.evaluation.micro210_policy import CHOICE_POLICY_VERSION

RESOURCE_DIRECTORY = Path(__file__).parent
CHOICE_SEED_NAMESPACE = f"micro210:{CHOICE_POLICY_VERSION}:"
EXPECTED_CHOICE_BENCHMARKS = {"truthfulqa", "gpqa-diamond", "mmlu-pro"}


def choice_seed(task_id: str) -> int:
    """Derive a stable signed 63-bit seed from the policy version and task identity."""
    digest = hashlib.sha256((CHOICE_SEED_NAMESPACE + task_id).encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def truthfulqa_choice_task(task: TaskSpec) -> TaskSpec:
    """Permute generated choices and the verifier's accepted letters together."""
    files = {resource.path: resource for resource in task.resources.verifier}
    config = json.loads(resource_bytes(files["config.json"]))
    spec = config["spec"]
    if spec["grader"]["name"] != "lm_eval:truthfulqa_mc2":
        raise ValueError(f"Unexpected TruthfulQA grader for {task.id}")
    targets = spec["doc"]["mc2_targets"]
    choices, labels = targets["choices"], targets["labels"]
    if len(choices) != len(labels) or len(choices) != len(spec["requests"]):
        raise ValueError(f"TruthfulQA choices, labels and requests disagree for {task.id}")
    seed = choice_seed(task.id)
    order = list(range(len(choices)))
    random.Random(seed).shuffle(order)
    original = "\n".join(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices))
    shuffled = "\n".join(f"{chr(65 + index)}. {choices[source]}" for index, source in enumerate(order))
    events = list(task.context.events)
    message = events[-1]
    if message.content.count(original) != 1:
        raise ValueError(f"TruthfulQA prompt does not match its verifier choices for {task.id}")
    events[-1] = message.model_copy(update={"content": message.content.replace(original, shuffled)})
    targets.update(choices=[choices[index] for index in order], labels=[labels[index] for index in order])
    spec["requests"] = [spec["requests"][index] for index in order]
    spec["target"] = order.index(spec["target"])
    resources = (
        *(resource for resource in task.resources.verifier if resource.path != "config.json"),
        inline_resource("config.json", json.dumps(config).encode()),
    )
    labels = {**task.labels, "micro210_choice_seed": str(seed), "micro210_choice_order": ",".join(map(str, order))}
    return task.model_copy(
        update={
            "context": task.context.model_copy(update={"events": tuple(events)}),
            "resources": task.resources.model_copy(update={"verifier": resources}),
            "labels": labels,
        }
    )


def letter_choice_positions(rows: list[dict]) -> dict[str, dict[str, set[str]]]:
    """Count accepted letters per unique task in every lettered-choice benchmark."""
    positions: dict[str, dict[str, set[str]]] = defaultdict(dict)
    seen: set[str] = set()
    for row in rows:
        if row["task_id"] in seen:
            continue
        seen.add(row["task_id"])
        task = TaskSpec.model_validate_json(row["task_spec"])
        files = {resource.path: resource for resource in task.resources.verifier}
        if "config.json" not in files:
            continue
        spec = json.loads(resource_bytes(files["config.json"]))["spec"]
        if spec["grader"]["name"] == "lm_eval:truthfulqa_mc2":
            accepted = {chr(65 + index) for index, label in enumerate(spec["doc"]["mc2_targets"]["labels"]) if label}
        elif re.fullmatch(r"[A-Z]", str(spec["target"])) and re.search(r"\bA[.)] ", task.context.events[-1].content):
            accepted = {spec["target"]}
        else:
            continue
        positions[row["benchmark"]][row["task_id"]] = accepted
    return positions


def reject_fixed_choice_positions(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Reject a benchmark that accepts one constant letter on every task."""
    positions = letter_choice_positions(rows)
    counts = {}
    for benchmark, tasks in positions.items():
        common = set.intersection(*tasks.values())
        if common:
            raise ValueError(f"{benchmark} accepts fixed choice position(s) {sorted(common)} on all {len(tasks)} tasks")
        counts[benchmark] = {
            letter: sum(letter in accepted for accepted in tasks.values())
            for letter in sorted(set.union(*tasks.values()))
        }
    return counts


def shuffle_micro210_inputs(source: str, destination: str) -> dict[str, dict[str, int]]:
    """Derive a new prepared table while preserving the frozen task selection."""
    table = pq.read_table(source)
    rows = []
    for row in table.to_pylist():
        if row["benchmark"] != "truthfulqa":
            rows.append(row)
            continue
        rollout = TaskRolloutSpec.model_validate_json(row["task_rollout_spec"])
        task = truthfulqa_choice_task(rollout.task)
        rollout = rollout.model_copy(update={"task": task})
        rows.append({**row, "task_spec": task.model_dump_json(), "task_rollout_spec": rollout.model_dump_json()})
    counts = reject_fixed_choice_positions(rows)
    if missing := EXPECTED_CHOICE_BENCHMARKS - counts.keys():
        raise ValueError(f"Micro210 choice audit missed benchmark(s): {sorted(missing)}")
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), destination)
    return counts


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
