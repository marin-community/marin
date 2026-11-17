"""Replay TaskForge text controls through the original Docker grader."""

import argparse
import asyncio
import dataclasses
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import MachineSpec
from taskcompendium.models import ConversationTrace, GradingAttempt, TaskSpec, TextMessage
from taskcompendium.runtime.grading import grade_in_sandbox
from taskcompendium.runtime.resources import resource_bytes

RUNTIME_REVISION = "8385245f5331b44d5ca0b44a7f558ebe4bd1d50b"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_reference(task: TaskSpec) -> str | None:
    resource = next(
        (resource for resource in task.resources.oracle if resource.path == "taskforge/reference_control.json"), None
    )
    if resource is None:
        return None
    payload = json.loads(resource_bytes(resource))
    turns = payload.get("turns", [])
    if len(turns) == 1 and turns[0]["type"] == "message":
        return turns[0]["content"]
    return None


def controls(task: TaskSpec) -> tuple[tuple[str, str], ...]:
    reference = text_reference(task)
    if reference is None:
        raise ValueError(f"Task has no direct text reference: {task.id}")
    return (("reference", reference), ("incorrect", "I do not know."), ("empty", ""))


def write_results(output: Path, results: list[dict]) -> None:
    temporary = output / "control-results.json.tmp"
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(output / "control-results.json")
    flat = [
        {
            **{key: value for key, value in row.items() if key not in ("detail", "diagnostics")},
            "detail_json": json.dumps(row["detail"]),
            "diagnostics_json": json.dumps(row["diagnostics"]),
        }
        for row in results
    ]
    pq.write_table(pa.Table.from_pylist(flat), output / "control-results.parquet")


async def run(args: argparse.Namespace) -> None:
    tasks = [TaskSpec.model_validate_json(line) for line in args.tasks.read_text().splitlines()]
    by_id = {task.id: task for task in tasks}
    if len(tasks) != len(by_id):
        raise ValueError("Task IDs must be unique")
    eligible = [task for task in tasks if task.grader.kind == "script" and text_reference(task)]
    if args.task_ids:
        ids = json.loads(args.task_ids.read_text())
        selected = [by_id[task_id] for task_id in ids]
        if len(ids) != len(set(ids)) or any(task not in eligible for task in selected):
            raise ValueError("Task IDs must name unique eligible script tasks")
    else:
        count = min(args.sample_size, len(eligible))
        if count == 0:
            raise ValueError("No eligible tasks")
        indices = [index * (len(eligible) - 1) // max(1, count - 1) for index in range(count)]
        selected = [eligible[index] for index in indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "control-results.json"
    provenance_path = args.output_dir / "control-provenance.json"
    input_hash = sha256(args.tasks)
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    prior = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    if results and prior.get("converted_rows_sha256") != input_hash:
        raise ValueError("Saved results belong to a different task input file")
    finished = set()
    for row in results:
        task = by_id[row["task_id"]]
        responses = dict(controls(task))
        response_hash = hashlib.sha256(responses[row["control"]].encode()).hexdigest()
        task_hash = hashlib.sha256(task.model_dump_json().encode()).hexdigest()
        if row.get("task_sha256", task_hash) != task_hash or row.get("response_sha256", response_hash) != response_hash:
            raise ValueError("Saved control differs from the current task or response")
        row.update(task_sha256=task_hash, response_sha256=response_hash)
        finished.add((task.id, row["control"], task_hash, response_hash))

    provenance = {
        **prior,
        "updated": datetime.now(UTC).isoformat(),
        "runtime_revision": args.runtime_revision,
        "converter_sha256": sha256(args.converter) if args.converter else None,
        "controls_script_sha256": sha256(Path(__file__)),
        "converted_rows_sha256": input_hash,
        "network_policy": "deny",
        "backend": "docker",
        "platform": "linux/amd64",
        "selected_task_ids": [task.id for task in selected],
        "python_empty_submission": "TaskCompendium rejects empty assistant submissions before script execution",
    }
    if results:
        provenance["resumed_prior_controls_script_sha256"] = prior.get("controls_script_sha256")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    factory = DockerMachineFactory()
    for task in selected:
        spec = MachineSpec(source=RegistryImage(task.grader.environment.docker_image), workdir=task.grader.cwd)
        task_hash = hashlib.sha256(task.model_dump_json().encode()).hexdigest()
        for label, response in controls(task):
            response_hash = hashlib.sha256(response.encode()).hexdigest()
            if (task.id, label, task_hash, response_hash) in finished:
                continue
            trace = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=response)))
            grade = await grade_in_sandbox(task, GradingAttempt(trace), factory, spec)
            results.append(
                {
                    "task_id": task.id,
                    "source_row": task.source.row,
                    "control": label,
                    "task_sha256": task_hash,
                    "response_sha256": response_hash,
                    **dataclasses.asdict(grade),
                }
            )
            write_results(args.output_dir, results)
            print(task.source.row, label, grade.status, grade.reward, flush=True)

    write_results(args.output_dir, results)
    provenance.update(
        task_count=len({row["task_id"] for row in results}),
        control_count=len(results),
        images=sorted({by_id[row["task_id"]].grader.environment.docker_image for row in results}),
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved {len(results)} controls over {provenance['task_count']} tasks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--converter", type=Path)
    parser.add_argument("--task-ids", type=Path, help="JSON array naming the exact task sample")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--runtime-revision", default=RUNTIME_REVISION)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
