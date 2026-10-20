"""Run pinned Harbor validation jobs from the trainer's installed runtime."""

import argparse
import asyncio
import json
import math
import signal
import socket
import subprocess
import traceback
from pathlib import Path

from harbor.job import Job
from harbor.models.job.config import JobConfig
from harbor.models.job.result import JobResult
from harbor.models.trial.config import TaskConfig
from skyrl_train.inline_harbor_client import CONTROL_MESSAGE_LIMIT


class HarborTaskCache:
    """Prepare each immutable cohort before its concurrent trials start."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._prepared: set[tuple[str, ...]] = set()

    async def prepare(self, config: JobConfig) -> JobConfig:
        key = tuple(task.model_dump_json() for task in config.tasks)
        async with self._lock:
            if key not in self._prepared:
                groups: dict[tuple[bool, Path | None], list[TaskConfig]] = {}
                for task in config.tasks:
                    groups.setdefault((task.overwrite, task.download_dir), []).append(task)
                for tasks in groups.values():
                    await Job._cache_tasks(tasks)
                self._prepared.add(key)
        # Honor overwrite once for the cohort, then reuse the prepared inputs.
        return config.model_copy(
            update={"tasks": [task.model_copy(update={"overwrite": False}) for task in config.tasks]}
        )


def validation_metrics(result: JobResult, config: JobConfig, reward_key: str) -> dict[str, float]:
    """Count only completed, verified trials; expose missing scores separately."""
    expected = {str(task.get_task_id()) for task in config.tasks}
    observed = [str(trial.task_id) for trial in result.trial_results]
    if len(expected) != len(config.tasks) or config.n_attempts != 1:
        raise ValueError("Inline validation requires unique tasks and one attempt per task")
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("Harbor validation returned missing, duplicate, or unexpected tasks")
    scores = []
    for trial in result.trial_results:
        if trial.exception_info is not None and trial.exception_info.exception_type != "AgentTimeoutError":
            continue
        verifier = trial.verifier_result
        if verifier is None or verifier.rewards is None or reward_key not in verifier.rewards:
            continue
        score = float(verifier.rewards[reward_key])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid normalized reward for {trial.task_name}: {score}")
        scores.append(score)
    return {
        "num_attempted": float(len(expected)),
        "num_scored": float(len(scores)),
        "reward_sum": math.fsum(scores),
        "avg_verifier_score": math.fsum(scores) / len(scores) if scores else 0.0,
    }


async def serve_trials(control_fd: int) -> None:
    """Execute only trials admitted by the parent; stdout remains available for Harbor logs."""
    connection = socket.socket(fileno=control_fd)
    connection.setblocking(False)
    reader, writer = await asyncio.open_connection(sock=connection, limit=CONTROL_MESSAGE_LIMIT)
    current = asyncio.current_task()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, current.cancel)
    task_cache = HarborTaskCache()

    async def execute(message):
        try:
            cohort = message["cohort"]
            config = JobConfig.model_validate(cohort["job"])
            if config.datasets or not config.tasks or len(config.agents) != 1 or config.n_attempts != 1:
                raise ValueError("Inline validation requires explicit tasks, one agent and one attempt")
            task_ids = [str(task.get_task_id()) for task in config.tasks]
            if len(task_ids) != len(set(task_ids)):
                raise ValueError("duplicate validation tasks")
            if any(not task.git_url or not task.git_commit_id for task in config.tasks):
                raise ValueError("Inline validation task sources must have immutable Git revisions")
            config = await task_cache.prepare(config)
            index = message["index"]
            config = config.model_copy(
                update={
                    "tasks": [config.tasks[index]],
                    "n_concurrent_trials": 1,
                    "job_name": f"{config.job_name}-task-{index:05d}",
                }
            )
            job = await Job.create(config)
            result = await job.run()
            response = {"id": message["id"], "metrics": validation_metrics(result, config, cohort["reward_key"])}
        except Exception as error:
            if isinstance(error, subprocess.CalledProcessError) and error.stderr:
                error.add_note(f"Subprocess stderr: {error.stderr!r}")
            response = {"id": message["id"], "error": traceback.format_exc()}
        writer.write((json.dumps(response) + "\n").encode())
        await writer.drain()

    try:
        async with asyncio.TaskGroup() as tasks:
            while line := await reader.readline():
                tasks.create_task(execute(json.loads(line)))
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
        writer.close()
        await writer.wait_closed()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-fd", type=int, required=True)
    args = parser.parse_args()
    asyncio.run(serve_trials(args.control_fd))


if __name__ == "__main__":
    main()
