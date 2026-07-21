# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a Harbor registry dataset against an already served model.

The served-model analog of ``marin.evaluation.lm_eval``: the agent (an LLM
loop such as terminus-2) runs in this process and talks to the model's
OpenAI-compatible endpoint, while each trial's sandbox runs as an Iris job via
``marin.harbor.iris_environment.IrisEnvironment``. Must run inside an Iris job
so sandboxes can reach the controller directly.
"""

import asyncio
import dataclasses
import json
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from iris.cluster.client.job_info import get_job_info

from marin.execution.artifact import Artifact
from marin.inference.types import RunningModel

IRIS_ENVIRONMENT_IMPORT_PATH = "marin.harbor.iris_environment:IrisEnvironment"
HARBOR_JOB_NAME = "harbor"

# hosted_vllm agents require token limits and (free) serving costs up front.
DEFAULT_HOSTED_VLLM_MODEL_INFO: Mapping[str, int | float] = {
    "max_input_tokens": 32768,
    "max_output_tokens": 8192,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}


class HarborEvalResults(Artifact):
    """A lazy reference to a Harbor job's results and trial artifacts."""

    results_path: str


@dataclass(frozen=True)
class HarborEvalRun:
    """A single Harbor dataset run against an already served model."""

    dataset: str
    dataset_version: str
    registry_url: str
    # Sandbox security profile; see marin.harbor.iris_environment.IrisEnvironment.
    container_profile: str
    agent_name: str = "terminus-2"
    n_tasks: int | None = None
    n_concurrent_trials: int = 4
    model_info: Mapping[str, int | float] = field(default_factory=lambda: dict(DEFAULT_HOSTED_VLLM_MODEL_INFO))


def run_harbor_eval(model: RunningModel, run: HarborEvalRun, output_path: str):
    """Run the dataset's trials against the served model and persist results.

    Harbor writes its full job output (per-trial logs, ``result.json``) under
    ``output_path``; a flat ``results.json`` summary is added at the root.
    Returns the harbor ``JobResult``.
    """
    from harbor.job import Job  # noqa: PLC0415  # optional dep: harbor
    from harbor.models.job.config import DatasetConfig, JobConfig  # noqa: PLC0415  # optional dep: harbor
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig  # noqa: PLC0415  # optional dep: harbor

    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("run_harbor_eval must run inside an Iris job.")

    config = JobConfig(
        job_name=HARBOR_JOB_NAME,
        jobs_dir=Path(output_path),
        datasets=[
            DatasetConfig(
                name=run.dataset,
                version=run.dataset_version,
                registry_url=run.registry_url,
                n_tasks=run.n_tasks,
            )
        ],
        agents=[
            AgentConfig(
                name=run.agent_name,
                model_name=f"hosted_vllm/{model.endpoint.model}",
                kwargs={
                    "api_base": model.endpoint.base_url,
                    "model_info": dict(run.model_info),
                },
            )
        ],
        environment=EnvironmentConfig(
            import_path=IRIS_ENVIRONMENT_IMPORT_PATH,
            kwargs={
                "controller_url": job_info.controller_address,
                "container_profile": run.container_profile,
            },
        ),
        n_concurrent_trials=run.n_concurrent_trials,
        quiet=True,
    )

    async def create_and_run():
        job = await Job.create(config)
        return await job.run()

    result = asyncio.run(create_and_run())
    summary_path = Path(output_path) / "results.json"
    summary_path.write_text(json.dumps(dataclasses.asdict(summarize_job_result(result)), indent=2))
    return result


@dataclass(frozen=True)
class TrialSummary:
    task_name: str
    # None when the verifier produced no reward (counted as 0.0 in the mean).
    reward: float | None
    exception: str | None


@dataclass(frozen=True)
class HarborEvalSummary:
    n_total_trials: int
    n_errored_trials: int
    mean_reward: float | None
    trials: dict[str, TrialSummary]


def summarize_job_result(result) -> HarborEvalSummary:
    """Flatten a Harbor ``JobResult`` into per-trial rewards and a mean."""
    trials = {}
    rewards = []
    for trial in result.trial_results:
        reward = None
        if trial.verifier_result is not None and trial.verifier_result.rewards:
            reward = trial.verifier_result.rewards.get("reward")
        reward = float(reward) if isinstance(reward, int | float) else None
        rewards.append(reward if reward is not None else 0.0)
        trials[trial.trial_name] = TrialSummary(
            task_name=trial.task_name,
            reward=reward,
            exception=trial.exception_info.exception_type if trial.exception_info else None,
        )
    return HarborEvalSummary(
        n_total_trials=result.n_total_trials,
        n_errored_trials=result.stats.n_errored_trials,
        mean_reward=statistics.mean(rewards) if rewards else None,
        trials=trials,
    )
