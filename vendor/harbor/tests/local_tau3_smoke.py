# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a prepared Tau3 task with real models through the local Shellbox adapter."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from harbor.environments.model_relay import JudgeAccess
from harbor.host_capabilities import MODEL_RELAY_PORT
from harbor.host_capabilities import host_capabilities_from_environment
from harbor.job import Job
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import JobConfig


async def run(config: JobConfig, api_key: str) -> None:
    if config.environment.type is not EnvironmentType.LOCAL:
        raise ValueError("This smoke run requires the local Shellbox adapter")
    endpoint = config.environment.kwargs["model_endpoint"]
    model = config.environment.kwargs["model_name"]
    capabilities = {
        "OPENAI_BASE_URL": f"http://127.0.0.1:{MODEL_RELAY_PORT}/v1",
        "OPENAI_API_KEY": "EMPTY",
        "TAU2_USER_MODEL": f"openai/{model}",
        "TAU2_NL_ASSERTIONS_MODEL": f"openai/{model}",
        "TAU2_USER_REASONING_EFFORT": "",
    }
    config = config.model_copy(
        update={
            "environment": config.environment.model_copy(
                update={
                    "kwargs": {
                        **config.environment.kwargs,
                        "judge_access": JudgeAccess.RUNTIME,
                    }
                }
            ),
            "user_agent": (
                config.user_agent.model_copy(update={"env": capabilities})
                if config.user_agent is not None
                else None
            ),
        }
    )
    job = await Job.create(
        config,
        guest_environment=capabilities,
        host_capabilities=host_capabilities_from_environment(
            dict(os.environ), model_authentication={endpoint: api_key}
        ),
    )
    result = await job.run()
    if result.stats.n_errored_trials:
        raise RuntimeError(
            f"Tau3 smoke had {result.stats.n_errored_trials} infrastructure errors"
        )
    job_dir = Path(str(config.jobs_dir)) / config.job_name
    reports = list(job_dir.glob("*/attempts/*/verifier/result.json"))
    if not reports:
        raise RuntimeError("No Tau3 verifier report was collected")
    for path in reports:
        report = json.loads(path.read_text())
        info = report["reward_info"]["info"]
        if (
            not report["used_tau2_evaluator"]
            or not info["runtime_log"]
            or not info["trajectory_message_count"]
        ):
            raise RuntimeError(
                "The verifier did not evaluate a real Tau3 runtime trajectory"
            )
        if not report["reward_info"]["nl_assertions"]:
            raise RuntimeError(
                "Choose a Tau3 task whose canonical reward includes an NL assertion"
            )
        print(
            json.dumps(
                {
                    "report": str(path),
                    "reward": report["reward"],
                    "messages": info["trajectory_message_count"],
                }
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="Harbor JobConfig JSON for prepared local Tau3 tasks"
    )
    parser.add_argument(
        "--api-key-env",
        required=True,
        help="Host-only environment variable holding the model API key",
    )
    args = parser.parse_args()
    config = JobConfig.model_validate_json(args.config.read_text())
    asyncio.run(run(config, os.environ[args.api_key_env]))


if __name__ == "__main__":
    main()
