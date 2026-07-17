# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference: Harbor's hello-world task on Iris sandboxes.

Demonstrates ``marin.harbor.sandbox.iris_sandbox``, the Daytona-style context
manager for Harbor environments on Iris compute: each sandbox is an Iris job
running the task's prebuilt ``docker_image``, bin-packed onto spare host CPU
of cluster workers, and torn down when the context exits.

The task is Harbor's canonical hello-world (write ``hello.txt``, verify its
contents), inlined below so the reference is self-contained. Each episode
walks the full Harbor trial shape by hand:

  1. start a sandbox from the task directory (``[environment]`` in task.toml)
  2. agent: upload ``solution/`` and run ``solve.sh``
  3. verifier: upload ``tests/`` and run ``test.sh``
  4. download ``reward.txt``

Episodes fan out with plain ``asyncio.gather`` — one context per sandbox, no
pool machinery. Requires the ``harbor`` extra and credentials for the target
Iris cluster.
"""

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.namespacing import user_namespaced_name
from marin.harbor.sandbox import iris_sandbox
from rigging.filesystem import open_url

logger = logging.getLogger(__name__)

CLUSTER = "marin"
EPISODES = 3

_TASK_TOML = """\
schema_version = "1.1"

[task]
name = "marin/hello-world"
description = "Harbor's hello-world on a prebuilt image."
authors = []
keywords = []

[environment]
docker_image = "python:3.13-slim"
cpus = 1
memory_mb = 1024
storage_mb = 2048
allow_internet = true

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 120.0
"""

_INSTRUCTION = 'Create a file hello.txt in /app containing exactly "Hello, world!".\n'

_SOLVE_SH = """\
#!/bin/bash
mkdir -p /app
echo "Hello, world!" > /app/hello.txt
"""

_TEST_SH = """\
#!/bin/bash
mkdir -p /logs/verifier
if grep -q "Hello, world!" /app/hello.txt; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""


def _write_hello_world_task(task_dir: Path) -> None:
    """Materialize Harbor's hello-world task layout (task.toml, solution/, tests/)."""
    (task_dir / "solution").mkdir(parents=True)
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "task.toml").write_text(_TASK_TOML)
    (task_dir / "instruction.md").write_text(_INSTRUCTION)
    (task_dir / "solution" / "solve.sh").write_text(_SOLVE_SH)
    (task_dir / "tests" / "test.sh").write_text(_TEST_SH)


@dataclass(frozen=True)
class HelloWorldConfig:
    cluster: str
    episodes: int
    output_path: str


async def _run_episode(task_dir: Path, cluster: str, index: int) -> float:
    """One Harbor trial by hand: agent solves, verifier scores, reward comes back."""
    async with iris_sandbox(task_dir=task_dir, cluster=cluster, name=f"hello-world-{index}") as sandbox:
        await sandbox.upload_dir(task_dir / "solution", "/solution")
        agent = await sandbox.exec("bash /solution/solve.sh")
        assert agent.return_code == 0, agent

        await sandbox.upload_dir(task_dir / "tests", "/tests")
        verifier = await sandbox.exec("bash /tests/test.sh")
        assert verifier.return_code == 0, verifier

        with tempfile.TemporaryDirectory() as tmp:
            reward_path = Path(tmp) / "reward.txt"
            await sandbox.download_file("/logs/verifier/reward.txt", reward_path)
            reward = float(reward_path.read_text())
    logger.info("episode %d reward=%s", index, reward)
    return reward


def run_hello_world(config: HelloWorldConfig) -> None:
    async def run() -> list[float]:
        with tempfile.TemporaryDirectory(prefix="hello-world-task-") as tmp:
            task_dir = Path(tmp)
            _write_hello_world_task(task_dir)
            episodes = (_run_episode(task_dir, config.cluster, i) for i in range(config.episodes))
            return list(await asyncio.gather(*episodes))

    rewards = asyncio.run(run())
    with open_url(os.path.join(config.output_path, "results.json"), "w") as f:
        json.dump({"rewards": rewards, "mean_reward": sum(rewards) / len(rewards)}, f, indent=2)


def build(*, version: str = "dev") -> ArtifactStep[Artifact]:
    """Harbor's hello-world, run as concurrent Iris sandboxes, as a lazy artifact."""

    def build_config(ctx: StepContext) -> HelloWorldConfig:
        return HelloWorldConfig(
            cluster=CLUSTER,
            episodes=EPISODES,
            output_path=ctx.output_path,
        )

    return ArtifactStep(
        name=user_namespaced_name("references/harbor-sandbox", version),
        version=version,
        artifact_type=Artifact,
        run=run_hello_world,
        build_config=build_config,
    )


if __name__ == "__main__":
    StepRunner().run([build().lower()])
