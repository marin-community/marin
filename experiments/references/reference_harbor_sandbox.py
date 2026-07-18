# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference: Harbor's prepackaged hello-world task on Iris sandboxes.

Demonstrates ``marin.harbor.sandbox.iris_sandbox``, the Daytona-style context
manager for Harbor environments on Iris compute: each sandbox is an Iris job
running the task's prebuilt ``docker_image``, bin-packed onto spare host CPU
of cluster workers, and torn down when the context exits.

The task is Harbor's own ``examples/tasks/hello-world``, fetched from
marin-community/harbor at the same commit as the workspace harbor pin (read
from pyproject.toml). Its one incompatibility with the Iris backend (prebuilt
images only) is the ``[environment]`` build spec: upstream builds a Dockerfile
that is just ``FROM ubuntu:24.04`` + ``WORKDIR /app``, which we patch to the
equivalent ``docker_image`` + ``workdir`` after download.

Each episode walks the full Harbor trial shape by hand:

  1. start a sandbox from the task directory
  2. agent: upload ``solution/`` and run ``solve.sh``
  3. verifier: upload ``tests/`` and run ``test.sh`` (installs uv + pytest
     inside the sandbox, so episodes take a minute or two)
  4. download ``reward.txt``

Episodes fan out with plain ``asyncio.gather`` — one context per sandbox, no
pool machinery. Requires the ``harbor`` extra and credentials for the target
Iris cluster.
"""

import asyncio
import json
import logging
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

from harbor.models.task.id import GitTaskId
from harbor.tasks.client import TaskClient
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.namespacing import user_namespaced_name
from marin.harbor.sandbox import iris_sandbox
from rigging.filesystem import StoragePath

logger = logging.getLogger(__name__)

CLUSTER = "marin"
EPISODES = 3
HARBOR_GIT_URL = "https://github.com/marin-community/harbor.git"
HELLO_WORLD_PATH = Path("examples/tasks/hello-world")
# Upstream builds environment/Dockerfile, which is just this base image and
# workdir; the Iris backend (prebuilt images only) takes them directly.
UPSTREAM_ENVIRONMENT_SPEC = "[environment]\n"
PATCHED_ENVIRONMENT_SPEC = '[environment]\ndocker_image = "ubuntu:24.04"\nworkdir = "/app"\n'
# Where the task's test.sh writes its score inside the sandbox.
REWARD_PATH = "/logs/verifier/reward.txt"


def harbor_pin() -> str:
    """The workspace harbor git pin — the task is fetched at the same commit."""
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    with open(pyproject, "rb") as f:
        return tomllib.load(f)["tool"]["uv"]["sources"]["harbor"]["rev"]


async def fetch_hello_world_task(output_dir: Path) -> Path:
    """Download harbor's hello-world at the pinned commit and patch it to a prebuilt image."""
    task_id = GitTaskId(git_url=HARBOR_GIT_URL, git_commit_id=harbor_pin(), path=HELLO_WORLD_PATH)
    await TaskClient().download_tasks([task_id], output_dir=output_dir, export=True)
    task_dir = output_dir / HELLO_WORLD_PATH.name

    config_path = task_dir / "task.toml"
    config = config_path.read_text()
    assert UPSTREAM_ENVIRONMENT_SPEC in config, f"unexpected task.toml layout:\n{config}"
    config_path.write_text(config.replace(UPSTREAM_ENVIRONMENT_SPEC, PATCHED_ENVIRONMENT_SPEC, 1))
    return task_dir


@dataclass(frozen=True)
class HelloWorldConfig:
    cluster: str
    episodes: int
    output_path: str


async def _run_episode(task_dir: Path, cluster: str, index: int) -> float:
    """One Harbor trial by hand: agent solves, verifier scores, reward comes back."""
    async with iris_sandbox(
        task_dir=task_dir,
        cluster=cluster,
        name=f"hello-world-{index}",
        # The default gvisor profile gives the verifier's apt full in-container
        # root under runsc; requires runsc-equipped workers (post-#7339 boots).
    ) as sandbox:
        await sandbox.upload_dir(task_dir / "solution", "/solution")
        agent = await sandbox.exec("bash /solution/solve.sh")
        assert agent.return_code == 0, agent

        # A real Harbor trial creates the verifier log dir; do the same by hand.
        await sandbox.ensure_dirs([str(Path(REWARD_PATH).parent)])
        await sandbox.upload_dir(task_dir / "tests", "/tests")
        verifier = await sandbox.exec("bash /tests/test.sh")
        assert verifier.return_code == 0, verifier

        with tempfile.TemporaryDirectory() as tmp:
            reward_path = Path(tmp) / "reward.txt"
            await sandbox.download_file(REWARD_PATH, reward_path)
            reward = float(reward_path.read_text())
    logger.info("episode %d reward=%s", index, reward)
    return reward


def run_hello_world(config: HelloWorldConfig) -> None:
    async def run() -> list[float]:
        with tempfile.TemporaryDirectory(prefix="harbor-hello-world-") as tmp:
            task_dir = await fetch_hello_world_task(Path(tmp))
            episodes = (_run_episode(task_dir, config.cluster, i) for i in range(config.episodes))
            return list(await asyncio.gather(*episodes))

    rewards = asyncio.run(run())
    results = {"rewards": rewards, "mean_reward": sum(rewards) / len(rewards)}
    (StoragePath(config.output_path) / "results.json").write_text(json.dumps(results, indent=2))


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
