# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference: Harbor's prepackaged hello-world task on Iris sandboxes.

Demonstrates ``marin.harbor.sandbox.iris_sandbox``, the Daytona-style context
manager for Harbor environments on Iris compute: each sandbox is an Iris job
running the task's prebuilt ``docker_image``, bin-packed onto spare host CPU
of cluster workers, and torn down when the context exits.

The task in ``harbor_hello_world/`` is Harbor's own ``examples/tasks/hello-world``,
vendored verbatim except for the ``[environment]`` build spec: upstream builds a
Dockerfile that is just ``FROM ubuntu:24.04`` + ``WORKDIR /app``, which the Iris
backend (prebuilt images only) expresses as ``docker_image`` + ``workdir``.
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
from dataclasses import dataclass
from pathlib import Path

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.namespacing import user_namespaced_name
from marin.harbor.sandbox import iris_sandbox
from rigging.filesystem import StoragePath

logger = logging.getLogger(__name__)

CLUSTER = "marin"
EPISODES = 3
TASK_DIR = Path(__file__).parent / "harbor_hello_world"
# Where the task's test.sh writes its score inside the sandbox.
REWARD_PATH = "/logs/verifier/reward.txt"


@dataclass(frozen=True)
class HelloWorldConfig:
    cluster: str
    episodes: int
    output_path: str


async def _run_episode(cluster: str, index: int) -> float:
    """One Harbor trial by hand: agent solves, verifier scores, reward comes back."""
    async with iris_sandbox(
        task_dir=TASK_DIR,
        cluster=cluster,
        name=f"hello-world-{index}",
        # The task's verifier installs packages with apt, which needs setuid;
        # Iris's default profile drops all capabilities, so opt up. Admin-gated.
        container_profile="privileged",
    ) as sandbox:
        await sandbox.upload_dir(TASK_DIR / "solution", "/solution")
        agent = await sandbox.exec("bash /solution/solve.sh")
        assert agent.return_code == 0, agent

        # A real Harbor trial creates the verifier log dir; do the same by hand.
        await sandbox.ensure_dirs([str(Path(REWARD_PATH).parent)])
        await sandbox.upload_dir(TASK_DIR / "tests", "/tests")
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
        episodes = (_run_episode(config.cluster, i) for i in range(config.episodes))
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
