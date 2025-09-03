"""A minimal *hello-world* environment for Marin RL.

The environment pings the inference server once per second with a trivial
prompt ("Hello, world") and records a single-turn rollout containing the
response.  Rewards are dummy zeros.  This serves as an integration test for the
actor, config, and rollout plumbing.
"""

import logging
import time

import ray
from levanter.utils.ray_utils import RayResources
from ray.actor import ActorHandle

from ..config import AbstractEnvConfig
from ..datatypes import (
    InferenceEndpoint,
    Turn,
    Rollout,
)
from ..env import SimpleEnv

logger = logging.getLogger(__name__)

class HelloWorldEnv(SimpleEnv):
    """Simple environment that produces deterministic dummy rollouts."""

    def do_rollout(self) -> list[Rollout]:
        if not hasattr(self, "_counter"):
            self._counter = 0

        response_text = f"Hello #{self._counter}!"

        record = Rollout(
            environment="hello_env",
            example_id=f"hello-{self._counter}",
            rollout_uid=f"hello-{self._counter}",
            turns=[
                Turn.from_prompt("Hello, world", input_seed=None),
                Turn.assistant_text(
                    response_text,
                    reward=1.0 if self._counter % 2 == 0 else 0.0,
                    input_seed=None,
                ),
            ],
            created_ts=time.time(),
            metadata={},
            replica_id="hello",
        )

        self._counter += 1
        time.sleep(1.0)  # pace output without async knowledge
        return [record]

    async def on_shutdown(self) -> None:
        # Example clean-up
        logger.info("HelloWorldEnv closed")


class HelloEnvConfig(AbstractEnvConfig):
    """Config that spawns :class:`HelloWorldEnv`."""

    def resources(self) -> RayResources:
        return RayResources(cpu=1)

    def build(self, inference: InferenceEndpoint, seed: int) -> ActorHandle:
        ActorCls = ray.remote(num_cpus=1)(HelloWorldEnv)
        actor = ActorCls.remote(inference)
        return actor
