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
    RolloutGroup,
    RolloutRecord,
    RolloutSink,
    Turn,
)
from ..env import SimpleEnv

logger = logging.getLogger(__name__)


class HelloWorldEnv(SimpleEnv):
    """Simple environment that produces deterministic dummy rollouts."""

    def do_rollout(self) -> list[RolloutGroup]:
        if not hasattr(self, "_counter"):
            self._counter = 0

        response_text = f"Hello #{self._counter}!"

        record = RolloutRecord(
            environment="hello_env",
            example_id=f"hello-{self._counter}",
            policy_version="v0",
            rollout_uid=f"hello-{self._counter}",
            replica_id="hello",
            turns=[
                Turn(
                    message="Hello, world",
                    logprobs=None,
                    role="user",
                    reward=None,
                    inference_metadata={},
                ),
                Turn(
                    message=response_text,
                    logprobs=None,
                    role="assistant",
                    reward=1 if self._counter % 2 == 0 else 0,
                    inference_metadata={},
                ),
            ],
            metadata={},
            created_ts=time.time(),
        )
        group = RolloutGroup(
            id=f"hello-{self._counter}",
            environment="hello_env",
            example_id=f"hello-{self._counter}",
            policy_version="v0",
            rollouts=[record],
            sealed_ts=time.time(),
            metadata={},
        )

        self._counter += 1
        time.sleep(1.0)  # pace output without async knowledge
        return [group]

    async def shutdown(self) -> None:
        # Example clean-up
        logger.info("HelloWorldEnv closed")


class HelloEnvConfig(AbstractEnvConfig):
    """Config that spawns :class:`HelloWorldEnv`."""

    def resources(self) -> RayResources:
        return RayResources(cpu=1)

    def build(self, inference: InferenceEndpoint, rollout_sink: RolloutSink, seed: int) -> ActorHandle:
        ActorCls = ray.remote(num_cpus=1)(HelloWorldEnv)
        actor = ActorCls.remote(inference, rollout_sink)
        actor.run.remote()  # kick off event loop
        return actor
