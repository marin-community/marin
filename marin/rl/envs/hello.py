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
from ..env import SimpleEnv
from ..types import (
    InferenceEndpoint,
    Rollout,
    RolloutGroup,
    RolloutSink,
    Turn,
)

logger = logging.getLogger(__name__)


class HelloWorldEnv(SimpleEnv):
    """Simple environment that produces deterministic dummy rollouts."""

    def do_rollout(self) -> list[RolloutGroup]:
        if not hasattr(self, "_counter"):
            self._counter = 0

        # Simulate calling the inference server (here we just echo text)
        response_text = f"Hello #{self._counter} from {self._inference.address}"

        turn = Turn(
            message=response_text,
            role="assistant",
            logprobs=None,
            reward=0.0,
            inference_metadata={"model": "dummy"},
        )
        rollout = Rollout(turns=[turn], metadata={"iteration": self._counter})
        group = RolloutGroup(
            id=f"hello-{self._counter}",
            source="hello_env",
            created=time.time(),
            rollouts=[rollout],
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
