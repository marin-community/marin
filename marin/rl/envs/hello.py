"""A minimal *hello-world* environment for Marin RL.

The environment pings the inference server once per second with a trivial
prompt ("Hello, world") and records a single-turn rollout containing the
response.  Rewards are dummy zeros.  This serves as an integration test for the
actor, config, and rollout plumbing.
"""

import asyncio
import time

import ray
from levanter.utils.ray_utils import RayResources

from ..config import AbstractEnvConfig
from ..env import AbstractMarinEnv
from ..types import (
    InferenceEndpoint,
    Rollout,
    RolloutGroup,
    RolloutSink,
    Turn,
)


class HelloWorldEnv(AbstractMarinEnv):
    """Simple environment that produces deterministic dummy rollouts."""

    async def run(self) -> None:
        counter = 0
        while not await self._should_stop():
            # Simulate calling the inference server (here we just echo text)
            response_text = f"Hello #{counter} from {self._inference.address}"

            turn = Turn(
                message=response_text,
                role="assistant",
                logprobs=None,
                reward=0.0,
                inference_metadata={"model": "dummy"},
            )
            rollout = Rollout(turns=[turn], metadata={"iteration": counter})
            group = RolloutGroup(
                id=f"hello-{counter}",
                source="hello_env",
                created=time.time(),
                rollouts=[rollout],
                metadata={},
            )
            self._send_rollouts(group)

            counter += 1
            await asyncio.sleep(1.0)  # yield control and pace output

    async def close(self) -> None:
        # Example clean-up
        print("HelloWorldEnv closed")


class HelloEnvConfig(AbstractEnvConfig):
    """Config that spawns :class:`HelloWorldEnv`."""

    def resources(self) -> RayResources:
        return RayResources(cpu=1)

    def build(self, inference: InferenceEndpoint, rollout_sink: RolloutSink, replica_id: int):
        ActorCls = ray.remote(num_cpus=1)(HelloWorldEnv)
        actor = ActorCls.remote(inference, rollout_sink)
        actor.run.remote()  # kick off event loop
        return actor
