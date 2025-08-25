"""Environment that queries an OpenAI-compatible chat completion endpoint.

Each iteration sends a fixed prompt (default "Hello!") and records the
assistant's response. Reward is set to 0.0.  Useful for smoke-testing the
end-to-end pipeline with the real OpenAI client or the openai_responses mock.
"""

import asyncio
import time
from dataclasses import dataclass

import openai
import ray
from levanter.utils.ray_utils import RayResources

from ..config import AbstractEnvConfig
from ..datatypes import InferenceEndpoint, RolloutGroup, RolloutRecord, RolloutSink, Turn
from ..env import AbstractMarinEnv


class ChatEchoEnv(AbstractMarinEnv):
    """Simple environment that echoes a single prompt to OpenAI."""

    def __init__(
        self,
        inference: InferenceEndpoint,
        rollout_sink: RolloutSink,
        *,
        prompt: str = "Hello!",
        system_prompt: str = "You are a helpful assistant.",
        model: str = "gpt-3.5-turbo",
        max_iters: int | None = None,
        api_key: str | None = None,
    ):
        super().__init__(inference, rollout_sink)
        self._prompt = prompt
        self._system_prompt = system_prompt
        self._model = model
        self._max_iters = max_iters
        self._api_key = api_key or openai.api_key

        # Prepare client (v1 openai lib)
        self._client = openai.Client(api_key=self._api_key, base_url=inference.address)

    async def run(self) -> None:
        counter = 0
        while True:
            if not await self._wait_ready():
                break
            if self._max_iters is not None and counter >= self._max_iters:
                break

            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": self._prompt},
                ],
            )

            assistant_msg = completion.choices[0].message.content

            record = RolloutRecord(
                environment="chat_echo_env",
                example_id=f"chat-{counter}",
                policy_version="v0",
                rollout_uid=f"chat-{counter}",
                replica_id="chat",
                turns=[
                    Turn.system_text(self._system_prompt),
                    Turn.from_prompt(self._prompt, input_seed=None),
                    Turn.from_openai_response(completion, reward=0.0, input_seed=None),
                ],
                metadata={"response": assistant_msg},
                created_ts=time.time(),
            )
            group = RolloutGroup(
                id=f"chat-{counter}",
                environment="chat_echo_env",
                example_id=f"chat-{counter}",
                policy_version="v0",
                rollouts=[record],
                sealed_ts=time.time(),
                metadata={},
            )
            self._rollout_sink([group])

            counter += 1
            await asyncio.sleep(0)  # yield control

    async def on_shutdown(self) -> None:
        pass


@dataclass(frozen=True)
class ChatEchoEnvConfig(AbstractEnvConfig):
    """Config for :class:`ChatEchoEnv`."""

    prompt: str = "Hello!"

    def resources(self) -> RayResources:
        return RayResources(cpu=1)

    def build(self, inference: InferenceEndpoint, rollout_sink: RolloutSink, seed: int) -> ray.actor.ActorHandle:
        ActorCls = ray.remote(num_cpus=1)(ChatEchoEnv)
        actor = ActorCls.remote(inference, rollout_sink, prompt=self.prompt)
        actor.run.remote()
        return actor
