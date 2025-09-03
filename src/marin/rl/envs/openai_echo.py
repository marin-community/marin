"""Environment that queries an OpenAI-compatible chat completion endpoint.

Each iteration sends a fixed prompt (default "Hello!") and records the
assistant's response as a Rollout. Reward is set to 0.0. Useful for
smoke-testing the end-to-end pipeline with the real OpenAI client or the
openai_responses mock.
"""

import time
from dataclasses import dataclass

import openai
import ray
from levanter.utils.ray_utils import RayResources

from ..config import AbstractEnvConfig
from ..datatypes import InferenceEndpoint, Turn, Rollout
from ..env import SimpleEnv


class ChatEchoEnv(SimpleEnv):
    """Simple environment that echoes a single prompt to OpenAI."""

    def __init__(
        self,
        inference: InferenceEndpoint,
        *,
        prompt: str = "Hello!",
        system_prompt: str = "You are a helpful assistant.",
        model: str = "gpt-3.5-turbo",
        max_iters: int | None = None,
        api_key: str | None = None,
    ):
        super().__init__(inference)
        self._prompt = prompt
        self._system_prompt = system_prompt
        self._model = model
        self._max_iters = max_iters
        self._api_key = api_key or openai.api_key

        # Prepare client (v1 openai lib)
        self._client = openai.Client(api_key=self._api_key, base_url=inference.address)

    def do_rollout(self) -> list[Rollout]:
        if not hasattr(self, "_counter"):
            self._counter = 0

        if self._max_iters is not None and self._counter >= self._max_iters:
            time.sleep(0.05)
            return []

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": self._prompt},
            ],
        )

        assistant_msg = completion.choices[0].message.content

        record = Rollout(
            environment="chat_echo_env",
            example_id=f"chat-{self._counter}",
            rollout_uid=f"chat-{self._counter}",
            turns=[
                Turn.system_text(self._system_prompt),
                Turn.from_prompt(self._prompt, input_seed=None),
                Turn.from_openai_response(completion, reward=0.0, input_seed=None),
            ],
            created_ts=time.time(),
            metadata={"response": assistant_msg},
            replica_id="chat",
        )

        self._counter += 1
        return [record]

    async def on_shutdown(self) -> None:
        pass


@dataclass(frozen=True)
class ChatEchoEnvConfig(AbstractEnvConfig):
    """Config for :class:`ChatEchoEnv`."""

    prompt: str = "Hello!"

    def resources(self) -> RayResources:
        return RayResources(cpu=1)

    def build(self, inference: InferenceEndpoint, seed: int) -> ray.actor.ActorHandle:
        ActorCls = ray.remote(num_cpus=1)(ChatEchoEnv)
        actor = ActorCls.remote(inference, prompt=self.prompt)
        return actor
