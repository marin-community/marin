"""Math environment adapter for Marin RL.

This environment samples problems from the MATH dataset (via the
`DigitalLearningGmbH/MATH-lighteval` split), queries an
OpenAI-compatible inference endpoint for answers, evaluates the answer
against the dataset's ground-truth solution, and emits the interaction
as a :class:`~marin.rl.types.RolloutGroup`.

Both the dataset loading and answer checking logic are largely ported
from `marin.post_training.environments.math_env.MathEnv` but translated
to the new async push-based API defined in
:pyclass:`marin.rl.env.AbstractMarinEnv`.
"""

import asyncio
import random
import time
from dataclasses import dataclass

import datasets
import openai
import ray
from levanter.utils.ray_utils import RayResources

from marin.post_training.environments.math_utils import (
    grade_answer,
    last_boxed_only_string,
    remove_boxed,
)

# Utilities for answer formatting / grading live in the post-training package.
from marin.post_training.utils import validate_format

from ..config import AbstractEnvConfig
from ..env import AbstractMarinEnv
from ..types import (
    InferenceEndpoint,
    Rollout,
    RolloutGroup,
    RolloutSink,
    Turn,
)

__all__ = [
    "MathEnv",
    "MathEnvConfig",
]


class MathEnv(AbstractMarinEnv):
    """Environment that plays MATH problems and evaluates correctness.

    Each iteration samples a single problem (uniformly without
    replacement), sends the prompt to the inference endpoint, and
    constructs a two-turn rollout (user + assistant) with a reward of
    ``1.0`` for a correct answer and ``0.0`` otherwise.  Formatting
    errors (missing <answer> tags, etc.) are treated as incorrect.
    """

    # Static instruction appended to every problem.
    _INSTRUCTION = (
        "Show your work in <think> </think> tags. And return the final "
        "answer in <answer> </answer> tags. Assistant: Let me solve this step by step. <think>"
    )

    def __init__(
        self,
        inference: InferenceEndpoint,
        rollout_sink: RolloutSink,
        *,
        data_source: str,
        split: str = "train",  # "train" or "test"
        model: str = "gpt-3.5-turbo",
        max_iters: int | None = None,
        api_key: str | None = None,
        seed: int = 0,
    ):
        super().__init__(inference, rollout_sink)

        self._data_source = data_source
        self._split = split
        self._model = model
        self._max_iters = max_iters

        # Load dataset: this can take a couple seconds on first run.
        dataset = datasets.load_dataset(self._data_source, trust_remote_code=True)[split]

        # Pre-process into a list of dicts {prompt, answer} so that the async
        # event loop isn't doing heavy work every iteration.
        self._examples: list[dict[str, str]] = []
        for item in dataset:
            prompt = f"{item['problem']} {self._INSTRUCTION}"
            answer = remove_boxed(last_boxed_only_string(item["solution"]))
            self._examples.append({"prompt": prompt, "answer": answer})

        # Deterministic RNG (per-actor) so runs are reproducible.
        self._rng: random.Random = random.Random(seed)

        # Shuffle once to avoid skew (sampling without replacement below).
        self._rng.shuffle(self._examples)
        self._example_idx = 0  # pointer into the shuffled list

        # Prepare OpenAI client for the target inference server.
        self._client = openai.Client(api_key=api_key or openai.api_key, base_url=inference.address)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        iteration = 0
        while not await self._should_stop():
            if self._max_iters is not None and iteration >= self._max_iters:
                break

            # ------------------------------------------------------------------
            # Sample a problem (reuse the dataset when exhausted).
            # ------------------------------------------------------------------
            if self._example_idx >= len(self._examples):
                self._rng.shuffle(self._examples)
                self._example_idx = 0
            example = self._examples[self._example_idx]
            self._example_idx += 1

            user_prompt: str = example["prompt"]
            gt_answer: str = example["answer"]

            # ------------------------------------------------------------------
            # Call inference server.
            # ------------------------------------------------------------------
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )

            assistant_msg: str = completion.choices[0].message.content

            # ------------------------------------------------------------------
            # Reward calculation.
            # ------------------------------------------------------------------
            parsed = validate_format(assistant_msg + ">")  # util expects trailing >
            is_valid = parsed["is_valid"]
            extracted_answer = parsed["answer"]
            is_correct = grade_answer(extracted_answer, gt_answer) if is_valid else False
            reward = float(is_correct)

            # ------------------------------------------------------------------
            # Build rollout & emit.
            # ------------------------------------------------------------------
            turns = [
                Turn(
                    message=user_prompt,
                    role="user",
                    logprobs=None,
                    reward=None,
                    inference_metadata={},
                ),
                Turn(
                    message=assistant_msg,
                    role="assistant",
                    logprobs=None,  # logprobs not available via OpenAI client
                    reward=reward,
                    inference_metadata={
                        "model": self._model,
                        "finish_reason": completion.choices[0].finish_reason,
                    },
                ),
            ]
            rollout = Rollout(turns=turns, metadata={"problem": user_prompt})
            group = RolloutGroup(
                id=f"math-{iteration}",
                source="math_env",
                created=time.time(),
                rollouts=[rollout],
                metadata={"valid_format": is_valid, "correct": is_correct},
            )

            # Push to learner / sink.
            self._send_rollouts(group)

            iteration += 1
            await asyncio.sleep(0)  # yield control to Ray scheduler

    # ------------------------------------------------------------------
    # Optional cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MathEnvConfig(AbstractEnvConfig):
    """Configuration wrapper that spawns :class:`MathEnv`."""

    data_source: str = "DigitalLearningGmbH/MATH-lighteval"
    split: str = "train"
    model: str = "gpt-3.5-turbo"
    max_iters: int | None = None
    seed: int = 0

    def resources(self) -> RayResources:
        return RayResources(cpu=1)

    def build(self, inference: InferenceEndpoint, rollout_sink: RolloutSink):
        ActorCls = ray.remote(num_cpus=1)(MathEnv)
        actor = ActorCls.remote(
            inference,
            rollout_sink,
            data_source=self.data_source,
            split=self.split,
            model=self.model,
            max_iters=self.max_iters,
            seed=self.seed,
        )
        actor.run.remote()
        return actor
