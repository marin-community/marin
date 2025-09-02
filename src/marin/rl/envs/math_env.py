"""Math environment adapter for Marin RL.

This environment samples problems from the MATH dataset (via the
`DigitalLearningGmbH/MATH-lighteval` split), queries an
OpenAI-compatible inference endpoint for answers, evaluates the answer
against the dataset's ground-truth solution, and emits the interaction
as a :class:`~marin.rl.datatypes.RolloutGroup`.

Both the dataset loading and answer checking logic are largely ported
from `marin.post_training.environments.math_env.MathEnv` but translated
to the new async push-based API defined in
:pyclass:`marin.rl.env.SimpleEnv`.
"""

import asyncio
import random
import uuid
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
from ..datatypes import (
    InferenceEndpoint,
    RolloutRecord,
    RolloutSink,
    Turn, Rollout,
)
from ..env import SimpleEnv

__all__ = [
    "MathEnv",
    "MathEnvConfig",
]


class MathEnv(SimpleEnv):
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
        num_generations: int = 1,
        api_key: str | None = None,
        seed: int = 0,
    ):
        super().__init__(inference, rollout_sink)

        self._data_source = data_source
        self._split = split
        self._num_generations = max(1, int(num_generations))

        # Load dataset: this can take a couple seconds on first run.
        dataset = datasets.load_dataset(self._data_source, trust_remote_code=True)[split]

        # Pre-process into a list of dicts {prompt, answer} so that the async
        # event loop isn't doing heavy work every iteration.
        self._examples: list[dict[str, str]] = []
        for i, item in enumerate(dataset):
            prompt = f"{item['problem']} {self._INSTRUCTION}"
            answer = remove_boxed(last_boxed_only_string(item["solution"]))
            # Keep the original (unshuffled) index as the stable example_id
            self._examples.append({"prompt": prompt, "answer": answer, "example_id": str(i)})

        # Deterministic RNG (per-actor) so runs are reproducible.
        self._rng: random.Random = random.Random(seed)

        # Shuffle once to avoid skew (sampling without replacement below).
        self._rng.shuffle(self._examples)
        self._example_idx = 0  # pointer into the shuffled list

        # Prepare OpenAI client for the target inference server.
        self._client = openai.Client(api_key=api_key, base_url=inference.address)

    @property
    def env_name(self) -> str:
        return f"math_env(data_source={self._data_source},split={self._split})"

    def do_rollout(self) -> list[Rollout]:
        # Sample a problem (reuse the dataset when exhausted).
        if self._example_idx >= len(self._examples):
            self._rng.shuffle(self._examples)
            self._example_idx = 0
        example = self._examples[self._example_idx]
        self._example_idx += 1

        user_prompt: str = example["prompt"]
        gt_answer: str = example["answer"]
        example_id: str = example["example_id"]
        seed = self._rng.randint(0, 2**31 - 1)

        # Call inference server for N generations (one request with n when possible).
        completion = self._client.chat.completions.create(
            model=self._inference.model,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            n=self._num_generations,
            seed=seed,
        )

        # Build multiple rollouts (one per generation) and a single group per prompt.
        rollouts: list[RolloutRecord] = []
        valid_list: list[bool] = []
        correct_list: list[bool] = []

        class _SingleChoiceResponse:
            def __init__(self, base, choice):
                self.choices = [choice]
                self.created = getattr(base, "created", None)
                self.model = getattr(base, "model", None)
                self.usage = getattr(base, "usage", None)

        for j, choice in enumerate(completion.choices):
            assistant_msg: str = choice.message.content
            parsed = validate_format(assistant_msg + ">")
            is_valid = bool(parsed["is_valid"])  # type: ignore[index]
            extracted_answer = parsed["answer"]
            is_correct = grade_answer(extracted_answer, gt_answer) if is_valid else False
            reward = float(is_correct)

            single_resp = _SingleChoiceResponse(completion, choice)

            record = RolloutRecord(
                environment=self.env_name,
                example_id=example_id,
                rollout_uid=f"math-{example_id}-g{j}-{uuid.uuid4().hex}",
                replica_id="math",
                turns=[
                    Turn.from_prompt(user_prompt, input_seed=seed),
                    Turn.from_openai_response(single_resp, reward=reward, input_seed=seed),
                ],
                metadata={
                    "valid_format": is_valid,
                    "correct": is_correct,
                    "generation_index": j,
                    "seed": seed,
                },
                created_ts=time.time(),
            )
            rollouts.append(record)
            valid_list.append(is_valid)
            correct_list.append(is_correct)

        return [


        ]


@dataclass(frozen=True)
class MathEnvConfig(AbstractEnvConfig):
    """Configuration wrapper that spawns :class:`MathEnv`."""

    data_source: str = "DigitalLearningGmbH/MATH-lighteval"
    split: str = "train"
    num_generations: int = 1
    seed: int = 0

    def resources(self) -> RayResources:
        return RayResources(cpu=1)

    def build(self, inference: InferenceEndpoint, rollout_sink: RolloutSink, seed: int) -> ray.actor.ActorHandle:
        ActorCls = ray.remote(num_cpus=1)(MathEnv)
        actor = ActorCls.remote(
            inference,
            rollout_sink,
            data_source=self.data_source,
            split=self.split,
            num_generations=self.num_generations,
            seed=self.seed + seed,
        )
        actor.run.remote()
        return actor
