# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Screen the Reasoning Gym task types by the pass rate of a policy, to choose the capability gaps of the controlled
capability-gap gate.

Each generator named in ``--types-file`` (the types of the seed tasks) makes ``--instances`` fresh instances; a type
whose known answer does not score 1.0 here is skipped. The policy answers each ``--samples`` times, and the
reply is graded as the seed tasks are: the text after the last ``Answer:`` marker (the whole stripped reply without
one), scored by ``reasoning_gym.get_score_answer_fn``. Run it in an Iris GPU task through ``job.sh``::

    job.sh -m experiments.post_training.rsi.screen_gap --model Qwen/Qwen3-4B-Instruct-2507 --output-dir s3://...
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass

import click
import reasoning_gym
from reasoning_gym.factory import DATASETS
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from rolloutengine.models import ModelRequest

from experiments.post_training.rsi.evaluate import PolicyOptions, Served, policy_options, serve

logger = logging.getLogger(__name__)

ANSWER_MARKER = "Answer:"
SCREEN_FILE = "screen.json"


@dataclass(frozen=True)
class TypeScreen:
    task: str
    instances: int
    mean_reward: float
    pass_rate: float
    """The share of attempts with the full reward."""


def extract_answer(reply: str) -> str:
    """The text after the last ``Answer:`` marker, stripped; the whole stripped reply without one."""
    _, marker, answer = reply.rpartition(ANSWER_MARKER)
    return (answer if marker else reply).strip()


def known_answer_scores(task: str, instances: int, seed: int) -> bool:
    """Whether the scorer gives the full reward to the generator's own answer, as the seed grader does."""
    dataset = reasoning_gym.create_dataset(task, size=instances, seed=seed)
    score = reasoning_gym.get_score_answer_fn(task)
    return all(float(score(extract_answer(f"{ANSWER_MARKER} {entry['answer']}"), entry)) == 1.0 for entry in dataset)


async def screen_type(policy, task: str, instances: int, samples: int, seed: int) -> TypeScreen:
    dataset = reasoning_gym.create_dataset(task, size=instances, seed=seed)
    score = reasoning_gym.get_score_answer_fn(task)

    async def attempt(entry: dict) -> float:
        request = ModelRequest(messages=({"role": "user", "content": entry["question"]},), options={})
        turn = await policy(request)
        return float(score(extract_answer(turn.text), entry))

    rewards = await asyncio.gather(*(attempt(entry) for entry in dataset for _ in range(samples)))
    return TypeScreen(
        task=task,
        instances=instances,
        mean_reward=sum(rewards) / len(rewards),
        pass_rate=sum(reward == 1.0 for reward in rewards) / len(rewards),
    )


@click.command()
@policy_options
@click.option("--output-dir", required=True)
@click.option("--types-file", required=True, help="A JSON list of the Reasoning Gym task types to screen.")
@click.option("--instances", type=int, default=20, show_default=True, help="Instances of each task type.")
@click.option("--samples", type=int, default=4, show_default=True, help="Attempts at each instance.")
@click.option("--seed", type=int, default=0, show_default=True)
def main(output_dir: str, types_file: str, instances: int, samples: int, seed: int, **options) -> None:
    """Pass rate of a base policy on every Reasoning Gym task type."""
    configure_logging(logging.INFO)
    types = sorted(json.loads(StoragePath(types_file).read_text()))
    asyncio.run(_screen(PolicyOptions(**options), StoragePath(output_dir), types, instances, samples, seed))


async def _screen(
    options: PolicyOptions, output: StoragePath, types: list[str], instances: int, samples: int, seed: int
) -> None:
    unknown = set(types) - set(DATASETS)
    if unknown:
        raise ValueError(f"Not Reasoning Gym task types: {sorted(unknown)}")
    gradable = [task for task in types if known_answer_scores(task, instances, seed)]
    logger.info("%d of %d types grade their own answers; skipped: %s", len(gradable), len(types), sorted(set(types) - set(gradable)))
    results = []
    with serve(options) as server:
        policy = Served(options, server).policy
        for task in gradable:
            result = await screen_type(policy, task, instances, samples, seed)
            logger.info("%s: pass %.3f, reward %.3f", task, result.pass_rate, result.mean_reward)
            results.append(asdict(result))
            (output / SCREEN_FILE).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
