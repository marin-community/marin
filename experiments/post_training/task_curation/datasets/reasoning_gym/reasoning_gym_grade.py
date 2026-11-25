# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a generated Reasoning Gym reply against its regenerated entry and print the reward.

``/tests/config.json`` holds the row's contract: the recorded entry and the generation that produced
it. The script regenerates the entry with the grader packages' ``reasoning_gym`` and the seeds and JSON
encoding of ``generate.py`` (shipped beside it), exits nonzero when the regenerated entry or the
generator configuration differs from the recorded one, and otherwise scores the text after the reply's
last ``Answer:`` marker, or the whole reply, with the pinned scorer and targeted repairs. The scorer sees the
generator's Python values, not their JSON forms.
"""

import contextlib
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

import reasoning_gym
from verifyit.grade import InvalidTask
from verifyit.modes.grade_reasoning_gym import score_reasoning_gym_answer

sys.path.insert(0, "/tests")

from generate import ROWS_PER_TASK, encoded, task_seed, without_generation_day

CONFIG = Path("/tests/config.json")
ANSWER = Path("/app/answer.txt")


def regenerated_entry(generation: dict[str, Any]) -> dict[str, Any]:
    name, index = generation["task"], generation["index"]
    if generation["seed"] != task_seed(name) or not 0 <= index < ROWS_PER_TASK:
        raise ValueError("Recorded locator differs from the generated source")
    dataset = reasoning_gym.create_dataset(name, size=ROWS_PER_TASK, seed=generation["seed"])
    if encoded(dataclasses.asdict(dataset.config)) != generation["config"]:
        raise ValueError("Recorded generator configuration differs from the generator defaults")
    return dataset[index]


def candidate(reply: str) -> str:
    _, marker, after = reply.rpartition("Answer:")
    return after.strip() if marker else reply.strip()


def main() -> None:
    contract = json.loads(CONFIG.read_text())["contract"]
    generation = contract["generation"]
    # Some generators iterate over sets, so the entry regenerates only under the recorded hash seed.
    if os.environ.get("PYTHONHASHSEED") != str(generation["python_hash_seed"]):
        raise ValueError(f"PYTHONHASHSEED must be {generation['python_hash_seed']} to regenerate the entry")
    with contextlib.redirect_stdout(sys.stderr):
        entry = regenerated_entry(generation)
        if without_generation_day(encoded(entry)) != without_generation_day(contract["entry"]):
            raise ValueError("Regenerated entry differs from the recorded entry")
        answer = candidate(ANSWER.read_text(errors="replace"))
        try:
            reward = score_reasoning_gym_answer(generation["task"], entry, answer)
        except InvalidTask:
            raise
        except Exception as error:
            # Check the reference when available; some datasets have no reference answer.
            reference = entry["answer"]
            if reference is not None and not score_reasoning_gym_answer(generation["task"], entry, reference) > 0:
                raise ValueError("Reasoning Gym reference answer does not score") from error
            reward = 0.0
    print(reward)


if __name__ == "__main__":
    main()
