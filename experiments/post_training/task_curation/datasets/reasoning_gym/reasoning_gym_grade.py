# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a reply with the task's own Reasoning Gym scorer and write the reward JSON.

Usage: ``reasoning_gym_grade.py CONFIG ANSWER SCORE``. ``CONFIG`` holds ``{"mode", "contract"}``:

- ``generated``: regenerate the entry from its recorded generator seed and index, refuse a contract
  whose recorded entry or configuration differs from the regenerated one, and score the text after
  the last ``Answer:`` marker against the regenerated entry (which keeps the generator's Python types).
- ``ultra``: score the last ``<answer>`` block, else the last boxed answer, of a Nemotron Ultra
  reply against the row's question, answer and metadata.
"""

import dataclasses
import importlib
import json
import operator
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time
from fractions import Fraction
from numbers import Integral
from pathlib import Path
from typing import cast

import numpy as np

# These constants and json_value repeat generate.py: this script ships alone to /tests in the grader
# image, where generate.py and its generator checkout are absent. A mismatch fails loudly, because
# the locator and regenerated-entry checks in generated_input refuse the contract.
ROWS_PER_TASK = 1000
GENERATION_SEED = 42


@dataclass(frozen=True)
class ScorerInput:
    task_name: str
    entry: dict
    candidate: str


def json_value(value):
    """The JSON form the generator records for values JSON cannot represent directly."""
    if isinstance(value, Integral):
        return operator.index(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, date | datetime | time):
        return {"python_type": "datetime." + type(value).__name__, "isoformat": value.isoformat()}
    if isinstance(value, Fraction):
        return {"python_type": "fractions.Fraction", "numerator": value.numerator, "denominator": value.denominator}
    raise TypeError(f"Generated value of type {type(value).__name__} is not JSON serializable")


def generated_input(reasoning_gym, contract, answer):
    datasets = importlib.import_module("reasoning_gym.factory").DATASETS
    generation = contract["generation"]
    expected_seed = GENERATION_SEED + sorted(datasets).index(generation["task"])
    if generation["seed"] != expected_seed or not 0 <= generation["index"] < ROWS_PER_TASK:
        raise ValueError("Recorded locator differs from the generated source")
    dataset = reasoning_gym.create_dataset(generation["task"], size=ROWS_PER_TASK, seed=expected_seed)
    encoded_config = json.loads(json.dumps(dataclasses.asdict(dataset.config), default=json_value))
    if encoded_config != generation["config"]:
        raise ValueError("Recorded generator configuration differs from the generator defaults")
    entry = dataset[generation["index"]]
    if json.loads(json.dumps(entry, default=json_value)) != contract["entry"]:
        raise ValueError("Regenerated entry differs from the recorded entry")
    _, marker, candidate = answer.rpartition("Answer:")
    return ScorerInput(generation["task"], entry, candidate.strip() if marker else answer.strip())


def ultra_input(contract, answer):
    extraction = importlib.import_module("skyrl_gym.envs.nemotron_ultra.answer_extraction")
    entry = {"question": contract["question"], "answer": contract.get("answer"), "metadata": contract["metadata"]}
    text = extraction.final_answer_text(answer)
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    candidate = matches[-1].group(1).strip() if matches else extraction.last_boxed_answer(text) or text.strip()
    return ScorerInput(contract["metadata"]["source_dataset"], entry, candidate)


def main(config_path, answer_path, score_path):
    config = json.loads(Path(config_path).read_text())
    mode, contract = config["mode"], config["contract"]
    if mode == "generated":
        # Some generators iterate over sets, so regeneration needs the recorded hash seed.
        seed = str(contract["generation"]["python_hash_seed"])
        if os.environ.get("PYTHONHASHSEED") != seed:
            os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": seed})
    reasoning_gym = importlib.import_module("reasoning_gym")
    answer = Path(answer_path).read_text(errors="replace")
    if mode == "generated":
        inputs = generated_input(reasoning_gym, contract, answer)
    elif mode == "ultra":
        inputs = ultra_input(contract, answer)
    else:
        raise ValueError(f"Unknown Reasoning Gym grading mode: {mode}")
    # Upstream annotates the returned two-argument scorer as a zero-argument callable.
    scorer = cast(Callable[[str, dict], float], reasoning_gym.get_score_answer_fn(inputs.task_name))
    reward = float(scorer(inputs.candidate, inputs.entry))
    Path(score_path).write_text(json.dumps({"reward": reward, "detail": {"task_name": inputs.task_name, "mode": mode}}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
