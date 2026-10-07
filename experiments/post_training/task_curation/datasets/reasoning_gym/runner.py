# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bridge pinned Reasoning Gym packages to a native score file."""

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

ROWS_PER_TASK = 1000
GENERATION_SEED = 42


@dataclass(frozen=True)
class NativeInput:
    task_name: str
    entry: dict
    candidate: str


def json_value(value):
    if isinstance(value, Integral):
        return operator.index(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, date | datetime | time):
        return {"python_type": "datetime." + type(value).__name__, "isoformat": value.isoformat()}
    if isinstance(value, Fraction):
        return {"python_type": "fractions.Fraction", "numerator": value.numerator, "denominator": value.denominator}
    raise TypeError(f"Generated value of type {type(value).__name__} is not JSON serializable")


def native_input(reasoning_gym, mode, contract, answer):
    if mode == "generated":
        datasets = importlib.import_module("reasoning_gym.factory").DATASETS

        generation = contract["generation"]
        expected_seed = GENERATION_SEED + sorted(datasets).index(generation["task"])
        if generation["seed"] != expected_seed or not 0 <= generation["index"] < ROWS_PER_TASK:
            raise ValueError("Recorded locator differs from the original generated source")
        dataset = reasoning_gym.create_dataset(generation["task"], size=ROWS_PER_TASK, seed=expected_seed)
        encoded_config = json.loads(json.dumps(dataclasses.asdict(dataset.config), default=json_value))
        if encoded_config != generation["config"]:
            raise ValueError("Recorded generator configuration differs from its original native defaults")
        entry = dataset[generation["index"]]
        if json.loads(json.dumps(entry, default=json_value)) != contract["entry"]:
            raise ValueError("Regenerated native entry differs from the complete recorded transport")
        _, marker, candidate = answer.rpartition("Answer:")
        return NativeInput(generation["task"], entry, candidate.strip() if marker else answer.strip())
    if mode == "ultra":
        extraction = importlib.import_module("skyrl_gym.envs.nemotron_ultra.answer_extraction")

        task_name = contract["metadata"]["source_dataset"]
        entry = {"question": contract["question"], "answer": contract.get("answer"), "metadata": contract["metadata"]}
        text = extraction.final_answer_text(answer)
        matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
        candidate = matches[-1].group(1).strip() if matches else extraction.last_boxed_answer(text) or text.strip()
        return NativeInput(task_name, entry, candidate)
    raise ValueError("Unknown native Reasoning Gym source contract")


def main():
    source = json.loads(Path("/tests/reasoning_contract.json").read_text())
    mode, contract = source["mode"], source["contract"]
    if mode == "generated":
        seed = str(contract["generation"]["python_hash_seed"])
        if os.environ.get("PYTHONHASHSEED") != seed:
            os.execve(sys.executable, [sys.executable, __file__], {**os.environ, "PYTHONHASHSEED": seed})
    reasoning_gym = importlib.import_module("reasoning_gym")

    answer = Path("/app/answer.txt").read_text(errors="replace")
    inputs = native_input(reasoning_gym, mode, contract, answer)
    # Upstream annotates the returned two-argument scorer as a zero-argument callable.
    scorer = cast(Callable[[str, dict], float], reasoning_gym.get_score_answer_fn(inputs.task_name))
    reward = float(scorer(inputs.candidate, inputs.entry))
    Path("/logs/verifier/score.json").write_text(
        json.dumps({"reward": reward, "detail": {"task_name": inputs.task_name, "native_policy": mode}})
    )


if __name__ == "__main__":
    main()
