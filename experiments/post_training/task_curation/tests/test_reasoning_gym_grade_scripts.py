# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the shipped Reasoning Gym scripts at their numeric-output boundary."""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import reasoning_gym

from experiments.post_training.task_curation.datasets.reasoning_gym import generate

ROOT = Path(__file__).resolve().parents[4]
PROSE = "I need to find the prime factorization of 835."
GRADE_SCRIPT = "reasoning_gym_grade.py"
# The Nemotron Ultra script joins here when its lane lands #12238's script change.
SCRIPT_PATHS = (ROOT / "experiments/post_training/task_curation/datasets/reasoning_gym" / GRADE_SCRIPT,)
SCRIPT_RUNNER = """
import importlib.util
import sys
from pathlib import Path

script_path, config_path, answer_path = map(Path, sys.argv[1:4])
bad_reference = sys.argv[4] == "bad_reference"
bad_reply = sys.argv[5]
failure = sys.argv[6]
sys.path.insert(0, str(script_path.parent))
sys.path.insert(0, str(script_path.parent / "scorers"))
spec = importlib.util.spec_from_file_location("shipped_reasoning_gym_grade", script_path)
assert spec is not None and spec.loader is not None
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)
grader.CONFIG = config_path
grader.ANSWER = answer_path
real_score = grader.score_reasoning_gym_answer

def score(dataset, entry, candidate):
    if candidate == bad_reply:
        if failure == "invalid_task":
            entry["metadata"].pop("example_answer", None)
            return real_score(dataset, entry, candidate)
        raise RuntimeError("scorer could not parse candidate")
    if bad_reference and candidate == entry["answer"]:
        return 0.0
    return real_score(dataset, entry, candidate)

grader.score_reasoning_gym_answer = score
grader.main()
"""


def contract_for(script: Path, task: str) -> dict:
    seed = generate.task_seed(task)
    dataset = reasoning_gym.create_dataset(task, size=generate.ROWS_PER_TASK, seed=seed)
    if script.parent.name == "nemotron_ultra":
        return generate.encoded(dataset[0])
    return {
        "entry": generate.encoded(dataset[0]),
        "generation": {
            "task": task,
            "seed": seed,
            "index": 0,
            "config": generate.encoded(dataclasses.asdict(dataset.config)),
            "python_hash_seed": 0,
        },
    }


@pytest.mark.parametrize("script", SCRIPT_PATHS, ids=("generated",))
@pytest.mark.parametrize(
    ("task", "candidate", "bad_reference", "failure", "expected_reward", "expected_error"),
    [
        ("prime_factorization", PROSE, False, "runtime_error", 0.0, None),
        ("prime_factorization", "reference", False, "runtime_error", 1.0, None),
        ("prime_factorization", PROSE, True, "runtime_error", None, "ValueError"),
        ("propositional_logic", PROSE, False, "invalid_task", None, "InvalidTask"),
    ],
    ids=("unparseable_candidate", "valid_answer", "broken_reference", "invalid_task_with_null_answer"),
)
def test_reasoning_gym_grade_script_scores_or_reports_task_errors(
    script, task, candidate, bad_reference, failure, expected_reward, expected_error, tmp_path
):
    contract = contract_for(script, task)
    entry = contract.get("entry", contract)
    answer = entry["answer"] if candidate == "reference" else candidate
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"contract": contract}))
    answer_path = tmp_path / "answer.txt"
    answer_path.write_text(answer)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            SCRIPT_RUNNER,
            str(script),
            str(config_path),
            str(answer_path),
            "bad_reference" if bad_reference else "valid_reference",
            PROSE,
            failure,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "0"},
        check=False,
    )
    if expected_error is not None:
        assert result.returncode != 0 and not result.stdout.strip(), result.stderr
        assert expected_error in result.stderr
    else:
        output = result.stdout.strip().splitlines()
        assert result.returncode == 0 and len(output) == 1, result.stderr
        assert float(output[0]) == expected_reward
