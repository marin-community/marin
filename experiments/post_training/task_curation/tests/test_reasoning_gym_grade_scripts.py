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
SCRIPT_PATHS = (
    ROOT / "experiments/post_training/task_curation/datasets/reasoning_gym" / GRADE_SCRIPT,
    ROOT / "experiments/post_training/task_curation/datasets/nemotron_ultra" / GRADE_SCRIPT,
)
SCRIPT_RUNNER = """
import importlib.util
import sys
from pathlib import Path

script_path, config_path, answer_path = map(Path, sys.argv[1:4])
bad_reference = sys.argv[4] == "bad_reference"
bad_reply = sys.argv[5]
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
        raise RuntimeError("scorer could not parse candidate")
    if bad_reference and candidate == entry["answer"]:
        return 0.0
    return real_score(dataset, entry, candidate)

grader.score_reasoning_gym_answer = score
grader.main()
"""


def contract_for(script: Path) -> dict:
    if script.parent.name == "nemotron_ultra":
        return {
            "question": "Find the prime factorization of 835.",
            "answer": "5 \u00d7 167",
            "metadata": {"source_dataset": "prime_factorization"},
        }
    seed = generate.task_seed("prime_factorization")
    dataset = reasoning_gym.create_dataset("prime_factorization", size=generate.ROWS_PER_TASK, seed=seed)
    return {
        "entry": generate.encoded(dataset[0]),
        "generation": {
            "task": "prime_factorization",
            "seed": seed,
            "index": 0,
            "config": generate.encoded(dataclasses.asdict(dataset.config)),
            "python_hash_seed": 0,
        },
    }


@pytest.mark.parametrize("script", SCRIPT_PATHS, ids=("generated", "nemotron_ultra"))
@pytest.mark.parametrize(
    ("candidate", "bad_reference", "expected"),
    [(PROSE, False, 0.0), ("reference", False, 1.0), (PROSE, True, None)],
    ids=("unparseable_candidate", "valid_answer", "broken_reference"),
)
def test_reasoning_gym_grade_script_scores_or_reports_broken_reference(
    script, candidate, bad_reference, expected, tmp_path
):
    contract = contract_for(script)
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
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "0"},
        check=False,
    )
    if expected is None:
        assert result.returncode != 0 and not result.stdout.strip(), result.stderr
    else:
        output = result.stdout.strip().splitlines()
        assert result.returncode == 0 and len(output) == 1, result.stderr
        assert float(output[0]) == expected
