# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a code reply with the image-installed SkyRL LiveCodeBench evaluator: 1 when every case passes.

Usage: lcb_grade.py SKYRL_GYM_ROOT CONFIG ANSWER SCORE. The config's ``test_cases`` hold the
source's hidden cases in any layout the evaluator normalizes.
"""

import importlib
import json
import sys
from pathlib import Path

LCB_MODULE = "skyrl_gym.envs.lcb.livecodebench"


# Each grade script ships alone to /tests in the grader image, so sql_grade.py keeps its own copy.
def source_module(root: Path, name: str):
    """Import ``name`` from the checkout at ``root``, refusing a copy installed elsewhere."""
    sys.path.insert(0, str(root))
    module = importlib.import_module(name)
    expected = (root / (name.replace(".", "/") + ".py")).resolve()
    if module.__file__ is None or Path(module.__file__).resolve() != expected:
        raise RuntimeError(f"Unexpected installed source module: {name}")
    return module


def code_score(root: Path, config: dict, answer: str) -> dict:
    lcb = source_module(root, LCB_MODULE)
    tests = json.loads(lcb.normalize_lcb_ground_truth(config["test_cases"]))
    if tests[0]["testtype"] == lcb.FUNCTIONAL_TEST_TYPE:
        # Malformed functional arguments are a task defect; fail before running the reply.
        for case in tests:
            for argument in case["input"].split("\n"):
                json.loads(argument)
            json.loads(case["output"])
    code = lcb.extract_code_from_model(answer)
    if code is None:
        return {"reward": 0.0, "detail": {}}
    results, diagnostics = lcb.lcb_execution_result(tests, code, execution_mode=lcb.TestExecutionMode.stop_on_failure)
    return {"reward": 1.0 if all(result is True for result in results) else 0.0, "detail": diagnostics}


def main() -> None:
    root, config_path, answer_path, score_path = map(Path, sys.argv[1:5])
    config = json.loads(config_path.read_text())
    answer = answer_path.read_text() if answer_path.exists() else ""
    score_path.write_text(json.dumps(code_score(root, config, answer), allow_nan=False))


if __name__ == "__main__":
    main()
