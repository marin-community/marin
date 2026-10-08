# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade code cases through the original image-installed LiveCodeBench evaluator."""

import importlib
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path("/opt/skyrl_gym")
LCB_PATH = SOURCE_ROOT / "skyrl_gym/envs/lcb/livecodebench.py"


def source_module(name: str, expected: Path):
    sys.path.insert(0, str(SOURCE_ROOT))
    module = importlib.import_module(name)
    if module.__file__ is None or Path(module.__file__).resolve() != expected:
        raise RuntimeError(f"Unexpected installed source module: {name}")
    return module


def code_score(config: dict, answer: str) -> dict:
    lcb = source_module("skyrl_gym.envs.lcb.livecodebench", LCB_PATH)
    ground_truth = config["test_cases"]
    tests = json.loads(lcb.normalize_lcb_ground_truth(ground_truth))
    if tests[0]["testtype"] == lcb.FUNCTIONAL_TEST_TYPE:
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
    config_path, answer_path, score_path = map(Path, sys.argv[1:4])
    config = json.loads(config_path.read_text())
    answer = answer_path.read_text() if answer_path.exists() else ""
    score_path.write_text(json.dumps(code_score(config, answer), allow_nan=False))


if __name__ == "__main__":
    main()
