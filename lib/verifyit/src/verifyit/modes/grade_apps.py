# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt the pinned APPS evaluator's case results to a native reward file."""

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path


def main():
    source, config_path, answer_path, reward_path = map(Path, sys.argv[1:])
    config = json.loads(config_path.read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["apps_source_sha256"]:
        raise ValueError("APPS evaluator source hash mismatch")
    spec = importlib.util.spec_from_file_location("apps_testing_util", source)
    assert spec is not None and spec.loader is not None
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)

    problem = {"input_output": json.loads(config["input_output"])}
    answer = answer_path.read_text()
    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", answer, re.DOTALL)
    code = blocks[-1].strip() if blocks else None
    reward = 0.0
    if code is not None:
        results = evaluator.run_test(problem=problem, test=code)
        reward = 1.0 if results and all(result == 1 for result in results) else 0.0
    reward_path.write_text(json.dumps({"reward": reward}))


if __name__ == "__main__":
    main()
