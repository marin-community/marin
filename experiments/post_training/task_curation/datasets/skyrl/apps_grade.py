# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score an APPS reply with the image-installed APPS evaluator: 1 when every hidden case passes.

Usage: apps_grade.py TESTING_UTIL CONFIG ANSWER SCORE. ``TESTING_UTIL`` is the evaluator's
``testing_util.py``, checked against ``apps_source_sha256`` in the config before it runs.
"""

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path


def main() -> None:
    source, config_path, answer_path, score_path = map(Path, sys.argv[1:5])
    config = json.loads(config_path.read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["apps_source_sha256"]:
        raise ValueError("APPS evaluator source hash mismatch")
    spec = importlib.util.spec_from_file_location("apps_testing_util", source)
    assert spec is not None and spec.loader is not None
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)

    problem = {"input_output": json.loads(config["input_output"])}
    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", answer_path.read_text(), re.DOTALL)
    reward = 0.0
    if blocks:
        results = evaluator.run_test(problem=problem, test=blocks[-1].strip())
        reward = 1.0 if results and all(result == 1 for result in results) else 0.0
    score_path.write_text(json.dumps({"reward": reward}))


if __name__ == "__main__":
    main()
