# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score an APPS reply with the vendored APPS evaluator: 1 when every hidden case passes.

The evaluator runs the last fenced code block of ``/app/answer.txt`` against ``input_output`` from
``/tests/config.json``. It executes the program in the calling process after disabling process and
file functions there, so this script calls it from a child process: a program that ends the child,
for example with ``sys.exit()`` in a call-based solution, or that runs past ``DEADLINE`` scores 0.
"""

import json
import multiprocessing
import os
import re
import sys
from multiprocessing.connection import Connection
from pathlib import Path

sys.path.insert(0, "/tests")

import apps_testing_util

CODE_BLOCK = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
DEADLINE = 300.0
"""Seconds for all cases together, below the grader's timeout so that a stuck program scores 0."""


def run_tests(problem: dict, program: str, results: Connection) -> None:
    os.dup2(2, 1)  # The program's prints go to stderr; stdout carries only the reward.
    results.send(apps_testing_util.run_test(problem=problem, test=program))


def passes(problem: dict, program: str) -> bool:
    receiver, sender = multiprocessing.Pipe(duplex=False)
    child = multiprocessing.Process(target=run_tests, args=(problem, program, sender))
    child.start()
    sender.close()
    try:
        results = receiver.recv() if receiver.poll(DEADLINE) else None
    except EOFError:
        results = None
    finally:
        child.kill()
        child.join()
    return bool(results) and all(result == 1 for result in results)


def main() -> None:
    config = json.loads(Path("/tests/config.json").read_text())
    blocks = CODE_BLOCK.findall(Path("/app/answer.txt").read_text())
    passed = bool(blocks) and passes({"input_output": config["input_output"]}, blocks[-1].strip())
    print(1.0 if passed else 0.0)


if __name__ == "__main__":
    main()
