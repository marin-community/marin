# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure Evalchemy Python source preparation; this module never executes a candidate.

Adapted from the MIT Evalchemy integrations and lm-evaluation-harness v0.4.12.
See lm-evaluation-harness-MIT.txt for the harness license.
"""

import ast
import re
import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

from verifyit.adapters.completion import END_OF_TURN

HUMANEVAL_STOPS = ("\nclass", "\ndef", "\n#", "\nif", "\nprint", "\n```", *END_OF_TURN)
HUMANEVAL_REQUEST_STOPS = HUMANEVAL_STOPS[:4]
# The alarm for the whole test program. HumanEval and CruxEval keep their sources' alarms. HumanEval+ and MBPP+ run
# every base and plus input in one program, so they get evalplus's 60-second limit for a whole task.
EXECUTION_TIMEOUT = {"humaneval": 3, "humanevalplus": 60, "mbppplus": 60, "cruxeval": 10}
PYTHON_IMPORTS = (
    "import math",
    "import re",
    "import sys",
    "import copy",
    "import datetime",
    "import itertools",
    "import collections",
    "import heapq",
    "import functools",
    "import hashlib",
    "import numpy",
    "import numpy as np",
    "import string",
    "from typing import *",
    "from collections import *",
)
CRUX_CALL = re.compile(r"f\(.*?\)")
CRUX_ANSWER_SPAN = re.compile(r"\[ANSWER\](.*?)\[/ANSWER\]", flags=re.DOTALL)
FUNCTION_INSTRUCTION = (
    "Please continue to complete the function. You are not allowed to modify the given code and do the completion only. "
    "Please return all completed function in a codeblock. Here is the given code to do completion:\n```python\n{}\n```"
)


def function_prompt(problem: Mapping[str, Any]) -> str:
    return FUNCTION_INSTRUCTION.format(problem["prompt"].strip())


def mbpp_example(question: str, tests: Sequence[str], code: str = "") -> str:
    prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(question.strip(), "\n".join(tests))
    if code:
        code = code.replace("\r", "").replace("\t", "    ")
        prompt += f"\n>>> Code:\n```python\n{code}\n```"
    return prompt


def mbpp_prompt(problem: Mapping[str, Any], examples: Sequence[Mapping[str, Any]]) -> str:
    shots = "\n\n".join(
        f"- Example {index}:\n" + mbpp_example(example["prompt"], example["test_list"], example["code"])
        for index, example in enumerate(examples, 1)
    )
    return (
        "Please refer the given examples and generate a python function for my problem.\n"
        "Examples are listed as follows:\n{}\n\nHere is my problem:\n{}"
    ).format(shots, mbpp_example(problem["prompt"], problem["test_list"]))


def generation_code(problem: Mapping[str, Any], response: str, suite: str) -> str:
    """Extract exactly the source Python candidate without parsing or executing it."""
    if suite == "humaneval":
        stops = [index for stop in HUMANEVAL_STOPS if (index := response.find(stop)) >= 0]
        return problem["prompt"] + (response[: min(stops)] if stops else response)
    blocks = re.findall(r"```python\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
    code_block = blocks[0] if blocks else response
    code = textwrap.dedent(code_block).strip("\n")
    if suite == "mbppplus":
        return code
    if suite != "humanevalplus":
        raise ValueError(f"Unknown Python generation suite: {suite}")
    question = problem["prompt"].strip()
    lines = [line for line in question.split("\n") if line.strip()]
    function_index = [index for index, line in enumerate(lines) if line.startswith("def ")][-1]
    function_name = lines[function_index].split("(")[0].strip()
    prefix = "\n".join(lines[:function_index])
    if re.search(rf"(?m)^[ \t]*{re.escape(function_name)}[ \t]*\(", code):
        return prefix + "\n" + code + "\n"
    return question + "\n" + code_block.strip("\n") + "\n"


def crux_answer(response: str, direction: str) -> str:
    """Extract the first answer block holding a CruxEval candidate."""
    spans = [span.strip() for span in CRUX_ANSWER_SPAN.findall(response)]
    answer = spans[0] if spans else response
    for span in spans:
        candidate = _crux_expression(span, direction)
        if direction == "input":
            if CRUX_CALL.search(span) is not None:
                answer = span
                break
        else:
            try:
                ast.parse(candidate, mode="eval")
            except SyntaxError:
                continue
            answer = span
            break
    return _crux_expression(answer, direction)


def _crux_expression(answer: str, direction: str) -> str:
    """Read an answer block by CRUXEval's harness rule."""
    # Split at the first "==" as CRUXEval's harness does (inference/tasks/*_prediction.py): an input answer is the
    # call before it, and an output answer is the text after it, including any later "==" where the harness stops.
    if "==" in answer:
        answer = answer.split("==", 1)[0 if direction == "input" else 1].strip()
    if direction == "input" and "assert f" in answer:
        answer = "f" + answer.split("assert f", 1)[1].strip()
    return answer


def test_program(problem: Mapping[str, Any], response: str, suite: str, direction: str = "") -> str:
    """Compile a private test program for execution only inside a grading sandbox."""
    if suite == "cruxeval":
        answer = crux_answer(response, direction)
        # Like CRUXEval's harness (evaluation/utils_general.py), fail an output answer containing f(<input>) and an
        # input answer without "f(" without running them.
        if (direction == "output" and f"f({problem['input']})" in answer) or (
            direction == "input" and "f(" not in answer
        ):
            return "raise AssertionError"
        # The legacy output task compares literals; it does not recompute f(input).
        return f"{problem['code']}\nassert {problem['output']} == {answer}"
    generation = generation_code(problem, response, suite)
    if suite == "humaneval":
        return generation + "\n" + problem["test"] + f"\ncheck({problem['entry_point']})"
    tests = problem["test"]
    if suite == "humanevalplus":
        tests += f"\ncheck({problem['entry_point']})"
    return "\n".join(PYTHON_IMPORTS) + "\n" + generation + "\n" + tests + "\n"


def execution_program(program: str, timeout: int) -> str:
    """Return the grader command's script: the test program in a forked child, with an alarm after ``timeout`` seconds.

    The child has one more second before it is killed, and the reward goes to reward.json beside the script.
    The receipt and exception handler reject early exits, including os._exit(0).
    """
    return (
        "import json\n"
        "import multiprocessing\n"
        "import signal\n"
        "from pathlib import Path\n"
        "import numpy\n"
        "\n"
        "def timeout_handler(signum, frame):\n"
        "    raise TimeoutError('Candidate timed out')\n"
        "\n"
        "def execute(connection):\n"
        "    signal.signal(signal.SIGALRM, timeout_handler)\n"
        f"    signal.setitimer(signal.ITIMER_REAL, {timeout})\n"
        "    try:\n"
        f"        exec(compile({program!r}, '<candidate>', 'exec'), {{}})\n"
        "    except BaseException:\n"
        "        passed = False\n"
        "    else:\n"
        "        passed = True\n"
        "    finally:\n"
        "        signal.setitimer(signal.ITIMER_REAL, 0)\n"
        "    connection.send(passed)\n"
        "\n"
        "context = multiprocessing.get_context('fork')\n"
        "receiver, sender = context.Pipe(duplex=False)\n"
        "process = context.Process(target=execute, args=(sender,))\n"
        "process.start()\n"
        "sender.close()\n"
        f"process.join({timeout + 1})\n"
        "if process.is_alive():\n"
        "    process.kill()\n"
        "    process.join()\n"
        "passed = False\n"
        "if receiver.poll():\n"
        "    try:\n"
        "        passed = receiver.recv() is True and process.exitcode == 0\n"
        "    except EOFError:\n"
        "        pass\n"
        "Path(__file__).with_name('reward.json').write_text(json.dumps({'reward': float(passed)}))\n"
    )
