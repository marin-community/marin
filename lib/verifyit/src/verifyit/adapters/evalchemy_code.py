# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure Evalchemy Python source preparation; this module never executes a candidate.

Adapted from the MIT Evalchemy integrations and lm-evaluation-harness v0.4.12.
See licenses/lm-evaluation-harness-MIT.txt for the harness license.
"""

import re
import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

END_OF_TURN = (
    "<|im_end|>",
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|endoftext|>",
    "</s>",
    "\nYou are an AI assistant",
    "\nQuestion:",
    "\nQ:",
    "\n[Question]",
    "\nUser:",
    "\nuser\n",
    "\nAssistant:",
)
HUMANEVAL_STOPS = ("\nclass", "\ndef", "\n#", "\nif", "\nprint", "\n```", *END_OF_TURN)
HUMANEVAL_REQUEST_STOPS = HUMANEVAL_STOPS[:4]
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
    """Extract the source's first answer block and shallow function-call match."""
    answer = response
    if "[ANSWER]" in answer and "[/ANSWER]" in answer:
        answer = answer.split("[ANSWER]", 1)[1].split("[/ANSWER]", 1)[0].strip()
    call = re.search(r"f\(.*?\)", answer)
    if call is not None:
        if direction == "input":
            return call.group(0)
        return answer.replace(call.group(0), "").replace("=", "").replace("assert", "").strip()
    return answer


def test_program(problem: Mapping[str, Any], response: str, suite: str, direction: str = "") -> str:
    """Compile a private test program for execution only inside a grading sandbox."""
    if suite == "cruxeval":
        answer = crux_answer(response, direction)
        # The legacy output task compares literals; it does not recompute f(input).
        return f"{problem['code']}\nassert {problem['output']} == {answer}"
    generation = generation_code(problem, response, suite)
    if suite == "humaneval":
        return generation + "\n" + problem["test"] + f"\ncheck({problem['entry_point']})"
    tests = problem["test"]
    if suite == "humanevalplus":
        tests += f"\ncheck({problem['entry_point']})"
    return "\n".join(PYTHON_IMPORTS) + "\n" + generation + "\n" + tests + "\n"
