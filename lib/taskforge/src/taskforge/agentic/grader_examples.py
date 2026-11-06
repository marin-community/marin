# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worked graders for draft tasks: one complete ``TaskSpec`` for each common kind, which ``submit`` accepts.

A draft agent gets this file as ``/workspace/reference/grader_examples.py``. Copy the function of the kind that the
proposal's grader design names into the build program, and replace its data. Each function takes the task id and
draws its data from it, so other task ids give other data and other reference answers.

- ``numeric_task``: one number, graded with a tolerance. No solver machine: the solver reads only the prompt.
- ``exact_task``: one short text, compared after normalizing case and whitespace.
- ``json_task``: a JSON object with fixed fields, compared field by field.
- ``choice_task``: one option letter. The correct option's position comes from the task id, so answers vary.
- ``judge_task``: prose, which a judge model compares with reference answers. Leave the judge's model empty.
- ``script_task``: a Python grader that runs in its own Linux machine, with its expected values in a file of
  ``resources.verifier``. It prints the reward, a number from 0 to 1, as its last line.
- ``shell_task``: a solver with a shell and files in /workspace; see ``/workspace/example_build.py``.
"""

import json
import random

from pydantic import JsonValue
from shellbox.machine import PackageSpec
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    JsonValueAnswer,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    StdoutReward,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import ExactSpec, JudgeSpec, McqSpec, NumericSpec, StructuredExactSpec

SOURCE = Source(dataset="taskforge-agentic-drafts", revision="example", row="example", importer_revision="example")
NO_MACHINE = EnvironmentRequirements()
"""A solver with no shell and no files: no capabilities, no package spec and no worker resources."""


def _task(task_id: str, prompt: str, grader, **fields) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
        environment_requirements=fields.pop("environment_requirements", NO_MACHINE),
        answer_type=fields.pop("answer_type", AnswerType.TEXT),
        answer_format=fields.pop("answer_format", PlainText()),
        grader=grader.grader,
        source=SOURCE,
        resources=ResourceGroups(verifier=grader.resources),
        **fields,
    )


def numeric_task(task_id: str) -> TaskSpec:
    rng = random.Random(task_id)
    rate, hours = rng.randint(20, 90), rng.randint(3, 12)
    prompt = (
        f"A pump moves {rate} litres per minute. How many litres does it move in {hours} hours? "
        "Reply with the number alone."
    )
    return _task(
        task_id,
        prompt,
        verifyit_package(NumericSpec(expected=str(rate * 60 * hours), tolerance_abs=0.0, tolerance_rel=0.0)),
    )


def exact_task(task_id: str) -> TaskSpec:
    rng = random.Random(task_id)
    # A large space of answers: a small one repeats across ids, and submit refuses that.
    word = "".join(rng.choice("bcdfghklmnprstvz") + rng.choice("aeiou") for _ in range(4))
    prompt = f"Write the made-up word {word!r} backwards. Reply with that text alone."
    return _task(task_id, prompt, verifyit_package(ExactSpec(expected=(word[::-1],))))


def json_task(task_id: str) -> TaskSpec:
    rng = random.Random(task_id)
    values = [rng.randint(1, 99) for _ in range(rng.randint(5, 9))]
    prompt = (
        f"For the values {values}, reply with one JSON object and nothing else, with the integer fields "
        '"count", "minimum" and "maximum".'
    )
    expected: dict[str, JsonValue] = {"count": len(values), "minimum": min(values), "maximum": max(values)}
    return _task(
        task_id,
        prompt,
        verifyit_package(StructuredExactSpec(expected=expected)),
        answer_type=AnswerType.JSON,
        answer_format=JsonValueAnswer(),
    )


def choice_task(task_id: str) -> TaskSpec:
    rng = random.Random(task_id)
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    right = a * b
    options = [right, right + rng.randint(1, 9), right - rng.randint(1, 9), right + 10 * rng.randint(1, 5)]
    rng.shuffle(options)  # The correct letter depends on the id, so the reference answers vary.
    letters = "ABCD"
    listing = "\n".join(f"{letter}. {value}" for letter, value in zip(letters, options, strict=True))
    prompt = f"What is {a} times {b}?\n{listing}\nReply with the letter of the correct option alone."
    expected = letters[options.index(right)]
    return _task(task_id, prompt, verifyit_package(McqSpec(expected=expected, options=len(options))))


def judge_task(task_id: str) -> TaskSpec:
    rng = random.Random(task_id)
    price, discount = rng.randint(20, 400), rng.choice((10, 15, 20, 25, 30, 40))
    question = (
        f"A coat costs {price} euros and is on sale at {discount}% off. In one or two sentences, tell a customer "
        "what they pay and how much they save."
    )
    saving = price * discount / 100
    reference = f"You pay {price - saving:g} euros, a saving of {saving:g} euros on the price of {price} euros."
    # The reference rubric: the judge decides whether the answer matches a reference. Its model stays empty.
    spec = JudgeSpec(references=(reference,), question=question)
    return _task(task_id, question, verifyit_package(spec))


GRADE_SCRIPT = """\
import json
import sys

expected = json.load(open("/tests/expected.json"))
answer = open("/app/answer.txt").read().split()
# The reward is the share of the expected primes that the answer lists, and 0 for any number that is not one.
found = {int(word) for word in answer if word.isdigit()}
print(0.0 if found - set(expected) else len(found & set(expected)) / len(expected))
"""


def script_task(task_id: str) -> TaskSpec:
    rng = random.Random(task_id)
    low = rng.randint(10, 5000)
    high = low + rng.randint(20, 40)
    primes = [n for n in range(low, high + 1) if n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))]
    prompt = f"List every prime from {low} to {high}, separated by spaces. Reply with the numbers alone."
    grader = ScriptGrader(
        argv=("python3", "/tests/grade.py"),
        # The grader's machine: Linux with Python. List PyPI packages with their versions when it needs them.
        environment=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, package_spec=PackageSpec(pypi_packages=())
        ),
        answer_path="/app/answer.txt",
        reward=StdoutReward(),
    )
    verifier = (
        inline_resource("tests/grade.py", GRADE_SCRIPT.encode()),
        inline_resource("tests/expected.json", json.dumps(primes).encode()),
    )
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
        environment_requirements=NO_MACHINE,
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader,
        source=SOURCE,
        resources=ResourceGroups(verifier=verifier),
    )


EXAMPLES = {
    "numeric": numeric_task,
    "exact": exact_task,
    "json": json_task,
    "choice": choice_task,
    "judge": judge_task,
    "script": script_task,
}
