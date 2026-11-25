# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
from collections import Counter

import pytest
from rolloutengine.test_support import ReplayModel, engine, machine_runtime, rollout_spec
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend
from taskcompendium.models import TextMessage

from experiments.post_training.task_curation.qwen_sampler_lab import FAMILIES, Difficulty, lab_examples


@pytest.fixture(scope="module")
def examples():
    return lab_examples()


def test_qwen_sampler_pool_has_distinct_prompts_and_balanced_cells(examples):
    prompts = [r.task.context.events[0].content for r in examples]
    assert len(set(prompts)) == len(set(r.task.id for r in examples)) == 4752
    assert set(Counter((r.strata.family, r.strata.difficulty) for r in examples if r.split == "train").values()) == {512}
    assert set(Counter((r.strata.family, r.strata.difficulty) for r in examples if r.fast_dev).values()) == {4}
    assert set(Counter((r.strata.family, r.strata.difficulty) for r in examples if r.split == "dev").values()) == {16}


def reference_answer(prompt, family, difficulty):
    # Independently calculate the labels from each rendered instruction, never from the private grader arguments.
    if family == "arithmetic":
        values = [int(x) for x in re.findall(r"\d+", prompt)]
        if difficulty == "easy":
            return str(sum(values))
        if difficulty == "intermediate":
            a, b, c = values
            return str(a * c + b * c)
        a, b, d, c, modulus = values
        return str((a * d + b * d - c) % modulus)
    if family == "string":
        if difficulty == "easy":
            word = prompt.split("of ", 1)[1].split(".", 1)[0]
            return word[::-1]
        if difficulty == "intermediate":
            word = prompt.split("of ", 1)[1].split(",", 1)[0]
            return "".join(reversed([v for i, v in enumerate(word) if i % 2 == 0]))
        tokens = prompt.split()
        word, n = tokens[1], int(tokens[4])
        return "".join(sorted(word[(i + n) % len(word)] for i in range(0, len(word), 2)))
    [array] = re.findall(r"\[([^\]]+)\]", prompt)
    values = [int(v) for v in array.split(",")]
    if difficulty == "easy":
        output = sorted(values)
    elif difficulty == "intermediate":
        output = [v + v for v in reversed(values)]
    else:
        [divisor] = re.findall(r"modulo (\d+)", prompt)
        modulus = int(divisor)
        output = [sum(values[:i]) % modulus for i in range(1, len(values) + 1)]
    return ",".join(map(str, output))


@pytest.mark.asyncio
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("difficulty", Difficulty)
async def test_one_turn_shellsim_grades_each_transform(examples, family, difficulty):
    task = next(r.task for r in examples if r.strata.family == family and r.strata.difficulty == difficulty)
    event = task.context.events[0]
    assert isinstance(event, TextMessage)
    expected = reference_answer(event.content, family, difficulty)
    runtime = machine_runtime(backend=Backend.SHELLSIM)
    spec = rollout_spec(task, verifier_machine=runtime)
    for answer, reward in [(expected, 1.0), ("incorrect", 0.0)]:
        model = ReplayModel([{"role": "assistant", "content": rf"The result is \boxed{{{answer}}}."}])
        result = await engine(model, {Backend.SHELLSIM: ShellSimMachineFactory()}).run_task(spec)
        assert result.grade.reward == reward
