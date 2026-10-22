# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The X task family: CruxEval-style Python input/output prediction, graded by
sandboxed execution (reusing the #8225-era grading primitives).

Functions are synthesized from seeded template families parameterized by ``k``, so no
public CruxEval prompt can enter the set; a normalized-text exclusion against the
public CruxEval prompts still runs at build time — the prompt file is a required
input — and records its count. Train/probe splits are assigned per function, by the
seeded hash of a function's ``(template family, k)`` identity, so no function is ever
shared between the probe split and the training split no matter which calls are drawn
against it. Each function contributes several calls, so the 1,000-prompt ``P_X``
spans roughly a tenth of the function pool rather than a handful of functions.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.post_training.curriculum_sft.code_tasks import (
    literals_equal,
    trace_value,
)
from verifyit.modes.extract import extract_boxed

# Prediction directions, as in CruxEval.
DIRECTION_OUTPUT = "output"  # given code + input, predict the return value
DIRECTION_INPUT = "input"  # given code + output, find an input that produces it

TRAIN_SPLIT = "train"
PROBE_SPLIT = "probe"

# The probe split is P_X's share of the function pool.
PROBE_FRACTION = 0.1
_PROBE_PERCENT = int(PROBE_FRACTION * 100)

TRACE_TIMEOUT_SECONDS = 10.0

# Parameters each template family ranges over; the (family, parameter) cross product
# is the function pool: 12 families x 99 parameters = 1,188 functions, about 120 of
# them in the probe split.
PARAMETER_RANGE = range(2, 101)

# Distinct calls drawn per function before the pool counts as exhausted.
MAX_CALLS_PER_FUNCTION = 32


@dataclass(frozen=True)
class XTask:
    """One graded prediction task with its ground truth, prompt, and split."""

    task_id: str
    direction: str
    code: str
    call: str
    answer: str
    split: str
    prompt: str
    template_name: str
    parameter: int


def _normalize_prompt(text: str) -> str:
    """Lowercase, collapse whitespace, and strip punctuation for exclusion checks."""
    return " ".join("".join(character if character.isalnum() else " " for character in text.lower()).split())


def _render_function(template_lines: tuple[str, ...], parameter: int) -> str:
    """One template family member: a short pure function over ints."""
    return "\n".join(line.format(k=parameter) for line in template_lines)


# One program shape per family — loops, slicing, containers, sorting, branches —
# toward CruxEval's diversity of traced constructs. Determinism of every family is
# execution-verified per task, not assumed.
TEMPLATE_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "mod_loop",
        (
            "def f(x):",
            "    total = 0",
            "    for step in range(x % {k} + 3):",
            "        total += step * {k}",
            "    return total % {k}",
        ),
    ),
    (
        "cycle_length",
        (
            "def f(x):",
            "    seen = set()",
            "    value = x % {k}",
            "    while value not in seen and len(seen) < {k}:",
            "        seen.add(value)",
            "        value = (value * 7 + {k}) % {k}",
            "    return len(seen)",
        ),
    ),
    (
        "digit_weights",
        (
            "def f(x):",
            "    digits = [int(d) for d in str(abs(x))]",
            "    return sum(d * {k} for d in digits) % 97",
        ),
    ),
    (
        "reverse_slice",
        (
            "def f(x):",
            "    return int(str(abs(x))[::-1][: {k}]) % {k}",
        ),
    ),
    (
        "square_list",
        (
            "def f(x):",
            "    values = [i * i % {k} for i in range(abs(x) % {k} + 2)]",
            "    return (sum(values) - max(values)) % {k}",
        ),
    ),
    (
        "sorted_pairs",
        (
            "def f(x):",
            "    pairs = sorted(((i * {k}) % 13, i) for i in range(x % {k} + 3))",
            "    return pairs[0][1] + pairs[-1][0]",
        ),
    ),
    (
        "char_counts",
        (
            "def f(x):",
            "    counts = {{}}",
            "    for ch in str(abs(x)):",
            "        counts[ch] = counts.get(ch, 0) + 1",
            "    return sum(counts.values()) * {k}",
        ),
    ),
    (
        "binary_build",
        (
            "def f(x):",
            "    out = ''",
            "    for i in range(1, x % {k} + 3):",
            "        out += str(i % 2)",
            "    return int(out) % {k}",
        ),
    ),
    (
        "digit_descent",
        (
            "def f(x):",
            "    n = abs(x)",
            "    steps = 0",
            "    while n > 0:",
            "        n //= {k}",
            "        steps += 1",
            "    return steps * {k} - steps",
        ),
    ),
    (
        "divmod_branch",
        (
            "def f(x):",
            "    a, b = divmod(abs(x), {k})",
            "    return (a + b) % {k} if a > b else (a * b + 1) % {k}",
        ),
    ),
    (
        "sign_chain",
        (
            "def f(x):",
            "    if x % {k} == 0:",
            "        return {k} * 2 - x % 3",
            "    if x % 3 == 0:",
            "        return x // 3 + {k}",
            "    return x % 7 + {k} // 2",
        ),
    ),
    (
        "length_filter",
        (
            "def f(x):",
            "    words = ['a' * (i + 1) for i in range({k})]",
            "    total = 0",
            "    for i, w in enumerate(words):",
            "        if len(w) > x % {k}:",
            "            total += i",
            "    return total",
        ),
    ),
)

_F_CALL = re.compile(r"(?<![\w.])f\(")


def _prompt_text(direction: str, *, code: str, call: str, answer: str) -> str:
    """The user prompt for one X task, by direction.

    The prompt never shows what the model must produce: an output task shows the call
    but not the value; an input task shows the target value but not the reference call.
    """
    header = f"Consider the following Python code.\n\n```python\n{code.strip()}\n```\n\n"
    if direction == DIRECTION_INPUT:
        return (
            header
            + f"Find an input `x` such that `f(x)` returns exactly {answer} (as a Python literal).\n"
            + "Trace the code carefully, then end your reply with the call `f(x)` on its own line."
        )
    if direction == DIRECTION_OUTPUT:
        return (
            header
            + f"What is the value of `{call}`? Trace the code carefully, then give the exact value as\n"
            + "a Python literal, as `repr` would print it, in \\boxed{}."
        )
    raise ValueError(f"unknown X direction {direction!r}")


def x_prompt(task: XTask) -> str:
    """The canonical user prompt for one X task."""
    return _prompt_text(task.direction, code=task.code, call=task.call, answer=task.answer)


def function_split(seed: int, template_name: str, parameter: int) -> str:
    """The split of one function, a seeded hash of its ``(template family, k)`` identity."""
    digest = hashlib.sha256(f"{seed}:{template_name}:{parameter}".encode()).hexdigest()
    return PROBE_SPLIT if int(digest[:8], 16) % 100 < _PROBE_PERCENT else TRAIN_SPLIT


def synthesize_x_tasks(*, train_count: int, probe_count: int, seed: int) -> list[XTask]:
    """Synthesize execution-graded prediction tasks until both split quotas are met.

    A function's split is fixed by :func:`function_split`, so train and probe never
    share a function. Passes over the shuffled function pool each draw one fresh call
    per function, until both quotas fill. Each call is executed through the
    curriculum's trace harness; calls whose execution errors, times out, or produces a
    non-literal value are dropped, as are repeats of a call already drawn, so every
    shipped task has an execution-verified ground truth.
    """
    generator = random.Random(seed)
    catalog = [(name, lines, parameter) for name, lines in TEMPLATE_FAMILIES for parameter in PARAMETER_RANGE]
    generator.shuffle(catalog)
    quotas = {TRAIN_SPLIT: train_count, PROBE_SPLIT: probe_count}
    tasks: list[XTask] = []
    drawn: set[tuple[str, int, str]] = set()
    for _ in range(MAX_CALLS_PER_FUNCTION):
        for name, lines, parameter in catalog:
            split = function_split(seed, name, parameter)
            if quotas[split] == 0:
                continue
            code = _render_function(lines, parameter)
            call = f"f({generator.randint(-10_000, 10_000)})"
            direction = DIRECTION_OUTPUT if generator.random() < 0.5 else DIRECTION_INPUT
            if (name, parameter, call) in drawn:
                continue
            drawn.add((name, parameter, call))
            answer, _ = trace_value(code, call, timeout=TRACE_TIMEOUT_SECONDS)
            if answer is None:
                continue
            task_digest = hashlib.sha256(f"{name}:{parameter}:{call}".encode()).hexdigest()[:16]
            tasks.append(
                XTask(
                    task_id=f"x-{task_digest}",
                    direction=direction,
                    code=code,
                    call=call,
                    answer=answer,
                    split=split,
                    prompt=_prompt_text(direction, code=code, call=call, answer=answer),
                    template_name=name,
                    parameter=parameter,
                )
            )
            quotas[split] -= 1
        if not any(quotas.values()):
            return tasks
    raise ValueError(
        f"the function pool ran out with {quotas} tasks still to draw; widen TEMPLATE_FAMILIES or PARAMETER_RANGE"
    )


def extract_input_call(completion: str, *, answer: str) -> str | None:
    """The candidate's ``f(...)`` call, parsed and validated.

    The call must parse as a genuine single-argument call of the function under test.
    A completion whose call argument just echoes the shown output literal is rejected:
    for input tasks the output is visible in the prompt, so echoing it back is the
    trivially rewarded non-answer the grader must not accept. The last valid call wins,
    since the prompt asks the model to end its reply with the call.
    """
    last_valid = None
    for match in _F_CALL.finditer(completion):
        open_paren = match.end() - 1
        close = None
        depth = 0
        for index in range(open_paren, len(completion)):
            if completion[index] == "(":
                depth += 1
            elif completion[index] == ")":
                depth -= 1
                if depth == 0:
                    close = index
                    break
        if close is None:
            continue
        call = completion[match.start() : close + 1]
        try:
            parsed = ast.parse(call, mode="eval").body
        except SyntaxError:
            continue
        if not (
            isinstance(parsed, ast.Call)
            and isinstance(parsed.func, ast.Name)
            and parsed.func.id == "f"
            and len(parsed.args) == 1
            and not parsed.keywords
        ):
            continue
        argument = completion[open_paren + 1 : close].strip()
        if not argument:
            continue
        if argument == answer.strip() or literals_equal(answer, argument):
            continue
        last_valid = call
    return last_valid


def grade_x_task(task: dict[str, Any], completion: str) -> bool:
    """Grade one completion against an X task, dispatched on the task's direction.

    Output tasks take the completion's last ``\\boxed{...}``, which the prompt asks
    for, and compare it with the reference value as literals; a completion without one
    fails. Grading is never ``eval`` of the completion itself, under which a bare echo
    of the value used to pass. Input tasks require a genuine ``f(...)`` call that
    differs from trivially echoing the shown output, and grade by executing the call.
    """
    direction = task["direction"]
    answer = str(task["answer"])
    if direction == DIRECTION_OUTPUT:
        boxed = extract_boxed(completion)
        return boxed is not None and literals_equal(answer, boxed.strip())
    if direction == DIRECTION_INPUT:
        call = extract_input_call(completion, answer=answer)
        if call is None:
            return False
        value, _ = trace_value(task["code"], call, timeout=TRACE_TIMEOUT_SECONDS)
        return value is not None and literals_equal(answer, value)
    raise ValueError(f"unknown X direction {direction!r}")


def exclude_public_cruxeval(tasks: list[XTask], cruxeval_prompts_path: str | Path) -> tuple[list[XTask], int]:
    """Drop tasks whose normalized function code appears inside a public CruxEval prompt.

    The prompt file is a required build input: a missing file raises rather than
    silently skipping the exclusion. CruxEval prompts wrap the function in prose, so
    the comparison is containment of the task's normalized function code inside a
    prompt's normalized text — whole-text equality would never collide and the count
    would always be zero.
    """
    path = Path(cruxeval_prompts_path)
    if not path.exists():
        raise FileNotFoundError(f"required CruxEval exclusion input is missing: {path}")
    prompts = [
        normalized
        for line in path.read_text().splitlines()
        if line.strip() and (normalized := _normalize_prompt(json.loads(line)["prompt"]))
    ]
    kept: list[XTask] = []
    excluded = 0
    for task in tasks:
        code = _normalize_prompt(task.code)
        if code and any(code in prompt for prompt in prompts):
            excluded += 1
            continue
        kept.append(task)
    return kept, excluded


def write_split(tasks: list[XTask], output_dir: Path) -> dict[str, int]:
    """Write train and probe jsonl files; returns the per-split counts."""
    counts = {TRAIN_SPLIT: 0, PROBE_SPLIT: 0}
    for split in (TRAIN_SPLIT, PROBE_SPLIT):
        with (output_dir / f"x_{split}.jsonl").open("w") as handle:
            for task in tasks:
                if task.split != split:
                    continue
                handle.write(json.dumps(asdict(task), sort_keys=True) + "\n")
                counts[split] += 1
    if counts[TRAIN_SPLIT] == 0 or counts[PROBE_SPLIT] == 0:
        raise ValueError(f"splits must both be nonempty, got {counts}")
    return counts
