# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checks and grading settings for code tasks whose hidden tests use the LiveCodeBench test-case layout."""

import ast
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from taskcompendium.convert.answers import source_defect
from taskcompendium.pipeline.models import ImportRejection

CODE_GRADER_MEMORY_MB = 5120
"""The LiveCodeBench child process may use 4 GiB; the grading machine also needs room for its runtime."""
THREAD_ENVIRONMENT = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
"""Run each test's numeric libraries on one thread."""
CODE_BLOCK = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
"""A fenced code block as the source scorers match one; they run the last block in a reply."""
_REPLY_CHOICE = (
    r"(\b(print|output|write|display|give|report|return|find|describe|choose|pick|select|use|satisfied|fine\s+with)\b"
    r"[^.\n]{0,20}\b(any|either)\b"
    r"|\b(any|either)\b[^.\n]{0,40}\b(accepted|acceptable|allowed|suitable|will do|suffices?"
    r"|(be|is|are)\s+(\w+\s+)?(printed|output|given|correct|fine|ok))\b)"
)
"""An instruction that lets the reply pick: a verb of output or choice before "any" ("print any of them", "he is fine
with any of them") or an acceptance after it ("any of them will be accepted"). Without one, "any" quantifies the input
("multiple edges between any pair of vertices")."""
_ANY_VALID_OUTPUT = re.compile(
    r"\b(if|when|in case)\b[^.\n]{0,40}\b(several|multiple|many|more than one)\b[^.\n]{0,80}"
    + _REPLY_CHOICE
    + r"|\b(several|multiple|many|more than one)\b[^.\n]{0,30}\b(answers?|solutions?|ways?|outputs?|results?)\b"
    r"[^.\n]{0,60}"
    + _REPLY_CHOICE
    + r"|\b(print|output|find|return)\s+any\s+(one\s+)?(of\s+(them|these)|valid|correct|possible|such|solution|answer)\b"
    r"|(?P<any_object>\bany\s+(valid|correct)\s+(answer|solution|output|sequence|permutation|arrangement|assignment"
    r"|construction)\b)"
    r"|\b(answers?|solutions?)\s+(is|are)\s+not\s+unique\b",
    re.IGNORECASE,
)
_EXISTENCE_BEFORE = re.compile(r"\b(if|whether|there\s+(is|are|exists?))\b[^.\n]*$", re.IGNORECASE)
"""A condition or existence claim earlier in the sentence, which makes "any valid ..." a quantifier rather than a
choice the reply makes ("Print YES if there is any valid solution, otherwise print NO")."""
_ONE_ANSWER_CHOSEN = re.compile(
    r"\blexicographically\s+(smallest|minimal|minimum|least|largest|maximal|maximum|greatest|first|last)\b",
    re.IGNORECASE,
)
"""A tie-breaker that orders whole answers. A partial one ("with the smallest first element") can leave several
answers, and a comparative ("a is lexicographically less than b") defines an order without choosing an answer."""
_FLOAT_TOLERANCE = re.compile(
    r"absolute or relative|relative or absolute"
    r"|error (does not|must not|should not|doesn't) exceed"
    r"|within (an? )?(error|tolerance)|accurate to within",
    re.IGNORECASE,
)
"""A clause accepting any answer within a numeric tolerance, worded with or without the word "error"."""
_ERROR_MEASURE = re.compile(r"\b(absolute|relative) errors?\b", re.IGNORECASE)
"""An error measure, which states a tolerance only when its sentence bounds it ("an absolute error of at most
10^{-6}"), not when the reply computes it ("Print the absolute error |a-b|")."""
_SENTENCE_OR_LINE_END = re.compile(r"\.(\s|$)|\n")
_CORRECT_IF = re.compile(r"\bconsidered correct if\b", re.IGNORECASE)
_CORRECT_IF_WINDOW = 200
"""How far past "considered correct if" its condition is read; some conditions continue in a list."""
_NUMERIC_TOLERANCE = re.compile(
    r"10\s*\^\s*\{?\s*[-\u2212]|\b10\s*(<=)?[-\u2212](<=)?\s*\d|\de\s*[-\u2212]\s*\d|\b0\.\d"
    r"|decimal (places|digits)|digits after the decimal point|\bepsilon\b|[εϵ]",
    re.IGNORECASE,
)
"""A tolerance in a condition: a negative power of ten (also as scraped superscripts or with a Unicode minus), a
decimal fraction, a number of decimal places, or an epsilon."""
_REPLY_SUBJECT = re.compile(r"\b(answers?|outputs?|solutions?|submission)\b[^.\n]{0,25}$", re.IGNORECASE)
"""A reply as the subject that ends just before "considered correct if", as in "Your answer will be"."""
_SENTENCE_END = re.compile(r"\.(\s|$)")
_ACCEPTED_VARIATION = re.compile(
    r"\b(any|possibly|at (most|least)|no (more|less|fewer) than|(not|n't) exceed|between|satisf(y|ies)|consistent"
    r"|suitable|valid)\b",
    re.IGNORECASE,
)
"""Freedom that a condition leaves the reply: a choice ("any suitable m"), an order ("possibly reordered"), a bound
("at most 1.3·b + 100 mistakes") or a property to satisfy ("consistent with s")."""
_INTERACTIVE = re.compile(r"(?P<negation>(\bnot|n't)\s+(an?\s+)?|\bnon-?\s*)?\binteractive problem\b", re.IGNORECASE)
"""An interactive-problem notice, or its negation ("This is NOT an interactive problem", "the usual, non-interactive
problem")."""
_FUNCTION_TEMPLATE = re.compile(
    r"complete the (given )?(function|method|class)(?![^.\n]{0,100}\bread(s|ing)?\b)", re.IGNORECASE
)
"""A prompt for a function body rather than a program, unless that sentence has the function read the input
("Complete the function solve() to read an integer and print its square")."""
_NO_INPUT_NOTICE = re.compile(
    r"you (don'?t|do not) (need|have) to (read|take) (any )?input|no need to read input", re.IGNORECASE
)
_FUNCTION_WORD = re.compile(r"\b(functions?|methods?)\b", re.IGNORECASE)
_FUNCTION_WINDOW = 200
"""How far from a no-input notice a function or method must be named for the notice to ask for a function body; a
program that reads nothing gets the same notice ("You don't need to read input. Print 42.")."""
_INPUT_FORMAT = re.compile(r"input format|-+\s*input\s*-+", re.IGNORECASE)


class CodeTestMode(StrEnum):
    """How hidden tests run a reply: on standard input, or by calling one function."""

    STDIN = "stdin"
    FUNCTIONAL = "functional"


def code_case_mode(cases: Mapping[str, Any] | list[Any]) -> CodeTestMode:
    """The test cases' mode. Raises for a layout the LiveCodeBench scorer cannot normalize."""
    if isinstance(cases, Mapping) and "test_cases" in cases:
        if cases.get("language") not in (None, "python"):
            raise ValueError("LCB supports only Python verification specs")
        cases = cases["test_cases"]
    elif isinstance(cases, Mapping):
        inputs, outputs = cases.get("inputs"), cases.get("outputs")
        if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs or len(inputs) != len(outputs):
            raise ValueError("LCB requires aligned nonempty input and output lists")
        kind = "functional" if cases.get("fn_name") is not None else "stdin"
        cases = [
            {"input": item, "output": result, "testtype": kind, "fn_name": cases.get("fn_name")}
            for item, result in zip(inputs, outputs, strict=True)
        ]
    if not isinstance(cases, list) or not cases or not all(isinstance(case, Mapping) for case in cases):
        raise ValueError("LCB requires a nonempty list of test cases")
    modes = set()
    names = set()
    for case in cases:
        kind = case.get("testtype", case.get("type"))
        if kind in ("stdin", "stdin_stdout"):
            if not isinstance(case.get("input"), str) or not isinstance(case.get("output"), str):
                raise ValueError("Standard-input LCB cases require string input and output")
            modes.add(CodeTestMode.STDIN)
        elif kind in ("functional", "call_based"):
            metadata = case.get("metadata")
            name = case.get("fn_name") or (metadata.get("func_name") if isinstance(metadata, Mapping) else None)
            if not isinstance(name, str) or not name:
                raise ValueError("Functional LCB cases require a function name")
            if not isinstance(case.get("input"), str | list):
                raise ValueError("Functional LCB input must be a list or encoded string")
            modes.add(CodeTestMode.FUNCTIONAL)
            names.add(name)
        else:
            raise ValueError("Unsupported LCB test-case type")
    if len(modes) != 1 or len(names) > 1:
        raise ValueError("LCB cases must share one execution mode and function name")
    return modes.pop()


class _Correctness(StrEnum):
    """What a "considered correct if" clause makes the grader check."""

    TOLERANCE = "tolerance"
    CONDITION = "condition"


def _condition(statement: str, clause: re.Match[str]) -> str:
    """The condition of a "considered correct if" clause: the rest of its sentence, or the list after a colon."""
    window = statement[clause.end() : clause.end() + _CORRECT_IF_WINDOW]
    if window.lstrip().startswith(":"):
        return window
    end = _SENTENCE_END.search(window)
    return window if end is None else window[: end.start()]


def _correctness(statement: str) -> _Correctness | None:
    """What the statement's "considered correct if" clauses make the grader check, a tolerance first, or None.

    A condition with a numeric tolerance accepts nearby numbers. A condition that leaves the reply some freedom is
    tested by a checker ("Your answer will be considered correct if you output any suitable m"). A condition that
    fixes the reply ("Your answer will be considered correct if it equals n*n") or a clause that defines a term
    ("A bracket sequence is considered correct if") says nothing that one stored output cannot check.
    """
    result = None
    for clause in _CORRECT_IF.finditer(statement):
        condition = _condition(statement, clause)
        if _NUMERIC_TOLERANCE.search(condition):
            return _Correctness.TOLERANCE
        if _REPLY_SUBJECT.search(statement, 0, clause.start()) and _ACCEPTED_VARIATION.search(condition):
            result = _Correctness.CONDITION
    return result


def _is_interactive(statement: str) -> bool:
    return any(notice["negation"] is None for notice in _INTERACTIVE.finditer(statement))


def _bounded_error(statement: str) -> bool:
    """Whether an absolute or relative error has a numeric tolerance in its sentence or line."""
    for measure in _ERROR_MEASURE.finditer(statement):
        start = max(statement.rfind(". ", 0, measure.start()), statement.rfind("\n", 0, measure.start())) + 1
        end = _SENTENCE_OR_LINE_END.search(statement, measure.end())
        if _NUMERIC_TOLERANCE.search(statement, start, end.start() if end else len(statement)):
            return True
    return False


def _asks_for_function(statement: str) -> bool:
    """Whether the statement asks for a function body: "complete the function", or a notice that the reply reads no
    input with a function or method named nearby ("You don't need to read input. Complete sum2() function.")."""
    if _FUNCTION_TEMPLATE.search(statement):
        return True
    return any(
        _FUNCTION_WORD.search(statement, max(0, notice.start() - _FUNCTION_WINDOW), notice.end() + _FUNCTION_WINDOW)
        for notice in _NO_INPUT_NOTICE.finditer(statement)
    )


def _output_choice(statement: str) -> re.Match[str] | None:
    """The first wording that lets the reply choose among several outputs, or None."""
    for match in _ANY_VALID_OUTPUT.finditer(statement):
        if not (match["any_object"] and _EXISTENCE_BEFORE.search(statement, 0, match.start())):
            return match
    return None


def ungradable_statement(statement: str, mode: CodeTestMode) -> ImportRejection | None:
    """Reject a statement whose replies comparison with one stored output cannot grade, else return None."""
    any_valid = _output_choice(statement)
    correctness = _correctness(statement)
    if any_valid and not _ONE_ANSWER_CHOSEN.search(statement, any_valid.end(), any_valid.end() + 120):
        reason = "any_valid_output"
    elif _FLOAT_TOLERANCE.search(statement) or _bounded_error(statement) or correctness == _Correctness.TOLERANCE:
        reason = "float_tolerance"
    elif correctness == _Correctness.CONDITION:
        reason = "any_valid_output"
    elif _is_interactive(statement):
        reason = "interactive_problem"
    elif mode == CodeTestMode.STDIN and _asks_for_function(statement) and not _INPUT_FORMAT.search(statement):
        reason = "function_template_with_example_tests"
    else:
        return None
    return source_defect(reason, "Comparing with one stored output cannot grade replies to this statement")


def has_code_block(response: str) -> bool:
    """Match a fenced code block the way the source scorers extract one."""
    return CODE_BLOCK.search(response) is not None


def python_reply(solution: Any) -> str | None:
    """A source solution as a reply the scorers extract code from.

    Returns ``None`` when there is no solution or when the program the scorers would run is not
    Python 3. Some sources keep Python 2 solutions; the graders run Python 3, so such a solution
    cannot serve as a known-correct submission.
    """
    if not isinstance(solution, str) or not solution.strip():
        return None
    reply = solution if has_code_block(solution) else f"```python\n{solution}\n```"
    try:
        ast.parse(CODE_BLOCK.findall(reply)[-1].strip())
    except SyntaxError:  # Includes TabError from Python 2 tab indentation.
        return None
    return reply
