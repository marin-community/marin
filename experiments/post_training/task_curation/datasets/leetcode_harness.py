# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The shared LeetCode harness core: the source's own environment, solution and check().

``harness_reward`` is the single implementation both the shipped grader script and the
converter's reference qualification run, so the CPU check is the grader by construction.
The namespace is the source's own runtime: the native ``prompt`` field is the functional
preamble (imports plus the ListNode/TreeNode helpers and constants), compiled in every
native row; the candidate's last fenced block carries the complete solution class. The
truncated ``starter_code`` stays in the config as provenance and is never executed.
"""

import re


class HarnessDefect(Exception):
    """The source harness itself cannot run; an infrastructure error, never a reward."""


def harness_reward(config: dict, answer_text: str) -> int:
    """1 when every shipped harness assertion passes; 0 on candidate-code failures.

    Raises :class:`HarnessDefect` when the source's own preamble or ``check()`` harness
    fails to execute — those are source-harness defects, not candidate failures.
    """
    namespace: dict = {}
    try:
        exec(config["preamble"], namespace)
    except Exception as error:
        raise HarnessDefect(f"the source preamble fails to execute: {error}") from None
    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", answer_text, re.DOTALL)
    code = blocks[-1] if blocks else answer_text
    try:
        exec(code, namespace)
        candidate = eval(config["entry_point"], namespace)
    except Exception:
        return 0
    try:
        exec(config["test_source"], namespace)
        check = namespace["check"]
    except Exception as error:
        raise HarnessDefect(f"the source test harness fails to execute: {error}") from None
    if not callable(check):
        raise HarnessDefect("the source harness defines no callable check(candidate)")
    try:
        check(candidate)
    except Exception:
        return 0
    return 1
