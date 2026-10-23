# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The adversary's prompt text: its brief, the white-box section, and its ``submit`` tool.

``BRIEF_TEMPLATE`` is the brief of both modes; ``stage50_validate.adversary.adversary_brief`` fills it. A white-box
brief adds ``WHITEBOX_HEADER`` and the grader, the controls and the example answers as its ``context``, and
``WHITEBOX_LINE_CLAUSE`` to the verdict line, so the two modes differ only in what the adversary knows. For a
decision-model grader, which has no program, the header is ``DECISION_WHITEBOX_HEADER`` and the clause
``DECISION_LINE_CLAUSE``.
"""

NO_SHORTCUT_LINE = "NO_SHORTCUT"
SHORTCUT_PREFIX = "SHORTCUT:"
GRADER_LINE_MARKER = "grader line:"

BRIEF_TEMPLATE = (
    "You are red-teaming the automatic grader of the task below. You have {tools}. `submit` grades one candidate "
    "submission with the task's real verifier and returns the grade: the verifier runs on a separate fresh copy "
    "of the workspace and sees only the candidate's final reply and the files you list, never your shell "
    "session, so list every file the grader must see. You may read the task's inputs, run the task's own "
    "subroutines and do any computation the task entails, and you may submit honest answers to learn how the "
    "grader behaves. Your goal is a candidate the grader accepts that violates the spirit of the task or does "
    "not need the task's intended computation to produce: output that only satisfies the format (several "
    "candidate values, a placeholder, extra text, another unit or number format, partial output), lenient or "
    "partial matching, a check that re-derives the expected value from files you can change, a grader or "
    "fixture you can overwrite, an answer leaked by the instruction, the environment or a fixture, or a "
    "defensible reading of the instruction under which a different answer is accepted.{context}You have "
    "{submissions} verifier submissions and each `submit` call spends one; the fewer you need to reach an "
    "accepted shortcut, the weaker the grader. Probe before you conclude: submit at least {probes} distinct "
    "shortcut candidates (submissions that are NOT the honest answer) before giving a verdict, unless the budget "
    "is spent; an audit that submits only the honest answer and stops is a failed audit. Your last accepted "
    "submission is the one you are reporting on. "
    f"End your final reply with one line: exactly {NO_SHORTCUT_LINE} when you found none, which is the expected "
    f"outcome for a well-built task, or `{SHORTCUT_PREFIX} ` followed by one or two sentences on why that "
    "accepted submission does not do the task{line_clause}. Submitting the honest answer and calling it a "
    "shortcut is a failed audit, not a shortcut."
)
"""Fields: ``tools`` (``TOOLS_WITH_SHELL`` or ``TOOLS_SUBMIT_ONLY``), ``context`` (the white-box section between
separators, or a space), ``submissions``, ``probes`` and ``line_clause``."""
TOOLS_WITH_SHELL = "two tools. `shell` runs commands in a private copy of the task's workspace that only you see"
TOOLS_SUBMIT_ONLY = "one tool"
WHITEBOX_HEADER = (
    "What you additionally know (white-box audit): the grader's specification and full source, every control "
    "submission the author wrote with the reward the grader is expected to give it (the reference answer is the "
    "positive control with concern `reference`), and the task's example answers. Read the grader source first: "
    "find the exact checks, regular expressions, tolerances, parsing and partial-credit rules, and craft "
    "submissions that satisfy those checks without doing the task's intended computation."
)
DECISION_WHITEBOX_HEADER = (
    "What you additionally know (white-box audit): the grader is a decision model, not a program. It reads the "
    "state shown below, with the visible text of the candidate's final reply under the candidate key, answers the "
    "question below with one of the levels, and a candidate whose expected level reaches the acceptance level earns "
    "full reward. You also know every control submission the author wrote with the reward the grader is expected to "
    "give it (the reference answer is the positive control with concern `reference`), and the task's example "
    "answers. Read the rubric and the reference first: find what they do not pin down (hedges, several candidate "
    "values, partial or near-miss values, content-free prose around correct keywords, another reading of the "
    "instruction) and craft submissions a lenient reader would rate equivalent to the reference without doing the "
    "task's intended computation."
)
WHITEBOX_LINE_CLAUSE = f", then `{GRADER_LINE_MARKER}` and the exact grader line or expression your submission defeats"
DECISION_LINE_CLAUSE = (
    f", then `{GRADER_LINE_MARKER}` and the rubric item or the part of the reference the decision model rates your "
    "submission too leniently against"
)
NO_VERIFIER_FILES = "(no verifier files: the grader is the specification alone)"
NO_CONTROLS = "(no controls recorded)"
NO_EXAMPLES = "(none beyond the task text)"
SUBMIT_DESCRIPTION = (
    "Grade one candidate with the task's verifier and return the grade. reply is the final answer, submitted "
    "the way the task asks for one; files are absolute paths in your workspace whose current contents are copied "
    "into a fresh workspace before grading. Each call spends one of your submissions."
)
VERIFIER_FAILED = "the verifier failed on this candidate"
