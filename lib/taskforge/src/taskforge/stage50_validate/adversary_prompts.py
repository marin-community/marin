# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The adversary's prompt text: its brief, the white-box section, and its tools.

``BRIEF_TEMPLATE`` is the brief of every adversary and both modes; ``stage50_validate.adversary.adversary_brief``
fills it, with an ``AdversarySpec``'s ``brief`` as its ``goal``. Every adversary reports through the same
``finish_search`` contract, so the goal is the only text that differs between them. The ``*_GOAL`` constants are the
default specs' goals; they are plain strings so they can move to prompt data files unchanged.
"""

FALSE_ACCEPT_GOAL = "Find substantively wrong, incomplete or evasive answers that receive full credit (reward 1)."
FALSE_REJECT_GOAL = "Find correct and complete answers that receive less than full credit (reward below 1)."
SHORTCUT_GOAL = (
    "Find candidates the grader accepts that violate the spirit of the task or do not need the task's intended "
    "computation to produce: output that only satisfies the format (several candidate values, a placeholder, extra "
    "text, another unit or number format, partial output), lenient or partial matching, a check that re-derives the "
    "expected value from files you can change, a grader or fixture you can overwrite, an answer leaked by the "
    "instruction, the environment or a fixture, or a defensible reading of the instruction under which a different "
    "answer is accepted."
)
"""The original shortcut audit's goal, kept as a spec so a roster can run it alone or beside false accepts."""

BRIEF_TEMPLATE = (
    "You are auditing the task's grader. {goal} You have {tools}. `submit` grades the final reply and "
    "every file you list on a fresh grading machine. Your private shell state is not submitted. "
    "Follow the task's actual delivery and format requirements. The reference can be wrong; the task "
    "contract determines correctness. Efficient solutions and permitted alternative formats are not defects. "
    "You have {submissions} submissions. Test at least {probes} plausible candidates. "
    "Then call finish_search with defects: each has submission_ids, explanation, and needs_addressing. "
    "Use an empty defects list if none were found, and summarize your search. Probe candidates are findings only "
    "when explicitly submitted as defects. "
    "A prose answer without finish_search is incomplete.{context}"
)

TOOLS_WITH_SHELL = "`submit`, `finish_search`, and `shell` in a private copy of the task workspace"
TOOLS_SUBMIT_ONLY = "`submit` and `finish_search`"
WHITEBOX_HEADER = (
    "What you additionally know (white-box audit): the grader's specification and full source, every control "
    "submission the author wrote with the reward the grader is expected to give it (the reference answer is the "
    "positive control with concern `reference`), and the task's example answers. Read the grader source first: "
    "find the exact checks, regular expressions, tolerances, parsing and partial-credit rules relevant to your goal."
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
