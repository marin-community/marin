# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Role-specific search instructions."""

SEARCH_GOALS = {
    "false_accept": "Find substantively wrong, incomplete or evasive answers that receive full credit (reward 1).",
    "false_reject": "Find correct and complete answers that receive less than full credit (reward below 1).",
}

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
    "find the exact checks, regular expressions, tolerances, parsing and partial-credit rules relevant to your role."
)
DECISION_WHITEBOX_HEADER = (
    "What you additionally know (white-box audit): the grader is a decision model, not a program. It reads the "
    "state shown below, with the visible text of the candidate's final reply under the candidate key, answers the "
    "question below with one of the levels, and a candidate whose expected level reaches the acceptance level earns "
    "full reward. You also know every control submission the author wrote with the reward the grader is expected to "
    "give it (the reference answer is the positive control with concern `reference`), and the task's example "
    "answers. Audit the rubric and reference against the task contract for the assigned search role."
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
