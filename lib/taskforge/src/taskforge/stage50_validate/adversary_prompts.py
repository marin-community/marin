# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The adversary's verdict markers and the short text its brief is filled with.

``AdversaryPrompts.brief`` is the brief of both modes; ``stage50_validate.adversary.adversary_brief`` fills it with
these. A white-box brief adds ``AdversaryPrompts.whitebox`` and the grader, the controls and the example answers as
its ``context``, and ``WHITEBOX_LINE_CLAUSE`` to the verdict line, so the two modes differ only in what the adversary
knows.
"""

NO_SHORTCUT_LINE = "NO_SHORTCUT"
SHORTCUT_PREFIX = "SHORTCUT:"
GRADER_LINE_MARKER = "grader line:"

TOOLS_WITH_SHELL = "two tools. `shell` runs commands in a private copy of the task's workspace that only you see"
TOOLS_SUBMIT_ONLY = "one tool"
WHITEBOX_LINE_CLAUSE = f", then `{GRADER_LINE_MARKER}` and the exact grader line or expression your submission defeats"
NO_VERIFIER_FILES = "(no verifier files: the grader is the specification alone)"
NO_CONTROLS = "(no controls recorded)"
NO_EXAMPLES = "(none beyond the task text)"
VERIFIER_FAILED = "the verifier failed on this candidate"
