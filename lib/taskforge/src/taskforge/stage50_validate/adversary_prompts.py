# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The short text the adversary's brief and tools are filled with.

``adversary.brief`` is the brief of every adversary and both modes; ``stage50_validate.adversary.adversary_brief``
fills it, with an ``AdversarySpec``'s ``brief`` as its ``goal``. Every adversary reports through the same
``finish_search`` contract, so the goal is the only text that differs between them. The default specs' goals are
``adversary.false_accept``, ``adversary.false_reject`` and ``adversary.shortcut``. A white-box brief adds
``adversary.whitebox`` and the grader, the controls and the example answers as its ``context``.
"""

TOOLS_WITH_SHELL = "`submit`, `finish_search`, and `shell` in a private copy of the task workspace"
TOOLS_SUBMIT_ONLY = "`submit` and `finish_search`"
NO_VERIFIER_FILES = "(no verifier files: the grader is the specification alone)"
NO_CONTROLS = "(no controls recorded)"
NO_EXAMPLES = "(none beyond the task text)"
VERIFIER_FAILED = "the verifier failed on this candidate"
