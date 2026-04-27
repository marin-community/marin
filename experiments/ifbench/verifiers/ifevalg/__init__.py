# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IFEvalG: 25 IFEval + 29 IFTrain constraints (54 verifier classes total).

Vendored from allenai/open-instruct:open_instruct/IFEvalG/. See ../_versions.txt
for the pinned commit. Only modification vs upstream: import paths rewritten
from `from open_instruct.IFEvalG import ...` to relative `from . import ...`.

INSTRUCTION_DICT is a strict superset of Google's original IFEval registry,
so we don't separately vendor google-research/instruction_following_eval.
"""

from .instructions_registry import INSTRUCTION_DICT
