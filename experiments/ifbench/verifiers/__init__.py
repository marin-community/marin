# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Vendored IFBench + IFEvalG verifier registries.

Exports INSTRUCTION_DICT_ALL: the union of both registries. Constraint IDs
are namespaced and disjoint by construction (IFBench uses `count:`, `ratio:`,
`words:`, `custom:` etc.; IFEvalG uses `keywords:`, `length_constraints:`,
`detectable_format:`, `paragraphs:`, `last_word:`, etc.). The union
collapses to a single dict with no key collisions.
"""

from .ifbench import INSTRUCTION_DICT as IFBENCH_DICT
from .ifevalg import INSTRUCTION_DICT as IFEVALG_DICT

_overlap = set(IFBENCH_DICT) & set(IFEVALG_DICT)
assert not _overlap, f"unexpected overlap between IFBench and IFEvalG registries: {_overlap}"

INSTRUCTION_DICT_ALL: dict = {**IFBENCH_DICT, **IFEVALG_DICT}

__all__ = ["IFBENCH_DICT", "IFEVALG_DICT", "INSTRUCTION_DICT_ALL"]
