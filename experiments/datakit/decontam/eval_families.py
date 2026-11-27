# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map decontamination eval record IDs to their benchmark families.

The Artificial Analysis corpora name their records with a benchmark-specific
subdirectory prefix; every other corpus falls back to the generic eval-ID
family splitter shared with the report viewers.
"""

import re

from experiments.datakit.decontam.prepare_eval_corpus import AA_EVALS

_SPLIT_RE = re.compile(r"^(.*)-(validation|test|training|train|dev|eval)-\d+$")
_UNCHEATABLE_ID_RE = re.compile(r"^uncheatable-(?P<subset>[a-z0-9_]+)-\d+$")
AA_SUBDIR_TO_NAME = {config.subdir: config.name for config in AA_EVALS}


def eval_id_to_family(eval_id: str) -> str:
    """The benchmark family of one generic eval record ID."""
    if uncheatable := _UNCHEATABLE_ID_RE.match(eval_id):
        return f"uncheatable_{uncheatable['subset']}"
    m = _SPLIT_RE.match(eval_id)
    return m.group(1) if m else eval_id


def eval_family(eval_id: str) -> str:
    """The benchmark family of one eval record ID, with the AA subdirectory names."""
    for subdir, name in AA_SUBDIR_TO_NAME.items():
        if eval_id == subdir or eval_id.startswith(f"{subdir}-"):
            return name
    return eval_id_to_family(eval_id)
