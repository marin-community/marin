# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-source crop verdicts from the 2026-09 TaskTrove audit.

``source_verdicts.json`` records, for each active TaskTrove source, whether its tasks are kept and
converted or dropped, with the reason. The pipeline reads it to skip dropped sources.
"""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

TASKTROVE_HF_ID = "open-thoughts/TaskTrove"
TASKTROVE_REVISION = "0292300"
"""v4.15 tree, 2026-09-08."""

_VERDICTS_PATH = Path(__file__).with_name("source_verdicts.json")


class SourceVerdict(StrEnum):
    KEEP = "keep"
    DROP = "drop"


@dataclass(frozen=True)
class SourceInfo:
    source: str
    verdict: SourceVerdict
    family: str
    reason: str


def load_source_verdicts() -> dict[str, SourceInfo]:
    raw = json.loads(_VERDICTS_PATH.read_text())
    return {s: SourceInfo(s, SourceVerdict(v["verdict"]), v["family"], v["reason"]) for s, v in raw.items()}
