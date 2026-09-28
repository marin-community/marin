# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""The paper's Uncheatable objective: fixed byte-share weights on the seven component BPBs.

Runs evaluated with BPB schema 2 (`eval/bpb_schema_version: 2`, logged from about 12 September 2026) report an
`eval/uncheatable_eval/bpb` that pools loss bits over scored bytes and reads 0.005 to 0.013 BPB above this weighting;
on legacy payloads the two agree to 1e-7. Score every run's Uncheatable loss with `fixed_uncheatable_bpb`.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

UNCHEATABLE_FIT = (
    Path(__file__).resolve().parent
    / "reference_outputs"
    / "delphi_frozen_procedure_validation_3e18_20260908"
    / "fits"
    / "fit_uncheatable.json"
)
RAW_UNCHEATABLE_KEY = "eval/uncheatable_eval/bpb"
BPB_SCHEMA_KEY = "eval/bpb_schema_version"


def uncheatable_weights() -> dict[str, float]:
    """Byte-share weight of each component, keyed by its metric (`eval/uncheatable_eval/<component>/bpb`)."""
    fit = json.loads(UNCHEATABLE_FIT.read_text())
    return {task["component"]: float(weight) for task, weight in zip(fit["tasks"], fit["task_weights"], strict=True)}


def fixed_uncheatable_bpb(metrics: Mapping[str, Any]) -> float:
    """Uncheatable BPB of one evaluation payload, from its seven component BPBs."""
    return sum(weight * float(metrics[key]) for key, weight in uncheatable_weights().items())
