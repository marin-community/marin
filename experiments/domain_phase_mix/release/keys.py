# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public content ids: hashes of a mixture's weights, of an exposure and of a two-phase schedule.

A run's observation id is its W&B run id, which the exporters read from their sources; nothing here computes it.
"""

import hashlib
from collections.abc import Mapping

ROUND = 9


def _canon(values: Mapping[str, float]) -> str:
    # Adding 0.0 turns the -0.0 that rounding leaves from tiny negative noise into 0.0.
    return ";".join(f"{k}={round(float(v), ROUND) + 0.0:.{ROUND}f}" for k, v in sorted(values.items()))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def mixture_id(weights: Mapping[str, float]) -> str:
    return "mix_" + _sha(_canon(weights))[:16]


def exposure_id(weights: Mapping[str, float], pool_fractions: Mapping[str, float], partition_revision: str) -> str:
    return "exp_" + _sha(f"{partition_revision}|{_canon(weights)}|{_canon(pool_fractions)}")[:16]


def schedule_id(w0: Mapping[str, float], w1: Mapping[str, float], boundary_step: int, total_steps: int) -> str:
    return "sch_" + _sha(f"{_canon(w0)}|{_canon(w1)}|{boundary_step}/{total_steps}")[:16]
