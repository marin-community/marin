# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Simulated-materialized epoch accounting for the StarCoder WSD80 panels."""

from __future__ import annotations

from dataclasses import dataclass

SIMULATED_EPOCH_TARGET_BUDGET = 5_729_908_864_777
NEMOTRON_SOURCE_TOKENS = 5_729_908_864_777
STARCODER_SOURCE_TOKENS = 216_567_300_822
PHASE_0_FRACTION = 0.8
PHASE_1_FRACTION = 1.0 - PHASE_0_FRACTION


@dataclass(frozen=True)
class DomainEpochs:
    """Effective passes over one source's simulated subset."""

    phase_0: float
    phase_1: float

    @property
    def total(self) -> float:
        return self.phase_0 + self.phase_1


@dataclass(frozen=True)
class StarCoderWsd80Epochs:
    """Effective simulated epochs for both sources in one 80/20 policy."""

    starcoder: DomainEpochs
    nemotron: DomainEpochs


def simulated_materialized_epochs(
    phase_0_starcoder: float,
    phase_1_starcoder: float,
) -> StarCoderWsd80Epochs:
    """Return effective epochs under the panel's fixed target-budget slicing."""
    if not 0.0 <= phase_0_starcoder <= 1.0 or not 0.0 <= phase_1_starcoder <= 1.0:
        raise ValueError("StarCoder phase weights must lie in [0, 1]")

    starcoder_multiplier = SIMULATED_EPOCH_TARGET_BUDGET / STARCODER_SOURCE_TOKENS
    nemotron_multiplier = SIMULATED_EPOCH_TARGET_BUDGET / NEMOTRON_SOURCE_TOKENS
    return StarCoderWsd80Epochs(
        starcoder=DomainEpochs(
            phase_0=PHASE_0_FRACTION * phase_0_starcoder * starcoder_multiplier,
            phase_1=PHASE_1_FRACTION * phase_1_starcoder * starcoder_multiplier,
        ),
        nemotron=DomainEpochs(
            phase_0=PHASE_0_FRACTION * (1.0 - phase_0_starcoder) * nemotron_multiplier,
            phase_1=PHASE_1_FRACTION * (1.0 - phase_1_starcoder) * nemotron_multiplier,
        ),
    )
