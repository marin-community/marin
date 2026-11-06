# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from forge.config import ForgeConfig
from forge.effects.context import Context
from forge.model.review import Accept, BandOutcome, Modify
from forge.model.workpiece import Workpiece
from forge.resources.bundle import Resources
from forge.stages.exits import Decided


async def review(_ctx: Context, workpiece: Workpiece, config: ForgeConfig, _resources: Resources) -> Decided:
    if not workpiece.evidence:
        raise ValueError("Review requires validation evidence")
    result = workpiece.evidence[-1]
    if workpiece.submission is None or result.task_digest != workpiece.submission.digest:
        raise ValueError("Validation evidence does not cover the current submission")
    decision = config.review(result, config.band)
    if workpiece.too_easy_baseline is not None and isinstance(decision, Modify):
        reverted = decision.placement is not BandOutcome.TOO_EASY
        reasons = ("accepted_with: too_easy",)
        if reverted:
            reasons += ("Hardening regressed; reverted to the previously validated submission", *decision.findings)
        decision = Accept(BandOutcome.TOO_EASY, reasons, revert=reverted)
    return Decided(decision)
