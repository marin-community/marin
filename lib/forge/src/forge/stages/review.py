# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from forge.config import ForgeConfig
from forge.effects.context import Context
from forge.model.review import Modify
from forge.model.workpiece import Workpiece
from forge.resources.bundle import Resources
from forge.stages.exits import Decided


async def review(_ctx: Context, workpiece: Workpiece, config: ForgeConfig, _resources: Resources) -> Decided:
    if not workpiece.evidence:
        raise ValueError("Review requires validation evidence")
    result = workpiece.evidence[-1]
    if workpiece.submission is None or result.task_digest != workpiece.submission.digest:
        raise ValueError("Validation evidence does not cover the current submission")
    decision = config.review(workpiece, config.band, config.budget)
    notes = tuple(f"{trial.name}: {reason}" for trial in result.adversary if trial.completed for reason in trial.reasons)
    if notes:
        decision = (
            replace(decision, findings=(*decision.findings, *notes))
            if isinstance(decision, Modify)
            else replace(decision, reasons=(*decision.reasons, *notes))
        )
    return Decided(decision)
