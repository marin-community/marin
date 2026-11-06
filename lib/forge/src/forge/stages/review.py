# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from forge.config import ForgeConfig
from forge.effects.context import Context
from forge.model.workpiece import Workpiece
from forge.resources.bundle import Resources
from forge.stages.exits import Decided


async def review(_ctx: Context, workpiece: Workpiece, config: ForgeConfig, _resources: Resources) -> Decided:
    if not workpiece.evidence:
        raise ValueError("Review requires validation evidence")
    result = workpiece.evidence[-1]
    if workpiece.submission is None or result.task_digest != workpiece.submission.digest:
        raise ValueError("Validation evidence does not cover the current submission")
    return Decided(config.review(result, config.band))
