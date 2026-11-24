# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from taskforge.config import TaskforgeConfig
from taskforge.effects.context import Context
from taskforge.model.workpiece import Workpiece
from taskforge.resources.bundle import Resources
from taskforge.stages.exits import Decided


async def review(_ctx: Context, workpiece: Workpiece, config: TaskforgeConfig, _resources: Resources) -> Decided:
    if not workpiece.evidence:
        raise ValueError("Review requires validation evidence")
    result = workpiece.evidence[-1]
    if workpiece.submission is None or result.task_digest != workpiece.submission.digest:
        raise ValueError("Validation evidence does not cover the current submission")
    return Decided(config.review(workpiece, config.band, config.budget))
